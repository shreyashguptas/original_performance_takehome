"""
# Anthropic's Original Performance Engineering Take-home (Release version)

Copyright Anthropic PBC 2026. Permission is granted to modify and use, but not
to publish or redistribute your solutions so it's hard to find spoilers.

# Task

- Optimize the kernel (in KernelBuilder.build_kernel) as much as possible in the
  available time, as measured by test_kernel_cycles on a frozen separate copy
  of the simulator.

Validate your results using `python tests/submission_tests.py` without modifying
anything in the tests/ folder.

We recommend you look through problem.py next.
"""

from collections import defaultdict
import random
import unittest

from problem import (
    Engine,
    DebugInfo,
    SLOT_LIMITS,
    VLEN,
    N_CORES,
    SCRATCH_SIZE,
    Machine,
    Tree,
    Input,
    HASH_STAGES,
    reference_kernel,
    build_mem_image,
    reference_kernel2,
)


class KernelBuilder:
    def __init__(self):
        self.instrs = []
        self.scratch = {}
        self.scratch_debug = {}
        self.scratch_ptr = 0
        self.const_map = {}

    def debug_info(self):
        return DebugInfo(scratch_map=self.scratch_debug)

    def build(self, slots: list[tuple[Engine, tuple]], vliw: bool = False):
        # Simple slot packing that just uses one slot per instruction bundle
        instrs = []
        for engine, slot in slots:
            instrs.append({engine: [slot]})
        return instrs

    def add(self, engine, slot):
        self.instrs.append({engine: [slot]})

    def add_vliw(self, slots):
        """Add a VLIW instruction bundle with multiple engine slots."""
        instr = {}
        for engine, slot in slots:
            if engine not in instr:
                instr[engine] = []
            instr[engine].append(slot)
        if instr:
            self.instrs.append(instr)

    def alloc_scratch(self, name=None, length=1):
        addr = self.scratch_ptr
        if name is not None:
            self.scratch[name] = addr
            self.scratch_debug[addr] = (name, length)
        self.scratch_ptr += length
        assert self.scratch_ptr <= SCRATCH_SIZE, "Out of scratch space"
        return addr

    def scratch_const(self, val, name=None):
        if val not in self.const_map:
            addr = self.alloc_scratch(name)
            self.add("load", ("const", addr, val))
            self.const_map[val] = addr
        return self.const_map[val]

    def build_hash(self, val_hash_addr, tmp1, tmp2, round, i):
        slots = []

        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            slots.append(("alu", (op1, tmp1, val_hash_addr, self.scratch_const(val1))))
            slots.append(("alu", (op3, tmp2, val_hash_addr, self.scratch_const(val3))))
            slots.append(("alu", (op2, val_hash_addr, tmp1, tmp2)))
            slots.append(("debug", ("compare", val_hash_addr, (round, i, "hash_stage", hi))))

        return slots

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Optimized VLIW SIMD kernel with deep pipelining.
        Key insight: 64 indirect loads (32 cycles) can overlap with ALL hash computation.
        """
        UNROLL = 8  # Process 8 vectors per batch iteration (64 elements)

        tmp1 = self.alloc_scratch("tmp1")
        tmp2 = self.alloc_scratch("tmp2")
        tmp3 = self.alloc_scratch("tmp3")

        # Scratch space addresses for init vars
        init_vars = [
            "rounds", "n_nodes", "batch_size", "forest_height",
            "forest_values_p", "inp_indices_p", "inp_values_p",
        ]
        for v in init_vars:
            self.alloc_scratch(v, 1)
        for i, v in enumerate(init_vars):
            self.add("load", ("const", tmp1, i))
            self.add("load", ("load", self.scratch[v], tmp1))

        # Allocate vector scratch space for UNROLL=8 vectors
        v_idx = [self.alloc_scratch(f"v_idx_{j}", VLEN) for j in range(UNROLL)]
        v_hash = [self.alloc_scratch(f"v_hash_{j}", VLEN) for j in range(UNROLL)]
        v_node_val = [self.alloc_scratch(f"v_node_val_{j}", VLEN) for j in range(UNROLL)]
        v_tmp1 = [self.alloc_scratch(f"v_tmp1_{j}", VLEN) for j in range(UNROLL)]
        v_tmp2 = [self.alloc_scratch(f"v_tmp2_{j}", VLEN) for j in range(UNROLL)]
        v_cond = [self.alloc_scratch(f"v_cond_{j}", VLEN) for j in range(UNROLL)]

        # Scalar addresses for indirect loads (8 per vector)
        s_addr = [[self.alloc_scratch(f"s_addr_{j}_{k}") for k in range(VLEN)] for j in range(UNROLL)]

        # Address base registers
        addr_base_idx = [self.alloc_scratch(f"addr_base_idx_{j}") for j in range(UNROLL)]
        addr_base_val = [self.alloc_scratch(f"addr_base_val_{j}") for j in range(UNROLL)]

        # Constants
        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)

        # Pre-broadcast hash constants to vectors
        v_hash_const1 = []
        v_hash_const3 = []
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            vc1 = self.alloc_scratch(f"v_hash_c1_{hi}", VLEN)
            vc3 = self.alloc_scratch(f"v_hash_c3_{hi}", VLEN)
            v_hash_const1.append(vc1)
            v_hash_const3.append(vc3)
            c1_scalar = self.scratch_const(val1)
            c3_scalar = self.scratch_const(val3)
            self.add("valu", ("vbroadcast", vc1, c1_scalar))
            self.add("valu", ("vbroadcast", vc3, c3_scalar))

        # Broadcast constants for index calculation
        v_zero = self.alloc_scratch("v_zero", VLEN)
        v_one = self.alloc_scratch("v_one", VLEN)
        v_two = self.alloc_scratch("v_two", VLEN)
        v_n_nodes = self.alloc_scratch("v_n_nodes", VLEN)
        self.add("valu", ("vbroadcast", v_zero, zero_const))
        self.add("valu", ("vbroadcast", v_one, one_const))
        self.add("valu", ("vbroadcast", v_two, two_const))
        self.add("valu", ("vbroadcast", v_n_nodes, self.scratch["n_nodes"]))

        self.add("flow", ("pause",))
        self.add("debug", ("comment", "Starting optimized VLIW SIMD loop"))

        n_batches = batch_size // (UNROLL * VLEN)  # 256 / 64 = 4

        for batch in range(n_batches):
            base_offset = batch * UNROLL * VLEN

            for rnd in range(rounds):
                # ============================================================
                # PHASE 1: Setup address bases and load v_idx, v_hash (round 0 only)
                # ============================================================
                if rnd == 0:
                    for j in range(UNROLL):
                        offset = base_offset + j * VLEN
                        offset_const = self.scratch_const(offset)
                        self.add_vliw([
                            ("alu", ("+", addr_base_idx[j], self.scratch["inp_indices_p"], offset_const)),
                            ("alu", ("+", addr_base_val[j], self.scratch["inp_values_p"], offset_const)),
                        ])
                    for j in range(0, UNROLL, 2):
                        self.add_vliw([
                            ("load", ("vload", v_idx[j], addr_base_idx[j])),
                            ("load", ("vload", v_idx[j+1], addr_base_idx[j+1])),
                        ])
                    for j in range(0, UNROLL, 2):
                        self.add_vliw([
                            ("load", ("vload", v_hash[j], addr_base_val[j])),
                            ("load", ("vload", v_hash[j+1], addr_base_val[j+1])),
                        ])

                # ============================================================
                # PHASE 2: Compute indirect addresses (use 8 ALU ops per cycle)
                # 64 addresses, 8 per cycle = 8 cycles
                # ============================================================
                for j in range(0, UNROLL, 1):
                    self.add_vliw([
                        ("alu", ("+", s_addr[j][0], self.scratch["forest_values_p"], v_idx[j] + 0)),
                        ("alu", ("+", s_addr[j][1], self.scratch["forest_values_p"], v_idx[j] + 1)),
                        ("alu", ("+", s_addr[j][2], self.scratch["forest_values_p"], v_idx[j] + 2)),
                        ("alu", ("+", s_addr[j][3], self.scratch["forest_values_p"], v_idx[j] + 3)),
                        ("alu", ("+", s_addr[j][4], self.scratch["forest_values_p"], v_idx[j] + 4)),
                        ("alu", ("+", s_addr[j][5], self.scratch["forest_values_p"], v_idx[j] + 5)),
                        ("alu", ("+", s_addr[j][6], self.scratch["forest_values_p"], v_idx[j] + 6)),
                        ("alu", ("+", s_addr[j][7], self.scratch["forest_values_p"], v_idx[j] + 7)),
                    ])

                # ============================================================
                # PHASE 3: Indirect loads with interleaved XOR and hash
                # Strategy: Load v0-v1 (8 cycles), then interleave v2-v7 loads
                # with v0-v1 hash computation
                # ============================================================

                # Load v0 and v1 node values (8 cycles)
                for k in range(0, VLEN, 2):
                    self.add_vliw([
                        ("load", ("load", v_node_val[0] + k, s_addr[0][k])),
                        ("load", ("load", v_node_val[0] + k + 1, s_addr[0][k + 1])),
                    ])
                for k in range(0, VLEN, 2):
                    self.add_vliw([
                        ("load", ("load", v_node_val[1] + k, s_addr[1][k])),
                        ("load", ("load", v_node_val[1] + k + 1, s_addr[1][k + 1])),
                    ])

                # XOR v0 and v1
                self.add_vliw([
                    ("valu", ("^", v_hash[0], v_hash[0], v_node_val[0])),
                    ("valu", ("^", v_hash[1], v_hash[1], v_node_val[1])),
                ])

                # Hash v0-v1 while loading v2-v7
                load_j, load_k = 2, 0

                for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                    # Cycle 1: tmp1 and tmp2 for v0-v1
                    slots = [
                        ("valu", (op1, v_tmp1[0], v_hash[0], v_hash_const1[hi])),
                        ("valu", (op3, v_tmp2[0], v_hash[0], v_hash_const3[hi])),
                        ("valu", (op1, v_tmp1[1], v_hash[1], v_hash_const1[hi])),
                        ("valu", (op3, v_tmp2[1], v_hash[1], v_hash_const3[hi])),
                    ]
                    if load_j < UNROLL:
                        slots.append(("load", ("load", v_node_val[load_j] + load_k, s_addr[load_j][load_k])))
                        load_k += 1
                        if load_k >= VLEN:
                            load_k = 0
                            load_j += 1
                    if load_j < UNROLL:
                        slots.append(("load", ("load", v_node_val[load_j] + load_k, s_addr[load_j][load_k])))
                        load_k += 1
                        if load_k >= VLEN:
                            load_k = 0
                            load_j += 1
                    self.add_vliw(slots)

                    # Cycle 2: combine for v0-v1
                    slots = [
                        ("valu", (op2, v_hash[0], v_tmp1[0], v_tmp2[0])),
                        ("valu", (op2, v_hash[1], v_tmp1[1], v_tmp2[1])),
                    ]
                    if load_j < UNROLL:
                        slots.append(("load", ("load", v_node_val[load_j] + load_k, s_addr[load_j][load_k])))
                        load_k += 1
                        if load_k >= VLEN:
                            load_k = 0
                            load_j += 1
                    if load_j < UNROLL:
                        slots.append(("load", ("load", v_node_val[load_j] + load_k, s_addr[load_j][load_k])))
                        load_k += 1
                        if load_k >= VLEN:
                            load_k = 0
                            load_j += 1
                    self.add_vliw(slots)

                # Finish remaining loads
                while load_j < UNROLL:
                    slots = []
                    for _ in range(2):
                        if load_j < UNROLL:
                            slots.append(("load", ("load", v_node_val[load_j] + load_k, s_addr[load_j][load_k])))
                            load_k += 1
                            if load_k >= VLEN:
                                load_k = 0
                                load_j += 1
                    if slots:
                        self.add_vliw(slots)

                # XOR v2-v7
                for j in range(2, UNROLL, 2):
                    self.add_vliw([
                        ("valu", ("^", v_hash[j], v_hash[j], v_node_val[j])),
                        ("valu", ("^", v_hash[j+1], v_hash[j+1], v_node_val[j+1])),
                    ])

                # Hash v2-v7 while overlapping v0-v1 index calculation
                # Index calc for v0-v1 needs: &, +, multiply_add, <, * = 5 ops each
                # We have 18 cycles for v2-v7 hash, can fit v0-v1 index calc (10 ops)

                # v0-v1 index step 1: val & 1
                self.add_vliw([
                    ("valu", (HASH_STAGES[0][0], v_tmp1[2], v_hash[2], v_hash_const1[0])),
                    ("valu", (HASH_STAGES[0][3], v_tmp2[2], v_hash[2], v_hash_const3[0])),
                    ("valu", (HASH_STAGES[0][0], v_tmp1[3], v_hash[3], v_hash_const1[0])),
                    ("valu", (HASH_STAGES[0][3], v_tmp2[3], v_hash[3], v_hash_const3[0])),
                    ("valu", ("&", v_cond[0], v_hash[0], v_one)),
                    ("valu", ("&", v_cond[1], v_hash[1], v_one)),
                ])
                self.add_vliw([
                    ("valu", (HASH_STAGES[0][0], v_tmp1[4], v_hash[4], v_hash_const1[0])),
                    ("valu", (HASH_STAGES[0][3], v_tmp2[4], v_hash[4], v_hash_const3[0])),
                    ("valu", (HASH_STAGES[0][0], v_tmp1[5], v_hash[5], v_hash_const1[0])),
                    ("valu", (HASH_STAGES[0][3], v_tmp2[5], v_hash[5], v_hash_const3[0])),
                    ("valu", ("+", v_cond[0], v_one, v_cond[0])),
                    ("valu", ("+", v_cond[1], v_one, v_cond[1])),
                ])
                self.add_vliw([
                    ("valu", (HASH_STAGES[0][0], v_tmp1[6], v_hash[6], v_hash_const1[0])),
                    ("valu", (HASH_STAGES[0][3], v_tmp2[6], v_hash[6], v_hash_const3[0])),
                    ("valu", (HASH_STAGES[0][0], v_tmp1[7], v_hash[7], v_hash_const1[0])),
                    ("valu", (HASH_STAGES[0][3], v_tmp2[7], v_hash[7], v_hash_const3[0])),
                    ("valu", ("multiply_add", v_idx[0], v_idx[0], v_two, v_cond[0])),
                    ("valu", ("multiply_add", v_idx[1], v_idx[1], v_two, v_cond[1])),
                ])
                self.add_vliw([
                    ("valu", (HASH_STAGES[0][2], v_hash[2], v_tmp1[2], v_tmp2[2])),
                    ("valu", (HASH_STAGES[0][2], v_hash[3], v_tmp1[3], v_tmp2[3])),
                    ("valu", (HASH_STAGES[0][2], v_hash[4], v_tmp1[4], v_tmp2[4])),
                    ("valu", (HASH_STAGES[0][2], v_hash[5], v_tmp1[5], v_tmp2[5])),
                    ("valu", ("<", v_cond[0], v_idx[0], v_n_nodes)),
                    ("valu", ("<", v_cond[1], v_idx[1], v_n_nodes)),
                ])
                self.add_vliw([
                    ("valu", (HASH_STAGES[0][2], v_hash[6], v_tmp1[6], v_tmp2[6])),
                    ("valu", (HASH_STAGES[0][2], v_hash[7], v_tmp1[7], v_tmp2[7])),
                    ("valu", ("*", v_idx[0], v_idx[0], v_cond[0])),
                    ("valu", ("*", v_idx[1], v_idx[1], v_cond[1])),
                ])

                # Hash stages 1-5 for v2-v7 with v2-v7 index calc overlap
                for hi in range(1, 6):
                    op1, val1, op2, op3, val3 = HASH_STAGES[hi]
                    self.add_vliw([
                        ("valu", (op1, v_tmp1[2], v_hash[2], v_hash_const1[hi])),
                        ("valu", (op3, v_tmp2[2], v_hash[2], v_hash_const3[hi])),
                        ("valu", (op1, v_tmp1[3], v_hash[3], v_hash_const1[hi])),
                        ("valu", (op3, v_tmp2[3], v_hash[3], v_hash_const3[hi])),
                        ("valu", (op1, v_tmp1[4], v_hash[4], v_hash_const1[hi])),
                        ("valu", (op3, v_tmp2[4], v_hash[4], v_hash_const3[hi])),
                    ])
                    self.add_vliw([
                        ("valu", (op1, v_tmp1[5], v_hash[5], v_hash_const1[hi])),
                        ("valu", (op3, v_tmp2[5], v_hash[5], v_hash_const3[hi])),
                        ("valu", (op1, v_tmp1[6], v_hash[6], v_hash_const1[hi])),
                        ("valu", (op3, v_tmp2[6], v_hash[6], v_hash_const3[hi])),
                        ("valu", (op1, v_tmp1[7], v_hash[7], v_hash_const1[hi])),
                        ("valu", (op3, v_tmp2[7], v_hash[7], v_hash_const3[hi])),
                    ])
                    self.add_vliw([
                        ("valu", (op2, v_hash[2], v_tmp1[2], v_tmp2[2])),
                        ("valu", (op2, v_hash[3], v_tmp1[3], v_tmp2[3])),
                        ("valu", (op2, v_hash[4], v_tmp1[4], v_tmp2[4])),
                        ("valu", (op2, v_hash[5], v_tmp1[5], v_tmp2[5])),
                        ("valu", (op2, v_hash[6], v_tmp1[6], v_tmp2[6])),
                        ("valu", (op2, v_hash[7], v_tmp1[7], v_tmp2[7])),
                    ])

                # ============================================================
                # PHASE 4: Index calculation for v2-v7 (10 cycles)
                # ============================================================
                for j in range(2, UNROLL, 2):
                    self.add_vliw([
                        ("valu", ("&", v_cond[j], v_hash[j], v_one)),
                        ("valu", ("&", v_cond[j+1], v_hash[j+1], v_one)),
                    ])
                for j in range(2, UNROLL, 2):
                    self.add_vliw([
                        ("valu", ("+", v_cond[j], v_one, v_cond[j])),
                        ("valu", ("+", v_cond[j+1], v_one, v_cond[j+1])),
                    ])
                for j in range(2, UNROLL, 2):
                    self.add_vliw([
                        ("valu", ("multiply_add", v_idx[j], v_idx[j], v_two, v_cond[j])),
                        ("valu", ("multiply_add", v_idx[j+1], v_idx[j+1], v_two, v_cond[j+1])),
                    ])
                for j in range(2, UNROLL, 2):
                    self.add_vliw([
                        ("valu", ("<", v_tmp1[j], v_idx[j], v_n_nodes)),
                        ("valu", ("<", v_tmp1[j+1], v_idx[j+1], v_n_nodes)),
                    ])
                for j in range(2, UNROLL, 2):
                    self.add_vliw([
                        ("valu", ("*", v_idx[j], v_idx[j], v_tmp1[j])),
                        ("valu", ("*", v_idx[j+1], v_idx[j+1], v_tmp1[j+1])),
                    ])

                # ============================================================
                # PHASE 5: Store (only on last round)
                # ============================================================
                if rnd == rounds - 1:
                    for j in range(0, UNROLL, 2):
                        self.add_vliw([
                            ("store", ("vstore", addr_base_val[j], v_hash[j])),
                            ("store", ("vstore", addr_base_val[j+1], v_hash[j+1])),
                        ])

        self.instrs.append({"flow": [("pause",)]})

BASELINE = 147734

def do_kernel_test(
    forest_height: int,
    rounds: int,
    batch_size: int,
    seed: int = 123,
    trace: bool = False,
    prints: bool = False,
):
    print(f"{forest_height=}, {rounds=}, {batch_size=}")
    random.seed(seed)
    forest = Tree.generate(forest_height)
    inp = Input.generate(forest, batch_size, rounds)
    mem = build_mem_image(forest, inp)

    kb = KernelBuilder()
    kb.build_kernel(forest.height, len(forest.values), len(inp.indices), rounds)
    # print(kb.instrs)

    value_trace = {}
    machine = Machine(
        mem,
        kb.instrs,
        kb.debug_info(),
        n_cores=N_CORES,
        value_trace=value_trace,
        trace=trace,
    )
    machine.prints = prints
    for i, ref_mem in enumerate(reference_kernel2(mem, value_trace)):
        machine.run()
        inp_values_p = ref_mem[6]
        if prints:
            print(machine.mem[inp_values_p : inp_values_p + len(inp.values)])
            print(ref_mem[inp_values_p : inp_values_p + len(inp.values)])
        assert (
            machine.mem[inp_values_p : inp_values_p + len(inp.values)]
            == ref_mem[inp_values_p : inp_values_p + len(inp.values)]
        ), f"Incorrect result on round {i}"
        inp_indices_p = ref_mem[5]
        if prints:
            print(machine.mem[inp_indices_p : inp_indices_p + len(inp.indices)])
            print(ref_mem[inp_indices_p : inp_indices_p + len(inp.indices)])
        # Updating these in memory isn't required, but you can enable this check for debugging
        # assert machine.mem[inp_indices_p:inp_indices_p+len(inp.indices)] == ref_mem[inp_indices_p:inp_indices_p+len(inp.indices)]

    print("CYCLES: ", machine.cycle)
    print("Speedup over baseline: ", BASELINE / machine.cycle)
    return machine.cycle


class Tests(unittest.TestCase):
    def test_ref_kernels(self):
        """
        Test the reference kernels against each other
        """
        random.seed(123)
        for i in range(10):
            f = Tree.generate(4)
            inp = Input.generate(f, 10, 6)
            mem = build_mem_image(f, inp)
            reference_kernel(f, inp)
            for _ in reference_kernel2(mem, {}):
                pass
            assert inp.indices == mem[mem[5] : mem[5] + len(inp.indices)]
            assert inp.values == mem[mem[6] : mem[6] + len(inp.values)]

    def test_kernel_trace(self):
        # Full-scale example for performance testing
        do_kernel_test(10, 16, 256, trace=True, prints=False)

    # Passing this test is not required for submission, see submission_tests.py for the actual correctness test
    # You can uncomment this if you think it might help you debug
    # def test_kernel_correctness(self):
    #     for batch in range(1, 3):
    #         for forest_height in range(3):
    #             do_kernel_test(
    #                 forest_height + 2, forest_height + 4, batch * 16 * VLEN * N_CORES
    #             )

    def test_kernel_cycles(self):
        do_kernel_test(10, 16, 256)


# To run all the tests:
#    python perf_takehome.py
# To run a specific test:
#    python perf_takehome.py Tests.test_kernel_cycles
# To view a hot-reloading trace of all the instructions:  **Recommended debug loop**
# NOTE: The trace hot-reloading only works in Chrome. In the worst case if things aren't working, drag trace.json onto https://ui.perfetto.dev/
#    python perf_takehome.py Tests.test_kernel_trace
# Then run `python watch_trace.py` in another tab, it'll open a browser tab, then click "Open Perfetto"
# You can then keep that open and re-run the test to see a new trace.

# To run the proper checks to see which thresholds you pass:
#    python tests/submission_tests.py

if __name__ == "__main__":
    unittest.main()
