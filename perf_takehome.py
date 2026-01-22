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

    def emit_hash_8vec(self, v_hash, v_tmp1, v_tmp2, v_hash_const1, v_hash_const3, v_hash_mult=None):
        """Emit hash for 8 vectors with optimized staging.
        For + combiner stages (0, 2, 4): use multiply_add fusion (1 op instead of 3)
        For ^ combiner stages (1, 3, 5): use standard 3 ops
        """
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            if op2 == "+" and v_hash_mult is not None and v_hash_mult[hi] is not None:
                # Fused: hash = hash * mult + const1 where mult = 1 + 2^shift
                # This combines: (hash + const1) + (hash << shift) into one multiply_add
                # 2 cycles for 8 vectors
                self.add_vliw([
                    ("valu", ("multiply_add", v_hash[0], v_hash[0], v_hash_mult[hi], v_hash_const1[hi])),
                    ("valu", ("multiply_add", v_hash[1], v_hash[1], v_hash_mult[hi], v_hash_const1[hi])),
                    ("valu", ("multiply_add", v_hash[2], v_hash[2], v_hash_mult[hi], v_hash_const1[hi])),
                    ("valu", ("multiply_add", v_hash[3], v_hash[3], v_hash_mult[hi], v_hash_const1[hi])),
                    ("valu", ("multiply_add", v_hash[4], v_hash[4], v_hash_mult[hi], v_hash_const1[hi])),
                    ("valu", ("multiply_add", v_hash[5], v_hash[5], v_hash_mult[hi], v_hash_const1[hi])),
                ])
                self.add_vliw([
                    ("valu", ("multiply_add", v_hash[6], v_hash[6], v_hash_mult[hi], v_hash_const1[hi])),
                    ("valu", ("multiply_add", v_hash[7], v_hash[7], v_hash_mult[hi], v_hash_const1[hi])),
                ])
            else:
                # Standard 3-op hash stage with ^ combiner
                # Cycle 1: op1/op3 for v0-v2
                self.add_vliw([
                    ("valu", (op1, v_tmp1[0], v_hash[0], v_hash_const1[hi])),
                    ("valu", (op3, v_tmp2[0], v_hash[0], v_hash_const3[hi])),
                    ("valu", (op1, v_tmp1[1], v_hash[1], v_hash_const1[hi])),
                    ("valu", (op3, v_tmp2[1], v_hash[1], v_hash_const3[hi])),
                    ("valu", (op1, v_tmp1[2], v_hash[2], v_hash_const1[hi])),
                    ("valu", (op3, v_tmp2[2], v_hash[2], v_hash_const3[hi])),
                ])
                # Cycle 2: op1/op3 for v3-v5
                self.add_vliw([
                    ("valu", (op1, v_tmp1[3], v_hash[3], v_hash_const1[hi])),
                    ("valu", (op3, v_tmp2[3], v_hash[3], v_hash_const3[hi])),
                    ("valu", (op1, v_tmp1[4], v_hash[4], v_hash_const1[hi])),
                    ("valu", (op3, v_tmp2[4], v_hash[4], v_hash_const3[hi])),
                    ("valu", (op1, v_tmp1[5], v_hash[5], v_hash_const1[hi])),
                    ("valu", (op3, v_tmp2[5], v_hash[5], v_hash_const3[hi])),
                ])
                # Cycle 3: op1/op3 for v6-v7 + op2 for v0-v1
                self.add_vliw([
                    ("valu", (op1, v_tmp1[6], v_hash[6], v_hash_const1[hi])),
                    ("valu", (op3, v_tmp2[6], v_hash[6], v_hash_const3[hi])),
                    ("valu", (op1, v_tmp1[7], v_hash[7], v_hash_const1[hi])),
                    ("valu", (op3, v_tmp2[7], v_hash[7], v_hash_const3[hi])),
                    ("valu", (op2, v_hash[0], v_tmp1[0], v_tmp2[0])),
                    ("valu", (op2, v_hash[1], v_tmp1[1], v_tmp2[1])),
                ])
                # Cycle 4: op2 for v2-v7
                self.add_vliw([
                    ("valu", (op2, v_hash[2], v_tmp1[2], v_tmp2[2])),
                    ("valu", (op2, v_hash[3], v_tmp1[3], v_tmp2[3])),
                    ("valu", (op2, v_hash[4], v_tmp1[4], v_tmp2[4])),
                    ("valu", (op2, v_hash[5], v_tmp1[5], v_tmp2[5])),
                    ("valu", (op2, v_hash[6], v_tmp1[6], v_tmp2[6])),
                    ("valu", (op2, v_hash[7], v_tmp1[7], v_tmp2[7])),
                ])

    def emit_index_calc_8vec(self, v_idx, v_hash, v_cond, v_tmp1, v_one, v_two, v_n_nodes):
        """Emit pipelined index calculation for 8 vectors (7 cycles instead of 10).
        Pipeline stages across vectors to maximize VALU utilization."""
        # Cycle 1: step1 for v0-5
        self.add_vliw([
            ("valu", ("&", v_cond[0], v_hash[0], v_one)),
            ("valu", ("&", v_cond[1], v_hash[1], v_one)),
            ("valu", ("&", v_cond[2], v_hash[2], v_one)),
            ("valu", ("&", v_cond[3], v_hash[3], v_one)),
            ("valu", ("&", v_cond[4], v_hash[4], v_one)),
            ("valu", ("&", v_cond[5], v_hash[5], v_one)),
        ])
        # Cycle 2: step1 for v6-7 + step2 for v0-3
        self.add_vliw([
            ("valu", ("&", v_cond[6], v_hash[6], v_one)),
            ("valu", ("&", v_cond[7], v_hash[7], v_one)),
            ("valu", ("+", v_cond[0], v_one, v_cond[0])),
            ("valu", ("+", v_cond[1], v_one, v_cond[1])),
            ("valu", ("+", v_cond[2], v_one, v_cond[2])),
            ("valu", ("+", v_cond[3], v_one, v_cond[3])),
        ])
        # Cycle 3: step2 for v4-7 + step3 for v0-1
        self.add_vliw([
            ("valu", ("+", v_cond[4], v_one, v_cond[4])),
            ("valu", ("+", v_cond[5], v_one, v_cond[5])),
            ("valu", ("+", v_cond[6], v_one, v_cond[6])),
            ("valu", ("+", v_cond[7], v_one, v_cond[7])),
            ("valu", ("multiply_add", v_idx[0], v_idx[0], v_two, v_cond[0])),
            ("valu", ("multiply_add", v_idx[1], v_idx[1], v_two, v_cond[1])),
        ])
        # Cycle 4: step3 for v2-7
        self.add_vliw([
            ("valu", ("multiply_add", v_idx[2], v_idx[2], v_two, v_cond[2])),
            ("valu", ("multiply_add", v_idx[3], v_idx[3], v_two, v_cond[3])),
            ("valu", ("multiply_add", v_idx[4], v_idx[4], v_two, v_cond[4])),
            ("valu", ("multiply_add", v_idx[5], v_idx[5], v_two, v_cond[5])),
            ("valu", ("multiply_add", v_idx[6], v_idx[6], v_two, v_cond[6])),
            ("valu", ("multiply_add", v_idx[7], v_idx[7], v_two, v_cond[7])),
        ])
        # Cycle 5: step4 for v0-5
        self.add_vliw([
            ("valu", ("<", v_tmp1[0], v_idx[0], v_n_nodes)),
            ("valu", ("<", v_tmp1[1], v_idx[1], v_n_nodes)),
            ("valu", ("<", v_tmp1[2], v_idx[2], v_n_nodes)),
            ("valu", ("<", v_tmp1[3], v_idx[3], v_n_nodes)),
            ("valu", ("<", v_tmp1[4], v_idx[4], v_n_nodes)),
            ("valu", ("<", v_tmp1[5], v_idx[5], v_n_nodes)),
        ])
        # Cycle 6: step4 for v6-7 + step5 for v0-3
        self.add_vliw([
            ("valu", ("<", v_tmp1[6], v_idx[6], v_n_nodes)),
            ("valu", ("<", v_tmp1[7], v_idx[7], v_n_nodes)),
            ("valu", ("*", v_idx[0], v_idx[0], v_tmp1[0])),
            ("valu", ("*", v_idx[1], v_idx[1], v_tmp1[1])),
            ("valu", ("*", v_idx[2], v_idx[2], v_tmp1[2])),
            ("valu", ("*", v_idx[3], v_idx[3], v_tmp1[3])),
        ])
        # Cycle 7: step5 for v4-7
        self.add_vliw([
            ("valu", ("*", v_idx[4], v_idx[4], v_tmp1[4])),
            ("valu", ("*", v_idx[5], v_idx[5], v_tmp1[5])),
            ("valu", ("*", v_idx[6], v_idx[6], v_tmp1[6])),
            ("valu", ("*", v_idx[7], v_idx[7], v_tmp1[7])),
        ])

    def build_kernel(
        self, forest_height: int, n_nodes: int, batch_size: int, rounds: int
    ):
        """
        Optimized VLIW SIMD kernel with round-major processing.
        Key optimizations:
        1. Round-major processing to share tree value loads
        2. 4-cycle hash stages (instead of 5)
        3. Better load/hash overlap
        """
        UNROLL = 8  # Process 8 vectors per sub-batch
        N_SUB_BATCHES = batch_size // (UNROLL * VLEN)  # 4 sub-batches

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

        # Allocate vector scratch space for all sub-batches
        # Each sub-batch has 8 vectors of 8 elements
        v_idx = [[self.alloc_scratch(f"v_idx_{b}_{j}", VLEN) for j in range(UNROLL)]
                 for b in range(N_SUB_BATCHES)]
        v_hash = [[self.alloc_scratch(f"v_hash_{b}_{j}", VLEN) for j in range(UNROLL)]
                  for b in range(N_SUB_BATCHES)]

        # Working scratch (shared across sub-batches)
        v_node_val = [self.alloc_scratch(f"v_node_val_{j}", VLEN) for j in range(UNROLL)]
        v_tmp1 = [self.alloc_scratch(f"v_tmp1_{j}", VLEN) for j in range(UNROLL)]
        v_tmp2 = [self.alloc_scratch(f"v_tmp2_{j}", VLEN) for j in range(UNROLL)]
        v_cond = [self.alloc_scratch(f"v_cond_{j}", VLEN) for j in range(UNROLL)]

        # Scalar addresses for indirect loads
        s_addr = [[self.alloc_scratch(f"s_addr_{j}_{k}") for k in range(VLEN)] for j in range(UNROLL)]

        # Address base registers
        addr_base_idx = [[self.alloc_scratch(f"addr_base_idx_{b}_{j}") for j in range(UNROLL)]
                         for b in range(N_SUB_BATCHES)]
        addr_base_val = [[self.alloc_scratch(f"addr_base_val_{b}_{j}") for j in range(UNROLL)]
                         for b in range(N_SUB_BATCHES)]

        # Constants
        zero_const = self.scratch_const(0)
        one_const = self.scratch_const(1)
        two_const = self.scratch_const(2)
        three_const = self.scratch_const(3)
        four_const = self.scratch_const(4)
        five_const = self.scratch_const(5)
        six_const = self.scratch_const(6)
        seven_const = self.scratch_const(7)

        # Pre-broadcast hash constants to vectors
        v_hash_const1 = []
        v_hash_const3 = []
        v_hash_mult = []  # Multipliers for fused stages (0, 2, 4)
        for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
            vc1 = self.alloc_scratch(f"v_hash_c1_{hi}", VLEN)
            vc3 = self.alloc_scratch(f"v_hash_c3_{hi}", VLEN)
            v_hash_const1.append(vc1)
            v_hash_const3.append(vc3)
            c1_scalar = self.scratch_const(val1)
            c3_scalar = self.scratch_const(val3)
            self.add("valu", ("vbroadcast", vc1, c1_scalar))
            self.add("valu", ("vbroadcast", vc3, c3_scalar))
            # For + combiner stages (0, 2, 4): mult = 1 + 2^shift
            if op2 == "+":
                mult_val = 1 + (1 << val3)  # 4097, 33, 9
                vm = self.alloc_scratch(f"v_hash_mult_{hi}", VLEN)
                mult_scalar = self.scratch_const(mult_val)
                self.add("valu", ("vbroadcast", vm, mult_scalar))
                v_hash_mult.append(vm)
            else:
                v_hash_mult.append(None)  # Placeholder for non-fused stages

        # Broadcast constants for index calculation
        v_zero = self.alloc_scratch("v_zero", VLEN)
        v_one = self.alloc_scratch("v_one", VLEN)
        v_two = self.alloc_scratch("v_two", VLEN)
        v_three = self.alloc_scratch("v_three", VLEN)
        v_n_nodes = self.alloc_scratch("v_n_nodes", VLEN)
        self.add("valu", ("vbroadcast", v_zero, zero_const))
        self.add("valu", ("vbroadcast", v_one, one_const))
        self.add("valu", ("vbroadcast", v_two, two_const))
        self.add("valu", ("vbroadcast", v_three, three_const))
        self.add("valu", ("vbroadcast", v_n_nodes, self.scratch["n_nodes"]))

        # Shared node value for level 0
        shared_node_val = self.alloc_scratch("shared_node_val")
        v_shared_node = self.alloc_scratch("v_shared_node", VLEN)

        # Level 1 tree values
        level1_val0 = self.alloc_scratch("level1_val0")
        level1_val1 = self.alloc_scratch("level1_val1")
        v_level1_0 = self.alloc_scratch("v_level1_0", VLEN)
        v_level1_1 = self.alloc_scratch("v_level1_1", VLEN)

        # Level 2 tree values
        level2_vals = [self.alloc_scratch(f"level2_val{i}") for i in range(4)]
        v_level2 = [self.alloc_scratch(f"v_level2_{i}", VLEN) for i in range(4)]

        # Pre-compute addresses for level 1 and 2
        addr_level1_0 = self.alloc_scratch("addr_level1_0")
        addr_level1_1 = self.alloc_scratch("addr_level1_1")
        addr_level2 = [self.alloc_scratch(f"addr_level2_{i}") for i in range(4)]

        # Preload and broadcast ALL tree values for special rounds (levels 0, 1, 2)
        # Level 0: forest_values[0]
        self.add("load", ("load", shared_node_val, self.scratch["forest_values_p"]))
        self.add("valu", ("vbroadcast", v_shared_node, shared_node_val))

        # Level 1: forest_values[1], forest_values[2]
        self.add_vliw([
            ("alu", ("+", addr_level1_0, self.scratch["forest_values_p"], one_const)),
            ("alu", ("+", addr_level1_1, self.scratch["forest_values_p"], two_const)),
        ])
        self.add_vliw([
            ("load", ("load", level1_val0, addr_level1_0)),
            ("load", ("load", level1_val1, addr_level1_1)),
        ])
        self.add_vliw([
            ("valu", ("vbroadcast", v_level1_0, level1_val0)),
            ("valu", ("vbroadcast", v_level1_1, level1_val1)),
        ])

        # Level 2: forest_values[3], forest_values[4], forest_values[5], forest_values[6]
        self.add_vliw([
            ("alu", ("+", addr_level2[0], self.scratch["forest_values_p"], three_const)),
            ("alu", ("+", addr_level2[1], self.scratch["forest_values_p"], four_const)),
            ("alu", ("+", addr_level2[2], self.scratch["forest_values_p"], five_const)),
            ("alu", ("+", addr_level2[3], self.scratch["forest_values_p"], six_const)),
        ])
        self.add_vliw([
            ("load", ("load", level2_vals[0], addr_level2[0])),
            ("load", ("load", level2_vals[1], addr_level2[1])),
        ])
        self.add_vliw([
            ("load", ("load", level2_vals[2], addr_level2[2])),
            ("load", ("load", level2_vals[3], addr_level2[3])),
        ])
        self.add_vliw([
            ("valu", ("vbroadcast", v_level2[0], level2_vals[0])),
            ("valu", ("vbroadcast", v_level2[1], level2_vals[1])),
            ("valu", ("vbroadcast", v_level2[2], level2_vals[2])),
            ("valu", ("vbroadcast", v_level2[3], level2_vals[3])),
        ])

        # Level 3 tree values: forest_values[7] through forest_values[14] (8 values)
        level3_vals = [self.alloc_scratch(f"level3_val{i}") for i in range(8)]
        v_level3 = [self.alloc_scratch(f"v_level3_{i}", VLEN) for i in range(8)]
        v_seven = self.alloc_scratch("v_seven", VLEN)
        self.add("valu", ("vbroadcast", v_seven, seven_const))

        # Compute addresses for level 3 values (7-14)
        addr_level3 = [self.alloc_scratch(f"addr_level3_{i}") for i in range(8)]
        for i in range(0, 8, 4):
            self.add_vliw([
                ("alu", ("+", addr_level3[i], self.scratch["forest_values_p"], self.scratch_const(7 + i))),
                ("alu", ("+", addr_level3[i+1], self.scratch["forest_values_p"], self.scratch_const(7 + i + 1))),
                ("alu", ("+", addr_level3[i+2], self.scratch["forest_values_p"], self.scratch_const(7 + i + 2))),
                ("alu", ("+", addr_level3[i+3], self.scratch["forest_values_p"], self.scratch_const(7 + i + 3))),
            ])
        # Load level 3 values (8 loads at 2/cycle = 4 cycles)
        for i in range(0, 8, 2):
            self.add_vliw([
                ("load", ("load", level3_vals[i], addr_level3[i])),
                ("load", ("load", level3_vals[i+1], addr_level3[i+1])),
            ])
        # Broadcast level 3 values (8 broadcasts at 6/cycle = 2 cycles)
        self.add_vliw([
            ("valu", ("vbroadcast", v_level3[0], level3_vals[0])),
            ("valu", ("vbroadcast", v_level3[1], level3_vals[1])),
            ("valu", ("vbroadcast", v_level3[2], level3_vals[2])),
            ("valu", ("vbroadcast", v_level3[3], level3_vals[3])),
            ("valu", ("vbroadcast", v_level3[4], level3_vals[4])),
            ("valu", ("vbroadcast", v_level3[5], level3_vals[5])),
        ])
        self.add_vliw([
            ("valu", ("vbroadcast", v_level3[6], level3_vals[6])),
            ("valu", ("vbroadcast", v_level3[7], level3_vals[7])),
        ])

        self.add("flow", ("pause",))
        self.add("debug", ("comment", "Starting round-major optimized kernel"))

        # ================================================================
        # PHASE 0: Load all idx and hash values for all sub-batches
        # ================================================================
        for b in range(N_SUB_BATCHES):
            base_offset = b * UNROLL * VLEN
            for j in range(UNROLL):
                offset = base_offset + j * VLEN
                offset_const = self.scratch_const(offset)
                self.add_vliw([
                    ("alu", ("+", addr_base_idx[b][j], self.scratch["inp_indices_p"], offset_const)),
                    ("alu", ("+", addr_base_val[b][j], self.scratch["inp_values_p"], offset_const)),
                ])
            for j in range(0, UNROLL, 2):
                self.add_vliw([
                    ("load", ("vload", v_idx[b][j], addr_base_idx[b][j])),
                    ("load", ("vload", v_idx[b][j+1], addr_base_idx[b][j+1])),
                ])
            for j in range(0, UNROLL, 2):
                self.add_vliw([
                    ("load", ("vload", v_hash[b][j], addr_base_val[b][j])),
                    ("load", ("vload", v_hash[b][j+1], addr_base_val[b][j+1])),
                ])

        # ================================================================
        # MAIN LOOP: Process rounds (round-major)
        # ================================================================
        for rnd in range(rounds):
            # Determine tree level
            if rnd <= 10:
                tree_level = rnd
            else:
                tree_level = rnd - 11  # After wrap

            if tree_level == 0:
                # Level 0: All elements access forest_values[0]
                # Tree values already preloaded at start
                for b in range(N_SUB_BATCHES):
                    # XOR
                    for j in range(0, UNROLL, 2):
                        self.add_vliw([
                            ("valu", ("^", v_hash[b][j], v_hash[b][j], v_shared_node)),
                            ("valu", ("^", v_hash[b][j+1], v_hash[b][j+1], v_shared_node)),
                        ])
                    # Hash with multiply_add fusion
                    self.emit_hash_8vec(v_hash[b], v_tmp1, v_tmp2, v_hash_const1, v_hash_const3, v_hash_mult)
                    # Index calc
                    self.emit_index_calc_8vec(v_idx[b], v_hash[b], v_cond, v_tmp1, v_one, v_two, v_n_nodes)

            elif tree_level == 1:
                # Level 1: Indices are 1 or 2
                # Tree values already preloaded at start
                for b in range(N_SUB_BATCHES):
                    # Selection mask: cond = idx & 1
                    for j in range(0, UNROLL, 2):
                        self.add_vliw([
                            ("valu", ("&", v_cond[j], v_idx[b][j], v_one)),
                            ("valu", ("&", v_cond[j+1], v_idx[b][j+1], v_one)),
                        ])
                    # vselect (1 per cycle due to flow engine limit)
                    for j in range(UNROLL):
                        self.add("flow", ("vselect", v_node_val[j], v_cond[j], v_level1_0, v_level1_1))
                    # XOR
                    for j in range(0, UNROLL, 2):
                        self.add_vliw([
                            ("valu", ("^", v_hash[b][j], v_hash[b][j], v_node_val[j])),
                            ("valu", ("^", v_hash[b][j+1], v_hash[b][j+1], v_node_val[j+1])),
                        ])
                    # Hash with multiply_add fusion
                    self.emit_hash_8vec(v_hash[b], v_tmp1, v_tmp2, v_hash_const1, v_hash_const3, v_hash_mult)
                    # Index calc
                    self.emit_index_calc_8vec(v_idx[b], v_hash[b], v_cond, v_tmp1, v_one, v_two, v_n_nodes)

            elif tree_level == 2:
                # Level 2: Indices are 3, 4, 5, 6
                # Tree values already preloaded at start
                for b in range(N_SUB_BATCHES):
                    # Arithmetic selection: result = sum(val[i] * (offset == i))
                    # Compute offset = idx - 3
                    self.add_vliw([
                        ("valu", ("-", v_cond[0], v_idx[b][0], v_three)),
                        ("valu", ("-", v_cond[1], v_idx[b][1], v_three)),
                        ("valu", ("-", v_cond[2], v_idx[b][2], v_three)),
                        ("valu", ("-", v_cond[3], v_idx[b][3], v_three)),
                        ("valu", ("-", v_cond[4], v_idx[b][4], v_three)),
                        ("valu", ("-", v_cond[5], v_idx[b][5], v_three)),
                    ])
                    self.add_vliw([
                        ("valu", ("-", v_cond[6], v_idx[b][6], v_three)),
                        ("valu", ("-", v_cond[7], v_idx[b][7], v_three)),
                    ])

                    # For each tree value, compute eq and accumulate
                    # eq0 = (offset == 0), t0 = val0 * eq0
                    self.add_vliw([
                        ("valu", ("==", v_tmp1[0], v_cond[0], v_zero)),
                        ("valu", ("==", v_tmp1[1], v_cond[1], v_zero)),
                        ("valu", ("==", v_tmp1[2], v_cond[2], v_zero)),
                        ("valu", ("==", v_tmp1[3], v_cond[3], v_zero)),
                        ("valu", ("==", v_tmp1[4], v_cond[4], v_zero)),
                        ("valu", ("==", v_tmp1[5], v_cond[5], v_zero)),
                    ])
                    self.add_vliw([
                        ("valu", ("==", v_tmp1[6], v_cond[6], v_zero)),
                        ("valu", ("==", v_tmp1[7], v_cond[7], v_zero)),
                    ])
                    self.add_vliw([
                        ("valu", ("*", v_tmp2[0], v_level2[0], v_tmp1[0])),
                        ("valu", ("*", v_tmp2[1], v_level2[0], v_tmp1[1])),
                        ("valu", ("*", v_tmp2[2], v_level2[0], v_tmp1[2])),
                        ("valu", ("*", v_tmp2[3], v_level2[0], v_tmp1[3])),
                        ("valu", ("*", v_tmp2[4], v_level2[0], v_tmp1[4])),
                        ("valu", ("*", v_tmp2[5], v_level2[0], v_tmp1[5])),
                    ])
                    self.add_vliw([
                        ("valu", ("*", v_tmp2[6], v_level2[0], v_tmp1[6])),
                        ("valu", ("*", v_tmp2[7], v_level2[0], v_tmp1[7])),
                    ])

                    # eq1, t1 = t0 + val1 * eq1
                    self.add_vliw([
                        ("valu", ("==", v_tmp1[0], v_cond[0], v_one)),
                        ("valu", ("==", v_tmp1[1], v_cond[1], v_one)),
                        ("valu", ("==", v_tmp1[2], v_cond[2], v_one)),
                        ("valu", ("==", v_tmp1[3], v_cond[3], v_one)),
                        ("valu", ("==", v_tmp1[4], v_cond[4], v_one)),
                        ("valu", ("==", v_tmp1[5], v_cond[5], v_one)),
                    ])
                    self.add_vliw([
                        ("valu", ("==", v_tmp1[6], v_cond[6], v_one)),
                        ("valu", ("==", v_tmp1[7], v_cond[7], v_one)),
                    ])
                    self.add_vliw([
                        ("valu", ("multiply_add", v_tmp2[0], v_level2[1], v_tmp1[0], v_tmp2[0])),
                        ("valu", ("multiply_add", v_tmp2[1], v_level2[1], v_tmp1[1], v_tmp2[1])),
                        ("valu", ("multiply_add", v_tmp2[2], v_level2[1], v_tmp1[2], v_tmp2[2])),
                        ("valu", ("multiply_add", v_tmp2[3], v_level2[1], v_tmp1[3], v_tmp2[3])),
                        ("valu", ("multiply_add", v_tmp2[4], v_level2[1], v_tmp1[4], v_tmp2[4])),
                        ("valu", ("multiply_add", v_tmp2[5], v_level2[1], v_tmp1[5], v_tmp2[5])),
                    ])
                    self.add_vliw([
                        ("valu", ("multiply_add", v_tmp2[6], v_level2[1], v_tmp1[6], v_tmp2[6])),
                        ("valu", ("multiply_add", v_tmp2[7], v_level2[1], v_tmp1[7], v_tmp2[7])),
                    ])

                    # eq2, t2 = t1 + val2 * eq2
                    self.add_vliw([
                        ("valu", ("==", v_tmp1[0], v_cond[0], v_two)),
                        ("valu", ("==", v_tmp1[1], v_cond[1], v_two)),
                        ("valu", ("==", v_tmp1[2], v_cond[2], v_two)),
                        ("valu", ("==", v_tmp1[3], v_cond[3], v_two)),
                        ("valu", ("==", v_tmp1[4], v_cond[4], v_two)),
                        ("valu", ("==", v_tmp1[5], v_cond[5], v_two)),
                    ])
                    self.add_vliw([
                        ("valu", ("==", v_tmp1[6], v_cond[6], v_two)),
                        ("valu", ("==", v_tmp1[7], v_cond[7], v_two)),
                    ])
                    self.add_vliw([
                        ("valu", ("multiply_add", v_tmp2[0], v_level2[2], v_tmp1[0], v_tmp2[0])),
                        ("valu", ("multiply_add", v_tmp2[1], v_level2[2], v_tmp1[1], v_tmp2[1])),
                        ("valu", ("multiply_add", v_tmp2[2], v_level2[2], v_tmp1[2], v_tmp2[2])),
                        ("valu", ("multiply_add", v_tmp2[3], v_level2[2], v_tmp1[3], v_tmp2[3])),
                        ("valu", ("multiply_add", v_tmp2[4], v_level2[2], v_tmp1[4], v_tmp2[4])),
                        ("valu", ("multiply_add", v_tmp2[5], v_level2[2], v_tmp1[5], v_tmp2[5])),
                    ])
                    self.add_vliw([
                        ("valu", ("multiply_add", v_tmp2[6], v_level2[2], v_tmp1[6], v_tmp2[6])),
                        ("valu", ("multiply_add", v_tmp2[7], v_level2[2], v_tmp1[7], v_tmp2[7])),
                    ])

                    # eq3, result = t2 + val3 * eq3
                    self.add_vliw([
                        ("valu", ("==", v_tmp1[0], v_cond[0], v_three)),
                        ("valu", ("==", v_tmp1[1], v_cond[1], v_three)),
                        ("valu", ("==", v_tmp1[2], v_cond[2], v_three)),
                        ("valu", ("==", v_tmp1[3], v_cond[3], v_three)),
                        ("valu", ("==", v_tmp1[4], v_cond[4], v_three)),
                        ("valu", ("==", v_tmp1[5], v_cond[5], v_three)),
                    ])
                    self.add_vliw([
                        ("valu", ("==", v_tmp1[6], v_cond[6], v_three)),
                        ("valu", ("==", v_tmp1[7], v_cond[7], v_three)),
                    ])
                    self.add_vliw([
                        ("valu", ("multiply_add", v_node_val[0], v_level2[3], v_tmp1[0], v_tmp2[0])),
                        ("valu", ("multiply_add", v_node_val[1], v_level2[3], v_tmp1[1], v_tmp2[1])),
                        ("valu", ("multiply_add", v_node_val[2], v_level2[3], v_tmp1[2], v_tmp2[2])),
                        ("valu", ("multiply_add", v_node_val[3], v_level2[3], v_tmp1[3], v_tmp2[3])),
                        ("valu", ("multiply_add", v_node_val[4], v_level2[3], v_tmp1[4], v_tmp2[4])),
                        ("valu", ("multiply_add", v_node_val[5], v_level2[3], v_tmp1[5], v_tmp2[5])),
                    ])
                    self.add_vliw([
                        ("valu", ("multiply_add", v_node_val[6], v_level2[3], v_tmp1[6], v_tmp2[6])),
                        ("valu", ("multiply_add", v_node_val[7], v_level2[3], v_tmp1[7], v_tmp2[7])),
                    ])

                    # XOR
                    self.add_vliw([
                        ("valu", ("^", v_hash[b][0], v_hash[b][0], v_node_val[0])),
                        ("valu", ("^", v_hash[b][1], v_hash[b][1], v_node_val[1])),
                        ("valu", ("^", v_hash[b][2], v_hash[b][2], v_node_val[2])),
                        ("valu", ("^", v_hash[b][3], v_hash[b][3], v_node_val[3])),
                        ("valu", ("^", v_hash[b][4], v_hash[b][4], v_node_val[4])),
                        ("valu", ("^", v_hash[b][5], v_hash[b][5], v_node_val[5])),
                    ])
                    self.add_vliw([
                        ("valu", ("^", v_hash[b][6], v_hash[b][6], v_node_val[6])),
                        ("valu", ("^", v_hash[b][7], v_hash[b][7], v_node_val[7])),
                    ])

                    # Hash with multiply_add fusion
                    self.emit_hash_8vec(v_hash[b], v_tmp1, v_tmp2, v_hash_const1, v_hash_const3, v_hash_mult)
                    # Index calc
                    self.emit_index_calc_8vec(v_idx[b], v_hash[b], v_cond, v_tmp1, v_one, v_two, v_n_nodes)

            elif tree_level == 3:
                # Level 3: Indices are 7-14 (8 values)
                # Tree values already preloaded at start
                # Pre-allocate comparison vectors for values 4, 5, 6 (only once)
                v_four = self.alloc_scratch("v_four", VLEN)
                v_five = self.alloc_scratch("v_five", VLEN)
                v_six = self.alloc_scratch("v_six", VLEN)
                self.add_vliw([
                    ("valu", ("vbroadcast", v_four, four_const)),
                    ("valu", ("vbroadcast", v_five, five_const)),
                    ("valu", ("vbroadcast", v_six, six_const)),
                ])
                v_cmp_vals = [v_zero, v_one, v_two, v_three, v_four, v_five, v_six, v_seven]

                for b in range(N_SUB_BATCHES):
                    # Arithmetic selection: result = sum(val[i] * (offset == i)) for i in 0..7
                    # Compute offset = idx - 7
                    self.add_vliw([
                        ("valu", ("-", v_cond[0], v_idx[b][0], v_seven)),
                        ("valu", ("-", v_cond[1], v_idx[b][1], v_seven)),
                        ("valu", ("-", v_cond[2], v_idx[b][2], v_seven)),
                        ("valu", ("-", v_cond[3], v_idx[b][3], v_seven)),
                        ("valu", ("-", v_cond[4], v_idx[b][4], v_seven)),
                        ("valu", ("-", v_cond[5], v_idx[b][5], v_seven)),
                    ])
                    self.add_vliw([
                        ("valu", ("-", v_cond[6], v_idx[b][6], v_seven)),
                        ("valu", ("-", v_cond[7], v_idx[b][7], v_seven)),
                    ])

                    # value 0: eq0 = (offset == 0), result = val0 * eq0
                    self.add_vliw([
                        ("valu", ("==", v_tmp1[0], v_cond[0], v_zero)),
                        ("valu", ("==", v_tmp1[1], v_cond[1], v_zero)),
                        ("valu", ("==", v_tmp1[2], v_cond[2], v_zero)),
                        ("valu", ("==", v_tmp1[3], v_cond[3], v_zero)),
                        ("valu", ("==", v_tmp1[4], v_cond[4], v_zero)),
                        ("valu", ("==", v_tmp1[5], v_cond[5], v_zero)),
                    ])
                    self.add_vliw([
                        ("valu", ("==", v_tmp1[6], v_cond[6], v_zero)),
                        ("valu", ("==", v_tmp1[7], v_cond[7], v_zero)),
                    ])
                    self.add_vliw([
                        ("valu", ("*", v_tmp2[0], v_level3[0], v_tmp1[0])),
                        ("valu", ("*", v_tmp2[1], v_level3[0], v_tmp1[1])),
                        ("valu", ("*", v_tmp2[2], v_level3[0], v_tmp1[2])),
                        ("valu", ("*", v_tmp2[3], v_level3[0], v_tmp1[3])),
                        ("valu", ("*", v_tmp2[4], v_level3[0], v_tmp1[4])),
                        ("valu", ("*", v_tmp2[5], v_level3[0], v_tmp1[5])),
                    ])
                    self.add_vliw([
                        ("valu", ("*", v_tmp2[6], v_level3[0], v_tmp1[6])),
                        ("valu", ("*", v_tmp2[7], v_level3[0], v_tmp1[7])),
                    ])

                    # values 1-7: eq_i, result += val[i] * eq_i
                    for vi in range(1, 8):
                        v_cmp = v_cmp_vals[vi]
                        self.add_vliw([
                            ("valu", ("==", v_tmp1[0], v_cond[0], v_cmp)),
                            ("valu", ("==", v_tmp1[1], v_cond[1], v_cmp)),
                            ("valu", ("==", v_tmp1[2], v_cond[2], v_cmp)),
                            ("valu", ("==", v_tmp1[3], v_cond[3], v_cmp)),
                            ("valu", ("==", v_tmp1[4], v_cond[4], v_cmp)),
                            ("valu", ("==", v_tmp1[5], v_cond[5], v_cmp)),
                        ])
                        self.add_vliw([
                            ("valu", ("==", v_tmp1[6], v_cond[6], v_cmp)),
                            ("valu", ("==", v_tmp1[7], v_cond[7], v_cmp)),
                        ])
                        self.add_vliw([
                            ("valu", ("multiply_add", v_tmp2[0], v_level3[vi], v_tmp1[0], v_tmp2[0])),
                            ("valu", ("multiply_add", v_tmp2[1], v_level3[vi], v_tmp1[1], v_tmp2[1])),
                            ("valu", ("multiply_add", v_tmp2[2], v_level3[vi], v_tmp1[2], v_tmp2[2])),
                            ("valu", ("multiply_add", v_tmp2[3], v_level3[vi], v_tmp1[3], v_tmp2[3])),
                            ("valu", ("multiply_add", v_tmp2[4], v_level3[vi], v_tmp1[4], v_tmp2[4])),
                            ("valu", ("multiply_add", v_tmp2[5], v_level3[vi], v_tmp1[5], v_tmp2[5])),
                        ])
                        self.add_vliw([
                            ("valu", ("multiply_add", v_tmp2[6], v_level3[vi], v_tmp1[6], v_tmp2[6])),
                            ("valu", ("multiply_add", v_tmp2[7], v_level3[vi], v_tmp1[7], v_tmp2[7])),
                        ])

                    # Copy result to v_node_val (use XOR with zero to avoid dependency issues)
                    self.add_vliw([
                        ("valu", ("^", v_node_val[0], v_tmp2[0], v_zero)),
                        ("valu", ("^", v_node_val[1], v_tmp2[1], v_zero)),
                        ("valu", ("^", v_node_val[2], v_tmp2[2], v_zero)),
                        ("valu", ("^", v_node_val[3], v_tmp2[3], v_zero)),
                        ("valu", ("^", v_node_val[4], v_tmp2[4], v_zero)),
                        ("valu", ("^", v_node_val[5], v_tmp2[5], v_zero)),
                    ])
                    self.add_vliw([
                        ("valu", ("^", v_node_val[6], v_tmp2[6], v_zero)),
                        ("valu", ("^", v_node_val[7], v_tmp2[7], v_zero)),
                    ])

                    # XOR
                    self.add_vliw([
                        ("valu", ("^", v_hash[b][0], v_hash[b][0], v_node_val[0])),
                        ("valu", ("^", v_hash[b][1], v_hash[b][1], v_node_val[1])),
                        ("valu", ("^", v_hash[b][2], v_hash[b][2], v_node_val[2])),
                        ("valu", ("^", v_hash[b][3], v_hash[b][3], v_node_val[3])),
                        ("valu", ("^", v_hash[b][4], v_hash[b][4], v_node_val[4])),
                        ("valu", ("^", v_hash[b][5], v_hash[b][5], v_node_val[5])),
                    ])
                    self.add_vliw([
                        ("valu", ("^", v_hash[b][6], v_hash[b][6], v_node_val[6])),
                        ("valu", ("^", v_hash[b][7], v_hash[b][7], v_node_val[7])),
                    ])

                    # Hash with multiply_add fusion
                    self.emit_hash_8vec(v_hash[b], v_tmp1, v_tmp2, v_hash_const1, v_hash_const3, v_hash_mult)
                    # Index calc
                    self.emit_index_calc_8vec(v_idx[b], v_hash[b], v_cond, v_tmp1, v_one, v_two, v_n_nodes)

            else:
                # Normal round: indirect loads required
                for b in range(N_SUB_BATCHES):
                    # Address calculation
                    for j in range(0, UNROLL, 2):
                        self.add_vliw([
                            ("alu", ("+", s_addr[j][0], self.scratch["forest_values_p"], v_idx[b][j] + 0)),
                            ("alu", ("+", s_addr[j][1], self.scratch["forest_values_p"], v_idx[b][j] + 1)),
                            ("alu", ("+", s_addr[j][2], self.scratch["forest_values_p"], v_idx[b][j] + 2)),
                            ("alu", ("+", s_addr[j][3], self.scratch["forest_values_p"], v_idx[b][j] + 3)),
                            ("alu", ("+", s_addr[j][4], self.scratch["forest_values_p"], v_idx[b][j] + 4)),
                            ("alu", ("+", s_addr[j][5], self.scratch["forest_values_p"], v_idx[b][j] + 5)),
                            ("alu", ("+", s_addr[j][6], self.scratch["forest_values_p"], v_idx[b][j] + 6)),
                            ("alu", ("+", s_addr[j][7], self.scratch["forest_values_p"], v_idx[b][j] + 7)),
                            ("alu", ("+", s_addr[j+1][0], self.scratch["forest_values_p"], v_idx[b][j+1] + 0)),
                            ("alu", ("+", s_addr[j+1][1], self.scratch["forest_values_p"], v_idx[b][j+1] + 1)),
                            ("alu", ("+", s_addr[j+1][2], self.scratch["forest_values_p"], v_idx[b][j+1] + 2)),
                            ("alu", ("+", s_addr[j+1][3], self.scratch["forest_values_p"], v_idx[b][j+1] + 3)),
                        ])
                        self.add_vliw([
                            ("alu", ("+", s_addr[j+1][4], self.scratch["forest_values_p"], v_idx[b][j+1] + 4)),
                            ("alu", ("+", s_addr[j+1][5], self.scratch["forest_values_p"], v_idx[b][j+1] + 5)),
                            ("alu", ("+", s_addr[j+1][6], self.scratch["forest_values_p"], v_idx[b][j+1] + 6)),
                            ("alu", ("+", s_addr[j+1][7], self.scratch["forest_values_p"], v_idx[b][j+1] + 7)),
                        ])

                    # Load node values with hash overlap
                    # Load v0-v1 first
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

                    # XOR v0-v1 and start hash while loading v2-v7
                    self.add_vliw([
                        ("valu", ("^", v_hash[b][0], v_hash[b][0], v_node_val[0])),
                        ("valu", ("^", v_hash[b][1], v_hash[b][1], v_node_val[1])),
                    ])

                    # Hash v0-v1 while loading v2-v7 (with multiply_add fusion for stages 0, 2, 4)
                    load_j, load_k = 2, 0
                    for hi, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                        if op2 == "+" and v_hash_mult[hi] is not None:
                            # Fused: hash = hash * mult + const1 (1 cycle instead of 2) + loads
                            slots = [
                                ("valu", ("multiply_add", v_hash[b][0], v_hash[b][0], v_hash_mult[hi], v_hash_const1[hi])),
                                ("valu", ("multiply_add", v_hash[b][1], v_hash[b][1], v_hash_mult[hi], v_hash_const1[hi])),
                            ]
                            for _ in range(2):
                                if load_j < UNROLL:
                                    slots.append(("load", ("load", v_node_val[load_j] + load_k, s_addr[load_j][load_k])))
                                    load_k += 1
                                    if load_k >= VLEN:
                                        load_k = 0
                                        load_j += 1
                            self.add_vliw(slots)
                        else:
                            # Standard 2-cycle hash stage + loads
                            # Cycle 1: op1/op3 for v0-v1 + loads
                            slots = [
                                ("valu", (op1, v_tmp1[0], v_hash[b][0], v_hash_const1[hi])),
                                ("valu", (op3, v_tmp2[0], v_hash[b][0], v_hash_const3[hi])),
                                ("valu", (op1, v_tmp1[1], v_hash[b][1], v_hash_const1[hi])),
                                ("valu", (op3, v_tmp2[1], v_hash[b][1], v_hash_const3[hi])),
                            ]
                            for _ in range(2):
                                if load_j < UNROLL:
                                    slots.append(("load", ("load", v_node_val[load_j] + load_k, s_addr[load_j][load_k])))
                                    load_k += 1
                                    if load_k >= VLEN:
                                        load_k = 0
                                        load_j += 1
                            self.add_vliw(slots)

                            # Cycle 2: op2 for v0-v1 + loads
                            slots = [
                                ("valu", (op2, v_hash[b][0], v_tmp1[0], v_tmp2[0])),
                                ("valu", (op2, v_hash[b][1], v_tmp1[1], v_tmp2[1])),
                            ]
                            for _ in range(2):
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
                            ("valu", ("^", v_hash[b][j], v_hash[b][j], v_node_val[j])),
                            ("valu", ("^", v_hash[b][j+1], v_hash[b][j+1], v_node_val[j+1])),
                        ])

                    # Hash v2-v7 stage 0 with multiply_add + overlapped v0-v1 index calc
                    # Stage 0 uses + combiner, so use multiply_add (2 cycles for 6 vectors)
                    # Plus 5 sequential steps for v0-v1 index calc
                    self.add_vliw([
                        ("valu", ("multiply_add", v_hash[b][2], v_hash[b][2], v_hash_mult[0], v_hash_const1[0])),
                        ("valu", ("multiply_add", v_hash[b][3], v_hash[b][3], v_hash_mult[0], v_hash_const1[0])),
                        ("valu", ("multiply_add", v_hash[b][4], v_hash[b][4], v_hash_mult[0], v_hash_const1[0])),
                        ("valu", ("multiply_add", v_hash[b][5], v_hash[b][5], v_hash_mult[0], v_hash_const1[0])),
                        ("valu", ("&", v_cond[0], v_hash[b][0], v_one)),
                        ("valu", ("&", v_cond[1], v_hash[b][1], v_one)),
                    ])
                    self.add_vliw([
                        ("valu", ("multiply_add", v_hash[b][6], v_hash[b][6], v_hash_mult[0], v_hash_const1[0])),
                        ("valu", ("multiply_add", v_hash[b][7], v_hash[b][7], v_hash_mult[0], v_hash_const1[0])),
                        ("valu", ("+", v_cond[0], v_one, v_cond[0])),
                        ("valu", ("+", v_cond[1], v_one, v_cond[1])),
                    ])
                    self.add_vliw([
                        ("valu", ("multiply_add", v_idx[b][0], v_idx[b][0], v_two, v_cond[0])),
                        ("valu", ("multiply_add", v_idx[b][1], v_idx[b][1], v_two, v_cond[1])),
                    ])
                    self.add_vliw([
                        ("valu", ("<", v_cond[0], v_idx[b][0], v_n_nodes)),
                        ("valu", ("<", v_cond[1], v_idx[b][1], v_n_nodes)),
                    ])
                    self.add_vliw([
                        ("valu", ("*", v_idx[b][0], v_idx[b][0], v_cond[0])),
                        ("valu", ("*", v_idx[b][1], v_idx[b][1], v_cond[1])),
                    ])

                    # Hash stages 1-5 for v2-v7 (with multiply_add fusion for stages 2 and 4)
                    for hi in range(1, 6):
                        op1, val1, op2, op3, val3 = HASH_STAGES[hi]
                        if op2 == "+" and v_hash_mult[hi] is not None:
                            # Fused: hash = hash * mult + const1 (2 cycles instead of 3)
                            self.add_vliw([
                                ("valu", ("multiply_add", v_hash[b][2], v_hash[b][2], v_hash_mult[hi], v_hash_const1[hi])),
                                ("valu", ("multiply_add", v_hash[b][3], v_hash[b][3], v_hash_mult[hi], v_hash_const1[hi])),
                                ("valu", ("multiply_add", v_hash[b][4], v_hash[b][4], v_hash_mult[hi], v_hash_const1[hi])),
                                ("valu", ("multiply_add", v_hash[b][5], v_hash[b][5], v_hash_mult[hi], v_hash_const1[hi])),
                                ("valu", ("multiply_add", v_hash[b][6], v_hash[b][6], v_hash_mult[hi], v_hash_const1[hi])),
                                ("valu", ("multiply_add", v_hash[b][7], v_hash[b][7], v_hash_mult[hi], v_hash_const1[hi])),
                            ])
                        else:
                            # Standard 3-op hash stage (3 cycles)
                            self.add_vliw([
                                ("valu", (op1, v_tmp1[2], v_hash[b][2], v_hash_const1[hi])),
                                ("valu", (op3, v_tmp2[2], v_hash[b][2], v_hash_const3[hi])),
                                ("valu", (op1, v_tmp1[3], v_hash[b][3], v_hash_const1[hi])),
                                ("valu", (op3, v_tmp2[3], v_hash[b][3], v_hash_const3[hi])),
                                ("valu", (op1, v_tmp1[4], v_hash[b][4], v_hash_const1[hi])),
                                ("valu", (op3, v_tmp2[4], v_hash[b][4], v_hash_const3[hi])),
                            ])
                            self.add_vliw([
                                ("valu", (op1, v_tmp1[5], v_hash[b][5], v_hash_const1[hi])),
                                ("valu", (op3, v_tmp2[5], v_hash[b][5], v_hash_const3[hi])),
                                ("valu", (op1, v_tmp1[6], v_hash[b][6], v_hash_const1[hi])),
                                ("valu", (op3, v_tmp2[6], v_hash[b][6], v_hash_const3[hi])),
                                ("valu", (op1, v_tmp1[7], v_hash[b][7], v_hash_const1[hi])),
                                ("valu", (op3, v_tmp2[7], v_hash[b][7], v_hash_const3[hi])),
                            ])
                            self.add_vliw([
                                ("valu", (op2, v_hash[b][2], v_tmp1[2], v_tmp2[2])),
                                ("valu", (op2, v_hash[b][3], v_tmp1[3], v_tmp2[3])),
                                ("valu", (op2, v_hash[b][4], v_tmp1[4], v_tmp2[4])),
                                ("valu", (op2, v_hash[b][5], v_tmp1[5], v_tmp2[5])),
                                ("valu", (op2, v_hash[b][6], v_tmp1[6], v_tmp2[6])),
                                ("valu", (op2, v_hash[b][7], v_tmp1[7], v_tmp2[7])),
                            ])

                    # Optimized index calc for v2-v7 (6 vectors = 6 ops per step = 1 cycle per step)
                    self.add_vliw([
                        ("valu", ("&", v_cond[2], v_hash[b][2], v_one)),
                        ("valu", ("&", v_cond[3], v_hash[b][3], v_one)),
                        ("valu", ("&", v_cond[4], v_hash[b][4], v_one)),
                        ("valu", ("&", v_cond[5], v_hash[b][5], v_one)),
                        ("valu", ("&", v_cond[6], v_hash[b][6], v_one)),
                        ("valu", ("&", v_cond[7], v_hash[b][7], v_one)),
                    ])
                    self.add_vliw([
                        ("valu", ("+", v_cond[2], v_one, v_cond[2])),
                        ("valu", ("+", v_cond[3], v_one, v_cond[3])),
                        ("valu", ("+", v_cond[4], v_one, v_cond[4])),
                        ("valu", ("+", v_cond[5], v_one, v_cond[5])),
                        ("valu", ("+", v_cond[6], v_one, v_cond[6])),
                        ("valu", ("+", v_cond[7], v_one, v_cond[7])),
                    ])
                    self.add_vliw([
                        ("valu", ("multiply_add", v_idx[b][2], v_idx[b][2], v_two, v_cond[2])),
                        ("valu", ("multiply_add", v_idx[b][3], v_idx[b][3], v_two, v_cond[3])),
                        ("valu", ("multiply_add", v_idx[b][4], v_idx[b][4], v_two, v_cond[4])),
                        ("valu", ("multiply_add", v_idx[b][5], v_idx[b][5], v_two, v_cond[5])),
                        ("valu", ("multiply_add", v_idx[b][6], v_idx[b][6], v_two, v_cond[6])),
                        ("valu", ("multiply_add", v_idx[b][7], v_idx[b][7], v_two, v_cond[7])),
                    ])
                    self.add_vliw([
                        ("valu", ("<", v_tmp1[2], v_idx[b][2], v_n_nodes)),
                        ("valu", ("<", v_tmp1[3], v_idx[b][3], v_n_nodes)),
                        ("valu", ("<", v_tmp1[4], v_idx[b][4], v_n_nodes)),
                        ("valu", ("<", v_tmp1[5], v_idx[b][5], v_n_nodes)),
                        ("valu", ("<", v_tmp1[6], v_idx[b][6], v_n_nodes)),
                        ("valu", ("<", v_tmp1[7], v_idx[b][7], v_n_nodes)),
                    ])
                    self.add_vliw([
                        ("valu", ("*", v_idx[b][2], v_idx[b][2], v_tmp1[2])),
                        ("valu", ("*", v_idx[b][3], v_idx[b][3], v_tmp1[3])),
                        ("valu", ("*", v_idx[b][4], v_idx[b][4], v_tmp1[4])),
                        ("valu", ("*", v_idx[b][5], v_idx[b][5], v_tmp1[5])),
                        ("valu", ("*", v_idx[b][6], v_idx[b][6], v_tmp1[6])),
                        ("valu", ("*", v_idx[b][7], v_idx[b][7], v_tmp1[7])),
                    ])

        # ================================================================
        # PHASE END: Store all hash values
        # ================================================================
        for b in range(N_SUB_BATCHES):
            for j in range(0, UNROLL, 2):
                self.add_vliw([
                    ("store", ("vstore", addr_base_val[b][j], v_hash[b][j])),
                    ("store", ("vstore", addr_base_val[b][j+1], v_hash[b][j+1])),
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
