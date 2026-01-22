# VLIW SIMD Kernel Optimization Findings

## Current Status
- Cycles: 6,239 (24x speedup from baseline 147,734)
- Target: < 2,164 cycles (68x speedup) for Opus 4 level
- Ultimate target: < 1,487 cycles (99x speedup)
- Need: ~3x more improvement for Opus 4, ~4x for ultimate target

## Architecture Constraints
- ALU slots: 12
- VALU slots: 6
- Load slots: 2
- Store slots: 2
- Flow slots: 1
- VLEN: 8
- Scratch: 1536 words

## Critical Bottleneck Analysis

### Indirect Loads (THE MAIN BOTTLENECK)
- 64 indirect loads per iteration (8 vectors × 8 elements)
- 2 loads/cycle max = 32 cycles MINIMUM just for indirect loads
- This is ~47% of each iteration's time!

### Per-Iteration Breakdown (UNROLL=8, 64 iterations)
| Operation | Cycles | Notes |
|-----------|--------|-------|
| Address calc (base, idx, val) | 3 | 8 ALU ops |
| Load idx/val vectors | 8 | 16 vloads, 2/cycle |
| Indirect addr calc | 6 | 64 ALU ops, 12/cycle |
| Indirect loads | 32 | 64 scalar loads, 2/cycle |
| XOR | 2 | 8 VALU ops |
| Hash (6 stages) | 18 | 8 vectors × 12 ops |
| Index calculation | 10 | mod, mul, cmp, sub, add, bounds |
| Stores | 8 | 16 vstores, 2/cycle |
| Loop overhead | ~3 | control flow |
| **TOTAL** | ~90 | Per iteration |

90 cycles × 64 iterations = 5,760 cycles (theoretical)
Actual: 6,749 cycles (some additional overhead)

## Optimization Strategies from Research

### 1. Cross-Iteration Software Pipelining (PROMISING)
**Source:** [Software Pipelining for VLIW](https://suif.stanford.edu/papers/lam-sp.pdf)

**Concept:** Start iteration N+1's loads while computing iteration N's results.

**Implementation:**
- While doing hash/index calc for iteration N, start loading indices for N+1
- Need double-buffering: two sets of scratch vectors
- Prolog: load first iteration
- Main loop: overlapped compute/load
- Epilog: compute final iteration

**Potential savings:** Could overlap ~32 cycles of loads with computation

### 2. Double Buffering
Use two sets of scratch:
- Set A: v_idx_a, v_val_a, v_node_val_a, v_hash_a
- Set B: v_idx_b, v_val_b, v_node_val_b, v_hash_b

Alternate each iteration:
- Iter N (even): Compute on A, load into B
- Iter N+1 (odd): Compute on B, load into A

### 3. Modulo Scheduling
Schedule operations across iterations with a fixed initiation interval (II).
Target II = max(resource-constrained II, recurrence-constrained II)

Resource-constrained II for our kernel:
- Loads: 80 total (16 vloads + 64 scalar) / 2 slots = 40 cycles
- Stores: 16 vstores / 2 slots = 8 cycles
- Minimum II = 40 (load-bound)

### 4. Loop Fusion/Elimination
Could we process multiple rounds in a single iteration?
- Combine round N's store with round N+1's load
- Problem: need round N's result before round N+1's tree lookup

### 5. Precomputation / Caching
- Could cache tree traversal results?
- Problem: traversal is data-dependent (based on hash results)

## Implemented Optimizations
1. [x] Vectorization (VLEN=8)
2. [x] VLIW instruction packing
3. [x] Loop unrolling (8x)
4. [x] vselect replaced with ALU ops
5. [x] Within-iteration pipelining (hash[0] with loads)
6. [ ] Cross-iteration pipelining (IN PROGRESS)
7. [ ] Double buffering
8. [ ] Modulo scheduling

## Key Insight
The 32-cycle indirect load minimum means we NEED to overlap computation with those loads. Cross-iteration pipelining is the most promising approach.

## Sources
- [Software Pipelining for VLIW](https://suif.stanford.edu/papers/lam-sp.pdf)
- [Automatic Vectorization of Tree Traversals](https://engineering.purdue.edu/~milind/docs/pact13.pdf)
- [IMP: Indirect Memory Prefetcher](https://pages.cs.wisc.edu/~yxy/pubs/imp.pdf)
- [Software Prefetching for Indirect Memory Accesses](https://dl.acm.org/doi/10.1145/3319393)
