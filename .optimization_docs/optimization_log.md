# VLIW SIMD Kernel Optimization Log

## Target
- Baseline: 147,734 cycles
- **PRIMARY GOAL: ≤ 1,400 cycles (~105x improvement)**
- Stretch Target: < 1,000 cycles (147x improvement)

## Optimization History

### Milestone 1: Initial Vectorization + VLIW + Unrolling
- **Cycles**: ~18,000
- **Speedup**: ~8x
- **Changes**: Basic vectorization with VLEN=8, VLIW instruction packing, UNROLL=8

### Milestone 2: multiply_add Optimization
- **Cycles**: ~12,000
- **Speedup**: ~12x
- **Changes**: Used `multiply_add` instruction for fused multiply-add in hash stages and index calculation

### Milestone 3: Round-based Loop Restructuring
- **Cycles**: ~6,000
- **Speedup**: ~25x
- **Changes**: Moved batch loop outer, rounds inner. Process 64 elements through all 16 rounds before storing.

### Milestone 4: Full Batch Loop Unrolling
- **Cycles**: 4,605
- **Speedup**: 32.1x
- **Changes**: Fully unrolled both batch loop (4 iterations) and round loop (16 rounds), eliminating all loop overhead

### Milestone 5: Interleaved Hash Pipeline
- **Cycles**: 4,082
- **Speedup**: 36.2x
- **Changes**: Interleaved v1 and v2-v5 hash computation with remaining vector loads. XOR and hash v1 while loading v3-v7.

### Milestone 6: Overlap v0-v5 Index Calc with v6-v7 Hash
- **Cycles**: 3,762
- **Speedup**: 39.3x
- **Changes**: During v6-v7 hash computation (10 cycles), also compute index calculations for v0-v5 using available VALU slots.

### Milestone 7: Overlap v0-v5 Address Calc with v6-v7 Index Calc
- **Cycles**: 3,522
- **Speedup**: 41.9x
- **Changes**: During v6-v7 index calculation (4 cycles), compute v0-v5 indirect addresses for the next round using ALU slots. Saves ~4 cycles per round × 15 rounds × 4 batches = 240 cycles.

### Milestone 8: Partial Batch Boundary Overlap
- **Cycles**: 3,516
- **Speedup**: 42.0x
- **Date**: 2026-01-21
- **Changes**: Overlapped previous batch's idx stores with current batch's addr_base setup (2 cycles of ALU during stores). Limited by address computation dependencies.

### Milestone 9: Overlap v6-v7 Address Calc with v0 Loads
- **Cycles**: 3,396
- **Speedup**: 43.5x
- **Date**: 2026-01-21
- **Changes**: For rounds 1-15, overlapped v6-v7 address calculation (16 ALU ops) with v0 loads (4 cycles). Saves 2 cycles per round × 15 rounds × 4 batches = 120 cycles. Key insight from slot utilization analysis showing only 9.9% ALU utilization.

### Milestone 10: Full v0 Pipelining - Skip Redundant Loads
- **Cycles**: 3,276
- **Speedup**: 45.1x
- **Date**: 2026-01-21
- **Changes**: Completed the v0 pipelining by removing redundant v0 loads at round start (rounds 1-15). v0 is now loaded at the END of the previous round (during v6-v7 idx calc), so the start of rounds 1-15 only needs v6-v7 address calc (2 cycles instead of 4). Saves 2 cycles × 15 rounds × 4 batches = 120 cycles.

### Milestone 11: Overlap v0 XOR + Hash Stage 0 with v6-v7 Addr Calc
- **Cycles**: 3,156
- **Speedup**: 46.8x
- **Date**: 2026-01-21
- **Changes**: For rounds 1-15, overlapped v0 XOR and v0 hash stage 0 with v6-v7 address calculation. Also included v1 loads[0-3] in these 2 cycles. Previously took 4 cycles (2 addr calc + 1 XOR + 1 hash stage 0), now takes 2 cycles. Saves 2 cycles × 15 rounds × 4 batches = 120 cycles.

---

## Current Status
- **Current Cycles**: 3,156
- **Current Speedup**: 46.8x
- **Gap to Target**: Need ~1,756 more cycles saved to reach ≤1,400 (2.25x more improvement needed)

## Architecture Constraints
| Resource | Limit per Cycle |
|----------|-----------------|
| ALU slots | 12 |
| VALU slots | 6 |
| Load slots | 2 |
| Store slots | 2 |
| Flow slots | 1 |

## Key Insights
1. Indirect loads (64 per round at 2/cycle = 32 cycles) are the fundamental bottleneck
2. VLIW semantics: reads at cycle start, writes at cycle end - enables overlapping reads with writes to same location
3. Different engine types (ALU, VALU, Load, Store) can operate in parallel
4. multiply_add is powerful: a*b+c in 1 cycle vs 2 cycles for separate multiply and add

## Potential Future Optimizations
1. Deeper pipelining of hash computation
2. Better overlap of batch boundary stores/loads
3. Reduce hash stages if possible (algorithm change)
4. Explore different UNROLL factors
5. Look for any remaining dead cycles in the pipeline
