# VLIW SIMD Kernel Optimization Log

## Starting Point
- Initial cycles: ~4000
- Target: <1400 cycles

## Optimizations Applied

### 1. Multiply-Add Fusion (3994 → 3695 cycles)
- Fused hash stages 0, 2, 4 which use + combiner
- `(hash + const1) + (hash << shift)` becomes `hash * mult + const1`
- Saves 1 cycle per fused stage (3 stages per hash = 3 cycles per hash)

### 2. Cross-Sub-Batch Pipelining (3352 → 3052 cycles)
- During hash v2-v7 stages 1-5 (VALU-only), overlap with:
  - Address calculation for next sub-batch (ALU)
  - v0-v1 tree value loads for next sub-batch (Load)
- For sub-batches 1-3, skip pre-computed v0-v1 address calc and loads
- Key: track prev_alu_ops_done for VLIW timing (writes at end of cycle)

### 3. Load/XOR Overlap
- XOR vectors as they finish loading
- Track which vectors are fully loaded before XORing

### 4. Level 1 vselect Pipelining
- Overlap vselect operations with XOR operations

### 5. Cross-Round Pipelining (3052 → 2952 cycles)
- Extended pipelining to work across round boundaries
- During round N's last sub-batch (b=3), pipeline for round N+1's first sub-batch
- When next round starts, skip pre-loaded work
- Saves ~80 cycles from normal→normal transitions
- Also added level 2 → level 3 pipelining (saves additional ~20 cycles)

### 6. Skip Last Round Index Calculation (2952 → 2916 cycles)
- Last round (round 15) doesn't need index calculation
- Indices are not used after the final round
- Saves 160 VALU ops (8 vectors × 5 ops × 4 sub-batches)
- Combined hash stage 0 into single 6-op cycle for last round

## Current State
- Cycles: 2916
- Speedup: 50.7x over baseline
- Total VALU ops: 9724 (reduced from 9884)
- VALU efficiency: 55.6%

## Bundle Analysis (2973 cycles)
- VALU-only: 1458 cycles (49.0%)
- Load-only: ~600 cycles (20.2%)
- 0 VALU slots: 653 cycles (22.0%)
- 2 VALU slots: 689 cycles (23.2%) - main inefficiency
- 6 VALU slots: 1264 cycles (42.5%) - full utilization

## VALU Efficiency Analysis
- Total VALU ops: 9884
- Maximum capacity: 17838 (2973 cycles × 6 slots)
- VALU efficiency: 55.4%
- The 689 2-slot cycles are from v0-v1 hash/index while loading v2-v7

## Remaining Challenges
1. 2-slot VALU inefficiency: 689 cycles use only 2 slots (data dependency limited)
2. Load-only cycles: Can't overlap with VALU due to data dependencies
3. Special rounds (level 0-2) have VALU-only work with idle Load/ALU

## Theoretical Minimum Analysis (Detailed)
- VALU ops per element: 18 (1 XOR + 12 hash + 5 index)
- Level 2 extra ops: 9 (arithmetic selection)
- Total element-ops: 256 × (14 rounds × 18 + 2 rounds × 27) = 78,336
- SIMD ops: 78,336 / 8 = 9,792
- Minimum VALU cycles (at 6/cycle): ~1,632
- Load cycles (10 normal rounds × 4 sub-batches × 32 = 1,280): ~1,280
- With perfect overlap: ~1,800 cycles theoretical minimum
- **Target <1400 is BELOW theoretical minimum** - may need different algorithm

## Key Insight
The 1400 cycle target appears to be below the theoretical VALU minimum of ~1630 cycles.
This suggests either:
1. A different algorithmic approach is needed (not just pipelining)
2. Mathematical simplifications to reduce VALU operations
3. The target is extremely aggressive/aspirational
