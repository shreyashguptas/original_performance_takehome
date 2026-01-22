# VLIW SIMD Kernel Optimization Log

## Current Status
- **Cycles**: ~4,837
- **Speedup**: ~30.5x (from baseline 147,734)
- **Target**: < 1,400 cycles (~105x speedup)

## Implemented Optimizations

### Level 0 Tree Optimization (Rounds 0, 11)
- **Insight**: At level 0, all 256 elements access forest_values[0]
- **Implementation**: 1 load + broadcast instead of 64 indirect loads
- **Savings**: ~30 cycles per batch-round

### Level 1 Tree Optimization (Rounds 1, 12)
- **Insight**: At level 1, indices are 1 or 2 (only 2 values)
- **Implementation**: 2 loads + vselect instead of 64 indirect loads
- **Savings**: ~24 cycles per batch-round (limited by vselect 1 slot/cycle)

### Level 2 Arithmetic Selection (Rounds 2, 13)
- **Insight**: At level 2, indices are 3, 4, 5, 6 (only 4 values)
- **Implementation**: Load 4 values, use `result = sum(val[i] * (offset == i))`
- **VALU operations**: offset calc + 4 eq checks + 4 multiply/add = ~18 cycles
- **Savings**: ~14 cycles per batch-round vs indirect loads (32 cycles)

### Improved Address Calculation
- **Previous**: 8 ALU ops per cycle (1 vector at a time) = 8 cycles
- **Improved**: 12 ALU ops per cycle (packed addresses) = 6 cycles
- **Savings**: 2 cycles per normal batch-round × 40 = 80 cycles total

## Attempted Optimizations (Not Successful)

### Level 3 Arithmetic Selection
- **Approach**: Load 8 tree values, use arithmetic to select correct one
- **Result**: SLOWER than indirect loads (5057 vs 4837 cycles)
- **Issue**: 8 equality checks + 8 multiply_add = 32 VALU cycles, same as 64 loads
- **Key insight**: Indirect loads can overlap with hash (uses load slots), but arithmetic selection competes with hash (both use VALU)

### UNROLL=16
- **Approach**: Process 128 elements per batch instead of 64
- **Result**: Requires complete rewrite of normal round pipelining
- **Issue**: Hash time scales sub-linearly but complexity explodes

## Bottleneck Analysis

### Load Bottleneck (Normal Rounds)
- 64 indirect loads per batch-round = 32 cycles (at 2 loads/cycle)
- 48 normal batch-rounds × 32 = 1,536 cycles (already > 1,400 target)

### Hash Bottleneck
- 6 stages × 5 cycles = 30 cycles per batch-round
- 64 batch-rounds × 30 = 1,920 cycles (but can overlap with loads)

### Fundamental Limit
- The load bottleneck (2/cycle) fundamentally limits optimization potential
- To reach 1,400 cycles, must dramatically reduce total loads
- Tree-level caching only helps for low levels (0, 1, maybe 2)

## Architecture Notes
- VALU: 6 slots/cycle, supports +, -, *, //, ^, &, |, <<, >>, <, ==
- No >= operator (must use: a >= b ≡ NOT(a < b) ≡ 1 - (a < b))
- Load: 2 slots/cycle
- Flow: 1 slot/cycle (vselect bottleneck)
- Scratch: 1,536 words
