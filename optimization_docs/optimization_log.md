# VLIW SIMD Kernel Optimization Log

## Current Status
- **Cycles**: ~4,950
- **Speedup**: ~30x (from baseline 147,734)
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

## Attempted Optimizations (Not Successful)

### Level 2 Arithmetic Selection
- **Approach**: Load 4 tree values, use arithmetic to select correct one
- **Result**: Too complex, bug-prone, marginal gains
- **Issue**: Scratch space management with intermediate values

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
