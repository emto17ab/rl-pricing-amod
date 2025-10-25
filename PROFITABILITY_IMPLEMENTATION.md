# Profitability-Aware Pricing Implementation

## Overview
This document describes the implementation of profitability constraints to prevent the "race to the bottom" in pricing between competing agents.

## Problem
The two agents were pricing each other down to 0, leading to:
- Prices approaching zero
- Revenue below operating costs  
- Negative rewards
- Unrealistic behavior

## Solution: Hybrid Profitability Constraints

### 1. Price Floor Enforcement
**Location**: `src/envs/amod_env_multi.py` - `match_step_simple()` method

- **Soft price floor**: Minimum price = `price_floor_ratio × baseline_price`
- **Default floor**: 60% of baseline price (configurable via `--price_floor_ratio`)
- **Tracking**: Counts price floor violations per agent per timestep
- **Logic**: 
  ```python
  price_floor = self.price_floor_ratio * baseline_price
  if proposed_price < price_floor:
      self.agent_pricing_violations[agent_id] += 1
      p = max(proposed_price, price_floor)
  ```

### 2. Loss Aversion Penalty
**Location**: `src/envs/amod_env_multi.py` - `match_step_simple()` method

- **Quadratic penalty** for unprofitable trips: `λ × (loss)²`
- **Default loss aversion**: λ = 2.0 (configurable via `--loss_aversion`)
- **Logic**:
  ```python
  base_reward = trip_revenue - trip_cost
  if base_reward < 0:
      loss_penalty = self.loss_aversion * (base_reward ** 2)
      adjusted_reward = base_reward - loss_penalty
  else:
      adjusted_reward = base_reward
  ```

### 3. Dual Reward Tracking
**Location**: `src/envs/amod_env_multi.py` - `match_step_simple()` method

Maintains separate tracking for:
- **True profit**: `revenue - cost` (before penalty)
- **Adjusted profit**: Includes loss aversion penalty (used for training)

This allows monitoring actual profitability while training with penalties.

## New Parameters

### AMoD Environment
```python
AMoD(
    ...,
    loss_aversion=2.0,        # Multiplier for loss penalty (λ)
    price_floor_ratio=0.6     # Minimum price as ratio of base (60%)
)
```

### Command-line Arguments (main_a2c_multi_agent.py)
```bash
--loss_aversion 2.0         # Default: 2.0
--price_floor_ratio 0.6     # Default: 0.6 (60% of baseline)
```

## New Metrics Tracked

### Per Agent (in wandb logs):
1. **`true_profit`**: Actual profit without penalties (revenue - cost)
2. **`adjusted_profit`**: Profit with loss aversion penalty (used for training)
3. **`pricing_violations`**: Count of times price hit the floor
4. **`unprofitable_trips`**: Count of trips with negative margin

### Combined Metrics:
- `combined/total_true_profit`
- `combined/total_adjusted_profit`
- `combined/total_pricing_violations`
- `combined/total_unprofitable_trips`

## Files Modified

1. **`src/envs/amod_env_multi.py`**:
   - Added `loss_aversion` and `price_floor_ratio` parameters to `__init__()`
   - Added violation tracking dictionaries
   - Modified pricing logic to enforce floor
   - Modified reward calculation to apply loss aversion
   - Added profitability metrics to `agent_info`

2. **`main_a2c_multi_agent.py`**:
   - Added command-line arguments for new parameters
   - Added episode-level profitability tracking
   - Added profitability metrics collection during episodes
   - Added profitability metrics to wandb logging
   - Passed new parameters to environment initialization

## Expected Behavior

### Training Phase
1. **Early episodes**: Agents may frequently violate price floor and take unprofitable trips
2. **Learning**: Quadratic penalty strongly discourages systematic underpricing
3. **Convergence**: Agents learn to balance:
   - Competitive pricing (lower prices win passengers)
   - Profitability (prices must cover costs + reasonable margin)

### Metrics to Monitor
- **pricing_violations**: Should decrease over training
- **unprofitable_trips**: Should decrease over training
- **true_profit vs adjusted_profit**: Gap shows penalty impact
- **mean_price_scalar**: Should stabilize above floor threshold

## Tuning Recommendations

### More Competitive Market (allow more price flexibility):
```bash
--loss_aversion 1.5
--price_floor_ratio 0.5
```

### More Profitable Market (enforce stricter pricing):
```bash
--loss_aversion 3.0
--price_floor_ratio 0.7
```

### Current Default (balanced):
```bash
--loss_aversion 2.0
--price_floor_ratio 0.6
```

## Theoretical Foundation

### Loss Aversion
Based on prospect theory (Kahneman & Tversky), losses are psychologically more impactful than equivalent gains. The quadratic penalty (`λ × loss²`) creates:
- **Mild discouragement** for small losses
- **Strong discouragement** for large systematic losses
- **No penalty** for profitable trips

### Price Floor
Represents minimum viable pricing in competitive markets:
- Covers variable costs
- Prevents predatory pricing
- Allows strategic discounting within bounds
- Reflects regulatory or operational constraints

## Validation

To verify the implementation is working:

1. **Check logs**: Monitor `pricing_violations` and `unprofitable_trips`
2. **Compare profits**: `true_profit` vs `adjusted_profit` shows penalty impact  
3. **Price distribution**: Check `mean_price_scalar` trends
4. **Revenue vs Cost**: Ensure `total_revenue > total_operating_cost` over time

## Example Usage

```bash
# Training with default profitability constraints
python main_a2c_multi_agent.py \
    --city san_francisco \
    --mode 1 \
    --checkpoint_path my_experiment \
    --loss_aversion 2.0 \
    --price_floor_ratio 0.6

# Testing with stricter constraints
python main_a2c_multi_agent.py \
    --city san_francisco \
    --mode 1 \
    --test \
    --checkpoint_path my_experiment \
    --loss_aversion 3.0 \
    --price_floor_ratio 0.7
```

## Notes

- The implementation maintains backward compatibility - existing code without these parameters will use defaults
- All metrics are logged to wandb for analysis
- The test function (`test_agents`) also uses these constraints when evaluating trained models
- Price floor is applied to the final price AFTER the agent's scalar is applied, not to the scalar itself
