# Entropy Regularization Scale Fix

## Problem Identified

The entropy regularization was **dominating** the policy gradient signal due to a scale mismatch:

### Scale Mismatch
- **Advantages**: Normalized to mean=0, std=1 → policy loss magnitude ~O(1)
- **Entropy**: Raw values in nats (natural logarithm units)
  - Dirichlet (66 dimensions): **-200 to -10 nats** (negative and huge!)
  - Beta (per region, averaged): **-1 to 2 nats** (moderate)
- **Original coefficient**: 0.2
- **Entropy bonus magnitude**: 0.2 × (-220) = **-44** for Dirichlet!

This meant entropy was **40x larger** than the policy gradient signal, completely overwhelming the learning.

## Why Entropy Can Be Negative

For high-dimensional distributions like Dirichlet with 66 regions:
- Entropy formula: `log(B(α)) - (α₀ - K)ψ(α₀) + Σᵢ(αᵢ - 1)ψ(αᵢ)`
- Where α₀ = Σαᵢ and K = number of regions
- When concentration is moderate (α ~ 2-5), the distribution is peaked
- High-dimensional peaked distributions have **negative entropy**

## Attempted Solution (Wrong)

❌ **Normalizing entropy to mean=0**:
```python
entropy_normalized = (entropy - entropy.mean()) / entropy.std()
entropy_bonus = coef * entropy_normalized.mean()  # = 0 always!
```

This makes `entropy_normalized.mean()` = 0, so entropy_bonus = 0, defeating the purpose entirely!

## Correct Solution

✓ **Use a much smaller entropy coefficient** to match the scale:

```python
# Keep raw entropy (don't normalize to mean=0)
entropy_mean = torch.stack(entropies_list).mean()
entropy_bonus = self.entropy_coef * entropy_mean

# Combined loss
a_loss = policy_loss_mean - entropy_bonus
```

**New default**: `entropy_coef = 0.01` (instead of 0.2)

### Scale Analysis with coef=0.01:
- **Dirichlet**: entropy_bonus = 0.01 × (-220) = **-2.2** → reasonable
- **Beta**: entropy_bonus = 0.01 × (-0.3) = **-0.003** → reasonable
- **Policy loss**: ~O(1)
- **Total loss**: policy gradient dominates, entropy provides mild regularization ✓

### Why This Works:
- Entropy bonus magnitude (2-3) is comparable to policy loss (1-2)
- Entropy prevents overconfidence without drowning out the reward signal
- The negative sign in `a_loss = policy_loss - entropy_bonus` means:
  - High entropy → more negative bonus → lower loss → encourages exploration ✓
  - Low entropy → less negative bonus → higher loss → discourages overconfidence ✓

## Implementation Changes

### Files Modified:
1. **src/algos/a2c_gnn.py**: Removed entropy normalization, kept raw entropy
2. **src/algos/a2c_gnn_multi_agent.py**: Same change
3. **main_a2c.py**: Changed default from 0.2 → 0.01
4. **main_a2c_multi_agent.py**: Changed default from 0.2 → 0.01

### Recommended Range:
- **Conservative**: 0.001-0.005 (very mild regularization)
- **Standard**: 0.01 (balanced, recommended default)
- **Aggressive**: 0.02-0.05 (strong exploration encouragement)

## Key Insight

**Don't normalize something to mean=0 if you want to use its magnitude as a regularizer!**

The correct approach with mismatched scales is to **tune the coefficient**, not normalize away the information.
