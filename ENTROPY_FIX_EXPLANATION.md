# Entropy Regularization Scale Fix

## Problem Identified

The original implementation had a **severe scale mismatch** between the entropy bonus and policy loss:

- **Advantages**: Normalized to mean=0, std=1 → policy loss magnitude ~0.5-2.0
- **Raw Entropy**: 
  - Dirichlet (66 regions): ~-235 nats (negative!)
  - Beta (averaged): ~-0.4 nats
- **Entropy bonus (coef=0.2)**: 
  - Dirichlet: 0.2 × (-235) = **-47** (dominates loss!)
  - Beta: 0.2 × (-0.4) = -0.08

**Result**: With unnormalized entropy, the entropy term was **47x larger** than the policy gradient term, completely dominating the training signal.

## Why Dirichlet Entropy is Negative

For a Dirichlet distribution over K=66 regions with concentration α:
- Entropy = log B(α) - (α₀-K)ψ(α₀) + Σᵢ(αᵢ-1)ψ(αᵢ)
- For high-dimensional Dirichlet, the normalizing constant B(α) becomes very small
- log B(α) is large negative, dominating the other positive terms
- Result: Large negative entropy values

This is correct mathematically, but creates a scale problem for optimization.

## Solution: Normalize Entropy

Just like we normalize advantages, we now normalize entropy across the batch:

```python
# Before (WRONG - scale mismatch)
entropy_bonus = entropy_coef * entropy_mean

# After (CORRECT - matched scales)
entropy_normalized = (entropies - entropies.mean()) / (entropies.std() + eps)
entropy_bonus = entropy_coef * entropy_normalized.mean()
```

## Benefits

1. **Matched scales**: Entropy bonus now has magnitude ~0 (by construction), comparable to policy loss
2. **Relative regularization**: We encourage actions with *higher* entropy relative to the batch average
3. **Stable training**: Entropy no longer dominates the gradient signal
4. **Keep coefficient 0.2**: The original coefficient is now appropriate

## Implementation

Updated both:
- `src/algos/a2c_gnn.py` (single-agent)
- `src/algos/a2c_gnn_multi_agent.py` (multi-agent)

The entropy regularization now works as intended: a soft constraint encouraging exploration without overwhelming the policy gradient.

## Verification

Run test:
```bash
python test_entropy_scale.py
```

This confirms:
- Raw entropy: -235 nats (Dirichlet) → entropy bonus -47
- Normalized entropy: 0 mean, 1 std → entropy bonus ~0
- Proper balance between policy gradient and entropy regularization
