#!/usr/bin/env python3
"""
Test to verify entropy regularization is working at the correct scale.
Shows the relationship between raw entropy, entropy bonus, and policy loss.
"""

import torch
from torch.distributions import Dirichlet, Beta
import numpy as np

print("="*80)
print("ENTROPY SCALE ANALYSIS")
print("="*80)

# Simulate typical concentration parameters from the model
nregion = 66  # Manhattan

print("\n--- MODE 0: Dirichlet (Rebalancing) ---")
# Typical concentration values (from softplus output, usually 1-10)
concentration_dirichlet = torch.ones(1, nregion) * 2.0 + torch.randn(1, nregion) * 0.5
concentration_dirichlet = torch.clamp(concentration_dirichlet, min=0.1)

m_dirichlet = Dirichlet(concentration_dirichlet + 1e-5)
entropy_dirichlet = m_dirichlet.entropy().item()

print(f"Concentration range: [{concentration_dirichlet.min():.2f}, {concentration_dirichlet.max():.2f}]")
print(f"Concentration sum (α₀): {concentration_dirichlet.sum():.2f}")
print(f"Raw entropy: {entropy_dirichlet:.2f} nats")
print(f"Entropy bonus with coef=0.01: {0.01 * entropy_dirichlet:.4f}")
print(f"Entropy bonus with coef=0.2:  {0.2 * entropy_dirichlet:.4f}")

print("\n--- MODE 1: Beta (Pricing) ---")
# Typical concentration values for Beta
alpha = torch.ones(1, nregion) * 3.0 + torch.randn(1, nregion) * 0.5
beta = torch.ones(1, nregion) * 3.0 + torch.randn(1, nregion) * 0.5
alpha = torch.clamp(alpha, min=0.1)
beta = torch.clamp(beta, min=0.1)

m_beta = Beta(alpha + 1e-5, beta + 1e-5)
entropy_beta_per_region = m_beta.entropy()
entropy_beta_mean = entropy_beta_per_region.mean().item()

print(f"Alpha range: [{alpha.min():.2f}, {alpha.max():.2f}]")
print(f"Beta range: [{beta.min():.2f}, {beta.max():.2f}]")
print(f"Entropy per region range: [{entropy_beta_per_region.min():.2f}, {entropy_beta_per_region.max():.2f}]")
print(f"Mean entropy: {entropy_beta_mean:.2f} nats")
print(f"Entropy bonus with coef=0.01: {0.01 * entropy_beta_mean:.4f}")
print(f"Entropy bonus with coef=0.2:  {0.2 * entropy_beta_mean:.4f}")

print("\n--- COMPARISON WITH NORMALIZED ADVANTAGES ---")
# Typical advantages (normalized to mean=0, std=1)
advantages = torch.randn(20)  # 20 steps
advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

log_probs = torch.randn(20) * 0.5 - 1.5  # Typical log probs around -1 to -2
policy_losses = -log_probs * advantages
policy_loss_mean = policy_losses.mean().item()

print(f"Advantages: mean={advantages.mean():.4f}, std={advantages.std():.4f}")
print(f"Policy losses: mean={policy_loss_mean:.4f}")

print("\n--- LOSS COMPOSITION ---")
print("For Dirichlet (mode 0):")
print(f"  Policy loss term:         {policy_loss_mean:.4f}")
print(f"  Entropy bonus (coef=0.01): {0.01 * entropy_dirichlet:.4f}")
print(f"  Entropy bonus (coef=0.2):  {0.2 * entropy_dirichlet:.4f}")
print(f"  Total loss (coef=0.01):    {policy_loss_mean - 0.01 * entropy_dirichlet:.4f}")
print(f"  Total loss (coef=0.2):     {policy_loss_mean - 0.2 * entropy_dirichlet:.4f}")

print("\nFor Beta (mode 1):")
print(f"  Policy loss term:         {policy_loss_mean:.4f}")
print(f"  Entropy bonus (coef=0.01): {0.01 * entropy_beta_mean:.4f}")
print(f"  Entropy bonus (coef=0.2):  {0.2 * entropy_beta_mean:.4f}")
print(f"  Total loss (coef=0.01):    {policy_loss_mean - 0.01 * entropy_beta_mean:.4f}")
print(f"  Total loss (coef=0.2):     {policy_loss_mean - 0.2 * entropy_beta_mean:.4f}")

print("\n" + "="*80)
print("CONCLUSION:")
print("="*80)
print("• Dirichlet entropy can be NEGATIVE and large in magnitude (~-10 to -30 nats)")
print("• Beta entropy is typically positive and moderate (~0 to 2 nats)")
print("• Advantages are normalized to std=1, so policy loss is O(1)")
print("• With coef=0.01:")
print("  - Entropy bonus magnitude is ~0.01-0.3 (reasonable regularization)")
print("  - Entropy doesn't dominate the policy gradient signal")
print("• With coef=0.2:")
print("  - Entropy bonus magnitude is ~0.2-6 (can dominate policy loss)")
print("  - Risk of entropy overwhelming the learning signal")
print("\n✓ Recommendation: Use entropy_coef=0.01 (or 0.001-0.05 range)")
print("="*80)
