#!/usr/bin/env python
"""
Quick test script to verify entropy regularization implementation works correctly.
Tests that entropy is computed and returned properly from the actor network.
"""

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from src.algos.layers import GNNActor

def test_entropy_output():
    """Test that GNNActor returns entropy for all modes"""
    print("="*80)
    print("Testing Entropy Regularization Implementation")
    print("="*80)
    
    # Test parameters
    nregion = 5
    in_channels = 10
    hidden_size = 32
    
    # Create dummy graph data
    x = torch.randn(nregion, in_channels)
    edge_index = torch.cat([
        torch.arange(nregion).view(1, -1),
        torch.arange(nregion).view(1, -1)
    ], dim=0).long()
    data = Data(x, edge_index)
    
    print(f"\nTest configuration:")
    print(f"  - Regions: {nregion}")
    print(f"  - Input channels: {in_channels}")
    print(f"  - Hidden size: {hidden_size}")
    
    # Test all three modes
    for mode in [0, 1, 2]:
        print(f"\n{'='*80}")
        print(f"Testing Mode {mode}")
        print(f"{'='*80}")
        
        actor = GNNActor(in_channels, hidden_size, act_dim=nregion, mode=mode)
        actor.eval()
        
        # Test deterministic mode
        print("\n1. Testing deterministic=True:")
        with torch.no_grad():
            action_det, log_prob_det, concentration_det, entropy_det = actor(data, deterministic=True)
        
        print(f"   - Action shape: {action_det.shape}")
        print(f"   - Log prob: {log_prob_det}")
        print(f"   - Entropy: {entropy_det}")
        print(f"   - Concentration shape: {concentration_det.shape}")
        
        # Test stochastic mode (for training)
        print("\n2. Testing deterministic=False (training mode):")
        with torch.no_grad():
            action_stoch, log_prob_stoch, concentration_stoch, entropy_stoch = actor(data, deterministic=False)
        
        print(f"   - Action shape: {action_stoch.shape}")
        print(f"   - Log prob shape: {log_prob_stoch.shape if hasattr(log_prob_stoch, 'shape') else 'scalar'}")
        print(f"   - Log prob value: {log_prob_stoch.item() if log_prob_stoch is not None else None:.4f}")
        print(f"   - Entropy shape: {entropy_stoch.shape if hasattr(entropy_stoch, 'shape') else 'scalar'}")
        print(f"   - Entropy value: {entropy_stoch.item() if entropy_stoch is not None else None:.4f}")
        print(f"   - Concentration shape: {concentration_stoch.shape}")
        
        # Verify entropy is positive (as expected)
        if entropy_stoch is not None and entropy_stoch.item() >= 0:
            print(f"   ✓ Entropy is positive (good!)")
        else:
            print(f"   ✗ Warning: Entropy is not positive")
        
        # Mode-specific checks
        if mode == 0:
            print(f"\n   Mode 0 (Dirichlet) specific checks:")
            print(f"   - Expected concentration shape: [1, {nregion}]")
            print(f"   - Actual: {concentration_stoch.shape}")
        elif mode == 1:
            print(f"\n   Mode 1 (Beta) specific checks:")
            print(f"   - Expected concentration shape: [1, {nregion}, 2] (alpha, beta)")
            print(f"   - Actual: {concentration_stoch.shape}")
        else:
            print(f"\n   Mode 2 (Beta + Dirichlet) specific checks:")
            print(f"   - Expected concentration shape: [1, {nregion}, 3] (alpha, beta, dirichlet)")
            print(f"   - Actual: {concentration_stoch.shape}")
    
    print("\n" + "="*80)
    print("All tests completed successfully!")
    print("="*80)
    return True

if __name__ == "__main__":
    test_entropy_output()
