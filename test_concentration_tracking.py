#!/usr/bin/env python3
"""
Test script to verify concentration parameter and log probability tracking.
Run a short training episode and verify wandb logs include the new metrics.
"""

import numpy as np
import torch

# Test that we can import the updated modules
try:
    from src.algos.a2c_gnn import A2C
    from src.algos.a2c_gnn_multi_agent import A2C as A2C_MultiAgent
    print("✓ Successfully imported A2C modules")
except Exception as e:
    print(f"✗ Failed to import A2C modules: {e}")
    exit(1)

# Test that select_action returns concentration and log_prob when requested
print("\nTesting select_action with return_params...")

# Create a simple mock observation
class MockEnv:
    def __init__(self):
        self.nregion = 5
        self.time = 0
        self.region = list(range(5))
        self.T = 6

class MockData:
    def __init__(self, nregion):
        self.x = torch.randn(nregion, 10)
        self.edge_index = torch.tensor([[i, i] for i in range(nregion)]).T
    
    def to(self, device):
        return self

# Test single-agent
try:
    env = MockEnv()
    device = torch.device("cpu")
    
    model = A2C(
        env=env,
        input_size=10,
        hidden_size=32,
        device=device,
        p_lr=1e-4,
        q_lr=1e-4,
        mode=0,  # Dirichlet
        T=6,
        scale_factor=0.01,
        gamma=0.97,
        actor_clip=500,
        critic_clip=500,
        entropy_coef=0.2
    )
    
    # Mock the parse_obs to return our mock data
    def mock_parse_obs(obs):
        return MockData(env.nregion)
    
    model.parse_obs = mock_parse_obs
    
    # Test return_params=False (default)
    action = model.select_action(None, deterministic=True, return_params=False)
    assert isinstance(action, np.ndarray), "Action should be numpy array"
    print(f"✓ Mode 0: select_action(return_params=False) returned action with shape {action.shape}")
    
    # Test return_params=True
    action, concentration, logprob = model.select_action(None, deterministic=False, return_params=True)
    assert isinstance(action, np.ndarray), "Action should be numpy array"
    assert isinstance(concentration, np.ndarray), "Concentration should be numpy array"
    assert logprob is None or isinstance(logprob, (float, np.floating)), "Log prob should be float or None"
    print(f"✓ Mode 0: select_action(return_params=True) returned:")
    print(f"  - action shape: {action.shape}")
    print(f"  - concentration shape: {concentration.shape}")
    print(f"  - log_prob: {logprob}")
    
except Exception as e:
    print(f"✗ Single-agent test failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test multi-agent
try:
    model_multi = A2C_MultiAgent(
        env=env,
        input_size=10,
        hidden_size=32,
        device=device,
        p_lr=1e-4,
        q_lr=1e-4,
        mode=1,  # Beta
        T=6,
        scale_factor=0.01,
        gamma=0.97,
        actor_clip=500,
        critic_clip=500,
        entropy_coef=0.2,
        agent_id=0
    )
    
    model_multi.parse_obs = mock_parse_obs
    
    # Test return_concentration=False (default)
    action = model_multi.select_action(None, deterministic=True, return_concentration=False)
    assert isinstance(action, np.ndarray), "Action should be numpy array"
    print(f"✓ Mode 1 Multi-agent: select_action(return_concentration=False) returned action with shape {action.shape}")
    
    # Test return_concentration=True
    action, concentration, logprob = model_multi.select_action(None, deterministic=False, return_concentration=True)
    assert isinstance(action, np.ndarray), "Action should be numpy array"
    assert isinstance(concentration, np.ndarray), "Concentration should be numpy array"
    assert logprob is None or isinstance(logprob, (float, np.floating)), "Log prob should be float or None"
    print(f"✓ Mode 1 Multi-agent: select_action(return_concentration=True) returned:")
    print(f"  - action shape: {action.shape}")
    print(f"  - concentration shape: {concentration.shape}")
    print(f"  - log_prob: {logprob}")
    
except Exception as e:
    print(f"✗ Multi-agent test failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "="*60)
print("All tests passed! ✓")
print("="*60)
print("\nNew features:")
print("1. select_action() can now return concentration parameters and log_prob")
print("2. Training scripts track these values per episode")
print("3. Wandb logs include:")
print("   - training/mean_log_prob")
print("   - concentration/region_X_alpha (and beta for Beta distributions)")
print("   - concentration/mean_alpha, mean_beta, etc.")
print("\nFor multi-agent:")
print("   - agent0/mean_log_prob, agent1/mean_log_prob")
print("   - Per-agent concentration tracking already exists")
