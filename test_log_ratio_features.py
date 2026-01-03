#!/usr/bin/env python
"""
Test that log-ratio features are calculated correctly:
1. Log-ratio for current competitive position
2. Historical log-ratio changes over 4 timesteps
"""
import torch
import numpy as np
import sys
sys.path.append('/work3/s233791/rl-pricing-amod')

from src.envs.amod_env_multi import Scenario, AMoD
from src.algos.a2c_gnn_multi_agent import A2C
import argparse

def test_log_ratio_features():
    print("\n" + "="*80)
    print("TESTING LOG-RATIO FEATURES")
    print("="*80)
    
    # Test parameters
    seed = 10
    json_hr = 19
    json_tstep = 3
    demand_ratio = 2.0
    beta = 0.2
    jitter = 1e-5
    max_wait = 60
    mode = 1  # Pricing only
    
    # Create scenario
    scenario = Scenario(
        json_file=f'/work3/s233791/rl-pricing-amod/data/scenario_san_francisco.json',
        demand_ratio=demand_ratio,
        json_hr=json_hr,
        json_tstep=json_tstep,
        sd=seed,
    )
    
    # Create environment
    env = AMoD(
        scenario, 
        mode=mode, 
        beta=beta,
        jitter=jitter,
        max_wait=max_wait,
        fix_agent=2,  # No fixed agent
        choice_intercept=16.32,
        wage=21.40,
        dynamic_wage=False,
        choice_price_mult=1.0,
        seed=seed,
    )
    
    # Create A2C model for agent 0
    model = A2C(
        env=env,
        input_size=6 + 2 + 6 * env.nregion,  # 68 features
        device=torch.device("cpu"),
        hidden_size=256,
        T=6,
        mode=mode,
        gamma=0.99,
        p_lr=1e-4,
        q_lr=1e-3,
        actor_clip=0.5,
        critic_clip=0.5,
        scale_factor=0.01,
        agent_id=0,
        json_file=f'/work3/s233791/rl-pricing-amod/data/scenario_san_francisco.json',
        use_od_prices=True,
    )
    
    print(f"\n1. INITIALIZED")
    print(f"   Regions: {env.nregion}")
    print(f"   Feature size: {6 + 2 + 6 * env.nregion}")
    
    # Reset environment
    obs_dict = env.reset()
    obs = obs_dict[0]  # Get agent 0's observation
    
    # Parse observation at t=0
    print(f"\n2. TIMESTEP 0 (Initial state)")
    data_t0 = model.obs_parser.parse_obs(obs)
    print(f"   Feature shape: {data_t0.x.shape}")
    
    # Check that we have zeros for historical changes (not enough history yet)
    # Features: 1 + 6 + 1 + 10 + 10 + 40 = 68
    # Historical changes are last 40 features (indices 28:68)
    historical_features_t0 = data_t0.x[:, 28:68]
    print(f"   Historical features (should be zeros): sum = {historical_features_t0.sum().item():.6f}")
    
    # Take a few steps to build history
    for t in range(1, 6):
        # Take random action
        action = np.random.uniform(0.5, 1.5, size=env.nregion)
        obs_dict, reward_dict, done, info = env.step({0: action, 1: action})
        obs = obs_dict[0]  # Agent 0's observation
        
        # Parse observation
        data = model.obs_parser.parse_obs(obs)
        
        # Extract log-ratio features (indices 18:28)
        log_ratio_features = data.x[:, 18:28]
        
        # Extract historical features (indices 28:68, 4 timesteps * 10 regions)
        historical_features = data.x[:, 28:68]
        
        print(f"\n   TIMESTEP {t}")
        print(f"   Log-ratio (current position) stats:")
        print(f"     mean: {log_ratio_features.mean().item():.6f}")
        print(f"     std:  {log_ratio_features.std().item():.6f}")
        print(f"     min:  {log_ratio_features.min().item():.6f}")
        print(f"     max:  {log_ratio_features.max().item():.6f}")
        
        if t >= 5:
            # After 5 timesteps, we should have 4 historical changes
            print(f"   Historical log-ratio changes (4 timesteps):")
            print(f"     sum:  {historical_features.sum().item():.6f}")
            print(f"     mean: {historical_features.mean().item():.6f}")
            print(f"     std:  {historical_features.std().item():.6f}")
            
            # Check a specific region
            region_0_historical = historical_features[0, :]  # All 40 features for region 0
            print(f"   Region 0 historical changes (4 timesteps * 10 OD pairs):")
            print(f"     First 10 values (t1-t2 changes): {region_0_historical[:10].numpy()}")
    
    print("\n" + "="*80)
    print("TEST COMPLETED")
    print("="*80)
    print("\nKEY OBSERVATIONS:")
    print("1. Log-ratio features represent competitive position: log(own_price / comp_price)")
    print("   - Negative = agent is cheaper")
    print("   - Zero = prices are equal")
    print("   - Positive = agent is more expensive")
    print("\n2. Historical features track competitive dynamics over 4 timesteps")
    print("   - Each timestep change: log(curr_ratio / prev_ratio)")
    print("   - Positive = became relatively more expensive")
    print("   - Negative = became relatively cheaper")
    print("\n3. Features are multiplicative (match actor's scalar outputs)")
    print("   - Actor outputs price scalars in [0.2, 2.0]")
    print("   - Log-ratios align with this multiplicative space")

if __name__ == '__main__':
    test_log_ratio_features()
