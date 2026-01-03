#!/usr/bin/env python
"""
Comprehensive test of data flow from JSON -> Parser -> Model -> Training
Tests all three modes (0, 1, 2) with use_od_prices=True
"""
import torch
import numpy as np
import sys
sys.path.append('/work3/s233791/rl-pricing-amod')

from src.envs.amod_env_multi import Scenario, AMoD
from src.algos.a2c_gnn_multi_agent import A2C
import argparse

def test_mode(mode):
    print("\n" + "="*80)
    print(f"TESTING MODE {mode}")
    print("="*80)
    
    # Test parameters
    seed = 10
    json_hr = 19
    json_tstep = 3
    demand_ratio = 2.0
    beta = 0.2
    jitter = 1e-5
    max_wait = 60
    fix_agent = 2  # No fixed agent
    choice_intercept = 16.32
    wage = 21.40
    dynamic_wage = False
    
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
        choice_price_mult=1.0,
        seed=seed,
        fix_agent=fix_agent,
        choice_intercept=choice_intercept,
        wage=wage,
        dynamic_wage=dynamic_wage
    )
    
    print(f"\n1. ENVIRONMENT INITIALIZED")
    print(f"   Number of regions: {env.nregion}")
    print(f"   Number of agents: {len(env.agents)}")
    print(f"   Mode: {mode}")
    
    # Calculate input size correctly based on use_od_prices
    T = 6  # look_ahead
    use_od_prices = True  # Test with OD prices enabled
    
    if use_od_prices:
        # New feature structure with log-ratios:
        # - current_avb: 1
        # - future_avb: T (6)
        # - queue_length: 1
        # - OD demand matrix: nregion (10)
        # - log_price_ratio: nregion (10)
        # - historical log-ratio changes: 4 * nregion (40)
        # Total: 1 + 6 + 1 + 10 + 10 + 40 = 68
        input_size = T + 2 + 6 * env.nregion
    else:
        input_size = T + 5
    
    print(f"   use_od_prices: {use_od_prices}")
    print(f"   Input size (features per node): {input_size}")
    if use_od_prices:
        print(f"   Expected: {T} + 2 + 6*{env.nregion} = {T + 2 + 6*env.nregion}")
    
    # Create A2C model for agent 0
    model = A2C(
        env=env,
        input_size=input_size,  # Use calculated input_size
        device=torch.device('cpu'),
        hidden_size=256,
        T=T,
        mode=mode,
        gamma=0.99,
        p_lr=0.0005,
        q_lr=0.001,
        actor_clip=10000,
        critic_clip=10000,
        scale_factor=0.01,
        agent_id=0,
        json_file=f'/work3/s233791/rl-pricing-amod/data/scenario_san_francisco.json',
        use_od_prices=use_od_prices,
        reward_scale=1000.0,
        entropy_coef_max=0.0,
        entropy_coef_min=0.0,
        entropy_decay_rate=0.0
    )
    
    print(f"\n2. A2C MODEL INITIALIZED")
    print(f"   Agent ID: 0")
    print(f"   use_od_prices: {use_od_prices}")
    print(f"   Mode: {mode}")
    
    # Reset environment
    agent_obs = env.reset()
    
    # Call match_step_simple to get proper observation
    if mode == 0:
        obs_dict, paxreward, done, info, system_info, _, _ = env.match_step_simple()
    else:
        initial_prices = {
            agent_id: np.ones((env.nregion, env.nregion)) * 0.5 
            for agent_id in env.agents
        }
        obs_dict, paxreward, done, info, system_info, _, _ = env.match_step_simple(price=initial_prices)
    
    obs = obs_dict[0]
    
    print(f"\n3. OBSERVATION FROM ENVIRONMENT")
    acc, time, dacc, demand = obs
    print(f"   acc: {type(acc)} with {len(acc)} regions")
    print(f"   time: {time}")
    print(f"   Sample acc values: {[(n, acc[n].get(time+1, 'N/A')) for n in list(env.region)[:3]]}")
    
    # Parse observation
    data = model.obs_parser.parse_obs(obs)
    
    print(f"\n4. PARSED OBSERVATION (GNN Data)")
    print(f"   data.x shape: {data.x.shape}")
    print(f"   data.edge_index shape: {data.edge_index.shape}")
    print(f"   Feature breakdown:")
    print(f"     - Current availability: 1")
    print(f"     - Future availability: 6 (T=6)")
    print(f"     - Queue length: 1")
    print(f"     - OD demand matrix: {env.nregion}")
    print(f"     - Price difference (own - competitor): {env.nregion}")
    print(f"     - Historical price changes (5 timesteps): {5 * env.nregion}")
    print(f"     TOTAL: {1+6+1+env.nregion+env.nregion+5*env.nregion} features")
    
    # Forward pass through actor
    print(f"\n5. ACTOR FORWARD PASS")
    action, log_prob, concentration, entropy = model.actor(data)
    
    print(f"   Action shape: {action.shape}")
    print(f"   Action values (first 3 regions): {action[:3].detach().numpy()}")
    print(f"   Concentration shape: {concentration.shape}")
    print(f"   Concentration values (first 3): {concentration[:3].detach().numpy()}")
    
    if mode == 0:
        print(f"   Mode 0: Rebalancing only")
        print(f"     - Action should be [nregion] with Dirichlet distribution")
        print(f"     - Sum of actions should ≈ 1.0: {action.sum().item():.6f}")
        assert action.shape == (env.nregion,), f"Expected ({env.nregion},), got {action.shape}"
    elif mode == 1:
        print(f"   Mode 1: Pricing only")
        print(f"     - Action should be [nregion] with Beta distribution")
        print(f"     - Actions should be in [0.2, 2.0]: min={action.min().item():.4f}, max={action.max().item():.4f}")
        assert action.shape == (env.nregion,), f"Expected ({env.nregion},), got {action.shape}"
    elif mode == 2:
        print(f"   Mode 2: Both pricing and rebalancing")
        print(f"     - Action should be [nregion, 2]")
        print(f"     - First column (prices) in [0.2, 2.0]: min={action[:, 0].min().item():.4f}, max={action[:, 0].max().item():.4f}")
        print(f"     - Second column (reb) sums to 1: {action[:, 1].sum().item():.6f}")
        assert action.shape == (env.nregion, 2), f"Expected ({env.nregion}, 2), got {action.shape}"
    
    # Forward pass through critic
    print(f"\n6. CRITIC FORWARD PASS")
    value = model.critic(data)
    print(f"   Value shape: {value.shape}")
    print(f"   Value estimate: {value.item():.4f}")
    assert value.shape == (1,), f"Expected (1,), got {value.shape}"
    
    # Test select_action (used during training)
    print(f"\n7. SELECT_ACTION (Combined Actor + Critic)")
    action_array, concentration_value, logprob_value = model.select_action(
        obs, deterministic=False, return_concentration=True
    )
    print(f"   action_array shape: {action_array.shape}")
    print(f"   concentration_value shape: {concentration_value.shape}")
    print(f"   log_prob: {logprob_value}")
    
    if mode == 0:
        assert action_array.shape == (env.nregion,), f"Expected ({env.nregion},), got {action_array.shape}"
        print(f"   ✓ Mode 0: Flattened to 1D array for rebalancing")
    elif mode == 1:
        assert action_array.shape == (env.nregion,), f"Expected ({env.nregion},), got {action_array.shape}"
        print(f"   ✓ Mode 1: Flattened to 1D array for pricing")
    elif mode == 2:
        assert action_array.shape == (env.nregion, 2), f"Expected ({env.nregion}, 2), got {action_array.shape}"
        print(f"   ✓ Mode 2: Kept as 2D array [price, reb] per region")
    
    print(f"\n8. SAVED ACTIONS (for training)")
    print(f"   Number of saved actions: {len(model.saved_actions)}")
    if len(model.saved_actions) > 0:
        saved_action = model.saved_actions[0]
        print(f"   SavedAction(log_prob={saved_action.log_prob}, value={saved_action.value.item():.4f})")
    
    print(f"\n{'='*80}")
    print(f"MODE {mode} TEST COMPLETED SUCCESSFULLY ✓")
    print(f"{'='*80}\n")
    
    return True

if __name__ == "__main__":
    print("\n" + "="*80)
    print("COMPREHENSIVE DATA FLOW TEST")
    print("Testing: JSON -> Environment -> Parser -> Model -> Training")
    print("="*80)
    
    success_count = 0
    for mode in [0, 1, 2]:
        try:
            if test_mode(mode):
                success_count += 1
        except Exception as e:
            print(f"\n✗ MODE {mode} FAILED:")
            print(f"   {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*80)
    print(f"FINAL RESULTS: {success_count}/3 modes passed")
    print("="*80)
