#!/usr/bin/env python
"""
Test script to verify shapes in GNNParser.parse_obs() with use_od_prices=True
"""
import torch
import numpy as np
import sys
sys.path.append('/work3/s233791/rl-pricing-amod')

from src.envs.amod_env_multi import Scenario, AMoD
from src.algos.a2c_gnn_multi_agent import GNNParser

# Test parameters
seed = 10
json_hr = 19
json_tstep = 3
demand_ratio = 2.0
beta = 0.2
mode = 1  # Test pricing mode
jitter = 1e-5
max_wait = 60
fix_agent = 2
choice_intercept = 16.32
wage = 21.40
dynamic_wage = False

print("="*80)
print("TESTING GNNParser with use_od_prices=True")
print("="*80)

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

print(f"\nEnvironment created:")
print(f"  Number of regions: {env.nregion}")
print(f"  Number of agents: {len(env.agents)}")
print(f"  Time horizon: {env.tf}")

# Create parser with use_od_prices=True
parser_with_od = GNNParser(
    env=env,
    T=6,
    scale_factor=0.01,
    agent_id=0,
    json_file=f'/work3/s233791/rl-pricing-amod/data/scenario_san_francisco.json',
    use_od_prices=True
)

# Create parser without use_od_prices
parser_without_od = GNNParser(
    env=env,
    T=6,
    scale_factor=0.01,
    agent_id=0,
    json_file=f'/work3/s233791/rl-pricing-amod/data/scenario_san_francisco.json',
    use_od_prices=False
)

# Reset environment and get observation
agent_obs = env.reset()

# In training, match_step_simple() is called first, which updates the observation
# For mode 0, it doesn't take a price argument
# For other modes, we need to provide initial prices per agent
if mode == 0:
    obs_dict, paxreward, done, info, system_info, _, _ = env.match_step_simple()
else:
    # Initialize base prices for all OD pairs per agent
    initial_prices = {
        agent_id: np.ones((env.nregion, env.nregion)) * 0.5 
        for agent_id in env.agents
    }
    obs_dict, paxreward, done, info, system_info, _, _ = env.match_step_simple(price=initial_prices)

obs = obs_dict[0]  # Get observation for agent 0 after match step
print(f"\nObservation structure (after match_step_simple):")
acc, time, dacc, demand = obs
print(f"  acc: dict with {len(acc)} regions")
print(f"  time: {time}")
print(f"  dacc: dict with {len(dacc)} regions")
print(f"  demand: dict with {len(demand)} OD pairs")

# Test with use_od_prices=True
print("\n" + "="*80)
print("TEST 1: use_od_prices=True")
print("="*80)
try:
    data_with_od = parser_with_od.parse_obs(obs)
    print(f"✓ Parser succeeded!")
    print(f"  data.x shape: {data_with_od.x.shape}")
    print(f"  data.edge_index shape: {data_with_od.edge_index.shape}")
    
    # Verify expected shape
    T = 6
    expected_features = 1 + T + 1 + 1 + env.nregion + env.nregion  # current_avb + future_avb + queue + demand + 2*OD_prices
    print(f"\n  Expected features per node: {expected_features}")
    print(f"    - Current availability: 1")
    print(f"    - Future availability: {T}")
    print(f"    - Queue length: 1")
    print(f"    - Current demand: 1")
    print(f"    - Own OD prices: {env.nregion}")
    print(f"    - Competitor OD prices: {env.nregion}")
    
    if data_with_od.x.shape == (env.nregion, expected_features):
        print(f"\n  ✓ Shape matches expected: [{env.nregion}, {expected_features}]")
    else:
        print(f"\n  ✗ Shape mismatch! Expected [{env.nregion}, {expected_features}]")
        
except Exception as e:
    print(f"✗ Parser failed with error:")
    print(f"  {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

# Test with use_od_prices=False
print("\n" + "="*80)
print("TEST 2: use_od_prices=False")
print("="*80)
try:
    data_without_od = parser_without_od.parse_obs(obs)
    print(f"✓ Parser succeeded!")
    print(f"  data.x shape: {data_without_od.x.shape}")
    print(f"  data.edge_index shape: {data_without_od.edge_index.shape}")
    
    # Verify expected shape
    expected_features = 1 + T + 1 + 1 + 1 + 1  # current_avb + future_avb + queue + demand + 2*aggregated_prices
    print(f"\n  Expected features per node: {expected_features}")
    print(f"    - Current availability: 1")
    print(f"    - Future availability: {T}")
    print(f"    - Queue length: 1")
    print(f"    - Current demand: 1")
    print(f"    - Own aggregated price: 1")
    print(f"    - Competitor aggregated price: 1")
    
    if data_without_od.x.shape == (env.nregion, expected_features):
        print(f"\n  ✓ Shape matches expected: [{env.nregion}, {expected_features}]")
    else:
        print(f"\n  ✗ Shape mismatch! Expected [{env.nregion}, {expected_features}]")
        
except Exception as e:
    print(f"✗ Parser failed with error:")
    print(f"  {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*80)
print("SHAPE VERIFICATION COMPLETE")
print("="*80)
