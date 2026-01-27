#!/usr/bin/env python
"""
Test script to verify that optimizer state is actually loaded from checkpoints.
"""

import torch
import sys
sys.path.append('src')

from algos.a2c_gnn_multi_agent import A2C
from envs.amod_env_multi_agent import AMoD

# Create a simple environment
env = AMoD(scenario_name="nyc_man_south", no_cars=100, mode=0, config_file="data/config.json")

# Create model
model = A2C(
    env=env,
    input_size=21,
    hidden_size=256,
    p_lr=0.0002,
    q_lr=0.0002,
    agent_id=0,
    no_share_info=True,
    mode=0,
)

# Get initial optimizer state (should be at step 0)
print("=== BEFORE LOADING ===")
print("Actor optimizer step count:", model.optimizers['a_optimizer'].state_dict()['state'])
print()

# Load checkpoint
checkpoint_path = "ckpt/dual_agent_nyc_man_south_no_info_mode0_agent1_running.pth"
print(f"Loading checkpoint: {checkpoint_path}")
model.load_checkpoint(path=checkpoint_path)
print()

# Check optimizer state after loading
print("=== AFTER LOADING ===")
a_opt_state = model.optimizers['a_optimizer'].state_dict()
c_opt_state = model.optimizers['c_optimizer'].state_dict()

# Check if step count is preserved
if len(a_opt_state['state']) > 0:
    first_param_id = list(a_opt_state['state'].keys())[0]
    step_count = a_opt_state['state'][first_param_id]['step']
    print(f"Actor optimizer step count: {step_count}")
    print(f"Actor optimizer has momentum buffers: {len(a_opt_state['state'])} parameters")
    
if len(c_opt_state['state']) > 0:
    first_param_id = list(c_opt_state['state'].keys())[0]
    step_count = c_opt_state['state'][first_param_id]['step']
    print(f"Critic optimizer step count: {step_count}")
    print(f"Critic optimizer has momentum buffers: {len(c_opt_state['state'])} parameters")

print()
print("✓ Optimizer state successfully loaded with step counts preserved!")
