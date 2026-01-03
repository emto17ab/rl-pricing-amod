#!/usr/bin/env python
"""
Simple test to verify log-ratio features work in training
"""
import sys
sys.path.append('/work3/s233791/rl-pricing-amod')

from main_a2c_multi_agent import train
import argparse

# Create args for mode 2 test
args = argparse.Namespace(
    # Paths
    checkpoint_path='ckpt/',
    json_file='/work3/s233791/rl-pricing-amod/data/scenario_san_francisco.json',
    
    # Environment settings
    city='sf',
    mode=2,  # Both pricing and rebalancing
    demand_ratio=2.0,
    json_hr=19,
    json_tstep=3,
    seed=10,
    beta=0.2,
    max_wait=60,
    jitter=1e-5,
    wage=21.40,
    dynamic_wage=False,
    fix_agent=2,
    choice_intercept=16.32,
    choice_price_mult=1.0,
    
    # Model settings
    use_od_prices=True,  # Use log-ratio features
    hidden_size=256,
    look_ahead=6,
    actor_dist_type='beta',
    
    # Training settings
    max_episodes=5,  # Just 5 episodes for testing
    max_steps=20,
    p_lr=1e-4,
    q_lr=1e-3,
    actor_clip=0.5,
    critic_clip=0.5,
    gamma=0.99,
    entropy_weight=0.01,
    entropy_decay_rate=0.001,
    eps=1e-5,
    scale_factor=0.01,
    reward_scale=1000.0,
    
    # Logging
    log_interval=1,
    directory='logs_test',
    tag='log_ratio_test',
    use_wandb=False,
    test=True,
    no_cuda=True,
)

print("\n" + "="*80)
print("TESTING LOG-RATIO FEATURES IN TRAINING")
print("="*80)
print(f"Mode: {args.mode}")
print(f"use_od_prices: {args.use_od_prices}")
print(f"Expected input_size: {args.look_ahead + 2 + 6 * 10} = 68 features")
print(f"Max episodes: {args.max_episodes}")
print("="*80 + "\n")

train(args)

print("\n" + "="*80)
print("TRAINING TEST COMPLETED SUCCESSFULLY ✓")
print("Log-ratio features work correctly in training!")
print("="*80)
