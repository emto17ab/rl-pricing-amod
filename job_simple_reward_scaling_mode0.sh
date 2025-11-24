#!/bin/bash
#BSUB -q hpc
#BSUB -J simple_reward_scaling_mode0_entropy_v5
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 24:00
#BSUB -o logs/simple_reward_scaling_mode0_entropy_v5_%J.out
#BSUB -e logs/simple_reward_scaling_mode0_entropy_v5_%J.err

source /work3/s233791/rl-pricing-amod/thesis_env/bin/activate

# Simplified approach with only reward scaling and advantage clipping
# - Removed GAE and entropy regularization (they caused instability)
# - reward_scale=0.00005: Scales 200k rewards down to ~10 for stable training
# - advantage_clip=10.0: Prevents extreme policy updates from critic errors
# - Standard Monte Carlo returns (no GAE complexity)

python main_a2c.py \
    --mode 0 \
    --city "nyc_manhattan" \
    --q_lr 0.0005 \
    --p_lr 0.0005 \
    --reward_scale 1 \
    --advantage_clip 10.0 \
    --actor_clip 1e10 \
    --critic_clip 1e10 \
    --max_episodes 40022 \
    --use_od_prices \
    --entropy_coef_start 0.2 \
    --entropy_coef_end 0.01 \
    --entropy_decay_episodes 5000 \
    --checkpoint_path simple_reward_scaling_mode0_entropy_v5