#!/bin/bash
#BSUB -q hpc
#BSUB -J "dual_agent_san_francisco[3]"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4GB]"
#BSUB -W 72:00
#BSUB -o logs/dual_agent_san_francisco_mode%I_%J.out
#BSUB -e logs/dual_agent_san_francisco_mode%I_%J.err

source /work3/s233791/rl-pricing-amod/thesis_env/bin/activate

# Subtract 1 from job index to get mode 0-2
MODE=$((LSB_JOBINDEX - 1))

python main_a2c_multi_agent.py --reward_scalar 10000 --critic_warmup_episodes 50 --mode $MODE --city "san_francisco" --q_lr 0.0005 --p_lr 0.0003 --actor_clip 5 --critic_clip 1000 --max_episodes 1000000 --use_od_prices --checkpoint_path dual_agent_san_francisco_mode${MODE}