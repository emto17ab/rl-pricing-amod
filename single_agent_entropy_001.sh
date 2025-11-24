#!/bin/bash
#BSUB -q hpc
#BSUB -J single_agent_reward_scale_mode2_continued
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 24:00
#BSUB -o logs/single_agent_reward_scale_mode2_continued_%J.out
#BSUB -e logs/single_agent_reward_scale_mode2_continued_%J.err

source /work3/s233791/rl-pricing-amod/thesis_env/bin/activate

python main_a2c.py --load --mode 2 --city "nyc_manhattan" --entropy_coef_start 0.0 --entropy_coef_end 0.000 --entropy_decay_episodes 0 --q_lr 0.0005 --p_lr 0.0005 --actor_clip 1e10 --critic_clip 1e10 --max_episodes 62020 --use_od_prices --reward_scale 0.0005 --checkpoint_path single_agent_mode2_reward_scaling_entropy