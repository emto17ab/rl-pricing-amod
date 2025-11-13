#!/bin/bash
#BSUB -q hpc
#BSUB -J base_case_single_agent_mode0_new_params
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 24:00
#BSUB -o logs/base_case_single_agent_mode0_new_params_%J.out
#BSUB -e logs/base_case_single_agent_mode0_new_params_%J.err

source /work3/s233791/rl-pricing-amod/thesis_env/bin/activate

python main_a2c.py --mode 0 --city "nyc_manhattan" --q_lr 0.0005 --p_lr 0.0005 --actor_clip 1 --critic_clip 500 --max_episodes 33011 --use_od_prices --checkpoint_path base_case_single_agent_mode0_new_params