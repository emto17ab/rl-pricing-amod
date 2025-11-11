#!/bin/bash
#BSUB -q hpc
#BSUB -J base_case_single_agent_fixed_policy
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 24:00
#BSUB -o logs/base_case_single_agent_fixed_policy_%J.out
#BSUB -e logs/base_case_single_agent_fixed_policy_%J.err

source /work3/s233791/rl-pricing-amod/thesis_env/bin/activate

python main_a2c.py --mode 2 --city "nyc_manhattan" --q_lr 0.0001 --p_lr 0.0001 --actor_clip 500 --critic_clip 500 --max_episodes 30001 --use_od_prices --fix_baseline --checkpoint_path base_case_single_agent_fixed_policy