#!/bin/bash
#BSUB -q hpc
#BSUB -J "single_agent_cars_1200_nyc_south_mode[1-3]"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 72:00
#BSUB -o logs/single_agent_cars_1200_nyc_south_mode%I_%J.out
#BSUB -e logs/single_agent_cars_1200_nyc_south_mode%I_%J.err

source /work3/s233791/rl-pricing-amod/thesis_env/bin/activate

# Subtract 1 from job index to get mode 0-2
MODE=$((LSB_JOBINDEX - 1))

python main_a2c.py --critic_warmup_episodes 500 --mode ${MODE} --city "nyc_man_south" --q_lr 0.001 --p_lr 0.001 --actor_clip 20000 --critic_clip 20000 --max_episodes 1000000 --use_od_prices --checkpoint_path single_agent_cars_1200_nyc_south_mode${MODE}_model_test