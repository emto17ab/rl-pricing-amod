#!/bin/bash
#BSUB -q hpc
#BSUB -J "dual_agent_cars_1200_sf_mode[1-3]"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 72:00
#BSUB -o logs/dual_agent_cars_1200_sf_mode%I_%J.out
#BSUB -e logs/dual_agent_cars_1200_sf_mode%I_%J.err

source /work3/s233791/rl-pricing-amod/thesis_env/bin/activate

MODE=$((LSB_JOBINDEX - 1))

python main_a2c_multi_agent.py --critic_warmup_episodes 500 --mode ${MODE} --city "san_francisco" --q_lr 0.001 --p_lr 0.001 --actor_clip 10000 --critic_clip 20000 --max_episodes 1000000 --use_od_prices --checkpoint_path dual_agent_cars_1200_sf_mode${MODE}