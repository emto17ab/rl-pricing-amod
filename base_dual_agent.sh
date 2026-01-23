#!/bin/bash
#BSUB -q hpc
#BSUB -J "dual_agent_nyc_man_south_cars_1850[3]"
#BSUB -n 6
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=200MB]"
#BSUB -W 72:00
#BSUB -o logs/dual_agent_nyc_man_south_cars_1850_mode%I_%J.out
#BSUB -e logs/dual_agent_nyc_man_south_cars_1850_mode%I_%J.err

source /work3/s233791/rl-pricing-amod/thesis_env/bin/activate

# Subtract 1 from job index to get mode 0-2
MODE=$((LSB_JOBINDEX - 1))

python main_a2c_multi_agent.py --reward_scalar 2000 --critic_warmup_episodes 50 --mode $MODE --city "nyc_man_south" --q_lr 0.0003 --p_lr 0.0003 --actor_clip 1000 --critic_clip 1000 --max_episodes 85000 --use_od_prices --checkpoint_path dual_agent_nyc_man_south_cars_1850_mode${MODE}