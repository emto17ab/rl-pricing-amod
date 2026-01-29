#!/bin/bash
#BSUB -q hpc
#BSUB -J "dual_nyc_man_south_cars_450_v2[3]"
#BSUB -n 6
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=800MB]"
#BSUB -W 72:00
#BSUB -o logs/dual_nyc_man_south_cars_450_v2_mode%I_%J.out
#BSUB -e logs/dual_nyc_man_south_cars_450_v2_mode%I_%J.err

source /work3/s233791/rl-pricing-amod/thesis_env/bin/activate

# Subtract 1 from job index to get mode 0-2
MODE=$((LSB_JOBINDEX - 1))

python main_a2c_multi_agent.py --reward_scalar 2000 --critic_warmup_episodes 50 --mode $MODE --city "nyc_man_south" --q_lr 0.0006 --p_lr 0.0002 --actor_clip 1000 --critic_clip 1000 --max_episodes 100000 --use_od_prices --no_share_info --checkpoint_path dual_nyc_man_south_cars_450_v2_mode${MODE}