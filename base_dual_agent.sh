#!/bin/bash
#BSUB -q hpc
#BSUB -J "dual_agent_cars_1200_high_learning[1-3]"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 72:00
#BSUB -o logs/base_case_dual_agent_mode%I_1200_cars_high_learning_%J.out
#BSUB -e logs/base_case_dual_agent_mode%I_1200_cars_high_learning_%J.err

source /work3/s233791/rl-pricing-amod/thesis_env/bin/activate

# Subtract 1 from job index to get mode 0-2
MODE=$((LSB_JOBINDEX - 1))

python main_a2c_multi_agent.py --mode $MODE --loss_aversion 0.0 --city "nyc_man_south" --q_lr 0.005 --p_lr 0.001 --actor_clip 20 --critic_clip 200 --max_episodes 1000000 --use_od_prices --checkpoint_path base_case_dual_agent_mode${MODE}_cars_1200_high_learning