#!/bin/bash
#BSUB -q hpc
#BSUB -J profitability_manhattan_mode2_40k_dual_agent_od_down_scaled
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 24:00
#BSUB -o logs/profitability_manhattan_mode2_40k_dual_agent_od_down_scaled_%J.out
#BSUB -e logs/profitability_manhattan_mode2_40k_dual_agent_od_down_scaled_%J.err

source /work3/s233791/rl-pricing-amod/thesis_env/bin/activate

python main_a2c_multi_agent.py --mode 2 --max_episodes 40002 --actor_clip 3 --critic_clip 3 --use_od_prices --q_lr 1e-4 --p_lr 1e-4 --city "nyc_manhattan" --checkpoint_path profitability_manhattan_mode2_40k_dual_agent_od_down_scaled --loss_aversion 2.0
