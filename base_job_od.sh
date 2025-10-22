#!/bin/bash
#BSUB -q hpc
#BSUB -J base_case_san_francisco_mode2_30k_dual_agent_OD
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 24:00
#BSUB -o logs/base_case_san_francisco_mode2_30k_dual_agent_OD_%J.out
#BSUB -e logs/base_case_san_francisco_mode2_30k_dual_agent_OD_%J.err

source /work3/s233791/rl-pricing-amod/thesis_env/bin/activate

python main_a2c_multi_agent.py --mode 2 --max_episodes 30001 --use_od_prices --clip 5 --q_lr 1e-4 --p_lr 1e-4 --checkpoint_path base_case_san_francisco_mode2_30k_dual_agent_OD