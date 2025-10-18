#!/bin/bash
#BSUB -q hpc
#BSUB -J base_case_mode2_dual_agent_30_average_no_normalization
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 24:00
#BSUB -o logs/base_case_mode2_dual_agent_30_average_no_normalization_%J.out
#BSUB -e logs/base_case_mode2_dual_agent_30_average_no_normalization_%J.err

source /work3/s233791/rl-pricing-amod/thesis_env/bin/activate

python main_a2c_multi_agent.py --mode 2 --max_episodes 30030 --p_lr 5e-4 --q_lr 5e-4 --clip 10 --checkpoint_path base_case_mode2_dual_agent_30k_average_no_normalization