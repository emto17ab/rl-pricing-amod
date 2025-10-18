#!/bin/bash
#BSUB -q hpc
#BSUB -J base_case_mode2_dual_agent_37_no_clip
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 24:00
#BSUB -o logs/base_case_mode2_dual_agent_37_no_clip_%J.out
#BSUB -e logs/base_case_mode2_dual_agent_37_no_clip_%J.err

source /work3/s233791/rl-pricing-amod/thesis_env/bin/activate

python main_a2c_multi_agent.py --mode 2 --max_episodes 37000 --p_lr 1e-4 --q_lr 1e-4 --checkpoint_path base_case_mode2_dual_agent_37_no_clip