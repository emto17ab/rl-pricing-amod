#!/bin/bash
#BSUB -q hpc
#BSUB -J base_case_mode2_dual_agent_8k
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 24:00
#BSUB -o logs/base_case_mode2_dual_agent_8k_%J.out
#BSUB -e logs/base_case_mode2_dual_agent_8k_%J.err

source /work3/s233791/rl-pricing-amod/thesis_env/bin/activate

python main_a2c2_multiagent.py --num_agents 2 --mode 2 --agent_mode 2 --max_episodes 8000 --checkpoint_path base_case_mode2_dual_agent_8k