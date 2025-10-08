#!/bin/bash
#BSUB -q hpc
#BSUB -J base_case_mode2_dual_agent_24k
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 24:00
#BSUB -o logs/base_case_mode2_dual_agent_24k_%J.out
#BSUB -e logs/base_case_mode2_dual_agent_24k_%J.err

source /work3/s233791/rl-pricing-amod/thesis_env/bin/activate

python main_a2c_multi_agent.py --mode 2 --max_episodes 24000 --checkpoint_path base_case_mode2_dual_agent_24k