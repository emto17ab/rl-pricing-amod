#!/bin/bash
#BSUB -q hpc
#BSUB -J fix_agent1_mode2_24k
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 24:00
#BSUB -o logs/fix_agent1_mode2_24k_%J.out
#BSUB -e logs/fix_agent1_mode2_24k_%J.err

source /work3/s233791/rl-pricing-amod/thesis_env/bin/activate

python main_a2c_multi_agent.py --mode 2 --max_episodes 24004 --checkpoint_path fix_agent1_mode2_24k --fix_agent 1
