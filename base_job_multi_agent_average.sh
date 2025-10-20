#!/bin/bash
#BSUB -q hpc
#BSUB -J test3
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 24:00
#BSUB -o logs/test3_%J.out
#BSUB -e logs/test3_%J.err

source /work3/s233791/rl-pricing-amod/thesis_env/bin/activate
hostname -s
python main_a2c_multi_agent.py --mode 2 --max_episodes 20 --p_lr 1e-4 --q_lr 1e-4 --clip 5 --fix_agent 0 --checkpoint_path test3