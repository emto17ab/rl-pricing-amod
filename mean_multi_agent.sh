#!/bin/bash
#BSUB -q hpc
#BSUB -J mean_mode2_dual_agent
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 24:00
#BSUB -o logs/mean_mode2_dual_agent_%J.out
#BSUB -e logs/mean_mode2_dual_agent_%J.err

source /work3/s233791/rl-pricing-amod/thesis_env/bin/activate

python main_a2c_multi_agent.py --mode 2 --max_episodes 30031 --p_lr 1e-4 --q_lr 1e-4 --clip 5 --checkpoint_path mean_mode2_dual_agent