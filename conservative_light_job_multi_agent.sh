#!/bin/bash
#BSUB -q hpc
#BSUB -J conservative_light_mode2_dual_agent
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 24:00
#BSUB -o logs/conservative_light_mode2_dual_agent_%J.out
#BSUB -e logs/conservative_light_mode2_dual_agent_%J.err

source /work3/s233791/rl-pricing-amod/thesis_env/bin/activate

python main_a2c_multi_agent.py --mode 2 --max_episodes 30007 --p_lr 5e-5 --q_lr 5e-5 --clip 1.0 --checkpoint_path conservative_light_mode2_dual_agent