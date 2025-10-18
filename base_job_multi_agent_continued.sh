#!/bin/bash
#BSUB -q hpc
#BSUB -J base_case_mode2_dual_agent_36_lower_learning_continued
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 24:00
#BSUB -o logs/base_case_mode2_dual_agent_36_lower_learning_continued_%J.out
#BSUB -e logs/base_case_mode2_dual_agent_36_lower_learning_continued_%J.err

source /work3/s233791/rl-pricing-amod/thesis_env/bin/activate

python main_a2c_multi_agent.py --load --mode 2 --max_episodes 36000 --p_lr 5e-5 --q_lr 5e-5 --clip 5 --checkpoint_path base_case_mode2_dual_agent_36k