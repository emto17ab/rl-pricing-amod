#!/bin/bash
#BSUB -q hpc
#BSUB -J entropy_reg_test_v1
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 24:00
#BSUB -o logs/entropy_reg_test_v1_%J.out
#BSUB -e logs/entropy_reg_test_v1_%J.err

source /work3/s233791/rl-pricing-amod/thesis_env/bin/activate

python main_a2c_multi_agent.py \
    --mode 2 \
    --max_episodes 40105 \
    --actor_clip 5 \
    --critic_clip 5 \
    --use_od_prices \
    --q_lr 1e-4 \
    --p_lr 1e-4 \
    --city "nyc_manhattan" \
    --entropy_coeff 1.0 \
    --checkpoint_path entropy_reg_test_v1
