#!/bin/bash
#BSUB -q hpc
#BSUB -J norm_both
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 24:00
#BSUB -o logs/norm_both_%J.out
#BSUB -e logs/norm_both_%J.err

source /work3/s233791/rl-pricing-amod/thesis_env/bin/activate

python main_a2c_multi_agent.py \
    --mode 2 \
    --max_episodes 40227 \
    --actor_clip 5 \
    --critic_clip 5 \
    --use_od_prices \
    --q_lr 1e-4 \
    --p_lr 1e-4 \
    --city "nyc_manhattan" \
    --entropy_coeff 0.005 \
    --checkpoint_path norm_both
