#!/bin/bash
#BSUB -q gpuv100
#BSUB -J test_job_gpu
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 24:00
#BSUB -o logs/test_job_%J.out
#BSUB -e logs/test_job_%J.err

source /zhome/7a/c/204061/Documents/thesis/rl-pricing-amod/thesis_env/bin/activate

python main_a2c2.py --mode 2 --cuda --max_episodes 16000