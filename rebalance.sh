#!/bin/bash
#BSUB -q hpc
#BSUB -J rebalance_dual_agent_24k
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8GB]"
#BSUB -W 24:00
#BSUB -o logs/rebalance_dual_agent_24k_%J.out
#BSUB -e logs/rebalance_dual_agent_24k_%J.err

source /work3/s233791/rl-pricing-amod/thesis_env/bin/activate

python main_a2c_multi_agent.py --mode 0 --max_episodes 24002 --p_lr 5e-4 --q_lr 5e-4 --clip 5 --checkpoint_path rebalance_dual_agent_24k