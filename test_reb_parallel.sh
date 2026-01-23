#!/bin/bash
#BSUB -J reb_test
#BSUB -o logs_test/reb_test.out
#BSUB -e logs_test/reb_test.err
#BSUB -n 6
#BSUB -q hpc
#BSUB -R "rusage[mem=2GB]"
#BSUB -R "span[hosts=1]"
#BSUB -R "select[model==XeonGold6226R]"
#BSUB -W 2:00

# Load environment
source /work3/s233791/rl-pricing-amod/thesis_env/bin/activate

# Create output directory
mkdir -p logs_test

# Run both sequential and parallel tests (one after the other)
python -u test_rebalancing_parallel.py --num_episodes 50 > logs_test/reb_test_results.txt 2>&1

echo "Rebalancing test completed"
