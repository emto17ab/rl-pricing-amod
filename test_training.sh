#!/bin/bash
# Quick test of fix_baseline training

source /work3/s233791/rl-pricing-amod/thesis_env/bin/activate

echo "========================================================================"
echo "Testing fix_baseline training (5 episodes, 5 steps each)"
echo "========================================================================"

python main_a2c.py \
    --mode 2 \
    --city "nyc_manhattan" \
    --max_episodes 5 \
    --max_steps 5 \
    --checkpoint_path "test_fix_baseline" \
    --fix_baseline \
    --seed 42

echo ""
echo "========================================================================"
echo "Test completed!"
echo "========================================================================"
