#!/usr/bin/env python3
"""
Quick test script to verify visualization data export works correctly.
Runs a short test with mode 3 (baseline, no learning) to generate visualization data.
"""

import subprocess
import sys
import os

print("="*80)
print("Testing Visualization Data Export")
print("="*80)

# Run a quick test with mode 3 (baseline mode - no models needed)
# This will run 10 test episodes and save visualization data from the last one
cmd = [
    sys.executable, "main_a2c_multi_agent.py",
    "--test",
    "--city", "nyc_manhattan",
    "--mode", "3",  # Baseline mode (no rebalancing, fixed prices)
    "--max_steps", "5",  # Short episode for quick test
    "--checkpoint_path", "viz_test_baseline"
]

print(f"\nRunning command:")
print(f"  {' '.join(cmd)}\n")

try:
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    # Check if visualization file was created
    viz_file = "saved_files/visualization_data/viz_data_nyc_manhattan_mode3_fixagent2_episodes15000.pkl"
    if os.path.exists(viz_file):
        print(f"\n✓ SUCCESS: Visualization data file created: {viz_file}")
        
        # Load and inspect the file
        import pickle
        import numpy as np
        with open(viz_file, 'rb') as f:
            data = pickle.load(f)
        
        print(f"\nData structure:")
        print(f"  Metadata: {data['metadata']}")
        for agent_id in [0, 1]:
            print(f"\n  Agent {agent_id}:")
            if len(data['agent_price_scalars'][agent_id]) > 0:
                print(f"    - Price scalars shape: {data['agent_price_scalars'][agent_id].shape}")
            else:
                print(f"    - Price scalars: empty")
            
            if len(data['agent_reb_actions'][agent_id]) > 0:
                print(f"    - Reb actions shape: {data['agent_reb_actions'][agent_id].shape}")
            else:
                print(f"    - Reb actions: empty")
            
            if len(data['agent_acc_temporal'][agent_id]) > 0:
                print(f"    - Acc temporal shape: {data['agent_acc_temporal'][agent_id].shape}")
            else:
                print(f"    - Acc temporal: empty")
        
        print(f"\n{'='*80}")
        print("✓ Test completed successfully!")
        print("You can now use this data in your visualization notebook.")
        print("="*80)
    else:
        print(f"\n✗ ERROR: Visualization data file not found: {viz_file}")
        sys.exit(1)
        
except subprocess.CalledProcessError as e:
    print(f"\n✗ ERROR: Command failed with return code {e.returncode}")
    print(f"STDOUT: {e.stdout}")
    print(f"STDERR: {e.stderr}")
    sys.exit(1)
except Exception as e:
    print(f"\n✗ ERROR: {e}")
    sys.exit(1)
