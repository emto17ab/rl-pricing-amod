"""
Quick training test to verify fix_baseline works in main_a2c.py
This runs a short training with fix_baseline to ensure integration is correct
"""

import sys
import subprocess

def test_training_with_fix_baseline():
    """Test that training runs with fix_baseline flag"""
    
    print("="*70)
    print("TESTING FIX_BASELINE TRAINING INTEGRATION")
    print("="*70)
    
    # Test with mode 2 (pricing + rebalancing)
    cmd = [
        "python", "main_a2c.py",
        "--mode", "2",
        "--city", "nyc_manhattan",
        "--max_episodes", "5",  # Just 5 episodes for quick test
        "--max_steps", "5",      # Just 5 steps per episode
        "--checkpoint_path", "test_fix_baseline",
        "--fix_baseline",        # The flag we're testing
        "--seed", "42"
    ]
    
    print("\nRunning command:")
    print(" ".join(cmd))
    print()
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120  # 2 minute timeout
        )
        
        # Check for success
        if result.returncode == 0:
            print("✓ Training completed successfully!")
            
            # Check output for expected messages
            if "FIXED BASELINE MODE ACTIVATED" in result.stdout:
                print("✓ Fixed baseline mode was activated")
            else:
                print("✗ WARNING: Fixed baseline activation message not found")
            
            if "Initial vehicles:" in result.stdout:
                print("✓ Initial vehicle info was printed")
            else:
                print("✗ WARNING: Initial vehicle info not printed")
            
            print("\n" + "="*70)
            print("TRAINING TEST PASSED!")
            print("="*70)
            return True
        else:
            print("✗ Training failed with return code:", result.returncode)
            print("\nSTDOUT:")
            print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
            print("\nSTDERR:")
            print(result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("✗ Training timed out after 2 minutes")
        return False
    except Exception as e:
        print(f"✗ Error running training: {e}")
        return False

if __name__ == "__main__":
    # Activate virtual environment first
    print("Note: Make sure to activate virtual environment before running this test:")
    print("  source /work3/s233791/rl-pricing-amod/thesis_env/bin/activate")
    print()
    
    success = test_training_with_fix_baseline()
    sys.exit(0 if success else 1)
