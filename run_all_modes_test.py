#!/usr/bin/env python
"""
Script to run all modes (0-4) in test mode and collect results into a formatted table.
"""

import subprocess
import re
import sys

# Configuration
CHECKPOINT_BASE = "single_agent_washington_dc_mode{}"
CITY = "washington_dc"
MODEL_TYPE = "running"

# Metrics to collect (in order for the table)
METRIC_PATTERNS = {
    "Reward": r"Rewards \(mean, std\): ([\d.-]+) ([\d.-]+)",
    "Rebalancing Costs": r"Rebalancing cost \(mean, std\): ([\d.-]+) ([\d.-]+)",
    "Rebalance Trips": r"Rebalancing trips \(mean, std\): ([\d.-]+) ([\d.-]+)",
    "Price": r"Price scalar \(mean, std\): ([\d.-]+) ([\d.-]+)",
    "Wait/mins": r"Waiting time \(mean, std\): ([\d.-]+) ([\d.-]+)",
    "Queue": r"Queue length \(mean, std\): ([\d.-]+) ([\d.-]+)",
    "Served Demand": r"Served demand \(mean, std\): ([\d.-]+) ([\d.-]+)",
    "Arrivals": r"Arrivals \(mean, std\): ([\d.-]+) ([\d.-]+)",
}

METRIC_ORDER = ["Reward", "Rebalancing Costs", "Rebalance Trips", "Price", "Wait/mins", "Queue", "Served Demand", "Arrivals"]

def run_mode(mode):
    """Run a single mode and return the output."""
    checkpoint_path = CHECKPOINT_BASE.format(mode)
    
    # Build the command
    cmd = [
        "python", "main_a2c.py",
        "--test",
        "--mode", str(mode),
        "--city", CITY,
        "--checkpoint_path", checkpoint_path,
        "--model_type", MODEL_TYPE,
        "--use_od_prices"
    ]
    
    # Only add --load for modes that need checkpoints (0, 1, 2)
    # Modes 3 and 4 are baseline modes that don't need trained models
    if mode in [0, 1, 2]:
        cmd.insert(3, "--load")  # Insert after --test
    
    print(f"Running mode {mode}...", file=sys.stderr)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        return result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        print(f"  Mode {mode} timed out!", file=sys.stderr)
        return ""
    except Exception as e:
        print(f"  Error running mode {mode}: {e}", file=sys.stderr)
        return ""

def parse_output(output, mode):
    """Parse the output and extract metrics."""
    results = {}
    for metric, pattern in METRIC_PATTERNS.items():
        match = re.search(pattern, output)
        if match:
            mean = float(match.group(1))
            std = float(match.group(2))
            results[metric] = (mean, std)
        else:
            # Price is not available for mode 0
            if metric == "Price" and mode == 0:
                results[metric] = (None, None)
            else:
                results[metric] = (None, None)
    return results

def format_value(mean, std):
    """Format a value as 'mean (std)' with 2 decimal places."""
    if mean is None:
        return "-"
    return f"{mean:.2f} ({std:.2f})"

def main():
    # Collect results for all modes
    all_results = {}
    
    for mode in range(5):  # Modes 0-4
        output = run_mode(mode)
        results = parse_output(output, mode)
        all_results[mode] = results
        print(f"  Mode {mode} completed.", file=sys.stderr)
    
    # Save to a TSV file for easy copy-paste into Excel
    output_file = "results_table.tsv"
    with open(output_file, "w") as f:
        # Header row
        f.write("Metric\t" + "\t".join([f"Mode {m}" for m in range(5)]) + "\n")
        
        # Data rows
        for metric in METRIC_ORDER:
            row = metric
            for mode in range(5):
                mean, std = all_results[mode].get(metric, (None, None))
                row += "\t" + format_value(mean, std)
            f.write(row + "\n")
    
    print(f"\nResults saved to: {output_file}", file=sys.stderr)
    print("Open this file and copy-paste directly into Excel.", file=sys.stderr)
    
    # Also print to stdout for reference
    print(f"\nResults:")
    with open(output_file, "r") as f:
        print(f.read())

if __name__ == "__main__":
    main()
