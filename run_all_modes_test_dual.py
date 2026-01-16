#!/usr/bin/env python
"""
Script to run all modes (0-4) in test mode for dual agent setup and collect results into a formatted table.
"""

import subprocess
import re
import sys

# Configuration
CHECKPOINT_BASE = "dual_agent_nyc_man_south_mode{}"
CITY = "nyc_man_south"
MODEL_TYPE = "test"

# Metrics to collect for Combined totals (in order for the table)
METRIC_PATTERNS = {
    "Reward": r"Total rewards \(mean, std\): ([\d.-]+), ([\d.-]+)",
    "Rebalancing Costs": r"Total rebalancing cost \(mean, std\): ([\d.-]+), ([\d.-]+)",
    "Rebalance Trips": r"Total rebalancing trips \(mean, std\): ([\d.-]+), ([\d.-]+)",
    "Price Agent 0": r"Agent 0 price scalar \(mean, std\): ([\d.-]+), ([\d.-]+)",
    "Price Agent 1": r"Agent 1 price scalar \(mean, std\): ([\d.-]+), ([\d.-]+)",
    "Wait/mins Agent 0": r"Agent 0 Metrics:[\s\S]*?Waiting time \(mean, std\): ([\d.-]+), ([\d.-]+)",
    "Wait/mins Agent 1": r"Agent 1 Metrics:[\s\S]*?Waiting time \(mean, std\): ([\d.-]+), ([\d.-]+)",
    "Queue Agent 0": r"Agent 0 Metrics:[\s\S]*?Queue length \(mean, std\): ([\d.-]+), ([\d.-]+)",
    "Queue Agent 1": r"Agent 1 Metrics:[\s\S]*?Queue length \(mean, std\): ([\d.-]+), ([\d.-]+)",
    "Served Demand": r"Total served demand \(mean, std\): ([\d.-]+), ([\d.-]+)",
    "Arrivals Agent 0": r"Agent 0 Metrics:[\s\S]*?Arrivals \(mean, std\): ([\d.-]+), ([\d.-]+)",
    "Arrivals Agent 1": r"Agent 1 Metrics:[\s\S]*?Arrivals \(mean, std\): ([\d.-]+), ([\d.-]+)",
    "Total Arrivals": r"Total arrivals \(mean, std\): ([\d.-]+), ([\d.-]+)",
}

METRIC_ORDER = ["Reward", "Rebalancing Costs", "Rebalance Trips", "Price Agent 0", "Price Agent 1", "Wait/mins Agent 0", "Wait/mins Agent 1", "Queue Agent 0", "Queue Agent 1", "Served Demand", "Arrivals Agent 0", "Arrivals Agent 1", "Total Arrivals"]

def run_mode(mode):
    """Run a single mode and return the output."""
    checkpoint_path = CHECKPOINT_BASE.format(mode)
    
    # Build the command
    cmd = [
        "python", "main_a2c_multi_agent.py",
        "--test",
        "--mode", str(mode),
        "--city", CITY,
        "--checkpoint_path", checkpoint_path,
        "--model_type", MODEL_TYPE,
        "--use_od_prices",
    ]
    
    # Only add --load for modes that need checkpoints (0, 1, 2)
    # Modes 3 and 4 are baseline modes that don't need trained models
    # Note: In dual agent test mode, --load is not needed as it always loads
    
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
            # Price is not available for mode 0, Rebalance Trips not for mode 1
            if "Price" in metric and mode == 0:
                results[metric] = (None, None)
            elif metric == "Rebalance Trips" and mode == 1:
                results[metric] = (None, None)
            else:
                results[metric] = (None, None)
    return results

def format_value(mean, std):
    """Format a value as 'mean (std)' with 1 decimal place."""
    if mean is None:
        return "-"
    # Use 2 decimals for price values (typically small numbers close to 1)
    if abs(mean) < 10:
        return f"{mean:.2f} ({std:.2f})"
    return f"{mean:.1f} ({std:.1f})"

def main():
    # Collect results for all modes
    all_results = {}
    modes_to_run = [2]
    
    for mode in modes_to_run:
        output = run_mode(mode)
        results = parse_output(output, mode)
        all_results[mode] = results
        print(f"  Mode {mode} completed.", file=sys.stderr)
    
    # Save to a TSV file for easy copy-paste into Excel
    output_file = "results_table_dual.tsv"
    with open(output_file, "w") as f:
        # Header row
        f.write("Metric\t" + "\t".join([f"Mode {m}" for m in modes_to_run]) + "\n")
        
        # Data rows
        for metric in METRIC_ORDER:
            row = metric
            for mode in modes_to_run:
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
