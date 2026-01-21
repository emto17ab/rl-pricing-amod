"""
Test script to measure CPLEX execution time: Sequential vs Parallel
"""
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from src.envs.amod_env_multi import Scenario, AMoD
from src.algos.reb_flow_solver_multi_agent import solveRebFlow
from src.misc.utils import dictsum

# Setup parameters
demand_ratio = {'nyc_man_south': 1.0}
json_hr = {'nyc_man_south': 19}
beta = {'nyc_man_south': 0.5}
choice_intercept = {'nyc_man_south': 9.84}
wage = {'nyc_man_south': 22.77}

city = 'nyc_man_south'
cplexpath = "/apps/cplex/cplex1210/opl/bin/x86-64_linux/"
directory = "saved_files"
max_episodes = 100
mode = 2  # Test mode 2 (both pricing and rebalancing)
seed = 10
max_steps = 20

print("Setting up environment...")
scenario = Scenario(
    json_file=f"data/scenario_{city}.json",
    demand_ratio=demand_ratio[city]*2,
    json_hr=json_hr[city],
    sd=seed,
    json_tstep=3,
    tf=max_steps,
    impute=0,
    supply_ratio=1.0,
    agent0_vehicle_ratio=0.5
)

env = AMoD(
    scenario, mode, 
    beta=beta[city], 
    jitter=1, 
    max_wait=2, 
    choice_price_mult=1.0, 
    seed=seed, 
    fix_agent=2,  # No fixed agent
    choice_intercept=choice_intercept[city], 
    wage=wage[city], 
    use_dynamic_wage_man_south=False
)

print(f"Environment created with {env.nregion} regions\n")

# Reset and prepare for testing
obs = env.reset()
env.match_step_simple()

# Create desired accumulations for both agents
desiredAcc = {}
for a in [0, 1]:
    current_total = dictsum(env.agent_acc[a], env.time + 1)
    base_per_region = current_total // env.nregion
    remainder = current_total % env.nregion
    desiredAcc[a] = {
        env.region[i]: base_per_region + (1 if i < remainder else 0)
        for i in range(env.nregion)
    }

print("=" * 60)
print("CPLEX TIMING TEST")
print("=" * 60)

# Test 1: Sequential execution (current implementation)
print("\n[Test 1] Sequential CPLEX calls:")
sequential_times = []
for run in range(5):
    start = time.time()
    rebAction_seq = {
        a: solveRebFlow(
            env, "nyc_manhattan", desiredAcc[a], cplexpath, directory, 
            a, max_episodes, mode, job_id=f"sequential_test_{run}"
        )
        for a in [0, 1]
    }
    elapsed = time.time() - start
    sequential_times.append(elapsed)
    print(f"  Run {run+1}: {elapsed:.4f}s")

seq_mean = np.mean(sequential_times)
seq_std = np.std(sequential_times)
print(f"  Mean: {seq_mean:.4f}s ± {seq_std:.4f}s")

# Test 2: Parallel execution
print("\n[Test 2] Parallel CPLEX calls:")

def solve_agent_rebalancing(agent_id, run_id):
    return solveRebFlow(
        env, "nyc_manhattan", desiredAcc[agent_id], cplexpath, directory, 
        agent_id, max_episodes, mode, job_id=f"parallel_test_{run_id}"
    )

parallel_times = []
for run in range(5):
    start = time.time()
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(
            lambda a: solve_agent_rebalancing(a, run), [0, 1]
        ))
    rebAction_par = {0: results[0], 1: results[1]}
    elapsed = time.time() - start
    parallel_times.append(elapsed)
    print(f"  Run {run+1}: {elapsed:.4f}s")

par_mean = np.mean(parallel_times)
par_std = np.std(parallel_times)
print(f"  Mean: {par_mean:.4f}s ± {par_std:.4f}s")

# Calculate speedup
print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
speedup = seq_mean / par_mean
time_saved_per_step = seq_mean - par_mean

print(f"Sequential:  {seq_mean:.4f}s ± {seq_std:.4f}s")
print(f"Parallel:    {par_mean:.4f}s ± {par_std:.4f}s")
print(f"Speedup:     {speedup:.2f}x")
print(f"Time saved:  {time_saved_per_step:.4f}s per rebalancing step")

# Calculate total time saved over full training
steps_with_rebalancing = max_episodes * max_steps
total_saved = time_saved_per_step * steps_with_rebalancing
print(f"\nProjected savings over {max_episodes} episodes × {max_steps} steps:")
print(f"  Total time saved: {total_saved:.1f}s ({total_saved/60:.1f} minutes)")

if speedup > 1.1:
    print(f"\n✓ Parallel execution provides {(speedup-1)*100:.1f}% speedup")
    print("  Recommendation: IMPLEMENT parallel CPLEX calls")
else:
    print(f"\n✗ Parallel execution provides minimal benefit ({(speedup-1)*100:.1f}%)")
    print("  Recommendation: Keep sequential implementation (less complexity)")
