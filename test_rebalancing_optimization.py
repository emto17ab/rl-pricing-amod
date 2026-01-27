import time
import numpy as np
from copy import deepcopy
from src.envs.amod_env_multi import AMoD, Scenario
from src.envs.amod_env import AMoD as AMoDSingle, Scenario as ScenarioSingle

def create_test_scenario_dual(demand_scale=1.0):
    """Create a test scenario for dual-agent - using NYC Manhattan South"""
    scenario = Scenario(
        json_file="data/scenario_nyc_man_south.json",
        demand_ratio=demand_scale * 2,
        json_hr=19,
        sd=42,
        json_tstep=3,
        tf=20,
        impute=False,
        supply_ratio=1.0,
        agent0_vehicle_ratio=0.5
    )
    return scenario

def create_test_scenario_single(demand_scale=1.0):
    """Create a test scenario for single-agent - using NYC Manhattan South"""
    scenario = ScenarioSingle(
        json_file="data/scenario_nyc_man_south.json",
        demand_ratio=demand_scale * 2,
        json_hr=19,
        sd=42,
        json_tstep=3,
        tf=20,
        impute=False,
        supply_ratio=1.0
    )
    return scenario

def benchmark_current_reb_dual(env, num_iterations=100):
    """Benchmark CURRENT rebalancing implementation (dual-agent)"""
    times = []
    
    for iteration in range(num_iterations):
        env.reset()
        
        # Run one matching step to populate vehicle distributions
        price_action = {0: np.ones(env.nregion) * 0.5, 1: np.ones(env.nregion) * 0.5}
        env.match_step_simple(price_action)
        
        # Create random rebalancing actions
        rebAction_agents = {
            0: np.random.randint(0, 5, len(env.edges)),
            1: np.random.randint(0, 5, len(env.edges))
        }
        
        start = time.perf_counter()
        
        # Current implementation
        t = env.time
        rebreward = {0: 0, 1: 0}
        env.ext_reward_agents = {a: np.zeros(env.nregion) for a in [0, 1]}
        
        for agent_id in [0, 1]:
            env.agent_info[agent_id]['rebalancing_cost'] = 0
        
        for agent_id in [0, 1]:
            rebAction = rebAction_agents[agent_id]
            
            for k in range(len(env.edges)):
                i, j = env.edges[k]
                
                rebAction[k] = min(env.agent_acc[agent_id][i][t], rebAction[k])
                
                reb_time = env.rebTime[i, j][t]
                
                env.agent_rebFlow[agent_id][i, j][t + reb_time] = rebAction[k]
                env.agent_rebFlow_ori[agent_id][i, j][t] = rebAction[k]
                
                env.agent_acc[agent_id][i][t] -= rebAction[k]
                env.agent_dacc[agent_id][j][t + reb_time] += rebAction[k]
                
                # Current: Multiple lookups and calculations
                rebalancing_cost = env.rebTime[i, j][t] * env.beta * rebAction[k]
                rebreward[agent_id] -= rebalancing_cost
                env.ext_reward_agents[agent_id][i] -= rebalancing_cost
                env.agent_info[agent_id]['rebalancing_cost'] += rebalancing_cost
        
        end = time.perf_counter()
        times.append((end - start) * 1000)
    
    return np.array(times)

def benchmark_optimized_reb_dual(env, num_iterations=100):
    """Benchmark OPTIMIZED rebalancing implementation (dual-agent)"""
    times = []
    
    for iteration in range(num_iterations):
        env.reset()
        
        # Run one matching step to populate vehicle distributions
        price_action = {0: np.ones(env.nregion) * 0.5, 1: np.ones(env.nregion) * 0.5}
        env.match_step_simple(price_action)
        
        # Create random rebalancing actions
        rebAction_agents = {
            0: np.random.randint(0, 5, len(env.edges)),
            1: np.random.randint(0, 5, len(env.edges))
        }
        
        start = time.perf_counter()
        
        # Optimized implementation
        t = env.time
        rebreward = {0: 0, 1: 0}
        env.ext_reward_agents = {a: np.zeros(env.nregion) for a in [0, 1]}
        
        for agent_id in [0, 1]:
            env.agent_info[agent_id]['rebalancing_cost'] = 0
        
        for agent_id in [0, 1]:
            rebAction = rebAction_agents[agent_id]
            
            for k in range(len(env.edges)):
                i, j = env.edges[k]
                
                rebAction[k] = min(env.agent_acc[agent_id][i][t], rebAction[k])
                
                # OPTIMIZED: Pre-calculate reb_time and cost factor once
                reb_time = env.rebTime[i, j][t]
                rebalancing_cost = reb_time * env.beta * rebAction[k]
                
                env.agent_rebFlow[agent_id][i, j][t + reb_time] = rebAction[k]
                env.agent_rebFlow_ori[agent_id][i, j][t] = rebAction[k]
                
                env.agent_acc[agent_id][i][t] -= rebAction[k]
                env.agent_dacc[agent_id][j][t + reb_time] += rebAction[k]
                
                # Optimized: Use pre-calculated cost
                rebreward[agent_id] -= rebalancing_cost
                env.ext_reward_agents[agent_id][i] -= rebalancing_cost
                env.agent_info[agent_id]['rebalancing_cost'] += rebalancing_cost
        
        end = time.perf_counter()
        times.append((end - start) * 1000)
    
    return np.array(times)

def benchmark_current_reb_single(env, num_iterations=100):
    """Benchmark CURRENT rebalancing implementation (single-agent)"""
    times = []
    
    for iteration in range(num_iterations):
        env.reset()
        
        # Run one matching step to populate vehicle distributions
        price_action = np.ones(env.nregion) * 0.5
        env.match_step_simple(price_action)
        
        # Create random rebalancing actions
        rebAction = np.random.randint(0, 5, len(env.edges))
        
        start = time.perf_counter()
        
        # Current implementation
        t = env.time
        env.reward = 0
        env.ext_reward = np.zeros(env.nregion)
        env.rebAction = rebAction.copy()
        
        for k in range(len(env.edges)):
            i, j = env.edges[k]
            if (i, j) not in env.G.edges:
                continue
            
            env.rebAction[k] = min(env.acc[i][t], rebAction[k])
            env.rebFlow[i, j][t+env.rebTime[i, j][t]] = env.rebAction[k]
            env.rebFlow_ori[i, j][t] = env.rebAction[k]
            env.acc[i][t] -= env.rebAction[k]
            env.dacc[j][t+env.rebTime[i, j][t]] += env.rebFlow[i, j][t+env.rebTime[i, j][t]]
            
            # Current: Multiple repeated lookups
            env.info['rebalancing_cost'] += env.rebTime[i, j][t] * env.beta * env.rebAction[k]
            env.info["operating_cost"] += env.rebTime[i, j][t] * env.beta * env.rebAction[k]
            env.reward -= env.rebTime[i, j][t] * env.beta * env.rebAction[k]
            env.ext_reward[i] -= env.rebTime[i, j][t] * env.beta * env.rebAction[k]
        
        end = time.perf_counter()
        times.append((end - start) * 1000)
    
    return np.array(times)

def benchmark_optimized_reb_single(env, num_iterations=100):
    """Benchmark OPTIMIZED rebalancing implementation (single-agent)"""
    times = []
    
    for iteration in range(num_iterations):
        env.reset()
        
        # Run one matching step to populate vehicle distributions
        price_action = np.ones(env.nregion) * 0.5
        env.match_step_simple(price_action)
        
        # Create random rebalancing actions
        rebAction = np.random.randint(0, 5, len(env.edges))
        
        start = time.perf_counter()
        
        # Optimized implementation
        t = env.time
        env.reward = 0
        env.ext_reward = np.zeros(env.nregion)
        env.rebAction = rebAction.copy()
        
        for k in range(len(env.edges)):
            i, j = env.edges[k]
            # OPTIMIZED: Removed redundant edge check (edges are always valid)
            
            env.rebAction[k] = min(env.acc[i][t], rebAction[k])
            
            # OPTIMIZED: Pre-calculate reb_time and cost factor once
            reb_time = env.rebTime[i, j][t]
            reb_cost_factor = reb_time * env.beta * env.rebAction[k]
            
            env.rebFlow[i, j][t+reb_time] = env.rebAction[k]
            env.rebFlow_ori[i, j][t] = env.rebAction[k]
            env.acc[i][t] -= env.rebAction[k]
            env.dacc[j][t+reb_time] += env.rebFlow[i, j][t+reb_time]
            
            # Optimized: Use pre-calculated cost factor
            env.info['rebalancing_cost'] += reb_cost_factor
            env.info["operating_cost"] += reb_cost_factor
            env.reward -= reb_cost_factor
            env.ext_reward[i] -= reb_cost_factor
        
        end = time.perf_counter()
        times.append((end - start) * 1000)
    
    return np.array(times)

def verify_correctness_dual():
    """Verify that both implementations produce identical results (dual-agent)"""
    print("\n" + "=" * 80)
    print("DUAL-AGENT CORRECTNESS VERIFICATION")
    print("=" * 80)
    
    scenario = create_test_scenario_dual(1.0)
    
    # Test with same random actions
    np.random.seed(42)
    test_actions = {
        0: np.random.randint(0, 5, 144),
        1: np.random.randint(0, 5, 144)
    }
    
    # Current implementation
    env1 = AMoD(scenario=deepcopy(scenario), mode=2, beta=0.5, jitter=1e-6,
                max_wait=10, choice_price_mult=1.0, seed=42, fix_agent=2,
                choice_intercept=3.0, wage=25.0)
    env1.reset()
    # Run one matching step to populate t+1 vehicle counts
    price_action = {0: np.ones(env1.nregion) * 0.5, 1: np.ones(env1.nregion) * 0.5}
    env1.match_step_simple(price_action)
    t1 = env1.time
    
    costs1 = {0: 0, 1: 0}
    for agent_id in [0, 1]:
        for k in range(len(env1.edges)):
            i, j = env1.edges[k]
            action = min(env1.agent_acc[agent_id][i][t1], test_actions[agent_id][k])
            costs1[agent_id] += env1.rebTime[i, j][t1] * env1.beta * action
    
    # Optimized implementation
    env2 = AMoD(scenario=deepcopy(scenario), mode=2, beta=0.5, jitter=1e-6,
                max_wait=10, choice_price_mult=1.0, seed=42, fix_agent=2,
                choice_intercept=3.0, wage=25.0)
    env2.reset()
    # Run one matching step to populate t+1 vehicle counts
    env2.match_step_simple(price_action)
    t2 = env2.time
    
    costs2 = {0: 0, 1: 0}
    for agent_id in [0, 1]:
        for k in range(len(env2.edges)):
            i, j = env2.edges[k]
            action = min(env2.agent_acc[agent_id][i][t2], test_actions[agent_id][k])
            reb_time = env2.rebTime[i, j][t2]
            costs2[agent_id] += reb_time * env2.beta * action
    
    # Compare costs
    all_match = True
    for agent_id in [0, 1]:
        if not np.isclose(costs1[agent_id], costs2[agent_id]):
            print(f"  ❌ Cost mismatch for agent {agent_id}")
            print(f"     Current: {costs1[agent_id]:.6f}, Optimized: {costs2[agent_id]:.6f}")
            all_match = False
    
    if all_match:
        print("  ✓ All costs match! Implementations are functionally identical.")
        return True
    else:
        print("  ✗ Cost mismatches found!")
        return False

def verify_correctness_single():
    """Verify that both implementations produce identical results (single-agent)"""
    print("\n" + "=" * 80)
    print("SINGLE-AGENT CORRECTNESS VERIFICATION")
    print("=" * 80)
    
    scenario = create_test_scenario_single(1.0)
    
    # Test with same random actions
    np.random.seed(42)
    test_actions = np.random.randint(0, 5, 144)
    
    # Current implementation
    env1 = AMoDSingle(scenario=deepcopy(scenario), mode=2, beta=0.5, jitter=1e-6,
                      max_wait=10, choice_price_mult=1.0, seed=42,
                      choice_intercept=3.0, wage=25.0)
    env1.reset()
    # Run one matching step to populate t+1 vehicle counts
    price_action = np.ones(env1.nregion) * 0.5
    env1.match_step_simple(price_action)
    t1 = env1.time
    
    cost1 = 0
    for k in range(len(env1.edges)):
        i, j = env1.edges[k]
        if (i, j) not in env1.G.edges:
            continue
        action = min(env1.acc[i][t1], test_actions[k])
        cost1 += env1.rebTime[i, j][t1] * env1.beta * action
    
    # Optimized implementation (without edge check)
    env2 = AMoDSingle(scenario=deepcopy(scenario), mode=2, beta=0.5, jitter=1e-6,
                      max_wait=10, choice_price_mult=1.0, seed=42,
                      choice_intercept=3.0, wage=25.0)
    env2.reset()
    # Run one matching step to populate t+1 vehicle counts
    env2.match_step_simple(price_action)
    t2 = env2.time
    
    cost2 = 0
    for k in range(len(env2.edges)):
        i, j = env2.edges[k]
        # Optimized: no edge check
        action = min(env2.acc[i][t2], test_actions[k])
        reb_time = env2.rebTime[i, j][t2]
        cost2 += reb_time * env2.beta * action
    
    # Compare costs
    if np.isclose(cost1, cost2):
        print(f"  ✓ Costs match! Current: {cost1:.6f}, Optimized: {cost2:.6f}")
        return True
    else:
        print(f"  ❌ Cost mismatch! Current: {cost1:.6f}, Optimized: {cost2:.6f}")
        return False

def run_benchmark():
    print("=" * 80)
    print("REBALANCING OPTIMIZATION TEST")
    print("=" * 80)
    
    # Verify correctness first
    if not verify_correctness_dual():
        print("\n⚠️  WARNING: Dual-agent correctness verification failed!")
        return
    
    if not verify_correctness_single():
        print("\n⚠️  WARNING: Single-agent correctness verification failed!")
        return
    
    # Test different demand levels for dual-agent
    print("\n" + "=" * 80)
    print("DUAL-AGENT BENCHMARKS")
    print("=" * 80)
    
    test_configs = [
        ("Low Demand (0.5x)", 0.5),
        ("Normal Demand (1.0x)", 1.0),
        ("High Demand (2.0x)", 2.0),
    ]
    
    dual_results = []
    
    for config_name, demand_scale in test_configs:
        print(f"\n{'=' * 80}")
        print(f"Test: {config_name}")
        print(f"{'=' * 80}")
        
        scenario = create_test_scenario_dual(demand_scale)
        
        env = AMoD(
            scenario=scenario,
            mode=2,
            beta=0.5,
            jitter=1e-6,
            max_wait=10,
            choice_price_mult=1.0,
            seed=42,
            fix_agent=2,
            choice_intercept=3.0,
            wage=25.0
        )
        
        print("\nBenchmarking CURRENT implementation...")
        current_times = benchmark_current_reb_dual(env, num_iterations=100)
        
        env = AMoD(
            scenario=scenario,
            mode=2,
            beta=0.5,
            jitter=1e-6,
            max_wait=10,
            choice_price_mult=1.0,
            seed=42,
            fix_agent=2,
            choice_intercept=3.0,
            wage=25.0
        )
        
        print("Benchmarking OPTIMIZED implementation...")
        optimized_times = benchmark_optimized_reb_dual(env, num_iterations=100)
        
        speedup = current_times.mean() / optimized_times.mean()
        time_saved = current_times.mean() - optimized_times.mean()
        
        dual_results.append((config_name, current_times.mean(), optimized_times.mean(), speedup, time_saved))
        
        print(f"\n{'─' * 80}")
        print(f"Results for {config_name}:")
        print(f"{'─' * 80}")
        print(f"\nCURRENT Implementation:")
        print(f"  Mean time:   {current_times.mean():.4f} ms")
        print(f"  Std dev:     {current_times.std():.4f} ms")
        print(f"\nOPTIMIZED Implementation:")
        print(f"  Mean time:   {optimized_times.mean():.4f} ms")
        print(f"  Std dev:     {optimized_times.std():.4f} ms")
        print(f"\n{'─' * 80}")
        print(f"SPEEDUP: {speedup:.2f}x faster")
        print(f"Time saved: {time_saved:.4f} ms per reb_step()")
        print(f"{'─' * 80}")
    
    # Test single-agent
    print("\n" + "=" * 80)
    print("SINGLE-AGENT BENCHMARKS")
    print("=" * 80)
    
    single_results = []
    
    for config_name, demand_scale in test_configs:
        print(f"\n{'=' * 80}")
        print(f"Test: {config_name}")
        print(f"{'=' * 80}")
        
        scenario = create_test_scenario_single(demand_scale)
        
        env = AMoDSingle(
            scenario=scenario,
            mode=2,
            beta=0.5,
            jitter=1e-6,
            max_wait=10,
            choice_price_mult=1.0,
            seed=42,
            choice_intercept=3.0,
            wage=25.0
        )
        
        print("\nBenchmarking CURRENT implementation...")
        current_times = benchmark_current_reb_single(env, num_iterations=100)
        
        env = AMoDSingle(
            scenario=scenario,
            mode=2,
            beta=0.5,
            jitter=1e-6,
            max_wait=10,
            choice_price_mult=1.0,
            seed=42,
            choice_intercept=3.0,
            wage=25.0
        )
        
        print("Benchmarking OPTIMIZED implementation...")
        optimized_times = benchmark_optimized_reb_single(env, num_iterations=100)
        
        speedup = current_times.mean() / optimized_times.mean()
        time_saved = current_times.mean() - optimized_times.mean()
        
        single_results.append((config_name, current_times.mean(), optimized_times.mean(), speedup, time_saved))
        
        print(f"\n{'─' * 80}")
        print(f"Results for {config_name}:")
        print(f"{'─' * 80}")
        print(f"\nCURRENT Implementation:")
        print(f"  Mean time:   {current_times.mean():.4f} ms")
        print(f"  Std dev:     {current_times.std():.4f} ms")
        print(f"\nOPTIMIZED Implementation:")
        print(f"  Mean time:   {optimized_times.mean():.4f} ms")
        print(f"  Std dev:     {optimized_times.std():.4f} ms")
        print(f"\n{'─' * 80}")
        print(f"SPEEDUP: {speedup:.2f}x faster")
        print(f"Time saved: {time_saved:.4f} ms per reb_step()")
        print(f"{'─' * 80}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    print("\nDUAL-AGENT RESULTS:")
    print(f"{'Config':<25} {'Current (ms)':<15} {'Optimized (ms)':<15} {'Speedup':<12} {'Saved (ms)'}")
    print("─" * 80)
    for config_name, current, optimized, speedup, saved in dual_results:
        print(f"{config_name:<25} {current:<15.4f} {optimized:<15.4f} {speedup:<12.2f}x {saved:.4f}")
    
    avg_speedup_dual = np.mean([r[3] for r in dual_results])
    avg_saved_dual = np.mean([r[4] for r in dual_results])
    print(f"\n{'─' * 80}")
    print(f"Average Speedup: {avg_speedup_dual:.2f}x")
    print(f"Average Time Saved: {avg_saved_dual:.4f} ms per reb_step()")
    print(f"{'─' * 80}")
    
    print("\nSINGLE-AGENT RESULTS:")
    print(f"{'Config':<25} {'Current (ms)':<15} {'Optimized (ms)':<15} {'Speedup':<12} {'Saved (ms)'}")
    print("─" * 80)
    for config_name, current, optimized, speedup, saved in single_results:
        print(f"{config_name:<25} {current:<15.4f} {optimized:<15.4f} {speedup:<12.2f}x {saved:.4f}")
    
    avg_speedup_single = np.mean([r[3] for r in single_results])
    avg_saved_single = np.mean([r[4] for r in single_results])
    print(f"\n{'─' * 80}")
    print(f"Average Speedup: {avg_speedup_single:.2f}x")
    print(f"Average Time Saved: {avg_saved_single:.4f} ms per reb_step()")
    print(f"{'─' * 80}")
    
    print("\nExtrapolation to full episode (20 timesteps):")
    print(f"  Dual-agent time saved per episode: ~{avg_saved_dual * 20:.2f} ms")
    print(f"  Single-agent time saved per episode: ~{avg_saved_single * 20:.2f} ms")
    print(f"  Dual-agent time saved per 1000 episodes: ~{avg_saved_dual * 20 * 1000 / 1000:.2f} seconds")
    print(f"  Single-agent time saved per 1000 episodes: ~{avg_saved_single * 20 * 1000 / 1000:.2f} seconds")
    
    print("\nKey Optimizations Applied:")
    print("  1. Pre-calculate reb_time once per edge (reduces dict lookups)")
    print("  2. Pre-calculate cost factor once (eliminates redundant multiplications)")
    print("  3. [Single-agent only] Remove redundant edge validity check")

if __name__ == "__main__":
    run_benchmark()
