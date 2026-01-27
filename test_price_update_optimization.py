import time
import numpy as np
from copy import deepcopy
from src.envs.amod_env_multi import AMoD, Scenario

def create_test_scenario(demand_scale=1.0):
    """Create a test scenario - using NYC Manhattan South"""
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

def benchmark_current_price_updates(env, num_iterations=100):
    """Benchmark CURRENT price update implementation"""
    times = []
    
    for iteration in range(num_iterations):
        env.reset()
        
        # Create random price actions (mode 1 - just scalars)
        price_action = {
            0: np.random.rand(env.nregion) * 0.5 + 0.25,
            1: np.random.rand(env.nregion) * 0.5 + 0.25
        }
        
        start = time.perf_counter()
        
        t = env.time
        # Current implementation: condition checked for EVERY (n,j) pair
        for n in env.region:
            for j in env.G[n]:
                if env.mode != 0 and price_action is not None and np.sum([np.sum(price_action[a]) for a in env.agents]) != 0:
                    for agent_id in env.agents:
                        baseline_price = env.agent_price[agent_id][n, j][t]
                        
                        if env.fix_agent == agent_id:
                            price_scalar = 0.5
                        else:
                            price_scalar = price_action[agent_id][n]
                            if isinstance(price_scalar, (list, np.ndarray)):
                                price_scalar = price_scalar[0]
                        
                        p = 2 * baseline_price * price_scalar
                        if p <= 1e-6:
                            p = env.jitter
                        
                        env.agent_price[agent_id][n, j][t] = p
        
        end = time.perf_counter()
        times.append((end - start) * 1000)
    
    return np.array(times)

def benchmark_optimized_price_updates(env, num_iterations=100):
    """Benchmark OPTIMIZED price update implementation"""
    times = []
    
    for iteration in range(num_iterations):
        env.reset()
        
        # Create random price actions
        price_action = {
            0: np.random.rand(env.nregion) * 0.5 + 0.25,
            1: np.random.rand(env.nregion) * 0.5 + 0.25
        }
        
        start = time.perf_counter()
        
        t = env.time
        
        # OPTIMIZED: Check condition once
        if env.mode != 0 and price_action is not None:
            # Compute sum once (not for every O-D pair)
            total_price_sum = sum(np.sum(price_action[a]) for a in env.agents)
            
            if total_price_sum != 0:
                # Pre-extract scalars once per region
                price_scalars = {}
                for agent_id in env.agents:
                    if env.fix_agent == agent_id:
                        price_scalars[agent_id] = {n: 0.5 for n in env.region}
                    else:
                        price_scalars[agent_id] = {}
                        for n in env.region:
                            scalar = price_action[agent_id][n]
                            if isinstance(scalar, (list, np.ndarray)):
                                scalar = scalar[0]
                            price_scalars[agent_id][n] = scalar
                
                # Update prices with pre-extracted scalars
                for n in env.region:
                    for j in env.G[n]:
                        for agent_id in env.agents:
                            baseline_price = env.agent_price[agent_id][n, j][t]
                            price_scalar = price_scalars[agent_id][n]
                            
                            p = 2 * baseline_price * price_scalar
                            if p <= 1e-6:
                                p = env.jitter
                            
                            env.agent_price[agent_id][n, j][t] = p
        
        end = time.perf_counter()
        times.append((end - start) * 1000)
    
    return np.array(times)

def verify_correctness():
    """Verify that both implementations produce identical results"""
    print("\n" + "=" * 80)
    print("CORRECTNESS VERIFICATION")
    print("=" * 80)
    
    scenario = create_test_scenario(1.0)
    
    # Test with same random prices
    np.random.seed(42)
    test_prices = {
        0: np.random.rand(69) * 0.5 + 0.25,
        1: np.random.rand(69) * 0.5 + 0.25
    }
    
    # Current implementation
    env1 = AMoD(scenario=deepcopy(scenario), mode=1, beta=0.5, jitter=1e-6,
                max_wait=10, choice_price_mult=1.0, seed=42, fix_agent=2,
                choice_intercept=3.0, wage=25.0)
    env1.reset()
    t1 = env1.time
    
    for n in env1.region:
        for j in env1.G[n]:
            if env1.mode != 0 and test_prices is not None and np.sum([np.sum(test_prices[a]) for a in env1.agents]) != 0:
                for agent_id in env1.agents:
                    baseline_price = env1.agent_price[agent_id][n, j][t1]
                    if env1.fix_agent == agent_id:
                        price_scalar = 0.5
                    else:
                        price_scalar = test_prices[agent_id][n]
                        if isinstance(price_scalar, (list, np.ndarray)):
                            price_scalar = price_scalar[0]
                    p = 2 * baseline_price * price_scalar
                    if p <= 1e-6:
                        p = env1.jitter
                    env1.agent_price[agent_id][n, j][t1] = p
    
    # Optimized implementation
    env2 = AMoD(scenario=deepcopy(scenario), mode=1, beta=0.5, jitter=1e-6,
                max_wait=10, choice_price_mult=1.0, seed=42, fix_agent=2,
                choice_intercept=3.0, wage=25.0)
    env2.reset()
    t2 = env2.time
    
    if env2.mode != 0 and test_prices is not None:
        total_price_sum = sum(np.sum(test_prices[a]) for a in env2.agents)
        if total_price_sum != 0:
            price_scalars = {}
            for agent_id in env2.agents:
                if env2.fix_agent == agent_id:
                    price_scalars[agent_id] = {n: 0.5 for n in env2.region}
                else:
                    price_scalars[agent_id] = {}
                    for n in env2.region:
                        scalar = test_prices[agent_id][n]
                        if isinstance(scalar, (list, np.ndarray)):
                            scalar = scalar[0]
                        price_scalars[agent_id][n] = scalar
            
            for n in env2.region:
                for j in env2.G[n]:
                    for agent_id in env2.agents:
                        baseline_price = env2.agent_price[agent_id][n, j][t2]
                        price_scalar = price_scalars[agent_id][n]
                        p = 2 * baseline_price * price_scalar
                        if p <= 1e-6:
                            p = env2.jitter
                        env2.agent_price[agent_id][n, j][t2] = p
    
    # Compare prices
    all_match = True
    for n in env1.region:
        for j in env1.G[n]:
            for agent_id in env1.agents:
                price1 = env1.agent_price[agent_id][n, j][t1]
                price2 = env2.agent_price[agent_id][n, j][t2]
                if not np.isclose(price1, price2):
                    print(f"  ❌ Mismatch at region {n}→{j}, agent {agent_id}")
                    print(f"     Current: {price1:.6f}, Optimized: {price2:.6f}")
                    all_match = False
    
    if all_match:
        print("  ✓ All prices match! Implementations are functionally identical.")
        return True
    else:
        print("  ✗ Price mismatches found!")
        return False

def run_benchmark():
    print("=" * 80)
    print("PRICE UPDATE OPTIMIZATION TEST")
    print("=" * 80)
    
    # Verify correctness first
    if not verify_correctness():
        print("\n⚠️  WARNING: Correctness verification failed!")
        return
    
    # Test different demand levels
    test_configs = [
        ("Low Demand (0.5x)", 0.5),
        ("Normal Demand (1.0x)", 1.0),
        ("High Demand (2.0x)", 2.0),
    ]
    
    all_results = []
    
    for config_name, demand_scale in test_configs:
        print(f"\n{'=' * 80}")
        print(f"Test: {config_name}")
        print(f"{'=' * 80}")
        
        scenario = create_test_scenario(demand_scale)
        
        env = AMoD(
            scenario=scenario,
            mode=1,
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
        current_times = benchmark_current_price_updates(env, num_iterations=100)
        
        env = AMoD(
            scenario=scenario,
            mode=1,
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
        optimized_times = benchmark_optimized_price_updates(env, num_iterations=100)
        
        current_mean = current_times.mean()
        optimized_mean = optimized_times.mean()
        speedup = current_mean / optimized_mean
        
        print(f"\n{'─' * 80}")
        print(f"Results for {config_name}:")
        print(f"{'─' * 80}")
        print(f"\nCURRENT Implementation:")
        print(f"  Mean time:   {current_mean:.4f} ms")
        print(f"  Std dev:     {current_times.std():.4f} ms")
        
        print(f"\nOPTIMIZED Implementation:")
        print(f"  Mean time:   {optimized_mean:.4f} ms")
        print(f"  Std dev:     {optimized_times.std():.4f} ms")
        
        print(f"\n{'─' * 80}")
        print(f"SPEEDUP: {speedup:.2f}x faster")
        print(f"Time saved: {current_mean - optimized_mean:.4f} ms per price update")
        print(f"{'─' * 80}")
        
        num_edges = len(list((n, j) for n in env.region for j in env.G[n]))
        print(f"\nScenario Statistics:")
        print(f"  Number of regions: {env.nregion}")
        print(f"  Number of edges: {num_edges}")
        print(f"  Redundant condition checks avoided: {num_edges}")
        
        all_results.append({
            'config': config_name,
            'current_mean': current_mean,
            'optimized_mean': optimized_mean,
            'speedup': speedup,
            'time_saved': current_mean - optimized_mean
        })
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\n{'Config':<25} {'Current (ms)':<15} {'Optimized (ms)':<15} {'Speedup':<10} {'Saved (ms)'}")
    print("─" * 80)
    for result in all_results:
        print(f"{result['config']:<25} {result['current_mean']:<15.4f} "
              f"{result['optimized_mean']:<15.4f} {result['speedup']:<10.2f}x "
              f"{result['time_saved']:.4f}")
    
    avg_speedup = np.mean([r['speedup'] for r in all_results])
    avg_saved = np.mean([r['time_saved'] for r in all_results])
    
    print(f"\n{'─' * 80}")
    print(f"Average Speedup: {avg_speedup:.2f}x")
    print(f"Average Time Saved: {avg_saved:.4f} ms per price update")
    print(f"{'─' * 80}")
    
    print(f"\nExtrapolation to full episode (20 timesteps):")
    print(f"  Time saved per episode: ~{avg_saved * 20:.2f} ms")
    print(f"  Time saved per 1000 episodes: ~{avg_saved * 20000 / 1000:.2f} seconds")
    
    print(f"\nKey Optimizations Applied:")
    print(f"  1. Condition check: O(edges) → O(1)")
    print(f"  2. Scalar extraction: O(edges × agents) → O(regions × agents)")
    print(f"  3. isinstance() calls reduced by ~{num_edges}x per update")

if __name__ == "__main__":
    run_benchmark()
