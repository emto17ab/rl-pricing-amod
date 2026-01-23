"""
Test script to compare parallel vs sequential rebalancing performance
Tests 4 configurations over 50 episodes in mode 2:
1. Sequential + default CPLEX threads
2. Sequential + 1 CPLEX thread
3. Parallel + default CPLEX threads
4. Parallel + 1 CPLEX thread
"""
import argparse
import numpy as np
import torch
import time
from src.envs.amod_env_multi import Scenario, AMoD
from src.algos.a2c_gnn_multi_agent import A2C
from src.algos.reb_flow_solver_multi_agent import solveRebFlow
from src.misc.utils import dictsum
from concurrent.futures import ThreadPoolExecutor


def run_rebalancing_test(use_parallel, num_episodes=50, test_name="test"):
    """
    Run rebalancing test for specified configuration
    
    Args:
        use_parallel: Whether to use ThreadPoolExecutor for parallel rebalancing
        num_episodes: Number of episodes to run
        test_name: Name for this test run (used in job_id to avoid file conflicts)
    
    Returns:
        dict with timing statistics
    """
    # Fixed parameters for testing
    city = "nyc_man_south"
    mode = 2
    seed = 10
    max_steps = 20
    cplexpath = "/apps/cplex/cplex1210/opl/bin/x86-64_linux/"
    directory = "saved_files"
    
    # City-specific parameters
    demand_ratio = 1.0
    json_hr = 19
    json_tstep = 3
    beta = 0.5
    jitter = 1
    maxt = 2
    choice_price_mult = 1.0
    choice_intercept = 9.84
    wage = 22.77
    
    # Create scenario
    scenario = Scenario(
        json_file=f"data/scenario_{city}.json",
        demand_ratio=demand_ratio * 2,
        json_hr=json_hr,
        sd=seed,
        json_tstep=json_tstep,
        tf=max_steps,
        impute=0,
        supply_ratio=1.0,
        agent0_vehicle_ratio=0.5
    )
    
    # Create environment
    env = AMoD(
        scenario,
        mode,
        beta=beta,
        jitter=jitter,
        max_wait=maxt,
        choice_price_mult=choice_price_mult,
        seed=seed,
        fix_agent=2,
        choice_intercept=choice_intercept,
        wage=wage,
        use_dynamic_wage_man_south=False
    )
    
    # Load trained models
    device = torch.device("cpu")
    look_ahead = 6
    use_od_prices = True  # Models were trained with OD prices
    
    # Calculate input size based on OD prices: T + 3 (current_avb, queue, demand) + 3*nregion (own, competitor, difference OD prices)
    input_size = look_ahead + 3 + 3 * env.nregion
    hidden_size = 256
    
    model_agents = {}
    for a in [0, 1]:
        model_agents[a] = A2C(
            env=env,
            input_size=input_size,
            hidden_size=hidden_size,
            device=device,
            p_lr=1e-4,
            q_lr=1e-4,
            T=look_ahead,
            scale_factor=0.01,
            json_file=f"data/scenario_{city}.json",
            mode=mode,
            actor_clip=500,
            critic_clip=500,
            gamma=0.97,
            agent_id=a,
            use_od_prices=use_od_prices,
            no_share_info=False,
            reward_scale=2000.0,
        )
        # Load checkpoint
        checkpoint_path = f"ckpt/dual_agent_{city}_cars_1050_mode{mode}_agent{a+1}_test.pth"
        model_agents[a].load_checkpoint(path=checkpoint_path)
        model_agents[a].eval()
    
    # Track timing for rebalancing only
    rebalancing_times = []
    total_start = time.time()
    
    for episode in range(num_episodes):
        obs = env.reset()
        action_rl = {0: np.zeros((env.nregion, 2)), 1: np.zeros((env.nregion, 2))}
        done = False
        episode_reb_times = []
        
        while not done:
            # Matching step
            obs, paxreward, done, info, system_info, _, _ = env.match_step_simple(action_rl)
            
            # Get actions from models
            action_rl = {}
            for a in [0, 1]:
                action_rl[a] = model_agents[a].select_action(obs[a], deterministic=True)
            
            # Compute desired accumulation
            desiredAcc = {}
            for a in [0, 1]:
                desiredAcc[a] = {
                    env.region[i]: int(action_rl[a][i, -1] * dictsum(env.agent_acc[a], env.time + 1))
                    for i in range(env.nregion)
                }
            
            # Time rebalancing step only
            reb_start = time.time()
            
            if use_parallel:
                # Parallel rebalancing with ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=2) as executor:
                    results = list(executor.map(
                        lambda a: solveRebFlow(
                            env, f"scenario_{city}", desiredAcc[a], cplexpath, directory,
                            a, 1000, mode, job_id=f"reb_{test_name}"
                        ),
                        [0, 1]
                    ))
                rebAction = {0: results[0], 1: results[1]}
            else:
                # Sequential rebalancing
                rebAction = {}
                for a in [0, 1]:
                    rebAction[a] = solveRebFlow(
                        env, f"scenario_{city}", desiredAcc[a], cplexpath, directory,
                        a, 1000, mode, job_id=f"reb_{test_name}"
                    )
            
            reb_time = time.time() - reb_start
            episode_reb_times.append(reb_time)
            
            # Execute rebalancing step
            _, rebreward, done, info, system_info, _, _ = env.reb_step(rebAction)
        
        # Track episode rebalancing time
        episode_total_reb_time = sum(episode_reb_times)
        rebalancing_times.append(episode_total_reb_time)
        
        if (episode + 1) % 10 == 0:
            avg_time = np.mean(rebalancing_times[-10:])
            print(f"  Episode {episode + 1}/{num_episodes}: Avg reb time (last 10) = {avg_time:.3f}s")
    
    total_time = time.time() - total_start
    
    # Compute statistics
    results = {
        "mean_reb_time": np.mean(rebalancing_times),
        "std_reb_time": np.std(rebalancing_times),
        "min_reb_time": np.min(rebalancing_times),
        "max_reb_time": np.max(rebalancing_times),
        "total_time": total_time,
        "episodes": num_episodes,
        "use_parallel": use_parallel
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Test rebalancing parallelization performance")
    parser.add_argument("--num_episodes", type=int, default=50, help="Number of test episodes")
    args = parser.parse_args()
    
    # Run both tests sequentially
    print("=" * 80)
    print("RUNNING REBALANCING PERFORMANCE TESTS")
    print("=" * 80)
    print(f"Testing {args.num_episodes} episodes for each configuration\n")
    
    # Test 1: Sequential
    print("\n" + "-" * 80)
    print("TEST 1: Sequential Rebalancing")
    print("-" * 80)
    results_seq = run_rebalancing_test(use_parallel=False, num_episodes=args.num_episodes, test_name="sequential")
    
    # Test 2: Parallel
    print("\n" + "-" * 80)
    print("TEST 2: Parallel Rebalancing (ThreadPoolExecutor)")
    print("-" * 80)
    results_par = run_rebalancing_test(use_parallel=True, num_episodes=args.num_episodes, test_name="parallel")
    
    # Print comparison
    print("\n" + "=" * 80)
    print("FINAL COMPARISON")
    print("=" * 80)
    print(f"\nSequential:")
    print(f"  Mean rebalancing time: {results_seq['mean_reb_time']:.3f}s ± {results_seq['std_reb_time']:.3f}s")
    print(f"  Total time: {results_seq['total_time']:.2f}s")
    
    print(f"\nParallel:")
    print(f"  Mean rebalancing time: {results_par['mean_reb_time']:.3f}s ± {results_par['std_reb_time']:.3f}s")
    print(f"  Total time: {results_par['total_time']:.2f}s")
    
    speedup = results_seq['mean_reb_time'] / results_par['mean_reb_time']
    print(f"\nSpeedup: {speedup:.2f}x ({'FASTER' if speedup > 1 else 'SLOWER'} with parallel)")
    print("=" * 80)


if __name__ == "__main__":
    main()
