"""
Script to capture actual rebalancing flows (rebAction) from one test episode.
This captures the CPLEX-optimized flows, not just the Dirichlet desired distribution.
"""
import os
import pickle
import numpy as np
import torch
from copy import deepcopy
from collections import defaultdict

from src.envs.amod_env_multi import Scenario, AMoD
from src.algos.a2c_gnn_multi_agent import A2C
from src.algos.reb_flow_solver_multi_agent import solveRebFlow
from src.misc.utils import dictsum


def capture_rebalancing_flows(
    city="nyc_man_south",
    checkpoint_base="dual_no_standardization_warmup_1200_mode2_scaled",
    mode=2,
    max_steps=20,
    output_file="saved_files/reb_flows_visualization.pkl"
):
    """
    Run one episode and capture actual rebAction flows for visualization.
    
    Args:
        city: City scenario to use
        checkpoint_base: Base name for checkpoint files
        mode: Simulation mode (0, 1, or 2)
        max_steps: Number of timesteps per episode
        output_file: Output pickle file path
    
    Returns:
        Dictionary with flow data for visualization
    """
    
    # Calibrated parameters
    demand_ratio = {
        'san_francisco': 2, 'washington_dc': 4.2, 'chicago': 1.8, 
        'nyc_man_north': 1.8, 'nyc_man_middle': 1.8, 'nyc_man_south': 2.0, 
        'nyc_brooklyn': 9, 'nyc_manhattan': 2.0, 'porto': 4, 'rome': 1.8
    }
    json_hr = {
        'san_francisco': 19, 'washington_dc': 19, 'chicago': 19, 
        'nyc_man_north': 19, 'nyc_man_middle': 19, 'nyc_man_south': 19, 
        'nyc_brooklyn': 19, 'nyc_manhattan': 19, 'porto': 8, 'rome': 8
    }
    beta = {
        'san_francisco': 0.3, 'washington_dc': 0.5, 'chicago': 0.5, 
        'nyc_man_north': 0.5, 'nyc_man_middle': 0.5, 'nyc_man_south': 0.3, 
        'nyc_brooklyn': 0.5, 'nyc_manhattan': 0.3, 'porto': 0.1, 'rome': 0.1
    }
    choice_intercept = {
        'san_francisco': 0.0, 'washington_dc': 0.0, 'chicago': 0.0,
        'nyc_man_north': 0.0, 'nyc_man_middle': 0.0, 'nyc_man_south': 12.1,
        'nyc_brooklyn': 0.0, 'nyc_manhattan': 0.0, 'porto': 0.0, 'rome': 0.0
    }
    wage = {
        'san_francisco': 21.40, 'nyc_man_south': 33.39, 'nyc_brooklyn': 12.16, 
        'washington_dc': 26.99, 'nyc_man_north': 33.39, 'nyc_man_middle': 33.39,
        'nyc_manhattan': 33.39, 'chicago': 26.99, 'porto': 21.40, 'rome': 21.40
    }
    
    # Device
    device = torch.device("cpu")
    
    # Create scenario and environment
    print(f"Loading scenario for {city}...")
    scenario = Scenario(
        json_file=f"data/scenario_{city}.json",
        demand_ratio=demand_ratio[city],
        json_hr=json_hr[city],
        sd=10,
        json_tstep=3,
        tf=max_steps,
        impute=0,
        supply_ratio=1.0
    )
    
    env = AMoD(
        scenario, mode, 
        beta=beta[city], 
        jitter=1, 
        max_wait=2, 
        choice_price_mult=1.0, 
        seed=10, 
        fix_agent=2,  # No fixing
        choice_intercept=choice_intercept[city],
        wage=wage[city],
        dynamic_wage=False
    )
    
    print(f"Environment created with {env.nregion} regions and {len(env.edges)} edges")
    print(f"Edges: {env.edges}")
    
    # Calculate input size based on mode
    T = 6  # look_ahead
    scale_factor = 0.01
    use_od_prices = True  # The checkpoint was trained with OD prices
    
    if use_od_prices:
        # OD prices: 1 + T + 1 + 1 + nregion + nregion (for OD price matrices)
        input_size = 1 + T + 1 + 1 + env.nregion + env.nregion
    else:
        # Aggregated prices: 1 + T + 1 + 1 + 1 + 1
        input_size = 1 + T + 1 + 1 + 1 + 1
    
    # Create model agents
    print("Creating model agents...")
    model_agents = {}
    for agent_id in [0, 1]:
        model_agents[agent_id] = A2C(
            env=env,
            input_size=input_size,
            device=device,
            hidden_size=256,
            T=T,
            mode=mode,
            gamma=0.97,
            p_lr=1e-4,
            q_lr=1e-4,
            actor_clip=500,
            critic_clip=500,
            scale_factor=scale_factor,
            agent_id=agent_id,
            json_file=f"data/scenario_{city}.json",
            use_od_prices=use_od_prices
        )
    
    # Load checkpoints
    print("Loading checkpoints...")
    for agent_id, agent_suffix in [(0, "agent1"), (1, "agent2")]:
        ckpt_path = f"ckpt/{checkpoint_base}_{agent_suffix}_running.pth"
        print(f"  Loading {ckpt_path}")
        model_agents[agent_id].load_checkpoint(ckpt_path)
        model_agents[agent_id].eval()
    
    # CPLEX path
    cplexpath = "/apps/cplex/cplex1210/opl/bin/x86-64_linux/"
    
    # Storage for visualization data
    flow_data = {
        'agent_reb_flows': {0: [], 1: []},  # Actual rebalancing flows per edge
        'agent_price_scalars': {0: [], 1: []},  # Price scalars per region
        'agent_acc': {0: [], 1: []},  # Vehicle availability per region
        'agent_desired_acc': {0: [], 1: []},  # Desired distribution per region
        'edges': list(env.edges),  # List of (origin, destination) tuples
        'regions': list(env.region),
        'metadata': {
            'city': city,
            'mode': mode,
            'num_regions': env.nregion,
            'num_timesteps': max_steps,
            'checkpoint': checkpoint_base,
            'zip_codes': ['10002', '10003', '10005', '10006', '10007', '10009', 
                         '10010', '10011', '10012', '10013', '10014', '10038']
        }
    }
    
    # Reset environment
    print("\nRunning episode...")
    obs = env.reset()
    done = False
    timestep = 0
    
    # Initial action placeholder
    action_rl = {a: np.zeros(env.nregion) for a in [0, 1]}
    
    while not done:
        print(f"  Timestep {timestep}...")
        
        if mode == 0:
            # Mode 0: Only rebalancing
            # Matching step
            obs, paxreward, done, info, system_info, _, _ = env.match_step_simple()
            
            # Get rebalancing actions from models
            action_rl = {}
            for a in [0, 1]:
                action_rl[a] = model_agents[a].select_action(obs[a], deterministic=True)
            
            # Compute desired accumulation
            desiredAcc = {}
            for a in [0, 1]:
                total_vehicles = dictsum(env.agent_acc[a], env.time + 1)
                desiredAcc[a] = {
                    env.region[i]: int(action_rl[a][i] * total_vehicles)
                    for i in range(env.nregion)
                }
            
            # Store desired distribution
            for a in [0, 1]:
                flow_data['agent_desired_acc'][a].append(
                    np.array([desiredAcc[a][r] for r in env.region])
                )
            
            # Solve rebalancing flow optimization
            rebAction = {}
            for a in [0, 1]:
                rebAction[a] = solveRebFlow(
                    env, f"viz_{city}", desiredAcc[a], 
                    cplexpath, "saved_files", a, max_steps, mode, 
                    job_id=f"viz_{timestep}"
                )
            
            # Store actual flows
            for a in [0, 1]:
                flow_data['agent_reb_flows'][a].append(np.array(rebAction[a]))
            
            # Store vehicle availability
            for a in [0, 1]:
                acc_current = np.array([
                    env.agent_acc[a].get(env.region[i], {}).get(env.time, 0) 
                    for i in range(env.nregion)
                ])
                flow_data['agent_acc'][a].append(acc_current)
            
            # Rebalancing step
            _, rebreward, done, info, system_info, _, _ = env.reb_step(rebAction)
            
        elif mode == 2:
            # Mode 2: Both pricing and rebalancing
            # Matching step with previous pricing action
            obs, paxreward, done, info, system_info, _, _ = env.match_step_simple(action_rl)
            
            # Get combined actions from models
            action_rl = {}
            for a in [0, 1]:
                action_rl[a] = model_agents[a].select_action(obs[a], deterministic=True)
            
            # Store price scalars (action[:, 0])
            for a in [0, 1]:
                flow_data['agent_price_scalars'][a].append(action_rl[a][:, 0].copy())
            
            # Compute desired accumulation from rebalancing action (action[:, 1])
            desiredAcc = {}
            for a in [0, 1]:
                total_vehicles = dictsum(env.agent_acc[a], env.time + 1)
                desiredAcc[a] = {
                    env.region[i]: int(action_rl[a][i, 1] * total_vehicles)
                    for i in range(env.nregion)
                }
            
            # Store desired distribution
            for a in [0, 1]:
                flow_data['agent_desired_acc'][a].append(
                    np.array([desiredAcc[a][r] for r in env.region])
                )
            
            # Solve rebalancing flow optimization
            rebAction = {}
            for a in [0, 1]:
                rebAction[a] = solveRebFlow(
                    env, f"viz_{city}", desiredAcc[a], 
                    cplexpath, "saved_files", a, max_steps, mode, 
                    job_id=f"viz_{timestep}"
                )
            
            # Store actual flows
            for a in [0, 1]:
                flow_data['agent_reb_flows'][a].append(np.array(rebAction[a]))
            
            # Store vehicle availability
            for a in [0, 1]:
                acc_current = np.array([
                    env.agent_acc[a].get(env.region[i], {}).get(env.time, 0) 
                    for i in range(env.nregion)
                ])
                flow_data['agent_acc'][a].append(acc_current)
            
            # Rebalancing step
            _, rebreward, done, info, system_info, _, _ = env.reb_step(rebAction)
        
        timestep += 1
    
    # Convert lists to numpy arrays
    for a in [0, 1]:
        flow_data['agent_reb_flows'][a] = np.array(flow_data['agent_reb_flows'][a])
        flow_data['agent_price_scalars'][a] = np.array(flow_data['agent_price_scalars'][a]) if flow_data['agent_price_scalars'][a] else None
        flow_data['agent_acc'][a] = np.array(flow_data['agent_acc'][a])
        flow_data['agent_desired_acc'][a] = np.array(flow_data['agent_desired_acc'][a])
    
    # Save to pickle
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'wb') as f:
        pickle.dump(flow_data, f)
    
    print(f"\nFlow data saved to {output_file}")
    print(f"Shape of agent_reb_flows[0]: {flow_data['agent_reb_flows'][0].shape}")
    print(f"Shape of agent_acc[0]: {flow_data['agent_acc'][0].shape}")
    print(f"Number of edges: {len(flow_data['edges'])}")
    
    return flow_data


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Capture rebalancing flows for visualization")
    parser.add_argument("--city", type=str, default="nyc_man_south", help="City to use")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Checkpoint base name")
    parser.add_argument("--mode", type=int, default=2, help="Simulation mode")
    parser.add_argument("--max_steps", type=int, default=20, help="Timesteps per episode")
    parser.add_argument("--supply_ratio", type=float, default=1.0, help="Supply ratio")
    parser.add_argument("--output_file", type=str, default=None, help="Output file path")
    
    args = parser.parse_args()
    
    # Determine output filename if not provided
    if args.output_file is None:
        # Estimate number of cars based on supply_ratio (approximate)
        num_cars = int(1200 * args.supply_ratio)  # Adjust base number as needed
        args.output_file = f"saved_files/reb_flows_mode{args.mode}_{num_cars}cars_{args.city}.pkl"
    
    flow_data = capture_rebalancing_flows(
        city=args.city,
        checkpoint_base=args.checkpoint_path,
        mode=args.mode,
        max_steps=args.max_steps,
        output_file=args.output_file
    )
