import argparse
import torch
from src.envs.amod_env_multi import Scenario, AMoD
from src.algos.a2c_gnn_multi_agent import A2C
from tqdm import trange
import numpy as np
from src.misc.utils import dictsum, nestdictsum
import copy, os
from src.algos.reb_flow_solver_multi import solveRebFlow
import json, pickle
import wandb
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Define calibrated simulation parameters
demand_ratio = {'san_francisco': 2, 'washington_dc': 4.2, 'chicago': 1.8, 'nyc_man_north': 1.8, 'nyc_man_middle': 1.8,
                'nyc_man_south': 1.8, 'nyc_brooklyn': 9, 'porto': 4, 'rome': 1.8, 'shenzhen_baoan': 2.5,
                'shenzhen_downtown_west': 2.5, 'shenzhen_downtown_east': 3, 'shenzhen_north': 3
               }
json_hr = {'san_francisco':19, 'washington_dc': 19, 'chicago': 19, 'nyc_man_north': 19, 'nyc_man_middle': 19,
           'nyc_man_south': 19, 'nyc_brooklyn': 19, 'porto': 8, 'rome': 8, 'shenzhen_baoan': 8,
           'shenzhen_downtown_west': 8, 'shenzhen_downtown_east': 8, 'shenzhen_north': 8
          }
beta = {'san_francisco': 0.2, 'washington_dc': 0.5, 'chicago': 0.5, 'nyc_man_north': 0.5, 'nyc_man_middle': 0.5,
                'nyc_man_south': 0.5, 'nyc_brooklyn':0.5, 'porto': 0.1, 'rome': 0.1, 'shenzhen_baoan': 0.5,
                'shenzhen_downtown_west': 0.5, 'shenzhen_downtown_east': 0.5, 'shenzhen_north': 0.5}

test_tstep = {'san_francisco': 3, 'nyc_brooklyn': 4, 'shenzhen_downtown_west': 3, 'nyc_man_middle': 3, 'nyc_man_south': 3, 'nyc_man_north': 3, 'washington_dc':3, 'chicago':3}

parser = argparse.ArgumentParser(description="SAC-GNN")

# Simulator parameters
parser.add_argument(
    "--seed", type=int, default=10, metavar="S", help="random seed (default: 10)"
)
parser.add_argument(
    "--demand_ratio",
    type=float,
    default=1,
    metavar="S",
    help="demand_ratio (default: 1)",
)
parser.add_argument(
    "--json_hr", type=int, default=7, metavar="S", help="json_hr (default: 7)"
)
parser.add_argument(
    "--json_tstep",
    type=int,
    default=3,
    metavar="S",
    help="minutes per timestep (default: 3min)",
)
parser.add_argument(
    '--mode', 
    type=int, 
    default=1,
    help='rebalancing mode. (0:manul, 1:pricing, 2:both. default 1)',
)

parser.add_argument(
    "--beta",
    type=float,
    default=0.5,
    metavar="S",
    help="cost of rebalancing (default: 0.5)",
)

# Model parameters
parser.add_argument(
    "--test", 
    action="store_true",
    default=False, 
    help="activates test mode for agent evaluation"
)
parser.add_argument(
    "--cplexpath",
    type=str,
    default="/apps/cplex/cplex1210/opl/bin/x86-64_linux/", # Changed to HPC PATH
    help="defines directory of the CPLEX installation",
)

parser.add_argument(
    "--directory",
    type=str,
    default="saved_files",
    help="defines directory where to save files",
)
parser.add_argument(
    "--max_episodes",
    type=int,
    default=10000,
    metavar="N",
    help="number of episodes to train agent (default: 10k)",
)
parser.add_argument(
    "--max_steps",
    type=int,
    default=20,
    metavar="N",
    help="number of steps per episode (default: 20)",
)
parser.add_argument(
    "--cuda", 
    action="store_true",
    default=False,
    help="Enables CUDA training",
)

parser.add_argument(
    "--batch_size",
    type=int,
    default=100,
    help="batch size for training (default: 100)",
)
parser.add_argument(
    "--jitter",
    type=int,
    default=1,
    help="jitter for demand 0 (default: 1)",
)
parser.add_argument(
    "--maxt",
    type=int,
    default=2,
    help="maximum passenger waiting time (default: 2mins)",
)

parser.add_argument(
    "--hidden_size",
    type=int,
    default=256,
    help="hidden size of neural networks (default: 256)",
)

parser.add_argument(
    "--checkpoint_path",
    type=str,
    default="A2C",
    help="name of checkpoint file to save/load (default: A2C)",
)
parser.add_argument(
    "--load",
    action="store_true",
    default=False,
    help="either to start training from checkpoint (default: False)",
)

parser.add_argument(
    "--clip",
    type=int,
    default=500,
    help="clip value for gradient clipping (default: 500)",
)

parser.add_argument(
    "--p_lr",
    type=float,
    default=1e-3,
    help="learning rate for policy network (default: 1e-3)",
)

parser.add_argument(
    "--q_lr",
    type=float,
    default=1e-3,
    help="learning rate for Q networks (default: 1e-3)",
)

parser.add_argument(
    "--city",
    type=str,
    default="san_francisco",
    help="city to train on",
)

parser.add_argument(
    "--impute",
    type=int,
    default=0,
    help="Whether impute the zero price (default: False)",
)

parser.add_argument(
    "--supply_ratio",
    type=float,
    default=1.0,
    help="supply scaling factor (default: 1)",
)

parser.add_argument(
    "--look_ahead",
    type=int,
    default=6,
    help="Time steps to look ahead (default: 6)",
)

parser.add_argument(
    "--scale_factor",
    type=float,
    default=0.01,
    help="Scale factor (default: 0.01)",
)

parser.add_argument(
    "--gamma",
    type=float,
    default=0.97,
    help="Discount factor (default: 0.97)",
)

parser.add_argument(
    "--choice_price_mult",
    type=float,
    default=1.0,
    help="Choice price multiplier (default: 1.0)",
)

# Parser arguments
args = parser.parse_args()

# Set device
args.cuda = args.cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")

# Set up weights and biases
wandb.login(key=os.getenv('WANDB_API_KEY'))

run = wandb.init(
    project="thesis",
    config=args,
)

# Set city
city = args.city

# Create the scenario
scenario = Scenario(
            json_file=f"data/scenario_{city}.json",
            demand_ratio=demand_ratio[city],
            json_hr=json_hr[city],
            sd=args.seed,
            json_tstep=args.json_tstep,
            tf=args.max_steps,
            impute=args.impute,
            supply_ratio=args.supply_ratio)

# Create the environment
env = AMoD(scenario, args.mode, beta=beta[city], jitter=args.jitter, max_wait=args.maxt, choice_price_mult=args.choice_price_mult, seed = args.seed)

model_agents = {
        a: A2C(
            env=env,
            input_size= args.look_ahead + 5, # 5 features + time encoding
            hidden_size=args.hidden_size,
            device=device,
            p_lr=args.p_lr,
            q_lr=args.q_lr,
            T = args.look_ahead,
            scale_factor = args.scale_factor,
            json_file=f"data/scenario_{city}.json",
            mode=args.mode,
            clip=args.clip,
            gamma=args.gamma,
            agent_id = a 
        )
        for a in [0, 1]
    }

if args.load:
    print("load checkpoint")
    for agent_id in [0, 1]:
        checkpoint_path = f"ckpt/{args.checkpoint_path}_agent{agent_id+1}_running.pth"
        model_agents[agent_id].load_checkpoint(path=checkpoint_path)
        print(f"Loaded checkpoint for agent {agent_id} from {checkpoint_path}")
    print("Loaded models from checkpoint successfully")

train_episodes = args.max_episodes  # set max number of training episodes
epochs = trange(train_episodes)  # epoch iterator

best_reward = -np.inf  # set best training reward
best_reward_test = -np.inf  # set best test reward

for agent_id in [0, 1]:
    model_agents[agent_id].train()

# Check metrics
epoch_demand_list = []
epoch_reward_list = []
epoch_waiting_list = []
epoch_servedrate_list = []
epoch_rebalancing_cost = []

# Get initial vehicles
initial_vehicles = env.get_initial_vehicles()

for i_episode in epochs:
    obs = env.reset()  # initialize environment

    action_rl = {
            a: [0.0] * env.nregion for a in [0, 1]
        }
    
    # Save original demand for reference
    demand_ori = nestdictsum(env.demand)

    if i_episode == train_episodes - 1:
        export = {"demand_ori":copy.deepcopy(env.demand)}

    episode_reward = {0: 0, 1: 0}
    episode_served_demand = {0: 0, 1: 0}
    episode_unserved_demand = {0: 0, 1: 0}
    episode_rebalancing_cost = {0: 0, 1: 0}
    episode_rejected_demand = {0: 0, 1: 0}
    episode_total_revenue = {0: 0, 1: 0}
    episode_total_operating_cost = {0: 0, 1: 0}
    episode_waiting = {0: 0, 1: 0}
    episode_rejection_rates = {0: [], 1: []}

    done = False
    step = 0

    while not done:
        if env.mode == 0:
            # Make Match Step
            obs, paxreward, done, info, _, _ = env.match_step_simple()

            # Update episode reward
            episode_reward = {a: episode_reward[a] + paxreward[a] for a in [0, 1]}

            action_rl = {a: model_agents[a].select_action(obs[a]) for a in [0,1]}

            desiredAcc = {
                    a: {
                        env.region[i]: int(action_rl[a][i] * dictsum(env.agent_acc[a], env.time + 1))
                        for i in range(env.nregion)
                    }
                    for a in [0, 1]
                }

            rebAction = {
                    a: solveRebFlow(env, "scenario_san_francisco4", desiredAcc[a], args.cplexpath, args.directory, a)
                    for a in [0, 1]
            }
            
            new_obs, rebreward, done, info, _, _ = env.reb_step(rebAction)
            episode_reward = {a: episode_reward[a] + rebreward[a] for a in [0, 1]}
            
            for agent_id in [0, 1]:
                model_agents[agent_id].rewards.append(paxreward[agent_id] + rebreward[agent_id])

        elif env.mode == 1:
            obs, paxreward, done, info, _, _ = env.match_step_simple(action_rl)

            episode_reward = {a: episode_reward[a] + paxreward[a] for a in [0, 1]}

            for agent_id in [0, 1]:
                model_agents[agent_id].rewards.append(paxreward[agent_id])

            action_rl = {a: model_agents[a].select_action(obs[a]) for a in [0,1]}

            # Matching update (global step)
            env.matching_update()
        
        elif env.mode == 2:
            # --- Matching step ---
            obs, paxreward, done, info, _, _ = env.match_step_simple(action_rl)
            
            episode_reward = {a: episode_reward[a] + paxreward[a] for a in [0, 1]}

            action_rl = {a: model_agents[a].select_action(obs[a]) for a in [0,1]}
                   
            # --- Desired Acc computation ---
            desiredAcc = {
                a: {
                    env.region[i]: int(action_rl[a][i][-1] * dictsum(env.agent_acc[a], env.time + 1))
                    for i in range(env.nregion)
                } for a in [0, 1]
            }
            
            
            # --- Rebalancing step ---
            rebAction = {
                a: solveRebFlow(env, "scenario_san_francisco4", desiredAcc[a], args.cplexpath, args.directory, a)
                for a in [0, 1]
            }
        
            new_obs, rebreward, done, info, _, _ = env.reb_step(rebAction)
        
            episode_reward = {a: episode_reward[a] + rebreward[a] for a in [0, 1]}
            for agent_id in [0, 1]:
                model_agents[agent_id].rewards.append(paxreward[agent_id] + rebreward[agent_id])
        else:
            raise ValueError("Only mode 0, 1, and 2 are allowed")

        for a in [0, 1]:
                episode_served_demand[a] += info[a]["served_demand"]
                episode_unserved_demand[a] += info[a]["unserved_demand"]
                episode_rebalancing_cost[a] += info[a]["rebalancing_cost"]
                episode_total_revenue[a] += info[a]["revenue"]
                episode_rejected_demand[a] += info[a]["rejected_demand"]
                episode_total_operating_cost[a] += info[a]["operating_cost"]
                episode_waiting[a] += info[a]["served_waiting"]
                episode_rejection_rates[a].append(info[a]["rejection_rate"])
    
        step += 1
    
    # Update both agent models after episode and collect training metrics
    grad_norms = {}
    for a in [0, 1]:
        grad_norms[a] = model_agents[a].training_step()  # update model after episode and get metrics

    # Get total vehicles for verification (returns dict with {agent_id: total_vehicles})
    total_vehicles = env.get_total_vehicles()

    # Calculate vehicle discrepancy: initial vehicles minus sum of both agents' vehicles
    total_vehicles_both_agents = total_vehicles[0] + total_vehicles[1]
    vehicle_discrepancy = abs(initial_vehicles - total_vehicles_both_agents)

    # Add training metrics to wandb
    wandb.log({
    "episode": i_episode + 1,
    # Agent 0 metrics
    "agent0/episode_reward": episode_reward[0],
    "agent0/episode_served_demand": episode_served_demand[0],
    "agent0/episode_rebalancing_cost": episode_rebalancing_cost[0],
    "agent0/episode_waiting_time": episode_waiting[0]/episode_served_demand[0] if episode_served_demand[0] > 0 else 0,
    "agent0/total_revenue": episode_total_revenue[0],
    "agent0/total_operating_cost": episode_total_operating_cost[0],
    "agent0/rejected_demand": episode_rejected_demand[0],
    "agent0/actor_grad_norm": grad_norms[0]["actor_grad_norm"],
    "agent0/critic_grad_norm": grad_norms[0]["critic_grad_norm"],
    "agent0/actor_loss": grad_norms[0]["actor_loss"],
    "agent0/critic_loss": grad_norms[0]["critic_loss"],
    # Agent 1 metrics
    "agent1/episode_reward": episode_reward[1],
    "agent1/episode_served_demand": episode_served_demand[1],
    "agent1/episode_rebalancing_cost": episode_rebalancing_cost[1],
    "agent1/episode_waiting_time": episode_waiting[1]/episode_served_demand[1] if episode_served_demand[1] > 0 else 0,
    "agent1/total_revenue": episode_total_revenue[1],
    "agent1/total_operating_cost": episode_total_operating_cost[1],
    "agent1/rejected_demand": episode_rejected_demand[1],
    "agent1/actor_grad_norm": grad_norms[1]["actor_grad_norm"],
    "agent1/critic_grad_norm": grad_norms[1]["critic_grad_norm"],
    "agent1/actor_loss": grad_norms[1]["actor_loss"],
    "agent1/critic_loss": grad_norms[1]["critic_loss"],
    # Combined metrics
    "combined/total_reward": episode_reward[0] + episode_reward[1],
    "combined/total_served_demand": episode_served_demand[0] + episode_served_demand[1],
    "combined/total_rebalancing_cost": episode_rebalancing_cost[0] + episode_rebalancing_cost[1],
    # Vehicle tracking
    "vehicles/agent0_total": total_vehicles[0],
    "vehicles/agent1_total": total_vehicles[1],
    "vehicles/combined_total": total_vehicles_both_agents,
    "vehicles/initial": initial_vehicles,
    "vehicles/discrepancy": vehicle_discrepancy
    })

    # Keep metrics for both agents
    epoch_reward_list.append(episode_reward)  # This is already a dict {0: reward0, 1: reward1}
    
    # Track total arrivals per agent
    total_arrivals = {0: env.agent_arrivals[0], 1: env.agent_arrivals[1]}
    epoch_demand_list.append(total_arrivals)

    # Calculate waiting time per agent (handle division by zero)
    epoch_waiting_list.append({
        0: episode_waiting[0]/episode_served_demand[0] if episode_served_demand[0] > 0 else 0,
        1: episode_waiting[1]/episode_served_demand[1] if episode_served_demand[1] > 0 else 0
    })
    # Calculate served rate per agent (handle division by zero)
    epoch_servedrate_list.append({
        0: episode_served_demand[0]/env.agent_arrivals[0] if env.agent_arrivals[0] > 0 else 0,
        1: episode_served_demand[1]/env.agent_arrivals[1] if env.agent_arrivals[1] > 0 else 0
    })
    epoch_rebalancing_cost.append(episode_rebalancing_cost)  # This is already a dict {0: cost0, 1: cost1}

    # Update progress bar with metrics from both agents
    total_reward = episode_reward[0] + episode_reward[1]
    epochs.set_description(
        f"Episode {i_episode+1} | "
        f"Total Reward: {total_reward:.2f} | "
        f"Agent0: R={episode_reward[0]:.2f}, AGrad={grad_norms[0]['actor_grad_norm']:.2f}, CGrad={grad_norms[0]['critic_grad_norm']:.2f}, ALoss={grad_norms[0]['actor_loss']:.2f}, CLoss={grad_norms[0]['critic_loss']:.2f} | "
        f"Agent1: R={episode_reward[1]:.2f}, AGrad={grad_norms[1]['actor_grad_norm']:.2f}, CGrad={grad_norms[1]['critic_grad_norm']:.2f}, ALoss={grad_norms[1]['actor_loss']:.2f}, CLoss={grad_norms[1]['critic_loss']:.2f}"
    )
    
    # Checkpoint best performing models (based on combined reward)
    if total_reward >= best_reward:
        for agent_id in [0, 1]:
            model_agents[agent_id].save_checkpoint(
                path=f"ckpt/{args.checkpoint_path}_agent{agent_id+1}_sample.pth")
        best_reward = total_reward
    
    # Save running checkpoints for both agents
    for agent_id in [0, 1]:
        model_agents[agent_id].save_checkpoint(
            path=f"ckpt/{args.checkpoint_path}_agent{agent_id+1}_running.pth")

# Training loop finished - save all metrics
wandb.finish()

metricPath = f"{args.directory}/train_logs/"
if not os.path.exists(metricPath):
    os.makedirs(metricPath)

# Save metrics for multi-agent setting
# epoch_reward_list, epoch_waiting_list, epoch_servedrate_list, epoch_rebalancing_cost, epoch_actor_loss, and epoch_critic_loss are lists of dicts
np.save(
    f"{args.directory}/train_logs/{city}_rewards_waiting_mode{args.mode}_{train_episodes}.npy", 
    np.array([epoch_reward_list, epoch_waiting_list, epoch_servedrate_list, epoch_demand_list, epoch_rebalancing_cost], dtype=object)
)

# Save export data with multi-agent structure
export["avail_distri"] = env.agent_acc  # Multi-agent vehicle distribution
export["demand_scaled"] = env.demand
with open(f"{args.directory}/train_logs/{city}_export_mode{args.mode}_{train_episodes}.pickle", 'wb') as f:
    pickle.dump(export, f)

print(f"\nTraining completed! Metrics saved to {metricPath}")