from __future__ import print_function
import argparse
from tqdm import trange
import numpy as np
import torch
from src.envs.amod_env import Scenario, AMoD
from src.algos.a2c_gnn_emil import A2C
from src.algos.reb_flow_solver import solveRebFlow
from src.misc.utils import dictsum, nestdictsum
import json, pickle
import copy, os

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

parser = argparse.ArgumentParser(description="A2C-GNN")

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
    "--test", type=bool, default=False, help="activates test mode for agent evaluation"
)
parser.add_argument(
    "--cplexpath",
    type=str,
    default="C:/Program Files/IBM/ILOG/CPLEX_Studio201/opl/bin/x64_win64/",
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
    help="number of steps per episode (default: T=20)",
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
    "--alpha",
    type=float,
    default=0.3,
    help="entropy coefficient (default: 0.3)",
)
parser.add_argument(
    "--hidden_size",
    type=int,
    default=30,
    help="hidden size of neural networks (default: 30)",
)
parser.add_argument(
    "--checkpoint_path",
    type=str,
    default="SAC",
    help="name of checkpoint file to save/load (default: SAC)",
)
parser.add_argument(
    "--load",
    type=bool,
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
    "--q_lag",
    type=int,
    default=1,
    help="update frequency of Q target networks (default: 1)",
)
parser.add_argument(
    "--city",
    type=str,
    default="san_francisco",
    help="city to train on",
)
parser.add_argument(
    "--rew_scale",
    type=float,
    default=0.1,
    help="reward scaling factor (default: 0.1)",
)

parser.add_argument(
    "--price_version",
    type=str,
    default="GNN-origin",
    help="price network version",
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
    "--small",
    type=bool,
    default=False,
    help="whether to run the small hypothetical case (default: False)",
)

# Parser arguments
args = parser.parse_args()

# Set device
args.cuda = args.cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
print("Using device:", device)

# Initialize CUDA context if using GPU
if args.cuda:
    torch.cuda.init()
    torch.cuda.empty_cache()
    print(f"CUDA device: {torch.cuda.get_device_name()}")
    print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Set city-specific parameters
city = args.city

# Create the scenario object
scenario = Scenario(
            json_file=f"data/scenario_{city}.json",
            demand_ratio=demand_ratio[city],
            json_hr=json_hr[city],
            sd=args.seed,
            json_tstep=args.json_tstep,
            tf=args.max_steps,
            impute=args.impute,
            supply_ratio=args.supply_ratio,
        )

# Create the AMoD environment
env = AMoD(scenario, args.mode, beta=beta[city], jitter=args.jitter, max_wait=args.maxt)

# Load the model 
model = A2C(
        env=env,
        input_size=10,
        hidden_size=args.hidden_size,
        device=device,
        p_lr=args.p_lr,
        q_lr=args.q_lr,
        T = 6, 
        json_file=f"data/scenario_{city}.json",
        scale_factor = 0.01,
        price_version = args.price_version,
        mode=args.mode,
        alpha=args.alpha,
        clip=args.clip
    )

if args.load:
        print("load checkpoint")
        model.load_checkpoint(path=f"ckpt/{args.checkpoint_path}.pth")

train_episodes = args.max_episodes  # set max number of training episodes
epochs = trange(train_episodes)  # epoch iterator
best_reward = -np.inf  # set best reward
best_reward_test = -np.inf  # set best reward
model.train()  # set model in train mode

# Check metrics
epoch_demand_list = []
epoch_reward_list = []
epoch_waiting_list = []
epoch_servedrate_list = []
epoch_rebalancing_cost = []
epoch_value1_list = []
epoch_value2_list = []

price_history = []

for i_episode in epochs:
    obs = env.reset()  # initialize environment

    # Save original demand for reference
    demand_ori = nestdictsum(env.demand)
    if i_episode == train_episodes - 1:
        export = {"demand_ori":copy.deepcopy(env.demand)}
    action_rl = [0]*env.nregion
    episode_reward = 0
    episode_served_demand = 0
    episode_rebalancing_cost = 0
    episode_waiting = 0
    actions = []

    current_eps = []
    done = False
    step = 0
    while not done:
        # take matching step (Step 1 in paper)
        if env.mode == 0:
            obs, paxreward, done, info, _, _ = env.match_step_simple()
            episode_reward += paxreward

            action_rl = model.select_action(obs)

            # transform sample from Dirichlet into actual vehicle counts (i.e. (x1*x2*..*xn)*num_vehicles)
            desiredAcc = {env.region[i]: int(action_rl[i] *dictsum(env.acc,env.time+1))for i in range(len(env.region))}

            # solve minimum rebalancing distance problem (Step 3 in paper)
            rebAction = solveRebFlow(
                env,
                "scenario_san_francisco4",
                desiredAcc,
                args.cplexpath,
                args.directory, 
            )

            # Take rebalancing action in environment
            new_obs, rebreward, done, info, _, _ = env.reb_step(rebAction)
            episode_reward += rebreward
            model.rewards.append(paxreward + rebreward)

        elif env.mode == 1:
            obs, paxreward, done, info, _, _ = env.match_step_simple(action_rl)

            episode_reward += paxreward
            model.rewards.append(paxreward)
            action_rl = model.select_action(obs)  

            env.matching_update()
        elif env.mode == 2:
            obs, paxreward, done, info, _, _ = env.match_step_simple(action_rl)

            episode_reward += paxreward

            action_rl = model.select_action(obs)

            # transform sample from Dirichlet into actual vehicle counts (i.e. (x1*x2*..*xn)*num_vehicles)
            desiredAcc = {
                env.region[i]: int(
                    action_rl[i][-1] * dictsum(env.acc, env.time + 1))
                for i in range(len(env.region))
            }
            # solve minimum rebalancing distance problem (Step 3 in paper)
            rebAction = solveRebFlow(
                env,
                "scenario_san_francisco4",
                desiredAcc,
                args.cplexpath,
                args.directory, 
            )
            # Take rebalancing action in environment
            new_obs, rebreward, done, info, _, _ = env.reb_step(rebAction)
            episode_reward += rebreward
            model.rewards.append(paxreward + rebreward)                
        else:
            raise ValueError("Only mode 0, 1, and 2 are allowed")                    

        # track performance over episode
        episode_served_demand += info["served_demand"]
        episode_rebalancing_cost += info["rebalancing_cost"]
        episode_waiting += info['served_waiting']
        actions.append(action_rl)

        if done:
            break
        step += 1

    grad_norms = model.training_step()  # update model after episode and get metrics
        
    # Keep metrics
    epoch_reward_list.append(episode_reward)
    epoch_demand_list.append(env.arrivals)
    epoch_waiting_list.append(episode_waiting/episode_served_demand)
    epoch_servedrate_list.append(episode_served_demand/env.arrivals)
    epoch_rebalancing_cost.append(episode_rebalancing_cost)

    # Keep price (only needed for pricing training)
    price_history.append(actions)

    epochs.set_description(
        f"Episode {i_episode+1} | Reward: {episode_reward:.2f} | Grad Norms: Actor={grad_norms['actor_grad_norm']:.2f}, Critic={grad_norms['critic_grad_norm']:.2f} | Loss: Actor={grad_norms['actor_loss']:.2f}, Critic={grad_norms['critic_loss']:.2f}"
    )

    # Checkpoint best performing model
    if episode_reward >= best_reward:
        model.save_checkpoint(
            path=f"ckpt/{args.checkpoint_path}_sample.pth")
        best_reward = episode_reward
    model.save_checkpoint(path=f"ckpt/{args.checkpoint_path}_running.pth")
    if i_episode % 100 == 0:
        test_reward, test_served_demand, test_rebalancing_cost = model.test_agent(
            10, env, args.cplexpath, args.directory
        )
        if test_reward >= best_reward_test:
            best_reward_test = test_reward
            model.save_checkpoint(
                path=f"ckpt/{args.checkpoint_path}_test.pth")
        
# Save metrics file
metricPath = f"{args.directory}/train_logs/"
if not os.path.exists(metricPath):
    os.makedirs(metricPath)
np.save(f"{args.directory}/train_logs/{city}_rewards_waiting_mode{args.mode}_{train_episodes}.npy", np.array([epoch_reward_list,epoch_waiting_list,epoch_servedrate_list,epoch_demand_list,epoch_rebalancing_cost]))
np.save(f"{args.directory}/train_logs/{city}_price_mode{args.mode}_{train_episodes}.npy", np.array(price_history))
np.save(f"{args.directory}/train_logs/{city}_q_mode{args.mode}_{train_episodes}.npy", np.array([epoch_value1_list,epoch_value2_list]))

export["avail_distri"] = env.acc
export["demand_scaled"] = env.demand
with open(f"{args.directory}/train_logs/{city}_export_mode{args.mode}_{train_episodes}.pickle", 'wb') as f:
    pickle.dump(export, f)