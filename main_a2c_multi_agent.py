import argparse
import torch
from src.envs.amod_env_multi import Scenario, AMoD
from src.algos.a2c_gnn_emil2 import A2C
from tqdm import trange
import numpy as np
from src.misc.utils import dictsum, nestdictsum
import copy, os
from src.algos.reb_flow_solver import solveRebFlow
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
env = AMoD(scenario, args.mode, beta=beta[city], jitter=args.jitter, max_wait=args.maxt, choice_price_mult=args.choice_price_mult)

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
        checkpoint_path = f"ckpt/{args.checkpoint_path}_agent_{agent_id}.pth"
        model_agents[agent_id].load_checkpoint(path=checkpoint_path)
        print(f"Loaded checkpoint for agent {agent_id} from {checkpoint_path}")
    print("Loaded models from checkpoint successfully")

#Initialize lists for logging
log = {'train_reward': [], 
        'train_served_demand': [], 
        'train_reb_cost': []}

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
epoch_value1_list = []
epoch_value2_list = []

price_history = []

# Get initial vehicles
initial_vehicles = env.get_initial_vehicles()

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
