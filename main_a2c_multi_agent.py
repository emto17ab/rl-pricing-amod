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

def test_agents(model_agents, test_episodes, env, cplexpath, directory, max_episodes, mode, fix_agent=2):
        epochs = range(test_episodes)  # epoch iterator
        episode_reward = []
        episode_served_demand = []
        episode_rebalancing_cost = []
        for _ in epochs:
            eps_reward = {0: 0, 1: 0}
            eps_served_demand = {0: 0, 1: 0}
            eps_rebalancing_cost = {0: 0, 1: 0}
            obs = env.reset()
            action_rl = [0]*env.nregion
            done = False
            while not done:
                if env.mode == 0:
                    # Make Match Step
                    obs, paxreward, done, info, _, _ = env.match_step_simple()

                    # Update episode reward
                    eps_reward = {a: eps_reward[a] + paxreward[a] for a in [0, 1]}

                    # Get actions
                    action_rl = {}
                    for a in [0, 1]:
                        if a == fix_agent:
                            # Fixed agent: use actual initial distribution for rebalancing
                            total_vehicles = sum(env.agent_initial_acc[a].values())
                            action_rl[a] = np.array([
                                env.agent_initial_acc[a][env.region[i]] / total_vehicles 
                                for i in range(env.nregion)
                            ])
                        else:
                            action_rl[a] = model_agents[a].select_action(obs[a], deterministic=True)

                    # Compute desired accumulation for all agents
                    desiredAcc = {}
                    for a in [0, 1]:
                        if a == fix_agent:
                            # For fixed agent, set desiredAcc to initial distribution
                            desiredAcc[a] = {env.region[i]: env.agent_initial_acc[a][env.region[i]] for i in range(env.nregion)}
                        else:
                            # For active agent, use action to determine desired distribution
                            desiredAcc[a] = {
                                env.region[i]: int(action_rl[a][i] * dictsum(env.agent_acc[a], env.time + 1))
                                for i in range(env.nregion)
                            }

                    # Compute rebalancing flows for both agents
                    rebAction = {
                        a: solveRebFlow(env, "scenario_san_francisco4", desiredAcc[a], cplexpath, directory, a, max_episodes, mode)
                        for a in [0, 1]
                    }
                    
                    _, rebreward, done, info, _, _ = env.reb_step(rebAction)
                    eps_reward = {a: eps_reward[a] + rebreward[a] for a in [0, 1]}
                    
                    
                elif env.mode == 1:
                    obs, paxreward, done, info, _, _ = env.match_step_simple(action_rl)

                    eps_reward = {a: eps_reward[a] + paxreward[a] for a in [0, 1]}

                    # Get actions
                    action_rl = {}
                    for a in [0, 1]:
                        if a == fix_agent:
                            # Fixed agent: environment handles price override to 0.5
                            action_rl[a] = np.array([0.5] * env.nregion)
                        else:
                            action_rl[a] = model_agents[a].select_action(obs[a], deterministic=True)

                    # Matching update (global step)
                    env.matching_update()
                
                elif env.mode == 2:
                    # --- Matching step ---
                    obs, paxreward, done, info, _, _ = env.match_step_simple(action_rl)
                    
                    eps_reward = {a: eps_reward[a] + paxreward[a] for a in [0, 1]}

                    # Get actions
                    action_rl = {}
                    for a in [0, 1]:
                        if a == fix_agent:
                            # Fixed agent: environment handles price override to 0.5
                            # Mode 2 action shape: [nregion, 2] where [:, 0] = price, [:, 1] = reb
                            total_vehicles = sum(env.agent_initial_acc[a].values())
                            reb_action = np.array([
                                env.agent_initial_acc[a][env.region[i]] / total_vehicles 
                                for i in range(env.nregion)
                            ])
                            action_rl[a] = np.column_stack([
                                np.array([0.5] * env.nregion),  # Price (will be overridden by env)
                                reb_action  # Rebalancing: actual initial distribution
                            ])
                        else:
                            action_rl[a] = model_agents[a].select_action(obs[a], deterministic=True)

                    # --- Desired Acc computation ---
                    # Compute desired accumulation for all agents
                    desiredAcc = {}
                    for a in [0, 1]:
                        if a == fix_agent:
                            # For fixed agent, set desiredAcc to initial distribution
                            desiredAcc[a] = {env.region[i]: env.agent_initial_acc[a][env.region[i]] for i in range(env.nregion)}
                        else:
                            # For active agent, use action to determine desired distribution
                            desiredAcc[a] = {
                                env.region[i]: int(action_rl[a][i][-1] * dictsum(env.agent_acc[a], env.time + 1))
                                for i in range(env.nregion)
                            }
                    
                    # --- Rebalancing step ---
                    # Compute rebalancing flows for both agents
                    rebAction = {
                        a: solveRebFlow(env, "scenario_san_francisco4", desiredAcc[a], cplexpath, directory, a, max_episodes, mode)
                        for a in [0, 1]
                    }
                
                    _, rebreward, done, info, _, _ = env.reb_step(rebAction)
                
                    eps_reward = {a: eps_reward[a] + rebreward[a] for a in [0, 1]}
                else:
                    raise ValueError("Only mode 0, 1, and 2 are allowed")  
                   
                for a in [0, 1]:
                    eps_served_demand[a] += info[a]["served_demand"]
                    eps_rebalancing_cost[a] += info[a]["rebalancing_cost"]
                
            episode_reward.append(eps_reward)
            episode_served_demand.append(eps_served_demand)
            episode_rebalancing_cost.append(eps_rebalancing_cost)

        # Calculate means for each agent
        mean_reward = {
            0: np.mean([ep[0] for ep in episode_reward]),
            1: np.mean([ep[1] for ep in episode_reward])
        }
        mean_served_demand = {
            0: np.mean([ep[0] for ep in episode_served_demand]),
            1: np.mean([ep[1] for ep in episode_served_demand])
        }
        mean_rebalancing_cost = {
            0: np.mean([ep[0] for ep in episode_rebalancing_cost]),
            1: np.mean([ep[1] for ep in episode_rebalancing_cost])
        }
        
        return (
            mean_reward,
            mean_served_demand,
            mean_rebalancing_cost,
        )

# Define calibrated simulation parameters
demand_ratio = {'san_francisco': 2, 'washington_dc': 4.2, 'chicago': 1.8, 'nyc_man_north': 1.8, 'nyc_man_middle': 1.8,
                'nyc_man_south': 1.8, 'nyc_brooklyn': 9, 'nyc_manhattan': 0.05, 'porto': 4, 'rome': 1.8, 'shenzhen_baoan': 2.5,
                'shenzhen_downtown_west': 2.5, 'shenzhen_downtown_east': 3, 'shenzhen_north': 3
               }
json_hr = {'san_francisco':19, 'washington_dc': 19, 'chicago': 19, 'nyc_man_north': 19, 'nyc_man_middle': 19,
           'nyc_man_south': 19, 'nyc_brooklyn': 19, 'nyc_manhattan': 19, 'porto': 8, 'rome': 8, 'shenzhen_baoan': 8,
           'shenzhen_downtown_west': 8, 'shenzhen_downtown_east': 8, 'shenzhen_north': 8
          }
beta = {'san_francisco': 0.2, 'washington_dc': 0.5, 'chicago': 0.5, 'nyc_man_north': 0.5, 'nyc_man_middle': 0.5,
                'nyc_man_south': 0.5, 'nyc_brooklyn':0.5, 'nyc_manhattan': 0.3, 'porto': 0.1, 'rome': 0.1, 'shenzhen_baoan': 0.5,
                'shenzhen_downtown_west': 0.5, 'shenzhen_downtown_east': 0.5, 'shenzhen_north': 0.5}

test_tstep = {'san_francisco': 3, 'nyc_brooklyn': 4, 'shenzhen_downtown_west': 3, 'nyc_manhattan': 3, 'nyc_man_middle': 3, 'nyc_man_south': 3, 'nyc_man_north': 3, 'washington_dc':3, 'chicago':3}

parser = argparse.ArgumentParser(description="A2C-GNN")

# Simulator parameters
parser.add_argument(
    "--seed", type=int, default=10, metavar="S", help="random seed (default: 10)"
)

parser.add_argument(
    "--model_type",
    type=str,
    default="running", 
    help="Defines the type of model (default: running)",
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
    "--actor_clip",
    type=float,
    default=500,
    help="clip value for actor gradient clipping (default: 500)",
)

parser.add_argument(
    "--critic_clip",
    type=float,
    default=500,
    help="clip value for critic gradient clipping (default: 500)",
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

parser.add_argument(
    "--fix_agent",
    type=int,
    default=2,
    choices=[0, 1, 2],
    help="Fix agent behavior for testing: 0=fix agent 0, 1=fix agent 1, 2=no fixing (default: 2)",
)

parser.add_argument(
    "--use_od_prices",
    action="store_true",
    default=False,
    help="Use OD price matrices instead of aggregated prices per region (default: False)",
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
    name=args.checkpoint_path,
    config=args,
)

# Set city
city = args.city
if not args.test:
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
    env = AMoD(scenario, args.mode, beta=beta[city], jitter=args.jitter, max_wait=args.maxt, choice_price_mult=args.choice_price_mult, seed = args.seed, fix_agent=args.fix_agent)
    
    # Print fixed agent information
    if args.fix_agent == 0:
        print("=" * 80)
        print("FIXED AGENT MODE: Agent 0 is FIXED")
        print("- Agent 0 uses BASE PRICES (scalar=0.5, no learning)")
        print("- Agent 0 is included in choice model and can receive demand")
        print("- Agent 0 vehicles reset to initial distribution each step")
        print("- Agent 1 is LEARNING (adjusts prices dynamically)")
        print("=" * 80)
    elif args.fix_agent == 1:
        print("=" * 80)
        print("FIXED AGENT MODE: Agent 1 is FIXED")
        print("- Agent 1 uses BASE PRICES (scalar=0.5, no learning)")
        print("- Agent 1 is included in choice model and can receive demand")
        print("- Agent 1 vehicles reset to initial distribution each step")
        print("- Agent 0 is LEARNING (adjusts prices dynamically)")
        print("=" * 80)
    else:
        print("=" * 80)
        print("NORMAL MODE: Both agents are active and learning")
        print("- Demand is split via choice model")
        print("- Both agents learn simultaneously")
        print("=" * 80)

    # Calculate input size based on price type
    if args.use_od_prices:
        # OD price matrices: T (future) + 3 (current_avb, queue, demand) + 2*nregion (own and competitor OD prices)
        input_size = args.look_ahead + 3 + 2 * env.nregion
    else:
        # Aggregated prices: T (future) + 3 (current_avb, queue, demand) + 2 (own and competitor aggregated prices)
        input_size = args.look_ahead + 5
    
    model_agents = {
            a: A2C(
                env=env,
                input_size=input_size,
                hidden_size=args.hidden_size,
                device=device,
                p_lr=args.p_lr,
                q_lr=args.q_lr,
                T = args.look_ahead,
                scale_factor = args.scale_factor,
                json_file=f"data/scenario_{city}.json",
                mode=args.mode,
                actor_clip=args.actor_clip,
                critic_clip=args.critic_clip,
                gamma=args.gamma,
                agent_id = a,
                use_od_prices = args.use_od_prices
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
        actions_price = {0: [], 1: []}  # Track price scalars during episode
        
        # Track concentration parameters during episode (different structures per mode)
        if env.mode == 0:
            # Mode 0: Only Dirichlet for rebalancing
            actions_concentration_dirichlet = {0: [], 1: []}
        elif env.mode == 1:
            # Mode 1: Only Beta (alpha, beta) for pricing
            actions_concentration_alpha = {0: [], 1: []}
            actions_concentration_beta = {0: [], 1: []}
        else:  # mode 2
            # Mode 2: Beta (alpha, beta) for pricing + Dirichlet for rebalancing
            actions_concentration_alpha = {0: [], 1: []}
            actions_concentration_beta = {0: [], 1: []}
            actions_concentration_dirichlet = {0: [], 1: []}

        done = False
        step = 0

        while not done:
            if env.mode == 0:
                # Make Match Step
                obs, paxreward, done, info, _, _ = env.match_step_simple()

                # Update episode reward
                episode_reward = {a: episode_reward[a] + paxreward[a] for a in [0, 1]}

                # Get actions and concentrations
                action_rl = {}
                concentrations = {}
                for a in [0, 1]:
                    if a == args.fix_agent:
                        # Fixed agent: use actual initial distribution for rebalancing
                        # Convert initial vehicle counts to proportions
                        total_vehicles = sum(env.agent_initial_acc[a].values())
                        action_rl[a] = np.array([
                            env.agent_initial_acc[a][env.region[i]] / total_vehicles 
                            for i in range(env.nregion)
                        ])
                        concentrations[a] = np.zeros((env.nregion, 1))  # Dummy for tracking
                    else:
                        action_conc = model_agents[a].select_action(obs[a], return_concentration=True)
                        action_rl[a] = action_conc[0]
                        concentrations[a] = action_conc[1]
                
                # Track concentration (mode 0: Dirichlet concentration for rebalancing)
                for a in [0, 1]:
                    if a != args.fix_agent:
                        actions_concentration_dirichlet[a].append(np.mean(concentrations[a]))

                # Determine which agents are active (not fixed)
                # Compute desired accumulation for all agents
                desiredAcc = {}
                for a in [0, 1]:
                    if a == args.fix_agent:
                        # For fixed agent, distribute vehicles uniformly across all regions
                        current_total = dictsum(env.agent_acc[a], env.time + 1)
                        base_per_region = current_total // env.nregion
                        remainder = current_total % env.nregion
                        # Distribute uniformly with remainder going to first regions
                        desiredAcc[a] = {
                            env.region[i]: base_per_region + (1 if i < remainder else 0)
                            for i in range(env.nregion)
                        }
                    else:
                        # For active agent, use action to determine desired distribution
                        desiredAcc[a] = {
                            env.region[i]: int(action_rl[a][i] * dictsum(env.agent_acc[a], env.time + 1))
                            for i in range(env.nregion)
                        }

                # Compute rebalancing flows for both agents
                rebAction = {
                    a: solveRebFlow(env, "scenario_san_francisco4", desiredAcc[a], args.cplexpath, args.directory, a, args.max_episodes, args.mode)
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

                # Get actions and concentrations
                action_rl = {}
                concentrations = {}
                for a in [0, 1]:
                    if a == args.fix_agent:
                        # Fixed agent: environment handles price override to 0.5
                        # Just provide any valid pricing action (will be ignored)
                        action_rl[a] = np.array([0.5] * env.nregion)
                        concentrations[a] = np.zeros((env.nregion, 2))  # Dummy for tracking
                    else:
                        action_conc = model_agents[a].select_action(obs[a], return_concentration=True)
                        action_rl[a] = action_conc[0]
                        concentrations[a] = action_conc[1]
                
                # Track prices during episode (mode 1: action_rl is price scalar)
                for a in [0, 1]:
                    if a == args.fix_agent:
                        # Fixed agent always uses 0.5 scalar
                        actions_price[a].append(1.0)  # 2 * 0.5 = 1.0 (base price)
                    else:
                        actions_price[a].append(np.mean(2 * np.array(action_rl[a])))
                
                # Track concentration (mode 1: Beta distribution - alpha and beta)
                for a in [0, 1]:
                    if a != args.fix_agent:
                        actions_concentration_alpha[a].append(np.mean(concentrations[a][:, 0]))
                        actions_concentration_beta[a].append(np.mean(concentrations[a][:, 1]))

                # Matching update (global step)
                env.matching_update()
            
            elif env.mode == 2:
                # --- Matching step ---
                obs, paxreward, done, info, _, _ = env.match_step_simple(action_rl)
                
                episode_reward = {a: episode_reward[a] + paxreward[a] for a in [0, 1]}

                # Get actions and concentrations
                action_rl = {}
                concentrations = {}
                for a in [0, 1]:
                    if a == args.fix_agent:
                        # Fixed agent: environment handles price override to 0.5
                        # Mode 2 action shape: [nregion, 2] where [:, 0] = price scalar, [:, 1] = reb action
                        total_vehicles = sum(env.agent_initial_acc[a].values())
                        reb_action = np.array([
                            env.agent_initial_acc[a][env.region[i]] / total_vehicles 
                            for i in range(env.nregion)
                        ])
                        action_rl[a] = np.column_stack([
                            np.array([0.5] * env.nregion),  # Price (will be overridden to 0.5 by env)
                            reb_action  # Rebalancing: actual initial distribution
                        ])
                        concentrations[a] = np.zeros((env.nregion, 3))  # Dummy for tracking
                    else:
                        action_conc = model_agents[a].select_action(obs[a], return_concentration=True)
                        action_rl[a] = action_conc[0]
                        concentrations[a] = action_conc[1]
                
                # Track prices during episode (mode 2: action_rl[:,0] is price scalar)
                for a in [0, 1]:
                    if a == args.fix_agent:
                        # Fixed agent always uses 0.5 scalar
                        actions_price[a].append(1.0)  # 2 * 0.5 = 1.0 (base price)
                    else:
                        actions_price[a].append(np.mean(2 * np.array(action_rl[a])[:, 0]))
                
                # Track concentration (mode 2: Beta + Dirichlet)
                for a in [0, 1]:
                    if a != args.fix_agent:
                        actions_concentration_alpha[a].append(np.mean(concentrations[a][:, 0]))
                        actions_concentration_beta[a].append(np.mean(concentrations[a][:, 1]))
                        actions_concentration_dirichlet[a].append(np.mean(concentrations[a][:, 2]))
                    
                # --- Desired Acc computation ---
                # Compute desired accumulation for all agents
                desiredAcc = {}
                for a in [0, 1]:
                    if a == args.fix_agent:
                        # For fixed agent, distribute vehicles uniformly across all regions
                        current_total = dictsum(env.agent_acc[a], env.time + 1)
                        base_per_region = current_total // env.nregion
                        remainder = current_total % env.nregion
                        # Distribute uniformly with remainder going to first regions
                        desiredAcc[a] = {
                            env.region[i]: base_per_region + (1 if i < remainder else 0)
                            for i in range(env.nregion)
                        }
                    else:
                        # For active agent, use action to determine desired distribution
                        desiredAcc[a] = {
                            env.region[i]: int(action_rl[a][i][-1] * dictsum(env.agent_acc[a], env.time + 1))
                            for i in range(env.nregion)
                        }
                
                # --- Rebalancing step ---
                # Compute rebalancing flows for both agents
                rebAction = {
                    a: solveRebFlow(env, "scenario_san_francisco4", desiredAcc[a], args.cplexpath, args.directory, a, args.max_episodes, args.mode)
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
            if a == args.fix_agent:
                # Fixed agent: skip learning, return dummy metrics
                grad_norms[a] = {
                    "actor_grad_norm": 0.0,
                    "critic_grad_norm": 0.0,
                    "actor_loss": 0.0,
                    "critic_loss": 0.0
                }
                # Clear the fixed agent's buffers without updating
                model_agents[a].rewards = []
                model_agents[a].saved_actions = []
                model_agents[a].saved_values = []
                model_agents[a].saved_logprobs = []
            else:
                grad_norms[a] = model_agents[a].training_step()  # update model after episode and get metrics

        # Get total vehicles for verification (returns dict with {agent_id: total_vehicles})
        total_vehicles = env.get_total_vehicles()

        # Calculate vehicle discrepancy: initial vehicles minus sum of both agents' vehicles
        total_vehicles_both_agents = total_vehicles[0] + total_vehicles[1]
        vehicle_discrepancy = abs(initial_vehicles - total_vehicles_both_agents)

        # Calculate mean price scalar per agent (for modes 1 and 2)
        mean_price_scalar = {0: 0, 1: 0}
        if env.mode != 0:
            for a in [0, 1]:
                mean_price_scalar[a] = np.mean(actions_price[a]) if len(actions_price[a]) > 0 else 0

        # Calculate mean concentration parameters per agent (mode-specific)
        if env.mode == 0:
            mean_concentration_dirichlet = {0: 0, 1: 0}
            for a in [0, 1]:
                mean_concentration_dirichlet[a] = np.mean(actions_concentration_dirichlet[a]) if len(actions_concentration_dirichlet[a]) > 0 else 0
        elif env.mode == 1:
            mean_concentration_alpha = {0: 0, 1: 0}
            mean_concentration_beta = {0: 0, 1: 0}
            for a in [0, 1]:
                mean_concentration_alpha[a] = np.mean(actions_concentration_alpha[a]) if len(actions_concentration_alpha[a]) > 0 else 0
                mean_concentration_beta[a] = np.mean(actions_concentration_beta[a]) if len(actions_concentration_beta[a]) > 0 else 0
        else:  # mode 2
            mean_concentration_alpha = {0: 0, 1: 0}
            mean_concentration_beta = {0: 0, 1: 0}
            mean_concentration_dirichlet = {0: 0, 1: 0}
            for a in [0, 1]:
                mean_concentration_alpha[a] = np.mean(actions_concentration_alpha[a]) if len(actions_concentration_alpha[a]) > 0 else 0
                mean_concentration_beta[a] = np.mean(actions_concentration_beta[a]) if len(actions_concentration_beta[a]) > 0 else 0
                mean_concentration_dirichlet[a] = np.mean(actions_concentration_dirichlet[a]) if len(actions_concentration_dirichlet[a]) > 0 else 0

        # Add training metrics to wandb
        log_dict = {
        "episode": i_episode + 1,
        # Agent 0 metrics
        "agent0/episode_reward": episode_reward[0],
        "agent0/episode_served_demand": episode_served_demand[0],
        "agent0/episode_unserved_demand": episode_unserved_demand[0],
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
        "agent1/episode_unserved_demand": episode_unserved_demand[1],
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
        "combined/total_unserved_demand": episode_unserved_demand[0] + episode_unserved_demand[1],
        "combined/total_rebalancing_cost": episode_rebalancing_cost[0] + episode_rebalancing_cost[1],
        # Vehicle tracking
        "vehicles/agent0_total": total_vehicles[0],
        "vehicles/agent1_total": total_vehicles[1],
        "vehicles/combined_total": total_vehicles_both_agents,
        "vehicles/initial": initial_vehicles,
        "vehicles/discrepancy": vehicle_discrepancy
        }
        
        # Add price scalar metrics only for modes 1 and 2
        if env.mode != 0:
            log_dict["agent0/mean_price_scalar"] = mean_price_scalar[0]
            log_dict["agent1/mean_price_scalar"] = mean_price_scalar[1]
        
        # Add concentration metrics (mode-specific)
        if env.mode == 0:
            log_dict["agent0/mean_concentration_dirichlet"] = mean_concentration_dirichlet[0]
            log_dict["agent1/mean_concentration_dirichlet"] = mean_concentration_dirichlet[1]
        elif env.mode == 1:
            log_dict["agent0/mean_concentration_alpha"] = mean_concentration_alpha[0]
            log_dict["agent0/mean_concentration_beta"] = mean_concentration_beta[0]
            log_dict["agent1/mean_concentration_alpha"] = mean_concentration_alpha[1]
            log_dict["agent1/mean_concentration_beta"] = mean_concentration_beta[1]
        else:  # mode 2
            log_dict["agent0/mean_concentration_alpha"] = mean_concentration_alpha[0]
            log_dict["agent0/mean_concentration_beta"] = mean_concentration_beta[0]
            log_dict["agent0/mean_concentration_dirichlet"] = mean_concentration_dirichlet[0]
            log_dict["agent1/mean_concentration_alpha"] = mean_concentration_alpha[1]
            log_dict["agent1/mean_concentration_beta"] = mean_concentration_beta[1]
            log_dict["agent1/mean_concentration_dirichlet"] = mean_concentration_dirichlet[1]
        
        wandb.log(log_dict)

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
        
        if i_episode % 100 == 0:
            for agent_id in [0, 1]:
                model_agents[agent_id].eval()
            test_reward, test_served_demand, test_rebalancing_cost = test_agents(
                    model_agents=model_agents, test_episodes=10, env=env, cplexpath=args.cplexpath, directory=args.directory, max_episodes=args.max_episodes, mode=args.mode)
            for agent_id in [0, 1]:
                model_agents[agent_id].train()

            test_reward = test_reward[0] + test_reward[1]
            if test_reward >= best_reward_test:
                        best_reward_test = test_reward
                        model_agents[0].save_checkpoint(
                            path=f"ckpt/{args.checkpoint_path}_agent1_test.pth")
                        model_agents[1].save_checkpoint(
                            path=f"ckpt/{args.checkpoint_path}_agent2_test.pth")

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

else:
    scenario = Scenario(
                json_file=f"data/scenario_{city}.json",
                demand_ratio=demand_ratio[city],
                json_hr=json_hr[city],
                sd=args.seed,
                json_tstep=args.json_tstep,
                tf=args.max_steps,
                impute=args.impute,
                supply_ratio=args.supply_ratio)

    env = AMoD(scenario, args.mode, beta=beta[city], jitter=args.jitter, max_wait=args.maxt, choice_price_mult=args.choice_price_mult, seed = args.seed, fix_agent=args.fix_agent)
    
    # Print fixed agent information
    if args.fix_agent == 0:
        print("=" * 80)
        print("TEST MODE - FIXED AGENT: Agent 0 is FIXED")
        print("- Agent 0 uses BASE PRICES (scalar=0.5, no learning)")
        print("- Agent 0 is included in choice model and can receive demand")
        print("- Agent 0 vehicles reset to initial distribution each step")
        print("- Agent 1 uses learned policy")
        print("=" * 80)
    elif args.fix_agent == 1:
        print("=" * 80)
        print("TEST MODE - FIXED AGENT: Agent 1 is FIXED")
        print("- Agent 1 uses BASE PRICES (scalar=0.5, no learning)")
        print("- Agent 1 is included in choice model and can receive demand")
        print("- Agent 1 vehicles reset to initial distribution each step")
        print("- Agent 0 uses learned policy")
        print("=" * 80)
    else:
        print("=" * 80)
        print("TEST MODE - NORMAL: Both agents are active")
        print("- Demand is split via choice model")
        print("=" * 80)

    # Calculate input size based on price type
    if args.use_od_prices:
        # OD price matrices: T (future) + 3 (current_avb, queue, demand) + 2*nregion (own and competitor OD prices)
        input_size = args.look_ahead + 3 + 2 * env.nregion
    else:
        # Aggregated prices: T (future) + 3 (current_avb, queue, demand) + 2 (own and competitor aggregated prices)
        input_size = args.look_ahead + 5
    
    model_agents = {
            a: A2C(
                env=env,
                input_size=input_size,
                hidden_size=args.hidden_size,
                device=device,
                p_lr=args.p_lr,
                q_lr=args.q_lr,
                T = args.look_ahead,
                scale_factor = args.scale_factor,
                json_file=f"data/scenario_{city}.json",
                mode=args.mode,
                actor_clip=args.actor_clip,
                critic_clip=args.critic_clip,
                gamma=args.gamma,
                agent_id = a,
                use_od_prices = args.use_od_prices
            )
            for a in [0, 1]
        }

    print("load models")
    for agent_id in [0, 1]:
        checkpoint_path = f"ckpt/{args.checkpoint_path}_agent{agent_id+1}_{args.model_type}.pth"
        model_agents[agent_id].load_checkpoint(path=checkpoint_path)
        print(f"Loaded checkpoint for agent {agent_id} from {checkpoint_path}")
    print("Loaded models from checkpoint successfully")

    best_reward = -np.inf  # set best reward
    best_reward_test = -np.inf  # set best reward
    for agent_id in [0, 1]:
        model_agents[agent_id].eval()
    
    # Initialize lists for logging
    log = {"test_reward": [], "test_served_demand": [], "test_reb_cost": []}
    
    epoch_reward_list = []
    epoch_demand_list = []
    epoch_rebalancing_cost = []
    epoch_waiting_list = []
    epoch_queue_length_list = []
    epoch_arrivals_list = []
    epoch_rebalancing_list = []
    epoch_price_mean_list = []
    
    # Initialize concentration tracking lists based on mode
    if env.mode == 0:
        epoch_concentration_dirichlet_list = []
    elif env.mode == 1:
        epoch_concentration_alpha_list = []
        epoch_concentration_beta_list = []
    else:  # mode 2
        epoch_concentration_alpha_list = []
        epoch_concentration_beta_list = []
        epoch_concentration_dirichlet_list = []

    # Storage for trip data from last episode
    trip_data_last_episode = []
    
    for episode in range(10):
        eps_reward = {0: 0, 1: 0}
        eps_demand = {0: 0, 1: 0}
        eps_rebalancing_cost = {0: 0, 1: 0}
        eps_waiting = {0: 0, 1: 0}
        eps_queue_length = {0: 0, 1: 0}
        eps_arrivals = {0: 0, 1: 0}
        eps_rebalancing = {0: 0, 1: 0}
        actions_price = {0: [], 1: []}
        
        # Initialize concentration tracking for episode based on mode
        if env.mode == 0:
            actions_concentration_dirichlet = {0: [], 1: []}
        elif env.mode == 1:
            actions_concentration_alpha = {0: [], 1: []}
            actions_concentration_beta = {0: [], 1: []}
        else:  # mode 2
            actions_concentration_alpha = {0: [], 1: []}
            actions_concentration_beta = {0: [], 1: []}
            actions_concentration_dirichlet = {0: [], 1: []}
        
        obs = env.reset()

        action_rl = [0]*env.nregion        
        done = False

        while not done:
            if env.mode == 0:
                # Make Match Step
                obs, paxreward, done, info, _, _ = env.match_step_simple()

                # Update episode reward
                eps_reward = {a: eps_reward[a] + paxreward[a] for a in [0, 1]}

                # Get actions and concentrations
                action_rl = {}
                concentrations = {}
                for a in [0, 1]:
                    if a == args.fix_agent:
                        # Fixed agent: use actual initial distribution for rebalancing
                        total_vehicles = sum(env.agent_initial_acc[a].values())
                        action_rl[a] = np.array([
                            env.agent_initial_acc[a][env.region[i]] / total_vehicles 
                            for i in range(env.nregion)
                        ])
                        concentrations[a] = np.zeros((env.nregion, 1))
                    else:
                        action_conc = model_agents[a].select_action(obs[a], deterministic=True, return_concentration=True)
                        action_rl[a] = action_conc[0]
                        concentrations[a] = action_conc[1]
                
                # Track concentration (mode 0: Dirichlet for rebalancing)
                for a in [0, 1]:
                    if a != args.fix_agent:
                        actions_concentration_dirichlet[a].append(np.mean(concentrations[a]))

                # Compute desired accumulation for all agents
                desiredAcc = {}
                for a in [0, 1]:
                    if a == args.fix_agent:
                        # For fixed agent, set desiredAcc to initial distribution
                        desiredAcc[a] = {env.region[i]: env.agent_initial_acc[a][env.region[i]] for i in range(env.nregion)}
                    else:
                        # For active agent, use action to determine desired distribution
                        desiredAcc[a] = {
                            env.region[i]: int(action_rl[a][i] * dictsum(env.agent_acc[a], env.time + 1))
                            for i in range(env.nregion)
                        }

                # Compute rebalancing flows for both agents
                rebAction = {
                    a: solveRebFlow(env, "scenario_san_francisco4", desiredAcc[a], args.cplexpath, args.directory, a, args.max_episodes, args.mode)
                    for a in [0, 1]
                }
                
                _, rebreward, done, info, _, _ = env.reb_step(rebAction)
                eps_reward = {a: eps_reward[a] + rebreward[a] for a in [0, 1]}
                
                
            elif env.mode == 1:
                obs, paxreward, done, info, _, _ = env.match_step_simple(action_rl)

                eps_reward = {a: eps_reward[a] + paxreward[a] for a in [0, 1]}

                # Get actions and concentrations
                action_rl = {}
                concentrations = {}
                for a in [0, 1]:
                    if a == args.fix_agent:
                        # Fixed agent: environment handles price override to 0.5
                        action_rl[a] = np.array([0.5] * env.nregion)
                        concentrations[a] = np.zeros((env.nregion, 2))
                    else:
                        action_conc = model_agents[a].select_action(obs[a], deterministic=True, return_concentration=True)
                        action_rl[a] = action_conc[0]
                        concentrations[a] = action_conc[1]
                
                # Track prices during episode (mode 1: action_rl is price scalar)
                for a in [0, 1]:
                    if a == args.fix_agent:
                        # Fixed agent always uses 0.5 scalar
                        actions_price[a].append(1.0)  # 2 * 0.5 = 1.0 (base price)
                    else:
                        actions_price[a].append(np.mean(2 * np.array(action_rl[a])))
                
                # Track concentration (mode 1: Beta distribution - alpha and beta)
                for a in [0, 1]:
                    if a != args.fix_agent:
                        actions_concentration_alpha[a].append(np.mean(concentrations[a][:, 0]))
                        actions_concentration_beta[a].append(np.mean(concentrations[a][:, 1]))

                # Matching update (global step)
                env.matching_update()
            
            elif env.mode == 2:
                # --- Matching step ---
                obs, paxreward, done, info, _, _ = env.match_step_simple(action_rl)
                
                eps_reward = {a: eps_reward[a] + paxreward[a] for a in [0, 1]}

                # Get actions and concentrations
                action_rl = {}
                concentrations = {}
                for a in [0, 1]:
                    if a == args.fix_agent:
                        # Fixed agent: environment handles price override to 0.5
                        # Mode 2 action shape: [nregion, 2] where [:, 0] = price scalar, [:, 1] = reb action
                        total_vehicles = sum(env.agent_initial_acc[a].values())
                        reb_action = np.array([
                            env.agent_initial_acc[a][env.region[i]] / total_vehicles 
                            for i in range(env.nregion)
                        ])
                        action_rl[a] = np.column_stack([
                            np.array([0.5] * env.nregion),  # Price (will be overridden by env)
                            reb_action  # Rebalancing: actual initial distribution
                        ])
                        concentrations[a] = np.zeros((env.nregion, 3))
                    else:
                        action_conc = model_agents[a].select_action(obs[a], deterministic=True, return_concentration=True)
                        action_rl[a] = action_conc[0]
                        concentrations[a] = action_conc[1]
                
                # Track prices during episode (mode 2: action_rl[:,0] is price scalar)
                for a in [0, 1]:
                    if a == args.fix_agent:
                        # Fixed agent always uses 0.5 scalar
                        actions_price[a].append(1.0)  # 2 * 0.5 = 1.0 (base price)
                    else:
                        actions_price[a].append(np.mean(2 * np.array(action_rl[a])[:, 0]))
                
                # Track concentration (mode 2: Beta + Dirichlet)
                for a in [0, 1]:
                    if a != args.fix_agent:
                        actions_concentration_alpha[a].append(np.mean(concentrations[a][:, 0]))
                        actions_concentration_beta[a].append(np.mean(concentrations[a][:, 1]))
                        actions_concentration_dirichlet[a].append(np.mean(concentrations[a][:, 2]))

                # --- Desired Acc computation ---
                # Compute desired accumulation for all agents
                desiredAcc = {}
                for a in [0, 1]:
                    if a == args.fix_agent:
                        # For fixed agent, set desiredAcc to initial distribution
                        desiredAcc[a] = {env.region[i]: env.agent_initial_acc[a][env.region[i]] for i in range(env.nregion)}
                    else:
                        # For active agent, use action to determine desired distribution
                        desiredAcc[a] = {
                            env.region[i]: int(action_rl[a][i][-1] * dictsum(env.agent_acc[a], env.time + 1))
                            for i in range(env.nregion)
                        }
                
                # --- Rebalancing step ---
                # Compute rebalancing flows for both agents
                rebAction = {
                    a: solveRebFlow(env, "scenario_san_francisco4", desiredAcc[a], args.cplexpath, args.directory, a, args.max_episodes, args.mode)
                    for a in [0, 1]
                }
            
                _, rebreward, done, info, _, _ = env.reb_step(rebAction)
            
                eps_reward = {a: eps_reward[a] + rebreward[a] for a in [0, 1]}
            else:
                raise ValueError("Only mode 0, 1, and 2 are allowed")
        
            for a in [0, 1]:
                eps_demand[a] += info[a]["served_demand"]
                eps_rebalancing_cost[a] += info[a]["rebalancing_cost"]
                eps_waiting[a] += info[a]["served_waiting"]/eps_demand[a] if eps_demand[a] > 0 else 0
        
        # After episode ends, capture episode-level metrics
        # Queue length: mean queue length across all regions for each agent (computed at episode end)
        for a in [0, 1]:
            eps_queue_length[a] = np.mean([len(env.agent_queue[a][i]) for i in env.agent_queue[a].keys()]) if len(env.agent_queue[a].keys()) > 0 else 0
            reb_od = 0
            for (o, d), flow in env.agent_rebFlow[a].items():
                reb_od += sum(flow.values())
            eps_rebalancing[a] = reb_od
        
        # Arrivals: total arrivals for the episode (tracked by environment)
        eps_arrivals = {0: env.agent_arrivals[0], 1: env.agent_arrivals[1]}
        
        # Price mean: average price across all steps for each agent (mode 1 and 2 only)
        eps_price_mean = {0: 0, 1: 0}
        if env.mode != 0:
            for a in [0, 1]:
                eps_price_mean[a] = np.mean(actions_price[a]) if len(actions_price[a]) > 0 else 0
        
        # Concentration mean: average concentration across all steps for each agent (mode-specific)
        if env.mode == 0:
            eps_concentration_dirichlet = {0: 0, 1: 0}
            for a in [0, 1]:
                eps_concentration_dirichlet[a] = np.mean(actions_concentration_dirichlet[a]) if len(actions_concentration_dirichlet[a]) > 0 else 0
        elif env.mode == 1:
            eps_concentration_alpha = {0: 0, 1: 0}
            eps_concentration_beta = {0: 0, 1: 0}
            for a in [0, 1]:
                eps_concentration_alpha[a] = np.mean(actions_concentration_alpha[a]) if len(actions_concentration_alpha[a]) > 0 else 0
                eps_concentration_beta[a] = np.mean(actions_concentration_beta[a]) if len(actions_concentration_beta[a]) > 0 else 0
        else:  # mode 2
            eps_concentration_alpha = {0: 0, 1: 0}
            eps_concentration_beta = {0: 0, 1: 0}
            eps_concentration_dirichlet = {0: 0, 1: 0}
            for a in [0, 1]:
                eps_concentration_alpha[a] = np.mean(actions_concentration_alpha[a]) if len(actions_concentration_alpha[a]) > 0 else 0
                eps_concentration_beta[a] = np.mean(actions_concentration_beta[a]) if len(actions_concentration_beta[a]) > 0 else 0
                eps_concentration_dirichlet[a] = np.mean(actions_concentration_dirichlet[a]) if len(actions_concentration_dirichlet[a]) > 0 else 0
        
        # Append episode results to epoch lists
        epoch_reward_list.append(eps_reward) # Done
        epoch_demand_list.append(eps_demand) # Done
        epoch_rebalancing_cost.append(eps_rebalancing_cost) # Done
        epoch_waiting_list.append(eps_waiting) # Done
        epoch_queue_length_list.append(eps_queue_length) # Done
        epoch_arrivals_list.append(eps_arrivals) # Done
        epoch_rebalancing_list.append(eps_rebalancing) # Done
        epoch_price_mean_list.append(eps_price_mean) # Done
        
        # Append concentration results (mode-specific)
        if env.mode == 0:
            epoch_concentration_dirichlet_list.append(eps_concentration_dirichlet)
        elif env.mode == 1:
            epoch_concentration_alpha_list.append(eps_concentration_alpha)
            epoch_concentration_beta_list.append(eps_concentration_beta)
        else:  # mode 2
            epoch_concentration_alpha_list.append(eps_concentration_alpha)
            epoch_concentration_beta_list.append(eps_concentration_beta)
            epoch_concentration_dirichlet_list.append(eps_concentration_dirichlet)
        
        # Capture trip data from the last episode
        if episode == 9:  # Last episode (0-indexed, so episode 9 is the 10th)
            trip_data_last_episode = env.get_trip_assignments()

    # After all episodes, compute statistics across episodes for multi-agent case
    
    # Save trip data from last episode to CSV
    if trip_data_last_episode:
        import pandas as pd
        df_trips = pd.DataFrame(trip_data_last_episode)
        trip_filename = f"{args.directory}/trip_data/trip_assignments_{city}_mode{args.mode}_fixagent{args.fix_agent}_episodes{args.max_episodes}.csv"
        os.makedirs(f"{args.directory}/trip_data", exist_ok=True)
        df_trips.to_csv(trip_filename, index=False)
        print(f"\nTrip assignment data saved to {trip_filename}")
        print(f"Total trips logged: {len(trip_data_last_episode)}")
    # Extract values for each agent across all episodes
    rewards_agent0 = [ep[0] for ep in epoch_reward_list]
    rewards_agent1 = [ep[1] for ep in epoch_reward_list]
    demands_agent0 = [ep[0] for ep in epoch_demand_list]
    demands_agent1 = [ep[1] for ep in epoch_demand_list]
    costs_agent0 = [ep[0] for ep in epoch_rebalancing_cost]
    costs_agent1 = [ep[1] for ep in epoch_rebalancing_cost]
    waiting_agent0 = [ep[0] for ep in epoch_waiting_list]
    waiting_agent1 = [ep[1] for ep in epoch_waiting_list]
    queue_agent0 = [ep[0] for ep in epoch_queue_length_list]
    queue_agent1 = [ep[1] for ep in epoch_queue_length_list]
    arrivals_agent0 = [ep[0] for ep in epoch_arrivals_list]
    arrivals_agent1 = [ep[1] for ep in epoch_arrivals_list]
    
    # Combined totals
    rewards_total = [ep[0] + ep[1] for ep in epoch_reward_list]
    demands_total = [ep[0] + ep[1] for ep in epoch_demand_list]
    costs_total = [ep[0] + ep[1] for ep in epoch_rebalancing_cost]
    arrivals_total = [ep[0] + ep[1] for ep in epoch_arrivals_list]
    
    print("\n" + "="*80)
    print("TEST RESULTS SUMMARY")
    print("="*80)
    
    print("\nAgent 0 Metrics:")
    print(f"  Rewards (mean, std): {np.mean(rewards_agent0):.2f}, {np.std(rewards_agent0):.2f}")
    print(f"  Served demand (mean, std): {np.mean(demands_agent0):.2f}, {np.std(demands_agent0):.2f}")
    print(f"  Rebalancing cost (mean, std): {np.mean(costs_agent0):.2f}, {np.std(costs_agent0):.2f}")
    print(f"  Waiting time (mean, std): {np.mean(waiting_agent0):.2f}, {np.std(waiting_agent0):.2f}")
    print(f"  Queue length (mean, std): {np.mean(queue_agent0):.2f}, {np.std(queue_agent0):.2f}")
    print(f"  Arrivals (mean, std): {np.mean(arrivals_agent0):.2f}, {np.std(arrivals_agent0):.2f}")
    
    print("\nAgent 1 Metrics:")
    print(f"  Rewards (mean, std): {np.mean(rewards_agent1):.2f}, {np.std(rewards_agent1):.2f}")
    print(f"  Served demand (mean, std): {np.mean(demands_agent1):.2f}, {np.std(demands_agent1):.2f}")
    print(f"  Rebalancing cost (mean, std): {np.mean(costs_agent1):.2f}, {np.std(costs_agent1):.2f}")
    print(f"  Waiting time (mean, std): {np.mean(waiting_agent1):.2f}, {np.std(waiting_agent1):.2f}")
    print(f"  Queue length (mean, std): {np.mean(queue_agent1):.2f}, {np.std(queue_agent1):.2f}")
    print(f"  Arrivals (mean, std): {np.mean(arrivals_agent1):.2f}, {np.std(arrivals_agent1):.2f}")
    
    print("\nCombined Metrics:")
    print(f"  Total rewards (mean, std): {np.mean(rewards_total):.2f}, {np.std(rewards_total):.2f}")
    print(f"  Total served demand (mean, std): {np.mean(demands_total):.2f}, {np.std(demands_total):.2f}")
    print(f"  Total rebalancing cost (mean, std): {np.mean(costs_total):.2f}, {np.std(costs_total):.2f}")
    print(f"  Total arrivals (mean, std): {np.mean(arrivals_total):.2f}, {np.std(arrivals_total):.2f}")
    
    # Only show rebalancing trips for modes 0 and 2 (not mode 1)
    if args.mode != 1:
        reb_agent0 = [ep[0] for ep in epoch_rebalancing_list]
        reb_agent1 = [ep[1] for ep in epoch_rebalancing_list]
        reb_total = [ep[0] + ep[1] for ep in epoch_rebalancing_list]
        print(f"  Agent 0 rebalancing trips (mean, std): {np.mean(reb_agent0):.2f}, {np.std(reb_agent0):.2f}")
        print(f"  Agent 1 rebalancing trips (mean, std): {np.mean(reb_agent1):.2f}, {np.std(reb_agent1):.2f}")
        print(f"  Total rebalancing trips (mean, std): {np.mean(reb_total):.2f}, {np.std(reb_total):.2f}")
    
    # Only show price scalar for modes 1 and 2 (not mode 0)
    if args.mode != 0:
        price_agent0 = [ep[0] for ep in epoch_price_mean_list]
        price_agent1 = [ep[1] for ep in epoch_price_mean_list]
        print(price_agent0)
        print(price_agent1)
        print(f"  Agent 0 price scalar (mean, std): {np.mean(price_agent0):.2f}, {np.std(price_agent0):.2f}")
        print(f"  Agent 1 price scalar (mean, std): {np.mean(price_agent1):.2f}, {np.std(price_agent1):.2f}")
    
    # Show concentration parameters (mode-specific)
    print("\nConcentration Parameters:")
    if args.mode == 0:
        # Mode 0: Only Dirichlet for rebalancing
        conc_dirichlet_agent0 = [ep[0] for ep in epoch_concentration_dirichlet_list]
        conc_dirichlet_agent1 = [ep[1] for ep in epoch_concentration_dirichlet_list]
        print(f"  Agent 0 Dirichlet concentration (mean, std): {np.mean(conc_dirichlet_agent0):.2f}, {np.std(conc_dirichlet_agent0):.2f}")
        print(f"  Agent 1 Dirichlet concentration (mean, std): {np.mean(conc_dirichlet_agent1):.2f}, {np.std(conc_dirichlet_agent1):.2f}")
    elif args.mode == 1:
        # Mode 1: Beta (alpha, beta) for pricing
        conc_alpha_agent0 = [ep[0] for ep in epoch_concentration_alpha_list]
        conc_alpha_agent1 = [ep[1] for ep in epoch_concentration_alpha_list]
        conc_beta_agent0 = [ep[0] for ep in epoch_concentration_beta_list]
        conc_beta_agent1 = [ep[1] for ep in epoch_concentration_beta_list]
        print(f"  Agent 0 Beta Alpha concentration (mean, std): {np.mean(conc_alpha_agent0):.2f}, {np.std(conc_alpha_agent0):.2f}")
        print(f"  Agent 0 Beta Beta concentration (mean, std): {np.mean(conc_beta_agent0):.2f}, {np.std(conc_beta_agent0):.2f}")
        print(f"  Agent 1 Beta Alpha concentration (mean, std): {np.mean(conc_alpha_agent1):.2f}, {np.std(conc_alpha_agent1):.2f}")
        print(f"  Agent 1 Beta Beta concentration (mean, std): {np.mean(conc_beta_agent1):.2f}, {np.std(conc_beta_agent1):.2f}")
    else:  # mode 2
        # Mode 2: Beta (alpha, beta) for pricing + Dirichlet for rebalancing
        conc_alpha_agent0 = [ep[0] for ep in epoch_concentration_alpha_list]
        conc_alpha_agent1 = [ep[1] for ep in epoch_concentration_alpha_list]
        conc_beta_agent0 = [ep[0] for ep in epoch_concentration_beta_list]
        conc_beta_agent1 = [ep[1] for ep in epoch_concentration_beta_list]
        conc_dirichlet_agent0 = [ep[0] for ep in epoch_concentration_dirichlet_list]
        conc_dirichlet_agent1 = [ep[1] for ep in epoch_concentration_dirichlet_list]
        print(f"  Agent 0 Beta Alpha concentration (mean, std): {np.mean(conc_alpha_agent0):.2f}, {np.std(conc_alpha_agent0):.2f}")
        print(f"  Agent 0 Beta Beta concentration (mean, std): {np.mean(conc_beta_agent0):.2f}, {np.std(conc_beta_agent0):.2f}")
        print(f"  Agent 0 Dirichlet concentration (mean, std): {np.mean(conc_dirichlet_agent0):.2f}, {np.std(conc_dirichlet_agent0):.2f}")
        print(f"  Agent 1 Beta Alpha concentration (mean, std): {np.mean(conc_alpha_agent1):.2f}, {np.std(conc_alpha_agent1):.2f}")
        print(f"  Agent 1 Beta Beta concentration (mean, std): {np.mean(conc_beta_agent1):.2f}, {np.std(conc_beta_agent1):.2f}")
        print(f"  Agent 1 Dirichlet concentration (mean, std): {np.mean(conc_dirichlet_agent1):.2f}, {np.std(conc_dirichlet_agent1):.2f}")