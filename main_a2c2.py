import argparse
import torch
from src.envs.amod_env2 import Scenario, AMoD
from src.algos.a2c_gnn_emil2 import A2C
from tqdm import trange
import numpy as np
from src.misc.utils import dictsum, nestdictsum
import copy, os
from src.algos.reb_flow_solver import solveRebFlow
import json, pickle

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

# Parser arguments
args = parser.parse_args()

# Set device
args.cuda = args.cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")

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
    env = AMoD(scenario, args.mode, beta=beta[city], jitter=args.jitter, max_wait=args.maxt)

    # Load the model 
    model = A2C(
            env=env,
            input_size= args.look_ahead + 4, # 4 features + time encoding
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
        )

    if args.load:
        print("load checkpoint")
        model.load_checkpoint(path=f"ckpt/{args.checkpoint_path}.pth")

    #Initialize lists for logging
    log = {'train_reward': [], 
            'train_served_demand': [], 
            'train_reb_cost': []}
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
                    f"Episode {i_episode+1} | Reward: {episode_reward:.2f} | Grad Norms: Actor={grad_norms['actor_grad_norm']:.2f}, Critic={grad_norms['critic_grad_norm']:.2f}| Loss: Actor ={grad_norms['actor_loss']:.2f}, Critic={grad_norms['critic_loss']:.2f}"
                )
        # Checkpoint best performing model
        if episode_reward >= best_reward:
            model.save_checkpoint(
                path=f"ckpt/{args.checkpoint_path}_sample.pth")
            best_reward = episode_reward
        model.save_checkpoint(path=f"ckpt/{args.checkpoint_path}_running.pth")
        if i_episode % 100 == 0:
                test_reward, test_served_demand, test_rebalancing_cost = model.test_agent(
                    10, env, args.cplexpath, args.directory)
                if test_reward >= best_reward_test:
                    best_reward_test = test_reward
                    model.save_checkpoint(
                        path=f"ckpt/{args.checkpoint_path}_test.pth")
    

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

else:
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
    env = AMoD(scenario, args.mode, beta=beta[city], jitter=args.jitter, max_wait=args.maxt)

    # Load the model 
    model = A2C(
            env=env,
            input_size= args.look_ahead + 4, # 4 features + time encoding
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
        )

    print("load model")
    model.load_checkpoint(path=f"ckpt/{args.checkpoint_path}.pth")

    test_episodes = args.max_episodes  # set max number of training episodes
    epochs = trange(test_episodes)  # epoch iterator
    best_reward = -np.inf  # set best reward
    best_reward_test = -np.inf  # set best reward
    model.train()  # set model in train mode

     # Initialize lists for logging
    log = {"test_reward": [], "test_served_demand": [], "test_reb_cost": []}

    rewards = []
    demands = []
    costs = []
    arrivals = []

    demand_original_steps = []
    demand_scaled_steps = []
    reb_steps = []
    reb_ori_steps = []
    reb_num = []
    pax_steps = []
    pax_wait = []
    actions_step = []
    price_mean = []
    available_steps = []
    rebalancing_cost_steps = []
    price_original_steps = []
    queue_steps = []
    waiting_steps = []

    for episode in range(10):
        actions = []
        actions_price = []
        rebalancing_cost = []
        rebalancing_num = []
        queue = []

        episode_reward = 0
        episode_served_demand = 0
        episode_price = []
        episode_rebalancing_cost = 0
        episode_waiting = 0
        obs = env.reset()
        # Original demand and price
        demand_original_steps.append(env.demand)
        price_original_steps.append(env.price)

        action_rl = [0]*env.nregion        
        done = False

        while not done:
            if env.mode == 0:
                obs, paxreward, done, info, _, _ = env.match_step_simple()
                episode_reward += paxreward
        
                action_rl = model.select_action(obs, deterministic=True)  # Choose an action
                actions.append(action_rl)

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
                _, rebreward, done, _, _, _ = env.reb_step(rebAction)
                eps_reward += rebreward
                
            elif env.mode == 1:
                obs, paxreward, done, info, _, _ = env.match_step_simple(action_rl)
                episode_reward += paxreward
                action_rl = model.select_action(obs, deterministic=True)  # Choose an action
                env.matching_update()

            elif env.mode == 2:
                obs, paxreward, done, info, _, _ = env.match_step_simple(action_rl)
                episode_reward += paxreward

                action_rl = model.select_action(obs, deterministic=True)  # Choose an action)

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
                _, rebreward, done, info, _, _ = env.reb_step(rebAction)
                episode_reward += rebreward 
            else:
                raise ValueError("Only mode 0, 1, and 2 are allowed")
            
            episode_served_demand += info["served_demand"]
            episode_rebalancing_cost += info["rebalancing_cost"]
            episode_waiting += info['served_waiting']
            actions.append(action_rl)
            if args.mode == 1:
                actions_price.append(np.mean(2*np.array(action_rl)))
            elif args.mode == 2:
                actions_price.append(np.mean(2*np.array(action_rl)[:,0]))
            rebalancing_cost.append(info["rebalancing_cost"])
            # queue.append([len(env.queue[i]) for i in sorted(env.queue.keys())])
            queue.append(np.mean([len(env.queue[i]) for i in env.queue.keys()]))
        
        # Send current statistics to screen
        epochs.set_description(
            f"Episode {episode+1} | Reward: {episode_reward:.2f} | ServedDemand: {episode_served_demand:.2f} | Reb. Cost: {episode_rebalancing_cost}"
        )
        # Log KPIs
        demand_scaled_steps.append(env.demand)
        available_steps.append(env.acc)
        reb_steps.append(env.rebFlow)
        reb_ori_steps.append(env.rebFlow_ori)
        pax_steps.append(env.paxFlow)
        pax_wait.append(env.paxWait)
        reb_od = 0
        for (o,d),flow in env.rebFlow.items():
            reb_od += sum(flow.values())
        reb_num.append(reb_od)
        actions_step.append(actions)
        if args.mode != 0:
            price_mean.append(np.mean(actions_price))
        
        rebalancing_cost_steps.append(rebalancing_cost)
        queue_steps.append(np.mean(queue))
        waiting_steps.append(episode_waiting/episode_served_demand)

        rewards.append(episode_reward)
        demands.append(episode_served_demand)
        costs.append(episode_rebalancing_cost)
        arrivals.append(env.arrivals)

    # Save metrics file
    np.save(f"{args.directory}/{city}_actions_mode{args.mode}.npy", np.array(actions_step))
    np.save(f"{args.directory}/{city}_queue_mode{args.mode}.npy", np.array(queue_steps))

    if env.mode != 1: 
        with open(f"{args.directory}/{city}_reb_mode{args.mode}.pickle", 'wb') as f:
            pickle.dump(reb_steps, f)
        with open(f"{args.directory}/{city}_reb_ori_mode{args.mode}.pickle", 'wb') as f:
            pickle.dump(reb_ori_steps, f)

    with open(f"{args.directory}/{city}_pax_mode{args.mode}.pickle", 'wb') as f:
        pickle.dump(pax_steps, f)
    with open(f"{args.directory}/{city}_pax_wait_mode{args.mode}.pickle", 'wb') as f:
        pickle.dump(pax_wait, f)                     
    with open(f"{args.directory}/{city}_demand_scaled_mode{args.mode}.pickle", 'wb') as f:
        pickle.dump(demand_scaled_steps, f)    

    print("Rewards (mean, std):", f"{np.mean(rewards):.2f}", f"{np.std(rewards):.2f}")
    print("Served demand (mean, std):", f"{np.mean(demands):.2f}", f"{np.std(demands):.2f}")
    print("Rebalancing cost (mean, std):", f"{np.mean(costs):.2f}", f"{np.std(costs):.2f}")
    print("Waiting time (mean, std):", f"{np.mean(waiting_steps):.2f}", f"{np.std(waiting_steps):.2f}")
    print("Queue length (mean, std):", f"{np.mean(queue_steps):.2f}", f"{np.std(queue_steps):.2f}")
    print("Arrivals (mean, std):", f"{np.mean(arrivals):.2f}", f"{np.std(arrivals):.2f}")
    print("Rebalancing trips (mean, std):", f"{np.mean(reb_num):.2f}", f"{np.std(reb_num):.2f}")
    if args.mode != 0:
        print("Price scalar (mean, std):", f"{np.mean(price_mean):.2f}", f"{np.std(price_mean):.2f}")