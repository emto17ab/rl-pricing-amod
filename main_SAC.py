from __future__ import print_function
import argparse
from tqdm import trange
import numpy as np
import torch
from src.envs.amod_env import Scenario, AMoD
from src.algos.sac import SAC
from src.algos.reb_flow_solver import solveRebFlow
from src.misc.utils import dictsum, nestdictsum
import json, pickle
from torch_geometric.data import Data
import copy, os


class GNNParser:
    """
    Parser converting raw environment observations to agent inputs (s_t).
    """

    def __init__(self, env, T=10, json_file=None, scale_factor=0.01):
        super().__init__()
        self.env = env
        self.T = T
        self.s = scale_factor
        self.json_file = json_file
        if self.json_file is not None:
            with open(json_file, "r") as file:
                self.data = json.load(file)

    def parse_obs(self, obs):
        # Takes input from the environemnt and returns a graph (with node features and connectivity)
        # Here we aggregate environment observations into node-wise features
        # In order, x is a collection of the following information:
        # 1) current availability scaled by factor, 
        # 2) Estimated availability (T timestamp) scaled by factor, 
        # 3) Estimated revenue (T timestamp) scaled by factor
        x = (
            torch.cat(
                (
                    # Current availability
                    torch.tensor(
                        [obs[0][n][self.env.time + 1] *
                            self.s for n in self.env.region]
                    )
                    .view(1, 1, self.env.nregion)
                    .float(),
                    # Estimated availability
                    torch.tensor(
                        [
                            [
                                (obs[0][n][self.env.time + 1] +
                                 self.env.dacc[n][t])
                                * self.s
                                for n in self.env.region
                            ]
                            for t in range(
                                self.env.time + 1, self.env.time + self.T + 1
                            )
                        ]
                    )
                    .view(1, self.T, self.env.nregion)
                    .float(),
                    # Queue length
                    torch.tensor(
                        [
                            len(self.env.queue[n]) * self.s for n in self.env.region
                        ]
                    )
                    .view(1, 1, self.env.nregion)
                    .float(),
                    # Current demand
                    torch.tensor(
                            [
                                sum(
                                    [
                                        (self.env.demand[i, j][self.env.time])
                                        # * (self.env.price[i, j][self.env.time])
                                        * self.s
                                        for j in self.env.region
                                    ]
                                )
                                for i in self.env.region
                            ]
                    )
                    .view(1, 1, self.env.nregion)
                    .float(),
                    # Current price
                    torch.tensor(
                            [
                                sum(
                                    [
                                        (self.env.price[i, j][self.env.time])
                                        * self.s
                                        for j in self.env.region
                                    ]
                                )
                                for i in self.env.region
                            ]
                    )
                    .view(1, 1, self.env.nregion)
                    .float(),                    
                ),
                dim=1,
            )
            .squeeze(0)
            .view(1 + self.T + 1 + 1 + 1, self.env.nregion)
            .T
        )       
        if self.json_file is not None:
            edge_index = torch.vstack(
                (
                    torch.tensor(
                        [edge["i"] for edge in self.data["topology_graph"]]
                    ).view(1, -1),
                    torch.tensor(
                        [edge["j"] for edge in self.data["topology_graph"]]
                    ).view(1, -1),
                )
            ).long()
        else:
            edge_index = torch.cat(
                (
                    torch.arange(self.env.nregion).view(1, self.env.nregion),
                    torch.arange(self.env.nregion).view(1, self.env.nregion),
                ),
                dim=0,
            ).long()
        data = Data(x, edge_index)
        return data


# Define calibrated simulation parameters
demand_ratio = {'san_francisco': 1, 'washington_dc': 4.2, 'chicago': 1.8, 'nyc_man_north': 1.8, 'nyc_man_middle': 1.8,
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

test_tstep = {'san_francisco': 3, 'nyc_brooklyn': 4, 'shenzhen_downtown_west': 3, 'nyc_man_middle': 3, 'nyc_man_south': 3}

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
    default=2,
    help='rebalancing mode. (0:manul, 1:pricing, 2:both. default 1)',
)

parser.add_argument(
    "--beta",
    type=int,
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
    help="number of episodes to train agent (default: 16k)",
)
parser.add_argument(
    "--max_steps",
    type=int,
    default=20,
    metavar="N",
    help="number of steps per episode (default: T=20)",
)
parser.add_argument(
    "--no-cuda", 
    type=bool, 
    default=True,
    help="disables CUDA training",
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
    help="maximum passenger waiting time (default: 6mins)",
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
    default=256,
    help="hidden size of neural networks (default: 256)",
)
parser.add_argument(
    "--checkpoint_path",
    type=str,
    default="SAC",
    help="name of checkpoint file to save/load (default: SAC)",
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
    help="learning rate for policy network (default: 1e-4)",
)
parser.add_argument(
    "--q_lr",
    type=float,
    default=1e-3,
    help="learning rate for Q networks (default: 4e-3)",
)
parser.add_argument(
    "--q_lag",
    type=int,
    default=1,
    help="update frequency of Q target networks (default: 10)",
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
    "--critic_version",
    type=int,
    default=4,
    help="critic version (default: 4)",
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

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
city = args.city


if not args.test:
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

    # d = {
    # (2, 3): 6,
    # (2, 0): 4,
    # (0, 3): 4,
    # "default": 1,
    # }
    # r = {
    # 0: [1, 1, 1, 2, 2, 3, 3, 1, 1, 1, 2, 2],
    # 1: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    # 2: [1, 1, 1, 2, 2, 3, 4, 4, 2, 1, 1, 1],
    # 3: [1, 1, 1, 2, 2, 2, 2, 2, 2, 1, 1, 1],
    # }
    # scenario = Scenario(tf=20, demand_input=d, demand_ratio=r, ninit=30, N1=2, N2=2)

    env = AMoD(scenario, args.mode, beta=beta[city], jitter=args.jitter, max_wait=args.maxt)

    parser = GNNParser(
        env, T=6, json_file=f"data/scenario_{city}.json"
    )  # Timehorizon T=6 (K in paper)

    # parser = GNNParser(
    #     env, T=6
    # )  # Timehorizon T=6 (K in paper)

    model = SAC(
        env=env,
        input_size=10,
        hidden_size=args.hidden_size,
        p_lr=args.p_lr,
        q_lr=args.q_lr,
        alpha=args.alpha,
        batch_size=args.batch_size,
        use_automatic_entropy_tuning=False,
        clip=args.clip,
        critic_version=args.critic_version,
        price_version = args.price_version,
        mode=args.mode,
        q_lag=args.q_lag
    ).to(device)

    train_episodes = args.max_episodes  # set max number of training episodes
    T = args.max_steps  # set episode length
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
            if step > 0:
                obs1 = copy.deepcopy(o)

            if env.mode == 0:
                obs, paxreward, done, info, _, _ = env.match_step_simple()
                # obs, paxreward, done, info, _, _ = env.pax_step(
                #                 CPLEXPATH=args.cplexpath, directory=args.directory, PATH="scenario_san_francisco4"
                #             )

                o = parser.parse_obs(obs=obs)
                episode_reward += paxreward
                if step > 0:
                    # store transition in memroy
                    rl_reward = paxreward + rebreward
                    model.replay_buffer.store(
                        obs1, action_rl, args.rew_scale * rl_reward, o
                    )

                action_rl = model.select_action(o)

                # transform sample from Dirichlet into actual vehicle counts (i.e. (x1*x2*..*xn)*num_vehicles)
                desiredAcc = {
                    env.region[i]: int(
                        action_rl[i] * dictsum(env.acc, env.time + 1))
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
            elif env.mode == 1:
                obs, paxreward, done, info, _, _ = env.match_step_simple(action_rl)

                o = parser.parse_obs(obs=obs)

                episode_reward += paxreward
                if step > 0:
                    # store transition in memroy
                    rl_reward = paxreward
                    model.replay_buffer.store(
                        obs1, action_rl, args.rew_scale * rl_reward, o
                    )

                action_rl = model.select_action(o,deterministic=True)  

                env.matching_update()
            elif env.mode == 2:
                obs, paxreward, done, info, _, _ = env.match_step_simple(action_rl)

                o = parser.parse_obs(obs=obs)
                episode_reward += paxreward
                if step > 0:
                    # store transition in memroy
                    rl_reward = paxreward + rebreward
                    model.replay_buffer.store(
                        obs1, action_rl, args.rew_scale * rl_reward, o
                    )

                action_rl = model.select_action(o)

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
            else:
                raise ValueError("Only mode 0, 1, and 2 are allowed")                    

            # track performance over episode
            episode_served_demand += info["served_demand"]
            episode_rebalancing_cost += info["rebalancing_cost"]
            episode_waiting += info['served_waiting']
            actions.append(action_rl)

            step += 1
            if i_episode > 10:
                # sample from memory and update model
                batch = model.replay_buffer.sample_batch(
                    args.batch_size, norm=False)
                grad_norms = model.update(data=batch)  
            else:
                grad_norms = {"actor_grad_norm":0, "critic1_grad_norm":0, "critic2_grad_norm":0, "actor_loss":0, "critic1_loss":0, "critic2_loss":0, "Q1_value":0, "Q2_value":0}
            
            # Keep track of loss
            epoch_value1_list.append(grad_norms["Q1_value"])
            epoch_value2_list.append(grad_norms["Q2_value"])

        # Keep metrics
        epoch_reward_list.append(episode_reward)
        epoch_demand_list.append(env.arrivals)
        epoch_waiting_list.append(episode_waiting/episode_served_demand)
        epoch_servedrate_list.append(episode_served_demand/env.arrivals)
        epoch_rebalancing_cost.append(episode_rebalancing_cost)

        # Keep price (only needed for pricing training)
        price_history.append(actions)

        epochs.set_description(
            f"Episode {i_episode+1} | Reward: {episode_reward:.2f} | Grad Norms: Actor={grad_norms['actor_grad_norm']:.2f}, Critic1={grad_norms['critic1_grad_norm']:.2f}, Critic2={grad_norms['critic2_grad_norm']:.2f}\
              | Loss: Actor ={grad_norms['actor_loss']:.2f}, Critic1={grad_norms['critic1_loss']:.2f}, Critic2={grad_norms['critic2_loss']:.2f}"
        )
        # Checkpoint best performing model
        if episode_reward >= best_reward:
            model.save_checkpoint(
                path=f"ckpt/{args.checkpoint_path}_sample.pth")
            best_reward = episode_reward
        model.save_checkpoint(path=f"ckpt/{args.checkpoint_path}_running.pth")
        if i_episode % 10 == 0:
            test_reward, test_served_demand, test_rebalancing_cost = model.test_agent(
                1, env, args.cplexpath, args.directory, parser=parser
            )
            if test_reward >= best_reward_test:
                best_reward_test = test_reward
                model.save_checkpoint(
                    path=f"ckpt/{args.checkpoint_path}_test.pth")
        # if (i_episode>=30 and i_episode<=50 and i_episode % 10 == 0):
        #     model.save_checkpoint(path=f"ckpt/{args.checkpoint_path}_running_{i_episode}.pth")          
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
else:
    scenario = Scenario(
        json_file=f"data/scenario_{city}.json",
        demand_ratio=demand_ratio[city],
        json_hr=json_hr[city],
        sd=args.seed,
        json_tstep=test_tstep[city],
        tf=args.max_steps,
        impute=args.impute,
        supply_ratio=args.supply_ratio
    )

    # d = {
    # (2, 3): 6,
    # (2, 0): 4,
    # (0, 3): 4,
    # "default": 1,
    # }
    # r = {
    # 0: [1, 1, 1, 2, 2, 3, 3, 1, 1, 1, 2, 2],
    # 1: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    # 2: [1, 1, 1, 2, 2, 3, 4, 4, 2, 1, 1, 1],
    # 3: [1, 1, 1, 2, 2, 2, 2, 2, 2, 1, 1, 1],
    # }
    # scenario = Scenario(tf=20, demand_input=d, demand_ratio=r, ninit=30, N1=2, N2=2)

    env = AMoD(scenario, args.mode, beta=beta[city], jitter=args.jitter, max_wait=args.maxt)

    parser = GNNParser(
        env, T=6, json_file=f"data/scenario_{city}.json"
    )  # Timehorizon T=6 (K in paper)

    # parser = GNNParser(
    #     env, T=6
    # )  # Timehorizon T=6 (K in paper)

    model = SAC(
        env=env,
        input_size=10,
        hidden_size=args.hidden_size,
        p_lr=args.p_lr,
        q_lr=args.q_lr,
        alpha=args.alpha,
        batch_size=args.batch_size,
        use_automatic_entropy_tuning=False,
        clip=args.clip,
        critic_version=args.critic_version,
        price_version = args.price_version,
        mode=args.mode,
        q_lag=args.q_lag
    ).to(device)

    print("load model")
    model.load_checkpoint(path=f"ckpt/{args.checkpoint_path}.pth")

    test_episodes = args.max_episodes  # set max number of training episodes
    T = args.max_steps  # set episode length
    epochs = trange(test_episodes)  # epoch iterator
    # Initialize lists for logging
    log = {"test_reward": [], "test_served_demand": [], "test_reb_cost": []}

    rewards = []
    demands = []
    costs = []
    arrivals = []

    demand_original_steps = []
    demand_scaled_steps = []
    reb_steps = []
    actions_step = []
    available_steps = []
    rebalancing_cost_steps = []
    price_original_steps = []
    queue_steps = []

    for episode in range(10):
        actions = []
        rebalancing_cost = []
        queue = []

        episode_reward = 0
        episode_served_demand = 0
        episode_rebalancing_cost = 0
        obs = env.reset()
        # Original demand and price
        demand_original_steps.append(env.demand)
        price_original_steps.append(env.price)

        action_rl = [0]*env.nregion        
        done = False
        while not done:

            if env.mode == 0:
                obs, paxreward, done, info, _, _ = env.match_step_simple()
                # obs, paxreward, done, info, _, _ = env.pax_step(
                #                 CPLEXPATH=args.cplexpath, directory=args.directory, PATH="scenario_san_francisco4"
                #             )

                o = parser.parse_obs(obs=obs)
                episode_reward += paxreward

                action_rl = model.select_action(o, deterministic=True)
                actions.append(action_rl)

                # transform sample from Dirichlet into actual vehicle counts (i.e. (x1*x2*..*xn)*num_vehicles)
                desiredAcc = {
                    env.region[i]: int(
                        action_rl[0][i] * dictsum(env.acc, env.time + 1))
                    for i in range(len(env.region))
                }
                # print(desiredAcc)
                # print({env.region[i]: env.acc[env.region[i]][env.time+1] for i in range(len(env.region))})
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
                episode_reward += rebreward

            elif env.mode == 1:
                obs, paxreward, done, info, _, _ = env.match_step_simple(action_rl)

                o = parser.parse_obs(obs=obs)

                episode_reward += paxreward

                action_rl = model.select_action(o, deterministic=True)  

                env.matching_update()
            elif env.mode == 2:
                obs, paxreward, done, info, _, _ = env.match_step_simple(action_rl)

                o = parser.parse_obs(obs=obs)
                episode_reward += paxreward

                action_rl = model.select_action(o, deterministic=True)

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
            actions.append(action_rl)
            rebalancing_cost.append(info["rebalancing_cost"])
            queue.append([len(env.queue[i]) for i in env.queue.keys()])
        # Send current statistics to screen
        epochs.set_description(
            f"Episode {episode+1} | Reward: {episode_reward:.2f} | ServedDemand: {episode_served_demand:.2f} | Reb. Cost: {episode_rebalancing_cost}"
        )
        # Log KPIs
        demand_scaled_steps.append(env.demand)
        available_steps.append(env.acc)
        reb_steps.append(env.rebFlow)
        actions_step.append(actions)
        rebalancing_cost_steps.append(rebalancing_cost)
        queue_steps.append(queue)

        rewards.append(episode_reward)
        demands.append(episode_served_demand)
        costs.append(episode_rebalancing_cost)
        arrivals.append(env.arrivals)

    # Save metrics file
    np.save(f"{args.directory}/{city}_actions_mode{args.mode}.npy", np.array(actions_step))
    np.save(f"{args.directory}/{city}_queue_mode{args.mode}.npy", np.array(queue_steps))
    np.save(f"{args.directory}/{city}_served_mode{args.mode}.npy", np.array([demands,arrivals]))
    if env.mode != 1: 
        np.save(f"{args.directory}/{city}_cost_mode{args.mode}.npy", np.array(rebalancing_cost_steps))
        with open(f"{args.directory}/{city}_reb_mode{args.mode}.pickle", 'wb') as f:
            pickle.dump(reb_steps, f)                    
    
    with open(f"{args.directory}/{city}_demand_ori_mode{args.mode}.pickle", 'wb') as f:
        pickle.dump(demand_original_steps, f)
    with open(f"{args.directory}/{city}_price_ori_mode{args.mode}.pickle", 'wb') as f:
        pickle.dump(price_original_steps, f)

    with open(f"{args.directory}/{city}_demand_scaled_mode{args.mode}.pickle", 'wb') as f:
        pickle.dump(demand_scaled_steps, f)    
    with open(f"{args.directory}/{city}_acc_mode{args.mode}.pickle", 'wb') as f:
        pickle.dump(available_steps, f)

    print("Rewards (mean, std):", np.mean(rewards), np.std(rewards))
    print("Served demand (mean, std):", np.mean(demands), np.std(demands))
    print("Rebalancing cost (mean, std):", np.mean(costs), np.std(costs))
    print("Arrivals (mean, std):", np.mean(arrivals), np.std(arrivals))
