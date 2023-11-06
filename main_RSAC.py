from __future__ import print_function
import argparse
from tqdm import trange
import numpy as np
import torch
from src.envs.amod_env import Scenario, AMoD
from src.algos.rsac import RSAC
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
        x = np.squeeze(
            np.concatenate(
                (
                    # Current availability
                    np.array(
                        [obs[0][n][self.env.time + 1] *
                            self.s for n in self.env.region]
                    )
                    .reshape(1, 1, self.env.nregion)
                    .astype(float),
                    # Estimated availability
                    np.array(
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
                    .reshape(1, self.T, self.env.nregion)
                    .astype(float),
                    # Queue length
                    np.array(
                        [
                            len(self.env.queue[n]) * self.s for n in self.env.region
                        ]
                    )
                    .reshape(1, 1, self.env.nregion)
                    .astype(float),
                    # Current demand
                    np.array(
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
                    .reshape(1, 1, self.env.nregion)
                    .astype(float),
                ),
                axis=1,
            ),0).reshape(1 + self.T + 1 + 1, self.env.nregion).T        
        
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

test_tstep = {'san_francisco': 3, 'nyc_brooklyn': 4, 'shenzhen_downtown_west': 3}

parser = argparse.ArgumentParser(description="SAC-GNN")

# Simulator parameters
parser.add_argument(
    "--seed", type=int, default=10, metavar="S", help="random seed (default: 10)"
)
parser.add_argument(
    "--demand_ratio",
    type=int,
    default=0.5,
    metavar="S",
    help="demand_ratio (default: 0.5)",
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
    default=32,
    help="batch size for training (default: 128)",
)
parser.add_argument(
    "--buffer_cap",
    type=int,
    default=1000,
    help="Buffer capacity (default: 1000)",
)
parser.add_argument(
    "--replay_ratio",
    type=int,
    default=4,
    help="Number of actions for each gradient update (default: 4)",
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
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
city = args.city
env_baseline = []


if not args.test:
    scenario = Scenario(
        json_file=f"data/scenario_{city}.json",
        demand_ratio=demand_ratio[city],
        json_hr=json_hr[city],
        sd=args.seed,
        json_tstep=args.json_tstep,
        tf=args.max_steps,
    )

    env = AMoD(scenario, args.mode, beta=beta[city])

    parser = GNNParser(
        env, T=6, json_file=f"data/scenario_{city}.json"
    )  # Timehorizon T=6 (K in paper)

    model = RSAC(
        env=env,
        recurrent_input_size=1,
        recurrent_hidden_size=2,
        other_input_size=8,
        hidden_size=args.hidden_size,
        sample_steps=1,
        p_lr=args.p_lr,
        q_lr=args.q_lr,
        alpha=args.alpha,
        batch_size=args.batch_size,
        buffer_cap=args.max_episodes,
        use_automatic_entropy_tuning=False,
        clip=args.clip,
        critic_version=args.critic_version,
        mode=args.mode,
        env_baseline=env_baseline,
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
    
    price_history = []

    for i_episode in epochs:
        obs = env.reset()  # initialize environment
        model.reinitialize_hidden()
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

            obs, paxreward, done, info, _, _ = env.match_step_simple(action_rl)

            o = parser.parse_obs(obs=obs)

            episode_reward += paxreward
            rl_reward = paxreward
            if step > 0:
                # store transition in memroy
                model.replay_buffer.store(
                    obs1.x, action_rl, args.rew_scale * rl_reward, o.x, done, o.edge_index
                )

            action_rl = model.select_action(o)  

            env.matching_update()                

            # track performance over episode
            episode_served_demand += info["served_demand"]
            episode_rebalancing_cost += info["rebalancing_cost"]
            episode_waiting += info['served_waiting']
            actions.append(action_rl)

            step += 1
            # update baseline
            if len(model.env_baseline) <= 1000:
                model.env_baseline.append(args.rew_scale * rl_reward)
            else:
                _ = model.env_baseline.pop(0)
                model.env_baseline.append(args.rew_scale * rl_reward)

            if i_episode > 10:
                # Sample from memory and update model. Start to sample when the buffer size is at least the lower bound.
                batch = model.replay_buffer.sample_batch()
                grad_norms = model.update(batch)  
            else:
                grad_norms = {"actor_grad_norm":0, "critic1_grad_norm":0, "critic2_grad_norm":0, "actor_loss":0, "critic1_loss":0, "critic2_loss":0}

        # Keep metrics
        epoch_reward_list.append(episode_reward)
        epoch_demand_list.append(env.arrivals/demand_ori)
        epoch_waiting_list.append(episode_waiting/episode_served_demand)
        epoch_servedrate_list.append(episode_served_demand/env.arrivals)

        # Keep price (only needed for pricing training)
        price_history.append(actions)

        epochs.set_description(
            f"Episode {i_episode+1} | Reward: {episode_reward:.2f} | Grad Norms: Actor={grad_norms['actor_grad_norm']:.2f}, Critic1={grad_norms['critic1_grad_norm']:.2f}, Critic2={grad_norms['critic2_grad_norm']:.2f}\
              | Actor loss: {grad_norms['actor_loss']:.2f} | Critic1 loss: {grad_norms['critic1_loss']:.2f} | Critic2 loss: {grad_norms['critic2_loss']:.2f}"
        )
        # Checkpoint best performing model
        if episode_reward >= best_reward:
            model.save_checkpoint(
                path=f"ckpt/{args.checkpoint_path}_sample.pth")
            best_reward = episode_reward
        model.save_checkpoint(path=f"ckpt/{args.checkpoint_path}_running.pth")

    # Save metrics file
    metricPath = f"{args.directory}/train_logs/"
    if not os.path.exists(metricPath):
        os.makedirs(metricPath)
    np.save(f"{args.directory}/train_logs/{city}_rewards_waiting_mode{args.mode}_{train_episodes}.npy", np.array([epoch_reward_list,epoch_waiting_list,epoch_servedrate_list,epoch_demand_list]))
    np.save(f"{args.directory}/train_logs/{city}_price_mode{args.mode}_{train_episodes}.npy", np.array(price_history))
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
    )

    env = AMoD(scenario, beta=beta[city])
    parser = GNNParser(env, T=6, json_file=f"data/scenario_{city}.json")

    model = RSAC(
        env=env,
        input_size=8,
        hidden_size=256,
        sample_steps=1,
        p_lr=1e-3,
        q_lr=1e-3,
        alpha=0.3,
        batch_size=args.batch_size,
        buffer_cap=args.buffer_cap,
        use_automatic_entropy_tuning=False,
        critic_version=args.critic_version,
        mode=args.mode
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

    for episode in range(10):
        episode_reward = 0
        episode_served_demand = 0
        episode_rebalancing_cost = 0
        obs = env.reset()
        done = False
        k = 0
        pax_reward = 0
        while not done:
            # take matching step (Step 1 in paper)
            obs, paxreward, done, info, _, _ = env.pax_step(
                CPLEXPATH=args.cplexpath,
                PATH="scenario_san_francisco4_test",
                directory=args.directory,
            )

            episode_reward += paxreward
            pax_reward += paxreward
            # use GNN-RL policy (Step 2 in paper)
            o = parser.parse_obs(obs=obs)
            action_rl = model.select_action(o, deterministic=True)

            if env.mode == 0:
                # transform sample from Dirichlet into actual vehicle counts (i.e. (x1*x2*..*xn)*num_vehicles)
                desiredAcc = {
                    env.region[i]: int(
                        action_rl[i] * dictsum(env.acc, env.time + 1))
                    for i in range(len(env.region))
                }
                # solve minimum rebalancing distance problem (Step 3 in paper)
                rebAction = solveRebFlow(
                    env, "scenario_san_francisco4_test", desiredAcc, args.cplexpath, args.directory
                )

                _, rebreward, done, info, _, _ = env.reb_step(rebAction)

                episode_reward += rebreward
            elif env.mode == 1:
                env.matching_update()
            else:
                pass
            # track performance over episode
            episode_served_demand += info["served_demand"]
            episode_rebalancing_cost += info["rebalancing_cost"]
            k += 1
        # Send current statistics to screen
        epochs.set_description(
            f"Episode {episode+1} | Reward: {episode_reward:.2f} | ServedDemand: {episode_served_demand:.2f} | Reb. Cost: {episode_rebalancing_cost}"
        )
        # Log KPIs

        rewards.append(episode_reward)
        demands.append(episode_served_demand)
        costs.append(episode_rebalancing_cost)

    print("Rewards (mean, std):", np.mean(rewards), np.std(rewards))
    print("Served demand (mean, std):", np.mean(demands), np.std(demands))
    print("Rebalancing cost (mean, std):", np.mean(costs), np.std(costs))
