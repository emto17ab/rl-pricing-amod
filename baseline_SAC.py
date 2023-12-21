from __future__ import print_function
import argparse
from tqdm import trange
import numpy as np
import torch
from src.envs.amod_env import Scenario, AMoD
from src.algos.sac import SAC
from src.algos.reb_flow_solver import solveRebFlow
from src.misc.utils import dictsum
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
                    torch.tensor(
                        [obs[0][n][self.env.time + 1] *
                            self.s for n in self.env.region]
                    )
                    .view(1, 1, self.env.nregion)
                    .float(),
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
                    torch.tensor(
                        [
                            [
                                sum(
                                    [
                                        (self.env.scenario.demand_input[i, j][t])
                                        * (self.env.price[i, j][t])
                                        * self.s
                                        for j in self.env.region
                                    ]
                                )
                                for i in self.env.region
                            ]
                            for t in range(
                                self.env.time + 1, self.env.time + self.T + 1
                            )
                        ]
                    )
                    .view(1, self.T, self.env.nregion)
                    .float(),
                ),
                dim=1,
            )
            .squeeze(0)
            .view(1 + self.T + self.T, self.env.nregion)
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

test_tstep = {'san_francisco': 3, 'nyc_brooklyn': 4, 'shenzhen_downtown_west': 3}

parser = argparse.ArgumentParser(description="SAC-GNN")

# Simulator parameters
parser.add_argument(
    "--seed", type=int, default=10, metavar="S", help="random seed (default: 10)"
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
    "--cplexpath",
    type=str,
    default="C:/Program Files/IBM/ILOG/CPLEX_Studio201/opl/bin/x64_win64/",
    help="defines directory of the CPLEX installation",
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
    "--city",
    type=str,
    default="san_francisco",
    help="city to train on",
)
parser.add_argument(
    "--directory",
    type=str,
    default="saved_files",
    help="defines directory where to save files",
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
    "--impute",
    type=bool,
    default=False,
    help="Whether impute the zero price (default: False)",
)
parser.add_argument(
    "--supply_ratio",
    type=float,
    default=1.0,
    help="supply scaling factor (default: 1)",
)
parser.add_argument(
    "--demand_ratio",
    type=float,
    default=0.5,
    metavar="S",
    help="demand_ratio (default: 0.5)",
)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
city = args.city


scenario = Scenario(
    json_file=f"data/scenario_{city}.json",
    demand_ratio=args.demand_ratio,
    json_hr=json_hr[city],
    sd=args.seed,
    json_tstep=args.json_tstep,
    tf=args.max_steps,
    impute=args.impute,
    supply_ratio=args.supply_ratio,
)

env = AMoD(scenario, args.mode, beta=beta[city], jitter=args.jitter, max_wait=args.maxt)

parser = GNNParser(
    env, T=6, json_file=f"data/scenario_{city}.json"
)  # Timehorizon T=6 (K in paper)


T = args.max_steps  # set episode length
epochs = trange(10)  # epoch iterator
best_reward = -np.inf  # set best reward


# Check metrics
epoch_demand_list = []
epoch_reward_list = []
epoch_waiting_list = []
epoch_servedrate_list = []

price_history = []

for i_episode in range(10):
    obs = env.reset()  # initialize environment

    action = [0]*env.nregion
    episode_reward = 0
    episode_served_demand = 0
    episode_rebalancing_cost = 0
    episode_waiting = 0

    current_eps = []
    done = False
    step = 0
    while not done:
        # take matching step (Step 1 in paper)
        if step > 0:
            obs1 = copy.deepcopy(o)

        if args.mode == 0:
            obs, paxreward, done, info, _, _ = env.match_step_simple()
            # obs, paxreward, done, info, _, _ = env.pax_step(
            #                 CPLEXPATH=args.cplexpath, PATH="scenario_nyc4", directory=args.directory
            #             )

            o = parser.parse_obs(obs=obs)
            episode_reward += paxreward
            if step > 0:
                # store transition in memroy
                rl_reward = paxreward + rebreward

            # transform sample from Dirichlet into actual vehicle counts (i.e. (x1*x2*..*xn)*num_vehicles)
            ed = 1/env.nregion
            desiredAcc = {
                env.region[i]: int(
                    ed * dictsum(env.acc, env.time + 1))
                for i in range(len(env.region))
            }
            # solve minimum rebalancing distance problem (Step 3 in paper)
            rebAction = solveRebFlow(
                env,
                "scenario_nyc4",
                desiredAcc,
                args.cplexpath,
                args.directory,
            )
            # Take rebalancing action in environment
            new_obs, rebreward, done, info, _, _ = env.reb_step(rebAction)
        elif args.mode == 1:
            obs, paxreward, done, info, _, _ = env.match_step_simple(action)

            o = parser.parse_obs(obs=obs)
            episode_reward += paxreward
            if step > 0:
                # store transition in memroy
                rl_reward = paxreward + rebreward
            env.matching_update()
            rebreward = 0
        
        episode_reward += rebreward
            

        # track performance over episode
        episode_served_demand += info["served_demand"]
        episode_rebalancing_cost += info["rebalancing_cost"]
        episode_waiting += info['served_waiting']

        step += 1 

    # Keep metrics
    epoch_reward_list.append(episode_reward)
    epoch_demand_list.append(env.arrivals)
    epoch_waiting_list.append(episode_waiting/episode_served_demand)
    epoch_servedrate_list.append(episode_served_demand/env.arrivals)

    epochs.set_description(
        f"Episode {i_episode+1} | Reward: {episode_reward:.2f} | ServedDemand: {episode_served_demand:.2f} | Episode served demand rate: {episode_served_demand/env.arrivals:.2f} | Waiting: {episode_waiting/episode_served_demand:.2f}"
    )

print(f"Average arrival: {np.mean(epoch_demand_list)}")
print(f"Average reward: {np.mean(epoch_reward_list)}")
# print(f"Avergaed served demand: {np.mean(epoch_demand_list)}")
print(f"Average waiting time: {np.mean(epoch_waiting_list)}")
print(f"Avergae serve rate: {np.mean(epoch_servedrate_list)}")
