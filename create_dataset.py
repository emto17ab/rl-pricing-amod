from __future__ import print_function
import argparse
from tqdm import trange
import numpy as np
import torch
import pickle
from collections import defaultdict
from src.envs.amod_env import Scenario, AMoD
from src.misc.utils import dictsum
import copy
from torch_geometric.data import Data
from gurobipy import Model, GRB, quicksum, Env

import json
import numpy as np
import torch


def DTV(env, demand):
    # Create the model
    gb_env = Env(empty=True)
    gb_env.setParam("OutputFlag", 0)
    gb_env.start()
    model = Model("mincostflow", env=gb_env)

    t = env.time

    accInitTuple = [(n, int(env.acc[n][t + 1])) for n in env.acc]
    edgeAttr = [(i, j, env.G.edges[i, j]["time"]) for i, j in env.G.edges]

    region = [i for (i, n) in accInitTuple]
    # add self loops to edgeAttr with time =0
    edgeAttr += [(i, i, 0) for i in region]
    time = {(i, j): t for (i, j, t) in edgeAttr}

    vehicles = {i: v for (i, v) in accInitTuple}

    num_vehicles = sum(vehicles.values())
    num_requests = sum(demand.values())

    vehicle_region = {}
    vehicle = 0
    for i in region:
        for _ in range(vehicles[i]):
            vehicle_region[vehicle] = i
            vehicle += 1

    request_region = {}
    request = 0
    for i in demand.keys():
        for _ in range(int(demand[i])):
            request_region[request] = i
            request += 1

    # calculate time for each vehicle to each request according to the region
    time_vehicle_request = {}
    for vehicle in vehicle_region:
        for request in request_region:
            time_vehicle_request[vehicle, request] = time[
                vehicle_region[vehicle], request_region[request]
            ]

    edge = [
        (vehicle, request) for vehicle in vehicle_region for request in request_region
    ]

    rebFlow = model.addVars(edge, vtype=GRB.BINARY, name="x")
    model.update()
    # Set objective
    model.setObjective(
        quicksum(rebFlow[e] * time_vehicle_request[e] for e in edge), GRB.MINIMIZE
    )

    # Add constraints
    model.addConstr(
        quicksum(rebFlow[v, k] for v, k in edge) == min(num_vehicles, num_requests)
    )

    # only one vehicle can be assigned to one request
    for request in request_region:
        model.addConstr(quicksum(rebFlow[v, request] for v in vehicle_region) <= 1)

    # only one request can be assigned to one vehicle
    for vehicle in vehicle_region:
        model.addConstr(quicksum(rebFlow[vehicle, k] for k in request_region) <= 1)

    # Optimize the model
    model.optimize()

    # get rebalancing flows
    flows = {e: 0 for e in env.edges}
    for var in model.getVars():
        if var.X != 0:
            substring = var.VarName[
                var.VarName.index("[") + 1 : var.VarName.index("]")
            ].split(",")
            i = vehicle_region[int(substring[0])]
            j = request_region[int(substring[1])]
            flows[i, j] += 1

    action = [flows[i, j] for i, j in env.edges]
    return action, flows

def SDE(env):
    """Heuristic for dynamic pricing. See in paper:https://zhouzimu.github.io/paper/sigmod18-tong.pdf"""
    t = env.time

    demand = defaultdict(float)
    for (i,_),d in env.demand.items():
        demand[i] += d[t]
    
    # Demand
    pax = [len(env.queue[i]) + demand[i] for i in env.region]
    # Supply
    veh = [env.acc[i][t] for i in env.region]

    price = []
    for i in env.region:
        w = veh[i]
        r = pax[i]
        if w < r:
            p = (1 + np.exp(w-r))/2
        else:
            p = (1 - np.exp(r-w))/2
        price.append(p)
    
    return price


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

def flow_to_dist(flows, acc):
    """AMoD
    Convert flow to distribution
    :param flows: rebflow decision in time step t (i,j)
    :param acc: idle vehicle distribtution in time step t+1
    :return: distribution
    """
    desiredAcc = acc.copy()
    for i, j in flows.keys():
        desiredAcc[i] -= flows[i, j]
        desiredAcc[j] += flows[i, j]

    total_acc = sum(acc.values())
    if total_acc == 0:
        distribution = {i: 1 / len(acc) for i in acc}
    else:
        distribution = {i: desiredAcc[i] / total_acc for i in desiredAcc}
    return distribution


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(
            obs=self.obs_buf[idxs],
            obs2=self.obs2_buf[idxs],
            act=self.act_buf[idxs],
            rew=self.rew_buf[idxs],
            done=self.done[idxs],
        )
        return {k: torch.as_tensor(v) for k, v in batch.items()}


demand_ratio = {
    "san_francisco": 2,
    "nyc_brooklyn": 9,
    "nyc_man_south": 1.8,
    "nyc_man_middle": 1.8,
}
json_hr = {
    "san_francisco": 19,
    "nyc_brooklyn": 19,
    "nyc_man_south": 19,
    "nyc_man_middle": 19,
}
beta = {
    "san_francisco": 0.2,
    "nyc_brooklyn": 0.5,
    "nyc_man_south": 0.5,
    "nyc_man_middle": 0.5,
}

test_tstep = {"san_francisco": 3, "nyc_brooklyn": 4, "nyc_man_south": 3, "nyc_man_middle": 3}

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
    "--beta",
    type=int,
    default=0.5,
    metavar="S",
    help="cost of rebalancing (default: 0.5)",
)

# Model parameters
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
    "--no_cuda", 
    type=int, 
    default=1,
    help="disables CUDA training",
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
    "--memory_path",
    type=str,
    default="nyc_brooklyn",
    help="name of the offline dataset file",
)
parser.add_argument(
    "--samples_buffer",
    type=int,
    default=10000,
    help="size of the replay buffer",
)
parser.add_argument(
    "--city",
    type=str,
    default="nyc_brooklyn",
    help="city to train on",
)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")

city = args.city


scenario = Scenario(
    json_file=f"data/scenario_{city}.json",
    demand_ratio=demand_ratio[city],
    json_hr=json_hr[city],
    sd=args.seed,
    json_tstep=args.json_tstep,
    tf=args.max_steps,
)

env = AMoD(scenario, 2, beta=beta[city], jitter=args.jitter, max_wait=args.maxt)

parser = GNNParser(
    env, T=6, json_file=f"data/scenario_{city}.json"
)

test_episodes = args.max_episodes  # set max number of training episodes
T = args.max_steps  # set episode length
epochs = trange(test_episodes)  # epoch iterator

obs_dim = (env.nregion, 10)

act_dim = (env.nregion, 2)
replay_buffer_rl = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=args.samples_buffer)

rewards = []
demands = []
costs = []
epsilon = 1e-6

for episode in epochs:
    episode_reward = 0
    episode_served_demand = 0
    episode_rebalancing_cost = 0

    obs = env.reset()
    done = False
    # while (not done):
    current_eps = []
    action_rl_price = [0]*env.nregion
    step = 0
    while not done:
        # take matching step (Step 1 in paper)
        if step > 0:
            obs1 = copy.deepcopy(o)
            action_rl_price = SDE(env)
        
        obs, paxreward, done, info, _, _ = env.match_step_simple(action_rl_price)
        o = parser.parse_obs(obs)

        episode_reward += paxreward
        open_requests = {}

        for i in env.region:
            open_requests[i] = len(env.queue[i])

        # DEFINE OPEN REQUESTS HERE
        if step > 0:
            rl_reward = paxreward + rebreward
            # Get combined action
            action_rl = np.array([action_rl_price,action_rl_reb]).T
            if step == T-1:
                replay_buffer_rl.store(obs1.x, action_rl, rl_reward, o.x, True)
            else:
                replay_buffer_rl.store(obs1.x, action_rl, rl_reward, o.x, False)

        rebAction, flows = DTV(env, open_requests)
        acc = {n: env.acc[n][env.time + 1] for n in env.region}
        action_rl_reb = flow_to_dist(flows, acc=acc)

        action_rl_reb = np.array([action_rl_reb[n] for n in env.region])
        non_zero_elements = action_rl_reb > 0
        zero_elements = action_rl_reb == 0
        num_non_zero = np.sum(non_zero_elements)
        num_zero = np.sum(zero_elements)

        # Subtract epsilon from non-zero elements and add to zero elements to get valid dirichlet distribution
        action_rl_reb[non_zero_elements] -= num_zero * epsilon / num_non_zero
        action_rl_reb[zero_elements] = epsilon
        action_rl_reb = list(action_rl_reb)

        new_obs, rebreward, done, info, _, _ = env.reb_step(rebAction)
        episode_reward += rebreward

        episode_served_demand += info["served_demand"]
        episode_rebalancing_cost += info["rebalancing_cost"]

        step += 1

    rewards.append(episode_reward)
    demands.append(episode_served_demand)
    costs.append(episode_rebalancing_cost)
    # stop episode if terminating conditions are met
    # Send current statistics to screen
    epochs.set_description(
        f"Episode {episode+1} | Reward: {episode_reward:.2f} | ServedDemand: {episode_served_demand:.2f} | Reb. Cost: {episode_rebalancing_cost}"
    )

print("Rewards (mean, std):", np.mean(rewards), np.std(rewards))
w = open(f"./Replaymemories/{args.city}_iql_heuristic_L.pkl", "wb")
pickle.dump(replay_buffer_rl, w)
w.close()
print("replay_buffer", replay_buffer_rl.size)
