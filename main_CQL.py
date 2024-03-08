from __future__ import print_function
import argparse
from tqdm import trange
import numpy as np
import torch
from src.envs.amod_env import Scenario, AMoD
from src.algos.cql import SAC
from src.algos.reb_flow_solver import solveRebFlow
from src.misc.utils import dictsum, nestdictsum
import random
import json, pickle
from torch_geometric.data import Data, Batch
import copy
import logging


def return_to_go(rewards):
    """
    Calculate the return-to-go for a given trajectory.
    """
    gamma = 0.97
    return_to_go = [0] * len(rewards)
    prev_return = 0
    for i in range(len(rewards)):
        return_to_go[-i - 1] = rewards[-i - 1] + gamma * prev_return
        prev_return = return_to_go[-i - 1]
    return np.array(return_to_go, dtype=np.float32)

class PairData(Data):
    def __init__(
        self,
        edge_index_s=None,
        x_s=None,
        reward=None,
        action=None,
        mc_returns=None,
        edge_index_t=None,
        x_t=None,
    ):
        super().__init__()
        self.edge_index_s = edge_index_s
        self.x_s = x_s
        self.reward = reward
        self.action = action
        self.mc_returns = mc_returns
        self.edge_index_t = edge_index_t
        self.x_t = x_t

    def __inc__(self, key, value, *args, **kwargs):
        if key == "edge_index_s":
            return self.x_s.size(0)
        if key == "edge_index_t":
            return self.x_t.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)


class ReplayData:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, device, rew_scale):
        self.device = device
        self.data_list = []
        self.rew_scale = rew_scale
        self.episode_data = {}
        self.episode_data["obs"] = []
        self.episode_data["act"] = []
        self.episode_data["rew"] = []
        self.episode_data["obs2"] = []

    def create_dataset(self, edge_index, memory_path, size=60000, st=False, sc=False, mc=False):
        """
        edge_index: Adjaency matrix of the graph
        memory_path: Path to the replay memory
        size: Size of the replay memory
        st: Standardize the rewards
        sc: min-max scaling of rewards
        """
        w = open(f"Replaymemories/{memory_path}.pkl", "rb")

        replay_buffer = pickle.load(w)
        data = replay_buffer.sample_all(size,mc)

        if st:
            mean = data["rew"].mean()
            std = data["rew"].std()
            data["rew"] = (data["rew"] - mean) / (std + 1e-16)
        elif sc:
            data["rew"] = (data["rew"] - data["rew"].min()) / (
                data["rew"].max() - data["rew"].min()
            )
        if mc:
            (state_batch, action_batch, reward_batch, next_state_batch, mc_returns) = (
                data["obs"],
                data["act"],
                args.rew_scale * data["rew"],
                data["obs2"],
                args.rew_scale * data["mc_returns"],
            )
            for i in range(len(state_batch)):
                self.data_list.append(
                    PairData(
                        edge_index,
                        state_batch[i],
                        reward_batch[i],
                        action_batch[i],
                        mc_returns[i],
                        edge_index,
                        next_state_batch[i],
                    )
                )
        else:
            (state_batch, action_batch, reward_batch, next_state_batch) = (
                data["obs"],
                data["act"],
                args.rew_scale * data["rew"],
                data["obs2"],
            )
            for i in range(len(state_batch)):
                self.data_list.append(
                    PairData(
                        edge_index,
                        state_batch[i],
                        reward_batch[i],
                        action_batch[i],
                        None,
                        edge_index,
                        next_state_batch[i],
                    )
                )       

    def store(self, data1, action, reward, data2):
        self.data_list.append(
            PairData(
                data1.edge_index,
                data1.x,
                torch.as_tensor(reward),
                torch.as_tensor(action),
                data2.edge_index,
                data2.x,
            )
        )
        self.rewards.append(reward)

    def size(self):
        return len(self.data_list)

    def sample_batch(self, batch_size=32, return_list=False):
        data = random.sample(self.data_list, batch_size)
        if return_list:
            return data
        else:
            return Batch.from_data_list(data, follow_batch=["x_s", "x_t"]).to(
                self.device
            )
        
    def store_episode_data(self, obs, action, reward, obs2, terminal=False):
        """
        store new transitions
        """
        self.episode_data["obs"].append(obs.x)
        self.episode_data["act"].append(torch.as_tensor(action))
        self.episode_data["rew"].append(torch.as_tensor(reward))
        self.episode_data["obs2"].append(obs2.x)
        if terminal:
            mc_returns = return_to_go(self.episode_data["rew"])
            for i in range(len(self.episode_data["obs"])):
                self.data_list.append(
                    PairData(
                        edge_index,
                        self.episode_data["obs"][i],
                        args.rew_scale * self.episode_data["rew"][i],
                        self.episode_data["act"][i],
                        args.rew_scale * torch.as_tensor(mc_returns[i]),
                        edge_index,
                        self.episode_data["obs2"][i],
                    )
                )

            self.episode_data["obs"] = []
            self.episode_data["act"] = []
            self.episode_data["rew"] = []
            self.episode_data["obs2"] = []

    def to_buffer(self):
        # Transform data buffer to replay buffer
        if self.size == 0:
            raise ValueError("Unable to transform empty data buffer to replay buffer")
        obs_dim = self.data_list[0].x_s.shape
        act_dim = self.data_list[0].action.shape
        size = self.size()
        buffer = ReplayBuffer(obs_dim, act_dim, size)

        for i in range(size):
            buffer.obs_buf[i] = self.data_list[i].x_s
            buffer.obs2_buf[i] = self.data_list[i].x_t
            buffer.act_buf[i] = self.data_list[i].action
            buffer.rew_buf[i] = self.data_list[i].reward
            buffer.mc_returns[i] = self.data_list[i].mc_returns

        buffer.ptr = size
        buffer.size = size
        
        return buffer


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    Necessary to load the offline datasets
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(combined_shape(
            size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(combined_shape(
            size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(
            size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.mc_returns = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def sample_all(self, samples, mc=False):
        if samples > self.size:
            samples = self.ptr
        if mc:
            batch = dict(
                obs=self.obs_buf[:samples],
                obs2=self.obs2_buf[:samples],
                act=self.act_buf[:samples],
                rew=self.rew_buf[:samples],
                mc_returns=self.mc_returns[:samples],
            )
        else:
            batch = dict(
                obs=self.obs_buf[:samples],
                obs2=self.obs2_buf[:samples],
                act=self.act_buf[:samples],
                rew=self.rew_buf[:samples],
            )            
        return {k: torch.as_tensor(v) for k, v in batch.items()}


parser = argparse.ArgumentParser(description="Cal-CQL-GNN")

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

test_tstep = {'san_francisco': 3, 'nyc_brooklyn': 4, 'shenzhen_downtown_west': 3, 'nyc_man_middle': 3, 'nyc_man_south': 3}

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
    help='rebalancing mode. (0:manul, 1:pricing, 2:both. default 2)',
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
    "--test", 
    type=bool, 
    default=False, 
    help="activates test mode for agent evaluation"
)
parser.add_argument(
    "--collection", 
    type=bool, 
    default=False, 
    help="activates data collection mode"
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
    "--no_cuda", 
    type=int, 
    default=1,
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
    default="CQL",
    help="name of checkpoint file to save/load (default: CQL)",
)
parser.add_argument(
    "--clip",
    type=int,
    default=500,
    help="clip value for gradient clipping (default: 500)",
)
parser.add_argument(
    "--q_lag",
    type=int,
    default=1,
    help="update frequency of Q target networks (default: 10)",
)
parser.add_argument(
    "--memory_path",
    type=str,
    default="nyc_brooklyn",
    help="name of the offline dataset file",
)
parser.add_argument(
    "--min_q_weight",
    type=float,
    default=5,
    help="CQL coefficient (eta in paper)",
)
parser.add_argument(
    "--samples_buffer",
    type=int,
    default=10000,
    help="size of the replay buffer",
)
parser.add_argument(
    "--lagrange_thresh",
    type=float,
    default=-1,
    help="threshold for the lagrange tuning of entropy (default: -1 =disabled)",
)
parser.add_argument(
    "--city",
    type=str,
    default="nyc_brooklyn",
    help="city to train on",
)
parser.add_argument(
    "--rew_scale",
    type=float,
    default=0.1,
    help="reward scaling factor (default: 0.1)",
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
    "--st",
    type=bool,
    default=False,
    help="standardize the rewards (default: False)",
)
parser.add_argument(
    "--sc",
    type=bool,
    default=False,
    help="min-max scale the rewards (default: False)",
)
parser.add_argument(
    "--enable_calql",
    type=bool,
    default=False,
    help="enable the Cal-CQL (default: False)",
)
parser.add_argument(
    "--load_ckpt",
    type=str,
    default="Cal_CQL",
    help="name of the checkpoint file to load for online-fine-tuning(default: SAC)",
)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
city = args.city

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

if args.collection:
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


    env = AMoD(scenario, args.mode, beta=beta[city], jitter=args.jitter, max_wait=args.maxt)

    model = SAC(
        env=env,
        input_size=10,
        hidden_size=args.hidden_size,
        p_lr=1e-4,
        q_lr=3e-4,
        alpha=args.alpha,
        batch_size=args.batch_size,
        use_automatic_entropy_tuning=False,
        lagrange_thresh=args.lagrange_thresh,
        device=device,
        json_file=f'data/scenario_{city}.json',
        min_q_version=3,
        mode=args.mode,
        q_lag=args.q_lag
    ).to(device)

    with open(f"data/scenario_{city}.json", "r") as file:
        data = json.load(file)

    edge_index = torch.vstack(
        (
            torch.tensor([edge["i"]
                         for edge in data["topology_graph"]]).view(1, -1),
            torch.tensor([edge["j"]
                         for edge in data["topology_graph"]]).view(1, -1),
        )
    ).long()

    replay_buffer = ReplayData(device=device, rew_scale=args.rew_scale)

    print("load model")
    model.load_checkpoint(path=f"ckpt/{args.checkpoint_path}.pth")

    episodes = args.max_episodes  # set max number of data collection episodes
    T = args.max_steps  # set episode length
    epochs = trange(episodes)  # epoch iterator
    model.eval()

    for episode in epochs:

        episode_reward = 0
        episode_served_demand = 0
        episode_rebalancing_cost = 0
        obs = env.reset()

        action_rl = [0]*env.nregion        
        done = False
        step = 0
        while not done:

            if step > 0:
                obs1 = copy.deepcopy(o)

            if env.mode == 0:
                obs, paxreward, done, info, _, _ = env.match_step_simple()
                # obs, paxreward, done, info, _, _ = env.pax_step(
                #                 CPLEXPATH=args.cplexpath, directory=args.directory, PATH="scenario_san_francisco4"
                #             )

                o = model.parse_obs(obs=obs)
                episode_reward += paxreward
                if step > 0:
                    # store transition in memroy
                    rl_reward = paxreward + rebreward
                    if step == T - 1:
                        replay_buffer.store_episode_data(
                            obs1, action_rl, rl_reward, o, terminal=True
                        )
                    else:
                        replay_buffer.store_episode_data(
                            obs1, action_rl, rl_reward, o, terminal=False
                        )                        

                action_rl = model.select_action(o, deterministic=False)

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

                o = model.parse_obs(obs=obs)

                episode_reward += paxreward
                if step > 0:
                    # store transition in memroy
                    rl_reward = paxreward
                    if step == T - 1:
                        replay_buffer.store_episode_data(
                            obs1, action_rl, rl_reward, o, terminal=True
                        )
                    else:
                        replay_buffer.store_episode_data(
                            obs1, action_rl, rl_reward, o, terminal=False
                        )

                action_rl = model.select_action(o)  

                env.matching_update()
            elif env.mode == 2:
                obs, paxreward, done, info, _, _ = env.match_step_simple(action_rl)

                o = model.parse_obs(obs=obs)
                episode_reward += paxreward
                if step > 0:
                    # store transition in memroy
                    rl_reward = paxreward + rebreward
                    if step == T - 1:
                        replay_buffer.store_episode_data(
                            obs1, action_rl, rl_reward, o, terminal=True
                        )
                    else:
                        replay_buffer.store_episode_data(
                            obs1, action_rl, rl_reward, o, terminal=False
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
            
            episode_served_demand += info["served_demand"]
            episode_rebalancing_cost += info["rebalancing_cost"]

            step += 1
        # Send current statistics to screen
        epochs.set_description(
            f"Episode {episode+1} | Reward: {episode_reward:.2f} | ServedDemand: {episode_served_demand:.2f} | Reb. Cost: {episode_rebalancing_cost}"
        )

    # Store buffer
    dataset = replay_buffer.to_buffer()
    pickle.dump(dataset, open(f'Replaymemories/{args.city}.pkl', 'wb'))
elif not args.test:
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


    env = AMoD(scenario, args.mode, beta=beta[city], jitter=args.jitter, max_wait=args.maxt)

    model = SAC(
        env=env,
        input_size=10,
        hidden_size=args.hidden_size,
        p_lr=1e-4,
        q_lr=3e-4,
        alpha=args.alpha,
        batch_size=args.batch_size,
        use_automatic_entropy_tuning=False,
        lagrange_thresh=args.lagrange_thresh,
        min_q_weight=args.min_q_weight,
        device=device,
        json_file=f'data/scenario_{city}.json',
        min_q_version=3,
        mode=args.mode,
        q_lag=args.q_lag
    ).to(device)

    with open(f"data/scenario_{city}.json", "r") as file:
        data = json.load(file)

    edge_index = torch.vstack(
        (
            torch.tensor([edge["i"]
                         for edge in data["topology_graph"]]).view(1, -1),
            torch.tensor([edge["j"]
                         for edge in data["topology_graph"]]).view(1, -1),
        )
    ).long()

    #######################################
    ############# Training Loop#############
    #######################################
    # Initialize dataset
    Dataset = ReplayData(device=device, rew_scale=args.rew_scale)
    Dataset.create_dataset(
        edge_index=edge_index,
        memory_path=args.memory_path,
        size=args.samples_buffer,
        st=args.st,
        sc=args.sc,
        mc=args.enable_calql
    )

    train_episodes = args.max_episodes  # set max number of training episodes
    T = args.max_steps  # set episode length
    training_steps = train_episodes * 20
    best_reward = -np.inf  # set best reward
    model.train()  # set model in train mode

    # Metrics
    loss_log = {"BQ1":[],"RQ1":[],"BQ2":[],"RQ2":[]}

    logging.info("Training start")
    for step in range(training_steps):

        batch = Dataset.sample_batch(args.batch_size)
        log = model.update(data=batch, conservative=True,
                     enable_calql=args.enable_calql)
        
        if step % 400 == 0:
            test_reward, test_served_demand, test_reb_cost = model.test_agent(10, env, args.cplexpath, args.directory)
            if test_reward > best_reward:
                best_reward = test_reward
                model.save_checkpoint(path=f"ckpt/offline/" + args.checkpoint_path + "_test.pth")
            logging.info(f"Training step {step} | Reward: {test_reward} | Q1 loss1: {log['Bellman loss Q1']} | Q1 loss2: {log['Regularizor loss Q1']} | Q1:{log['Q1']} | Q2 loss1: {log['Bellman loss Q2']} | Q2 loss2: {log['Regularizor loss Q2']} | Q2:{log['Q2']}")
        
        loss_log["BQ1"].append(log['Bellman loss Q1'])
        loss_log["RQ1"].append(log['Regularizor loss Q1'])
        loss_log["BQ2"].append(log['Bellman loss Q2'])
        loss_log["RQ2"].append(log['Regularizor loss Q2'])

        model.save_checkpoint(path=f"ckpt/offline/" + args.checkpoint_path + ".pth")

    with open(f"{args.directory}/{city}_loss_mode{args.mode}_w{args.min_q_weight}_{train_episodes}_L.pickle", 'wb') as f:
        pickle.dump(loss_log, f) 

else:
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


    env = AMoD(scenario, args.mode, beta=beta[city], jitter=args.jitter, max_wait=args.maxt)

    model = SAC(
        env=env,
        input_size=10,
        hidden_size=args.hidden_size,
        p_lr=1e-4,
        q_lr=3e-4,
        alpha=args.alpha,
        batch_size=args.batch_size,
        use_automatic_entropy_tuning=False,
        lagrange_thresh=args.lagrange_thresh,
        device=device,
        json_file=f'data/scenario_{city}.json',
        min_q_version=3,
        mode=args.mode,
        q_lag=args.q_lag
    ).to(device)

    print("load model")
    model.load_checkpoint(path=f"ckpt/{args.checkpoint_path}.pth")

    test_episodes = args.max_episodes  # set max number of testing episodes
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
                # obs, paxreward, done, info, _, _ = env.pax_step(
                #                 CPLEXPATH=args.cplexpath, directory=args.directory, PATH="scenario_san_francisco4"
                #             )

                o = model.parse_obs(obs=obs)
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

                o = model.parse_obs(obs=obs)

                episode_reward += paxreward

                action_rl = model.select_action(o, deterministic=True)  

                env.matching_update()
            elif env.mode == 2:
                obs, paxreward, done, info, _, _ = env.match_step_simple(action_rl)

                o = model.parse_obs(obs=obs)
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
    # np.save(f"{args.directory}/{city}_actions_mode{args.mode}.npy", np.array(actions_step))
    # np.save(f"{args.directory}/{city}_queue_mode{args.mode}.npy", np.array(queue_steps))
    # np.save(f"{args.directory}/{city}_served_mode{args.mode}.npy", np.array([demands,arrivals]))
    # if env.mode != 1: 
        # np.save(f"{args.directory}/{city}_cost_mode{args.mode}.npy", np.array(rebalancing_cost_steps))
    #     with open(f"{args.directory}/{city}_reb_mode{args.mode}.pickle", 'wb') as f:
    #         pickle.dump(reb_steps, f)
    #     with open(f"{args.directory}/{city}_reb_ori_mode{args.mode}.pickle", 'wb') as f:
    #         pickle.dump(reb_ori_steps, f)

    # with open(f"{args.directory}/{city}_pax_mode{args.mode}.pickle", 'wb') as f:
    #     pickle.dump(pax_steps, f)
    # with open(f"{args.directory}/{city}_pax_wait_mode{args.mode}.pickle", 'wb') as f:
    #     pickle.dump(pax_wait, f)                     
    
    # with open(f"{args.directory}/{city}_demand_ori_mode{args.mode}.pickle", 'wb') as f:
    #     pickle.dump(demand_original_steps, f)
    # with open(f"{args.directory}/{city}_price_ori_mode{args.mode}.pickle", 'wb') as f:
    #     pickle.dump(price_original_steps, f)

    # with open(f"{args.directory}/{city}_demand_scaled_mode{args.mode}.pickle", 'wb') as f:
    #     pickle.dump(demand_scaled_steps, f)    
    # with open(f"{args.directory}/{city}_acc_mode{args.mode}.pickle", 'wb') as f:
    #     pickle.dump(available_steps, f)

    print("Rewards (mean, std):", np.mean(rewards), np.std(rewards))
    print("Served demand (mean, std):", np.mean(demands), np.std(demands))
    print("Rebalancing cost (mean, std):", np.mean(costs), np.std(costs))
    print("Waiting time (mean, std):", np.mean(waiting_steps), np.std(waiting_steps))
    print("Queue length (mean, std):", np.mean(queue_steps), np.std(queue_steps))
    print("Arrivals (mean, std):", np.mean(arrivals), np.std(arrivals))
    print("Rebalancing trips (mean, std):", np.mean(reb_num), np.std(reb_num))
    if args.mode != 0:
        print("Price scalar (mean, std):", np.mean(price_mean), np.std(price_mean))