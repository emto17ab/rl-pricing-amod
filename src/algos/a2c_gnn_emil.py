import numpy as np
import torch 
from torch import nn
import json
from torch_geometric.data import Data
from src.algos.layers import GNNOrigin, GNNOD, MLPOD, MLPOrigin, GNNCritic, GNNActor_Emil
import torch.nn.functional as F
from collections import namedtuple

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

class GNNParser:
    """
    Parser converting raw environment observations to agent inputs (s_t).
    """

    def __init__(self, env, device, T, json_file, scale_factor=0.01):
        super().__init__()
        self.env = env
        self.T = T
        self.s = scale_factor
        self.json_file = json_file
        self.device = device
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
        # Based on the topology_graph in the data specifices how the nodes are connected.
        # List 1 start nodes, list 2 destination nodes        
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
            ).long().to(self.device)
        else:
            edge_index = torch.cat(
                (
                    torch.arange(self.env.nregion).view(1, self.env.nregion),
                    torch.arange(self.env.nregion).view(1, self.env.nregion),
                ),
                dim=0,
            ).long().to(self.device)
        # In x each row represents a region with the
        # 1. value being the curr availability,
        # 2. - (1+T) value the estimated availability for that region for time step t and T ahead
        # T+2 queue length at that node
        # T+3 current demand at that node
        # T+4 current price of that node

        # edge_index contains two lists 
        # first list start nodes
        # second list destination nodes for the start nodes in list 1
        data = Data(x, edge_index)
        return data

class A2C(nn.Module):
    """
    Advantage Actor Critic algorithm for the AMoD control problem. 
    """
    def __init__(
        self,
        env,
        input_size,
        device,
        hidden_size=32,
        T = 6,
        json_file = None,
        scale_factor = 0.01,  
        eps=np.finfo(np.float32).eps.item(),
        price_version = "GNN-origin",
        mode = 1,
        gamma = 0.97,
        alpha = 0.01,  # entropy coefficient
        p_lr = 1e-3,   # actor learning rate
        q_lr = 1e-3,   # critic learning rate
        clip = 10      # gradient clipping value

    ):
        
        super(A2C, self).__init__()
        # Set the environment and related attributes
        self.env = env
        self.act_dim = env.nregion

        # Set the observation parser
        self.obs_parser = GNNParser(self.env, device, T = T, json_file = json_file, scale_factor= scale_factor)

        # Set very small number to avoid division by zero
        self.eps = eps

        # Set gamma parameter
        self.gamma = gamma

        # set the mode 
        self.mode = mode

        # Set entropy coefficient
        self.alpha = alpha

        # Set learning rates
        self.p_lr = p_lr
        self.q_lr = q_lr

        # Set gradient clipping value
        self.clip = clip

        # Set the input size and hidden size
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Specify the device
        self.device = device

        # Set the action dimension and other attributes
        self.price = price_version
        self.edges = None
        self.actor = GNNActor_Emil(self.input_size, self.hidden_size, act_dim=self.act_dim, mode=mode)
        """
        # Set price based on GNN and origin-based action (single price per origin)
        if price_version == 'GNN-origin':
            self.actor = GNNOrigin(self.input_size, self.hidden_size, act_dim=self.act_dim, mode=mode)
        # Set price based on GNN and OD-based action (individual price per OD pair)    
        elif price_version == 'GNN-od':
            self.edges = torch.zeros(len(env.region)**2,2).long().to(device)
            k = 0
            for i in env.region:
                for j in env.region:
                    self.edges[k,0] = i
                    self.edges[k,1] = j
                    k += 1
            self.actor = GNNOD(self.edges,self.input_size,self.hidden_size, act_dim=self.act_dim, mode=mode)
        # Set price based on MLP and OD-based action (individual price per OD pair)
        elif price_version == 'MLP-od':
            self.actor = MLPOD(self.input_size,self.hidden_size, act_dim=self.act_dim, mode=mode)
        # Set price based on MLP and origin-based action (single price per origin)
        elif price_version == 'MLP-origin':
            self.actor = MLPOrigin(self.input_size,self.hidden_size, act_dim=self.act_dim, mode=mode)
        else:
            raise ValueError("Price version only allowed among 'GNN-origin', 'GNN-od', 'MLP-origin', and 'MLP-od'.")
        """
        # Initalize the critic
        self.critic = GNNCritic(self.input_size, self.hidden_size)

        # Inialize the optimizers
        self.optimizers = self.configure_optimizers()

        # action & reward buffer
        self.saved_actions = []
        self.rewards = []
        self.to(self.device)

    
    # Combines select action and forward steps of actor and critic
    def select_action(self, obs):
        # Parse the observation to get the graph data
        data = self.parse_obs(obs).to(self.device)

        # Forward pass through actor network to get action and log probability        
        a, logprob = self.actor(data)
        
        # Forward pass through critic network to get state value estimate
        # Note: For A2C, we don't pass actions to critic since it estimates V(s), not Q(s,a)
        value = self.critic(data)
        
        self.saved_actions.append(SavedAction(logprob, value))
    
        action_list = a.detach().cpu().numpy().tolist()
        
        # Return the action to be executed in the environment
        return action_list

    def training_step(self):
        # Check if we have any saved actions to train on
        if not self.saved_actions or not self.rewards:
            return {
                "actor_grad_norm": 0.0,
                "critic_grad_norm": 0.0,
                "actor_loss": 0.0,
                "critic_loss": 0.0
            }
        
        R = 0
        saved_actions = self.saved_actions
        policy_losses = [] # list to save actor (policy) loss
        value_losses = [] # list to save critic (value) loss
        returns = [] # list to save the true values

        # calculate the true value using rewards returned from the environment
        for r in self.rewards[::-1]:
            # calculate the discounted value
            R = r + self.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns).to(self.device)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + self.eps)
        # If only one return, don't normalize to avoid division by zero
        elif len(returns) == 1:
            returns = returns - returns.mean()  # Just center it

        for (log_prob, value), R in zip(saved_actions, returns):
            advantage = R - value.item()

            # calculate actor (policy) loss with entropy regularization
            policy_losses.append(-log_prob * advantage)

            # calculate critic (value) loss using L1 smooth loss
            value_losses.append(F.smooth_l1_loss(value, R.unsqueeze(0)))

        # take gradient steps with clipping
        self.optimizers['a_optimizer'].zero_grad()
        a_loss = torch.stack(policy_losses).sum()
        a_loss.backward()
        # Gradient clipping for actor
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.clip)
        self.optimizers['a_optimizer'].step()
        
        self.optimizers['c_optimizer'].zero_grad()
        v_loss = torch.stack(value_losses).sum()
        v_loss.backward()
        # Gradient clipping for critic
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.clip)
        self.optimizers['c_optimizer'].step()
        
        # reset rewards and action buffer
        del self.rewards[:]
        del self.saved_actions[:]

        # Return metrics for logging
        return {
            "actor_grad_norm": actor_grad_norm.item(),
            "critic_grad_norm": critic_grad_norm.item(),
            "actor_loss": a_loss.item(),
            "critic_loss": v_loss.item()
        }

    def parse_obs(self, obs):
        state = self.obs_parser.parse_obs(obs)
        return state
    
    def configure_optimizers(self):
        # Define dictionary to hold the optimizers
        optimizers = dict()

        # Define the optimizers for actor and critic
        actor_params = list(self.actor.parameters())
        critic_params = list(self.critic.parameters())
        optimizers["a_optimizer"] = torch.optim.Adam(actor_params, lr=self.p_lr)
        optimizers["c_optimizer"] = torch.optim.Adam(critic_params, lr=self.q_lr)
        return optimizers

    def save_checkpoint(self, path='ckpt.pth'):
        checkpoint = dict()
        checkpoint['model'] = self.state_dict()
        for key, value in self.optimizers.items():
            checkpoint[key] = value.state_dict()
        torch.save(checkpoint, path)
        
    def load_checkpoint(self, path="ckpt.pth"):
        checkpoint = torch.load(path, map_location=self.device)
        model_dict = self.state_dict()
        pretrained_dict = {
            k: v for k, v in checkpoint["model"].items() if k in model_dict
        }
        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)
        for key, value in self.optimizers.items():
            self.optimizers[key].load_state_dict(checkpoint[key])
    
    def log(self, log_dict, path='log.pth'):
        torch.save(log_dict, path)

    def test_agent(self, test_episodes, env, cplexpath, directory):
        """Test the trained agent and return average performance metrics."""
        from src.algos.reb_flow_solver import solveRebFlow
        from src.misc.utils import dictsum
        
        episode_reward = []
        episode_served_demand = []
        episode_rebalancing_cost = []
        
        for _ in range(test_episodes):
            eps_reward = 0
            eps_served_demand = 0
            eps_rebalancing_cost = 0
            obs = env.reset()
            action_rl = [0] * env.nregion
            done = False
            
            while not done:
                if env.mode == 0:
                    obs, paxreward, done, info, _, _ = env.match_step_simple()
                    eps_reward += paxreward

                    action_rl = self.select_action(obs, deterministic=True)
                    
                    # transform sample from Dirichlet into actual vehicle counts
                    desiredAcc = {
                        env.region[i]: int(
                            action_rl[i] * dictsum(env.acc, env.time + 1))
                        for i in range(len(env.region))
                    }
                    # solve minimum rebalancing distance problem
                    rebAction = solveRebFlow(
                        env, "scenario_san_francisco4", desiredAcc, cplexpath, directory
                    )
                    # Take rebalancing action in environment
                    _, rebreward, done, info, _, _ = env.reb_step(rebAction)
                    eps_reward += rebreward

                elif env.mode == 1:
                    obs, paxreward, done, info, _, _ = env.match_step_simple(action_rl)
                    eps_reward += paxreward
                    action_rl = self.select_action(obs)
                    env.matching_update()
                    
                elif env.mode == 2:
                    obs, paxreward, done, info, _, _ = env.match_step_simple(action_rl)
                    eps_reward += paxreward
                    action_rl = self.select_action(obs)

                    # transform sample from Dirichlet into actual vehicle counts
                    desiredAcc = {
                        env.region[i]: int(
                            action_rl[i][-1] * dictsum(env.acc, env.time + 1))
                        for i in range(len(env.region))
                    }
                    # solve minimum rebalancing distance problem
                    rebAction = solveRebFlow(
                        env, "scenario_san_francisco4", desiredAcc, cplexpath, directory
                    )
                    # Take rebalancing action in environment
                    _, rebreward, done, info, _, _ = env.reb_step(rebAction)
                    eps_reward += rebreward
                else:
                    raise ValueError("Only mode 0, 1, and 2 are allowed")

                eps_served_demand += info["served_demand"]
                eps_rebalancing_cost += info["rebalancing_cost"]

            episode_reward.append(eps_reward)
            episode_served_demand.append(eps_served_demand)
            episode_rebalancing_cost.append(eps_rebalancing_cost)

        return (
            np.mean(episode_reward),
            np.mean(episode_served_demand),
            np.mean(episode_rebalancing_cost),
        )
