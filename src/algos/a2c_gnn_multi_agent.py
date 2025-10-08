import numpy as np
import torch 
from torch import nn
import json
from torch_geometric.data import Data
from src.algos.layers2 import GNNCritic, GNNActor
import torch.nn.functional as F
from collections import namedtuple
from src.algos.reb_flow_solver import solveRebFlow
from src.misc.utils import dictsum, nestdictsum

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

class GNNParser:
    """
    Parser converting raw environment observations to agent inputs (s_t).
    """

    def __init__(self, env, T, scale_factor, agent_id, json_file=None):
        super().__init__()
        self.env = env
        self.T = T
        self.s = scale_factor
        self.json_file = json_file
        self.agent_id = agent_id
        self.opponent_id = 1 - agent_id
        if self.json_file is not None:
            with open(json_file, "r") as file:
                self.data = json.load(file)
    
    def parse_obs(self, obs):
        acc, time, dacc, demand = obs

        # Current availability at t+1
        current_avb = torch.tensor([acc[n][time + 1] * self.s for n in self.env.region]).view(1, 1, self.env.nregion).float()

        # Future availability from t+2 to t+T
        future_avb = torch.tensor([[(acc[n][time + 1] + dacc[n][t]) * self.s for n in self.env.region] for t in range(time + 1, time + self.T + 1)]).view(1, self.T, self.env.nregion).float()

        # Queue length 
        queue_length = torch.tensor([len(self.env.agent_queue[self.agent_id][n]) * self.s for n in self.env.region]).view(1, 1, self.env.nregion).float()

        # Current demand at t
        ############# LOOOK AT THIS. SHOULD IT USE DEMAND from OBS OR self.env.demand ##########################
        current_demand = torch.tensor([sum([(demand[i, j][time])* self.s for j in self.env.region]) for i in self.env.region]).view(1, 1, self.env.nregion).float()

        # Own current price at t
        own_current_price = torch.tensor([sum([(self.env.agent_price[self.agent_id][i, j][time])* self.s for j in self.env.region]) for i in self.env.region]).view(1, 1, self.env.nregion).float()

        # Competitor current price at t
        competitor_current_price = torch.tensor([sum([(self.env.agent_price[self.opponent_id][i, j][time])* self.s for j in self.env.region]) for i in self.env.region]).view(1, 1, self.env.nregion).float()

        x = (
            torch.cat(
                [current_avb, future_avb, queue_length, current_demand, own_current_price, competitor_current_price], 
                dim=1
            )
            .squeeze(0)
            .view(1 + self.T + 1 + 1 + 1 + 1, self.env.nregion)
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

class A2C(nn.Module):
    """
    Advantage Actor Critic algorithm for the AMoD control problem. 
    """
    def __init__(
        self,
        env,
        input_size,
        device,
        hidden_size,
        T,
        mode,
        gamma,
        p_lr,   # actor learning rate
        q_lr,   # critic learning rate
        clip,    # gradient clipping value
        scale_factor,
        agent_id,
        json_file = None,
        eps=np.finfo(np.float32).eps.item()
    ):
        
        super(A2C, self).__init__()
        # Set the environment and related attributes
        self.env = env
        self.act_dim = env.nregion
        self.agent_id = agent_id

        # Set the mode
        self.mode = mode

        # Set very small number to avoid division by zero
        self.eps = eps

        # Set the input size and hidden size
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Specify the device
        self.device = device

        # Set the Actor and Critic networks
        self.actor = GNNActor(self.input_size, self.hidden_size, act_dim=self.act_dim, mode=mode)
        self.critic = GNNCritic(self.input_size, self.hidden_size, act_dim=self.act_dim)
    
        # Set the observation parser
        self.obs_parser = GNNParser(self.env, T=T, json_file=json_file, scale_factor=scale_factor,
                                   agent_id=agent_id)

        # Set learning rates
        self.p_lr = p_lr
        self.q_lr = q_lr

        # Inialize the optimizers
        self.optimizers = self.configure_optimizers()

        # Set gamma parameter
        self.gamma = gamma

        # Set gradient clipping value
        self.clip = clip

        # Action & reward buffer
        self.saved_actions = []
        self.rewards = []
        self.to(self.device)

    def parse_obs(self, obs):
        state = self.obs_parser.parse_obs(obs)
        return state
    
    # Combines select action and forward steps of actor and critic
    def select_action(self, obs, deterministic=False):
        # Parse the observation to get the graph data
        state = self.parse_obs(obs).to(self.device)

        # Forward pass through actor network to get action and log probability        
        a, logprob = self.actor(state, deterministic=deterministic)
        
        # Forward pass through critic network to get state value estimate
        value = self.critic(state)
        
        # Only save actions for training (when not deterministic)
        if not deterministic:
            self.saved_actions.append(SavedAction(logprob, value))
    
        action_list = a.detach().cpu().numpy().tolist()
        
        # Return the action to be executed in the environment
        return action_list

    def training_step(self):
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
        returns = (returns - returns.mean()) / (returns.std() + self.eps)

        for (log_prob, value), R in zip(saved_actions, returns):
            advantage = R - value.item()

            # calculate actor (policy) loss with entropy regularization
            policy_losses.append(-log_prob * advantage)

            # calculate critic (value) loss using L1 smooth loss
            value_losses.append(F.smooth_l1_loss(value, torch.tensor([R]).to(self.device)))

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
    