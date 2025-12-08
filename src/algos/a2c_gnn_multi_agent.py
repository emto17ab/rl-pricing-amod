import numpy as np
import torch 
from torch import nn
import json
from torch_geometric.data import Data
from src.algos.layers import GNNCritic, GNNActor
import torch.nn.functional as F
from collections import namedtuple
from src.algos.reb_flow_solver import solveRebFlow
from src.misc.utils import dictsum, nestdictsum

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

class GNNParser:
    """
    Parser converting raw environment observations to agent inputs (s_t).
    """

    def __init__(self, env, T, scale_factor, agent_id, json_file=None, use_od_prices=False):
        super().__init__()
        self.env = env
        self.T = T
        self.s = scale_factor
        self.json_file = json_file
        self.agent_id = agent_id
        self.opponent_id = 1 - agent_id
        self.use_od_prices = use_od_prices
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

        # Price features (conditional on use_od_prices)
        if self.use_od_prices:
            # OD price matrices: shape [1, nregion, nregion] for each agent
            own_current_price = torch.tensor([[self.env.agent_price[self.agent_id][i, j].get(time, 0) * self.s 
                                              for j in self.env.region] 
                                             for i in self.env.region]).view(1, self.env.nregion, self.env.nregion).float()
            
            competitor_current_price = torch.tensor([[self.env.agent_price[self.opponent_id][i, j].get(time, 0) * self.s 
                                                     for j in self.env.region] 
                                                    for i in self.env.region]).view(1, self.env.nregion, self.env.nregion).float()
            
            # Concatenate: [1, 1+T+1+1+nregion+nregion, nregion]
            x = (
                torch.cat(
                    [current_avb, future_avb, queue_length, current_demand, own_current_price, competitor_current_price], 
                    dim=1
                )
                .squeeze(0)
                .view(1 + self.T + 1 + 1 + self.env.nregion + self.env.nregion, self.env.nregion)
                .T
            )
        else:
            # Aggregated prices: shape [1, 1, nregion]
            own_current_price = torch.tensor([sum([(self.env.agent_price[self.agent_id][i, j][time])* self.s for j in self.env.region]) for i in self.env.region]).view(1, 1, self.env.nregion).float()

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
        actor_clip,    # gradient clipping value for actor
        critic_clip,   # gradient clipping value for critic
        scale_factor,
        agent_id,
        json_file,
        use_od_prices,
        eps=np.finfo(np.float32).eps.item(),
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
                                   agent_id=agent_id, use_od_prices=use_od_prices)

        # Set learning rates
        self.p_lr = p_lr
        self.q_lr = q_lr

        # Inialize the optimizers
        self.optimizers = self.configure_optimizers()

        # Set gamma parameter
        self.gamma = gamma

        # Set gradient clipping values
        self.actor_clip = actor_clip
        self.critic_clip = critic_clip

        # Action & reward buffer
        self.saved_actions = []
        self.rewards = []
        self.concentration_history = []  # Track concentration parameters per step
        
        self.to(self.device)

    def parse_obs(self, obs):
        state = self.obs_parser.parse_obs(obs)
        return state
    
    # Combines select action and forward steps of actor and critic
    def select_action(self, obs, deterministic=False, return_concentration=False):
        # Parse the observation to get the graph data
        state = self.parse_obs(obs).to(self.device)

        # Forward pass through actor network to get action, log probability, concentration, and entropy
        a, logprob, concentration, entropy = self.actor(state, deterministic=deterministic)
        
        # Forward pass through critic network to get state value estimate
        value = self.critic(state)
        
        # Only save actions for training (when not deterministic)
        if not deterministic:
            # Store all relevant info for diagnostic logging
            self.last_concentration = concentration.detach()
            self.last_action = a.detach()
            self.last_value = value.detach()
            self.last_log_prob = logprob.detach() if logprob is not None else None
            
            # Track concentration for episode-level statistics
            self.concentration_history.append(concentration.detach().cpu().numpy())
            
            self.saved_actions.append(SavedAction(logprob, value))
    
        # Convert to numpy array
        action_np = a.detach().cpu().numpy()
        
        # Handle different action shapes based on mode:
        # Mode 0 & 1: shape [nregion, 1] -> flatten to [nregion]
        # Mode 2: shape [nregion, 2] -> keep as 2D [[price, reb], ...]
        if action_np.shape[-1] == 1:
            # Mode 0 or 1: squeeze last dimension to get 1D array
            action_array = action_np.squeeze(-1)
        else:
            # Mode 2: return 2D array as-is
            action_array = action_np
        
        # Return the action and optionally concentration parameter and log probability
        if return_concentration:
            concentration_value = concentration.detach().cpu().numpy()
            logprob_value = logprob.item() if logprob is not None else None
            return action_array, concentration_value, logprob_value
        else:
            return action_array

    def training_step(self):
        R = 0
        saved_actions = self.saved_actions
        policy_losses = [] # list to save actor (policy) loss
        value_losses = [] # list to save critic (value) loss
        returns = [] # list to save the true values

        # Normalize rewards (same as single agent)
        scaled_rewards = [(r - np.mean(self.rewards)) / (np.std(self.rewards) + self.eps) for r in self.rewards]
        
        # calculate the true value using scaled rewards
        for r in scaled_rewards[::-1]:
            # calculate the discounted value
            R = r + self.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns).to(self.device)
        
        # Calculate advantages
        advantages = []
        
        for (log_prob, value), R in zip(saved_actions, returns):
            advantage = R - value.item()
            advantages.append(advantage)
        
        # Calculate advantage statistics for logging
        advantages_tensor = torch.tensor(advantages)
        adv_mean = advantages_tensor.mean().item()
        adv_std = advantages_tensor.std().item()
        
        # Calculate losses using advantages (no normalization of advantages)
        for i, ((log_prob, value), R) in enumerate(zip(saved_actions, returns)):
            # Policy loss is just the advantage term
            policy_loss = -log_prob * advantages[i]
            policy_losses.append(policy_loss)
            
            # Value loss unchanged
            value_losses.append(F.smooth_l1_loss(value, torch.tensor([R]).to(self.device)))
        
        # Actor loss is the policy loss
        a_loss = torch.stack(policy_losses).mean()
        
        # take gradient steps
        self.optimizers['a_optimizer'].zero_grad()
        a_loss.backward()
        
        # Gradient clipping for actor
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.actor_clip)
        self.optimizers['a_optimizer'].step()
        
        self.optimizers['c_optimizer'].zero_grad()
        v_loss = torch.stack(value_losses).mean()
        v_loss.backward()
        
        # Gradient clipping for critic
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.critic_clip)
        self.optimizers['c_optimizer'].step()
        
        # reset rewards and action buffer
        del self.rewards[:]
        del self.saved_actions[:]
        del self.concentration_history[:]  # Clear concentration history

        # Return metrics for logging (same as single agent)
        return {
            "actor_grad_norm": actor_grad_norm.item(),
            "critic_grad_norm": critic_grad_norm.item(),
            "actor_loss": a_loss.item(),
            "critic_loss": v_loss.item(),
            "advantage_mean": adv_mean,
            "advantage_std": adv_std,
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
    