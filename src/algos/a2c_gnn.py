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

SavedAction = namedtuple('SavedAction', ['log_prob', 'value', 'entropy'])

class GNNParser:
    """
    Parser converting raw environment observations to agent inputs (s_t).
    """

    def __init__(self, env, T, scale_factor, json_file=None, use_od_prices=False):
        super().__init__()
        self.env = env
        self.T = T
        self.s = scale_factor
        self.json_file = json_file
        self.use_od_prices = use_od_prices
        if self.json_file is not None:
            with open(json_file, "r") as file:
                self.data = json.load(file)

    def parse_obs(self, obs):
        acc, time, dacc, demand = obs

        # Current availability at t+1
        current_avb = torch.tensor([acc[n][self.env.time + 1] * self.s for n in self.env.region]).view(1, 1, self.env.nregion).float()

        # Future availability from t+2 to t+T
        future_avb = torch.tensor([[(acc[n][self.env.time + 1] + self.env.dacc[n][t])* self.s for n in self.env.region] for t in range(self.env.time + 1, self.env.time + self.T + 1)]).view(1, self.T, self.env.nregion).float()

        # Queue length
        queue_len = torch.tensor([len(self.env.queue[n]) * self.s for n in self.env.region]).view(1, 1, self.env.nregion).float()
        
        # Current demand at t
        current_demand = torch.tensor([sum([(demand[i, j][time])* self.s for j in self.env.region]) for i in self.env.region]).view(1, 1, self.env.nregion).float()

        if self.use_od_prices:
            # Current OD prices at t - shape [nregion, nregion] for OD-specific prices
            current_price = torch.tensor([[(self.env.price[i, j][time])* self.s for j in self.env.region] for i in self.env.region]).view(1, self.env.nregion, self.env.nregion).float()
            
            x = torch.cat((current_avb, future_avb, queue_len, current_demand, current_price), dim=1).squeeze(0).view(1 + self.T + 1 + 1 + self.env.nregion, self.env.nregion).T
        else:
            # Current price at t
            current_price = torch.tensor([sum([(self.env.price[i, j][time])* self.s for j in self.env.region]) for i in self.env.region]).view(1, 1, self.env.nregion).float()                   

            x = torch.cat((current_avb, future_avb, queue_len, current_demand, current_price), dim=1).squeeze(0).view(1 + self.T + 1 + 1 + 1, self.env.nregion).T
              
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
        json_file = None,
        use_od_prices = False,
        entropy_coef = 0.2,  # entropy regularization coefficient
        eps=np.finfo(np.float32).eps.item()
    ):
        
        super(A2C, self).__init__()
        # Set the environment and related attributes
        self.env = env
        self.act_dim = env.nregion

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
        self.obs_parser = GNNParser(self.env, T=T, json_file=json_file, scale_factor=scale_factor, use_od_prices=use_od_prices)

        # Set learning rates
        self.p_lr = p_lr
        self.q_lr = q_lr

        # Inialize the optimizers
        self.optimizers = self.configure_optimizers()

        # Set gamma parameter
        self.gamma = gamma
        
        # Set entropy coefficient
        self.entropy_coef = entropy_coef

        # Set gradient clipping values
        self.actor_clip = actor_clip
        self.critic_clip = critic_clip

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

        # Forward pass through actor network to get action, log probability, and entropy
        a, logprob, concentration, entropy = self.actor(state, deterministic=deterministic)
        
        # Forward pass through critic network to get state value estimate
        value = self.critic(state)
        
        # Only save actions for training (when not deterministic)
        if not deterministic:
            self.saved_actions.append(SavedAction(logprob, value, entropy))
    
        action_np = a.detach().cpu().numpy()

        # Handle different action shapes based on mode:
        # Mode 0 & 1: shape [nregion, 1] -> flatten to [nregion]
        # Mode 2: shape [nregion, 2] -> keep as 2D [[price, reb], ...]
        if action_np.shape[-1] == 1:
            # Mode 0 or 1: squeeze last dimension to get 1D array
            return action_np.squeeze(-1)
        else:
            # Mode 2: return 2D array as-is
            return action_np

    def training_step(self):
        R = 0
        saved_actions = self.saved_actions
        policy_losses = [] # list to save actor (policy) loss
        value_losses = [] # list to save critic (value) loss
        returns = [] # list to save the true values
        advantages_list = [] # list to track advantages
        entropies_list = [] # list to track entropies

        # calculate the true value using rewards returned from the environment
        for r in self.rewards[::-1]:
            # calculate the discounted value
            R = r + self.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + self.eps)

        for (log_prob, value, entropy), R in zip(saved_actions, returns):
            advantage = R - value.item()
            advantages_list.append(advantage)

            # calculate actor (policy) loss: negative log prob * advantage
            policy_losses.append(-log_prob * advantage)
            
            # Store entropy for tracking and regularization
            if entropy is not None:
                entropies_list.append(entropy)

            # calculate critic (value) loss using L1 smooth loss
            #value_losses.append(F.smooth_l1_loss(value, R.unsqueeze(0)))
            value_losses.append(F.smooth_l1_loss(value, torch.tensor([R]).to(self.device)))

        # Calculate mean policy loss
        policy_loss_mean = torch.stack(policy_losses).mean()
        
        # Calculate entropy bonus (negative because we want to maximize entropy)
        if len(entropies_list) > 0:
            entropy_mean = torch.stack(entropies_list).mean()
            entropy_bonus = self.entropy_coef * entropy_mean
        else:
            entropy_mean = torch.tensor(0.0).to(self.device)
            entropy_bonus = torch.tensor(0.0).to(self.device)
        
        # Combined actor loss: policy loss - entropy bonus (subtract because we want to maximize entropy)
        a_loss = policy_loss_mean - entropy_bonus

        # take gradient steps with clipping
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

        # Return metrics for logging
        return {
            "actor_grad_norm": actor_grad_norm.item(),
            "critic_grad_norm": critic_grad_norm.item(),
            "actor_loss": a_loss.item(),
            "policy_loss": policy_loss_mean.item(),
            "entropy": entropy_mean.item(),
            "entropy_bonus": entropy_bonus.item(),
            "advantage_mean": np.mean(advantages_list),
            "advantage_std": np.std(advantages_list),
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

    def test_agent(self, test_episodes, env, cplexpath, directory):
        epochs = range(test_episodes)  # epoch iterator
        episode_reward = []
        episode_served_demand = []
        episode_rebalancing_cost = []
        for _ in epochs:
            eps_reward = 0
            eps_served_demand = 0
            eps_rebalancing_cost = 0
            obs = env.reset()
            action_rl = [0]*env.nregion
            actions = []
            done = False
            while not done:

                if env.mode == 0:
                    obs, paxreward, done, info, system_info, _, _ = env.match_step_simple()
                    eps_reward += paxreward
            
                    action_rl = self.select_action(obs, deterministic=True)  # Choose an action
                    actions.append(action_rl)

                    # transform sample from Dirichlet into actual vehicle counts (i.e. (x1*x2*..*xn)*num_vehicles)
                    desiredAcc = {env.region[i]: int(action_rl[i] *dictsum(env.acc,env.time+1))for i in range(len(env.region))}

                    # solve minimum rebalancing distance problem (Step 3 in paper)
                    rebAction = solveRebFlow(
                        env,
                        "nyc_manhattan",
                        desiredAcc,
                        cplexpath,
                        directory, 
                    )

                    # Take rebalancing action in environment
                    _, rebreward, done, _, system_info, _, _ = env.reb_step(rebAction)
                    eps_reward += rebreward
                    
                elif env.mode == 1:
                    obs, paxreward, done, info, system_info, _, _ = env.match_step_simple(action_rl)
                    eps_reward += paxreward
                    action_rl = self.select_action(obs, deterministic=True)  # Choose an action
                    env.matching_update()

                elif env.mode == 2:
                    obs, paxreward, done, info, system_info, _, _ = env.match_step_simple(action_rl)
                    eps_reward += paxreward

                    action_rl = self.select_action(obs, deterministic=True)  # Choose an action)

                    # transform sample from Dirichlet into actual vehicle counts (i.e. (x1*x2*..*xn)*num_vehicles)
                    desiredAcc = {
                        env.region[i]: int(
                            action_rl[i][-1] * dictsum(env.acc, env.time + 1))
                        for i in range(len(env.region))
                    }

                    # solve minimum rebalancing distance problem (Step 3 in paper)
                    rebAction = solveRebFlow(
                        env,
                        "nyc_manhattan",
                        desiredAcc,
                        cplexpath,
                        directory, 
                    )

                    # Take rebalancing action in environment
                    _, rebreward, done, info, system_info, _, _ = env.reb_step(rebAction)
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