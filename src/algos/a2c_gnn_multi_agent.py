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
import datetime
import os

SavedAction = namedtuple('SavedAction', ['log_prob', 'value', 'entropy'])

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
        entropy_coef = 0.2,  # entropy regularization coefficient
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
        
        # Set entropy coefficient
        self.entropy_coef = entropy_coef

        # Set gradient clipping values
        self.actor_clip = actor_clip
        self.critic_clip = critic_clip

        # Action & reward buffer
        self.saved_actions = []
        self.rewards = []
        self.concentration_history = []  # Track concentration parameters per step
        
        # Diagnostic logging for gradient explosion detection
     
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        job_id = os.environ.get('LSB_JOBID', os.environ.get('SLURM_JOB_ID', 'local'))
        self.explosion_log_file = f"gradient_explosion_diagnosis_agent{agent_id}_job{job_id}_{timestamp}.txt"
        
        self.step_counter = 0
        self.episode_counter = 0
        self.grad_norm_history = []
        self.log_prob_history = []  # Track log_prob values for healthy baseline
        self.enable_step_logging = False  # Can be enabled when gradients are high
        
        # Initialize logging file
        with open(self.explosion_log_file, 'w') as f:
            f.write("="*100 + "\n")
            f.write(f"GRADIENT EXPLOSION DIAGNOSTIC LOG - Agent {agent_id}\n")
            f.write(f"Purpose: Understand what causes gradient explosions through data\n")
            f.write("="*100 + "\n")
            f.write(f"Job ID: {job_id}\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Configuration:\n")
            f.write(f"  - Actor gradient clip: {actor_clip}\n")
            f.write(f"  - Critic gradient clip: {critic_clip}\n")
            f.write(f"  - Gamma: {gamma}\n")
            f.write(f"  - Actor LR: {p_lr}\n")
            f.write(f"  - Critic LR: {q_lr}\n")
            f.write(f"  - Log prob aggregation: MEAN (scale-invariant)\n")
            f.write(f"  - Entropy regularization: ENABLED (prevents peaked distributions)\n")
            f.write("="*100 + "\n\n")
        
        self.to(self.device)
    
    def log_episode_start(self, episode_num):
        """Mark the start of a new episode in the log"""
        self.episode_counter = episode_num
        if self.enable_step_logging:
            with open(self.explosion_log_file, 'a') as f:
                f.write("\n" + "="*100 + "\n")
                f.write(f"EPISODE {episode_num} - DETAILED STEP TRACKING\n")
                f.write("="*100 + "\n\n")
    
    def log_episode_end(self, episode_num, total_reward):
        """Mark the end of an episode in the log"""
        if self.enable_step_logging:
            with open(self.explosion_log_file, 'a') as f:
                f.write("\n" + "-"*100 + "\n")
                f.write(f"END OF EPISODE {episode_num}\n")
                f.write(f"Total episode reward: {total_reward:.6f}\n")
                f.write("-"*100 + "\n\n")
    
    def log_step_details(self, step_num, reward_received):
        """Log detailed step information for debugging"""
        if not hasattr(self, 'last_action'):
            return
        
        with open(self.explosion_log_file, 'a') as f:
            f.write(f"\n--- Step {step_num} Details (Agent {self.agent_id}) ---\n")
            f.write(f"Reward: {reward_received:.6f}\n")
            
            if hasattr(self, 'last_value') and self.last_value is not None:
                f.write(f"Value prediction: {self.last_value.item():.6f}\n")
            
            if hasattr(self, 'last_log_prob') and self.last_log_prob is not None:
                f.write(f"Log probability: {self.last_log_prob.item():.6f}\n")
            
            if hasattr(self, 'last_action') and self.last_action is not None:
                action_np = self.last_action.cpu().numpy()
                if self.mode == 0:  # Rebalancing only
                    f.write(f"Rebalancing actions (per node):\n")
                    for i, val in enumerate(action_np.flatten()):
                        f.write(f"  Node {i}: {val:.6f}\n")
                elif self.mode == 1:  # Pricing only
                    f.write(f"Price scalars (per node):\n")
                    for i, val in enumerate(action_np.flatten()):
                        f.write(f"  Node {i}: {val:.6f} (price = {2*val:.6f})\n")
                elif self.mode == 2:  # Both
                    f.write(f"Actions (per node) - [price_scalar, reb_action]:\n")
                    for i in range(action_np.shape[0]):
                        f.write(f"  Node {i}: price_scalar={action_np[i,0]:.6f} (price={2*action_np[i,0]:.6f}), reb={action_np[i,1]:.6f}\n")
            
            if hasattr(self, 'last_concentration') and self.last_concentration is not None:
                conc_np = self.last_concentration.cpu().numpy()
                if self.mode == 0:  # Dirichlet
                    f.write(f"Concentration parameters (Dirichlet, per node):\n")
                    for i, val in enumerate(conc_np.flatten()):
                        f.write(f"  Node {i}: {val:.6f}\n")
                elif self.mode == 1:  # Beta (alpha, beta)
                    f.write(f"Concentration parameters (Beta, per node) - [alpha, beta]:\n")
                    for i in range(conc_np.shape[1]):
                        f.write(f"  Node {i}: alpha={conc_np[0,i,0]:.6f}, beta={conc_np[0,i,1]:.6f}\n")
                elif self.mode == 2:  # Beta + Dirichlet
                    f.write(f"Concentration parameters (per node) - [alpha, beta, dirichlet]:\n")
                    for i in range(conc_np.shape[1]):
                        f.write(f"  Node {i}: alpha={conc_np[0,i,0]:.6f}, beta={conc_np[0,i,1]:.6f}, dirichlet={conc_np[0,i,2]:.6f}\n")
            
            f.write("\n")

    def parse_obs(self, obs):
        state = self.obs_parser.parse_obs(obs)
        return state
    
    # Combines select action and forward steps of actor and critic
    def select_action(self, obs, deterministic=False, return_concentration=False, log_step=False, step_num=None):
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
            
            # Optional: Log step details if enabled or requested
            if (self.enable_step_logging or log_step) and step_num is not None:
                self.log_step_details(step_num, reward_received=None)
            
            self.saved_actions.append(SavedAction(logprob, value, entropy))
    
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

        # calculate the true value using rewards returned from the environment
        for r in self.rewards[::-1]:
            # calculate the discounted value
            R = r + self.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns).to(self.device)
        
        # ==== DIAGNOSTIC TRACKING: Track values before any normalization ====
        raw_returns = returns.clone()
        raw_returns_mean = raw_returns.mean().item()
        raw_returns_std = raw_returns.std().item()
        raw_returns_min = raw_returns.min().item()
        raw_returns_max = raw_returns.max().item()
        
        # Note: Returns are NOT normalized (common choice - let critic learn the scale)
        # Advantages WILL be normalized below (standard practice for policy gradient stability)
        returns = (returns - returns.mean()) / (returns.std() + self.eps)
        
        # Calculate advantages
        advantages = []
        raw_advantages = []
        values_list = []
        log_probs_list = []
        entropies_list = []
        
        for (log_prob, value, entropy), R in zip(saved_actions, returns):
            advantage = R - value.item()
            advantages.append(advantage)
            raw_advantages.append(advantage)
            values_list.append(value.item())
            log_probs_list.append(log_prob.item())
            if entropy is not None:
                entropies_list.append(entropy)
        
        # ==== DIAGNOSTIC TRACKING: Advantage statistics ====
        raw_advantages_tensor = torch.tensor(raw_advantages)
        raw_adv_mean = raw_advantages_tensor.mean().item()
        raw_adv_std = raw_advantages_tensor.std().item()
        raw_adv_min = raw_advantages_tensor.min().item()
        raw_adv_max = raw_advantages_tensor.max().item()
        
        # Detect advantage outliers (beyond 3 std devs)
        adv_outliers = []
        for i, adv in enumerate(raw_advantages):
            if abs(adv - raw_adv_mean) > 3 * (raw_adv_std + self.eps):
                adv_outliers.append((i, adv))
        
        # ==== DIAGNOSTIC TRACKING: Value and log_prob statistics ====
        values_array = np.array(values_list)
        log_probs_array = np.array(log_probs_list)
        
        value_mean = values_array.mean()
        value_std = values_array.std()
        value_min = values_array.min()
        value_max = values_array.max()
        
        log_prob_mean = log_probs_array.mean()
        log_prob_std = log_probs_array.std()
        log_prob_min = log_probs_array.min()
        log_prob_max = log_probs_array.max()
        
        # Detect log_prob outliers
        log_prob_outliers = []
        for i, lp in enumerate(log_probs_list):
            if abs(lp - log_prob_mean) > 3 * (log_prob_std + self.eps):
                log_prob_outliers.append((i, lp))
        
        # Calculate losses using normalized advantages
        for i, ((log_prob, value, entropy), R) in enumerate(zip(saved_actions, returns)):
            policy_losses.append(-log_prob * advantages[i])
            value_losses.append(F.smooth_l1_loss(value, torch.tensor([R]).to(self.device)))
        
        # ==== DIAGNOSTIC TRACKING: Loss statistics before backward ====
        policy_losses_values = [pl.item() for pl in policy_losses]
        value_losses_values = [vl.item() for vl in value_losses]
        
        policy_loss_mean = np.mean(policy_losses_values)
        policy_loss_std = np.std(policy_losses_values)
        value_loss_mean = np.mean(value_losses_values)
        value_loss_std = np.std(value_losses_values)
        
        # Calculate mean policy loss
        policy_loss_base = torch.stack(policy_losses).mean()
        
        # Calculate entropy bonus
        # Note: Entropy is in nats (natural log units) and can be very large (e.g., 10-20 for Dirichlet)
        # or negative (high-dimensional Dirichlet). Advantages are normalized to std=1.
        # Use a small entropy_coef (e.g., 0.001-0.01) to balance the scales.
        if len(entropies_list) > 0:
            entropy_mean = torch.stack(entropies_list).mean()
            entropy_bonus = self.entropy_coef * entropy_mean
        else:
            entropy_mean = torch.tensor(0.0).to(self.device)
            entropy_bonus = torch.tensor(0.0).to(self.device)
        
        # Combined actor loss: policy loss - entropy bonus (subtract because we want to maximize entropy)
        a_loss = policy_loss_base - entropy_bonus
        
        # take gradient steps
        self.optimizers['a_optimizer'].zero_grad()
        a_loss.backward()
        
        # ==== DIAGNOSTIC TRACKING: Gradient norms BEFORE clipping ====
        actor_grad_norm_before_clip = 0.0
        actor_grad_max = 0.0
        actor_grad_by_layer = {}
        for name, param in self.actor.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                param_max = param.grad.data.abs().max().item()
                actor_grad_norm_before_clip += param_norm ** 2
                actor_grad_max = max(actor_grad_max, param_max)
                actor_grad_by_layer[name] = param_norm
        actor_grad_norm_before_clip = actor_grad_norm_before_clip ** 0.5
        
        # Gradient clipping for actor
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.actor_clip)
        self.optimizers['a_optimizer'].step()
        
        self.optimizers['c_optimizer'].zero_grad()
        v_loss = torch.stack(value_losses).mean()
        v_loss.backward()
        
        # ==== DIAGNOSTIC TRACKING: Critic gradient norms BEFORE clipping ====
        critic_grad_norm_before_clip = 0.0
        critic_grad_max = 0.0
        critic_grad_by_layer = {}
        for name, param in self.critic.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                param_max = param.grad.data.abs().max().item()
                critic_grad_norm_before_clip += param_norm ** 2
                critic_grad_max = max(critic_grad_max, param_max)
                critic_grad_by_layer[name] = param_norm
        critic_grad_norm_before_clip = critic_grad_norm_before_clip ** 0.5
        
        # Gradient clipping for critic
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.critic_clip)
        self.optimizers['c_optimizer'].step()
        
        # ==== DIAGNOSTIC LOGGING: Write detailed information if gradients are large ====
        self.step_counter += 1
        
        # Define threshold for "explosion warning"
        explosion_threshold = 0.8 * self.actor_clip  # 80% of clip value
        warning_threshold = 0.5 * self.actor_clip     # 50% of clip value
        
        # Enable step logging for next episode if gradients are concerning
        if actor_grad_norm_before_clip > warning_threshold or critic_grad_norm_before_clip > warning_threshold:
            self.enable_step_logging = True
            with open(self.explosion_log_file, 'a') as f:
                f.write(f"\n⚠️  WARNING: Gradients elevated at episode {self.episode_counter}. Enabling step-by-step logging for next episode.\n")
        else:
            self.enable_step_logging = False
        
        if actor_grad_norm_before_clip > explosion_threshold or critic_grad_norm_before_clip > explosion_threshold:
            with open(self.explosion_log_file, 'a') as f:
                f.write("\n" + "="*100 + "\n")
                f.write(f"⚠️  HIGH GRADIENT DETECTED - Step {self.step_counter}\n")
                f.write("="*100 + "\n\n")
                
                f.write("GRADIENT NORMS:\n")
                f.write(f"  Actor grad norm (before clip):  {actor_grad_norm_before_clip:.4f}\n")
                f.write(f"  Actor grad norm (after clip):   {actor_grad_norm.item():.4f}\n")
                f.write(f"  Actor grad max value:            {actor_grad_max:.4f}\n")
                f.write(f"  Critic grad norm (before clip):  {critic_grad_norm_before_clip:.4f}\n")
                f.write(f"  Critic grad norm (after clip):   {critic_grad_norm.item():.4f}\n")
                f.write(f"  Critic grad max value:           {critic_grad_max:.4f}\n\n")
                
                f.write("ACTOR GRADIENTS BY LAYER:\n")
                for name, norm in sorted(actor_grad_by_layer.items(), key=lambda x: -x[1])[:5]:
                    f.write(f"  {name}: {norm:.4f}\n")
                f.write("\n")
                
                f.write("CRITIC GRADIENTS BY LAYER:\n")
                for name, norm in sorted(critic_grad_by_layer.items(), key=lambda x: -x[1])[:5]:
                    f.write(f"  {name}: {norm:.4f}\n")
                f.write("\n")
                
                f.write("RETURNS (DISCOUNTED REWARDS):\n")
                f.write(f"  Raw returns mean: {raw_returns_mean:.6f}\n")
                f.write(f"  Raw returns std:  {raw_returns_std:.6f}\n")
                f.write(f"  Raw returns min:  {raw_returns_min:.6f}\n")
                f.write(f"  Raw returns max:  {raw_returns_max:.6f}\n")
                f.write(f"  Raw returns: {raw_returns.cpu().numpy()}\n\n")
                
                f.write("RAW REWARDS (per step):\n")
                f.write(f"  Rewards: {self.rewards}\n")
                f.write(f"  Mean: {np.mean(self.rewards):.6f}, Std: {np.std(self.rewards):.6f}\n")
                f.write(f"  Min: {np.min(self.rewards):.6f}, Max: {np.max(self.rewards):.6f}\n\n")
                
                f.write("ADVANTAGES:\n")
                f.write(f"  Raw advantage mean: {raw_adv_mean:.6f}\n")
                f.write(f"  Raw advantage std:  {raw_adv_std:.6f}\n")
                f.write(f"  Raw advantage min:  {raw_adv_min:.6f}\n")
                f.write(f"  Raw advantage max:  {raw_adv_max:.6f}\n")
                f.write(f"  Raw advantages: {raw_advantages}\n")
                if adv_outliers:
                    f.write(f"  ⚠️  ADVANTAGE OUTLIERS (>3 std): {adv_outliers}\n")
                f.write("\n")
                
                f.write("VALUES (Critic predictions):\n")
                f.write(f"  Value mean: {value_mean:.6f}\n")
                f.write(f"  Value std:  {value_std:.6f}\n")
                f.write(f"  Value min:  {value_min:.6f}\n")
                f.write(f"  Value max:  {value_max:.6f}\n")
                f.write(f"  Values: {values_list}\n\n")
                
                f.write("LOG PROBABILITIES:\n")
                f.write(f"  Log prob mean: {log_prob_mean:.6f}\n")
                f.write(f"  Log prob std:  {log_prob_std:.6f}\n")
                f.write(f"  Log prob min:  {log_prob_min:.6f}\n")
                f.write(f"  Log prob max:  {log_prob_max:.6f}\n")
                f.write(f"  Log probs: {log_probs_list}\n")
                if log_prob_outliers:
                    f.write(f"  ⚠️  LOG PROB OUTLIERS (>3 std): {log_prob_outliers}\n")
                f.write("\n")
                
                f.write("LOSSES:\n")
                f.write(f"  Actor loss (final):     {a_loss.item():.6f}\n")
                f.write(f"  Critic loss (final):    {v_loss.item():.6f}\n")
                f.write(f"  Policy losses mean:     {policy_loss_mean:.6f}\n")
                f.write(f"  Policy losses std:      {policy_loss_std:.6f}\n")
                f.write(f"  Value losses mean:      {value_loss_mean:.6f}\n")
                f.write(f"  Value losses std:       {value_loss_std:.6f}\n\n")
                
                # Track concentration parameters if available
                if hasattr(self, 'last_concentration'):
                    conc = self.last_concentration.cpu().numpy()
                    f.write("CONCENTRATION PARAMETERS (last action):\n")
                    f.write(f"  Shape: {conc.shape}\n")
                    f.write(f"  Mean: {conc.mean():.6f}\n")
                    f.write(f"  Std: {conc.std():.6f}\n")
                    f.write(f"  Min: {conc.min():.6f}\n")
                    f.write(f"  Max: {conc.max():.6f}\n")
                    
                    # Show per-node statistics for pricing modes
                    if self.mode == 1:  # Beta distribution (alpha, beta)
                        f.write(f"  Alpha (per node): mean={conc[:,:,0].mean():.4f}, std={conc[:,:,0].std():.4f}\n")
                        f.write(f"  Beta (per node):  mean={conc[:,:,1].mean():.4f}, std={conc[:,:,1].std():.4f}\n")
                        f.write(f"  Alpha values: {conc[:,:,0].flatten()}\n")
                        f.write(f"  Beta values: {conc[:,:,1].flatten()}\n")
                    elif self.mode == 2:  # Both pricing and rebalancing
                        f.write(f"  Alpha (pricing, per node): mean={conc[:,:,0].mean():.4f}, std={conc[:,:,0].std():.4f}\n")
                        f.write(f"  Beta (pricing, per node):  mean={conc[:,:,1].mean():.4f}, std={conc[:,:,1].std():.4f}\n")
                        f.write(f"  Dirichlet (reb, per node): mean={conc[:,:,2].mean():.4f}, std={conc[:,:,2].std():.4f}\n")
                    f.write("\n")
                
                f.write("="*100 + "\n\n")
        
        # ==== CONCENTRATION PARAMETER STATISTICS ====
        # Calculate max and min concentration across all steps in the episode
        concentration_max = 0.0
        concentration_min = 0.0
        
        if self.concentration_history:
            # Stack all concentration values from the episode
            all_concentrations = np.concatenate([c.flatten() for c in self.concentration_history])
            concentration_max = float(all_concentrations.max())
            concentration_min = float(all_concentrations.min())
        
        # reset rewards and action buffer
        del self.rewards[:]
        del self.saved_actions[:]
        del self.concentration_history[:]  # Clear concentration history

        # Return metrics for logging
        return {
            "actor_grad_norm": actor_grad_norm.item(),
            "critic_grad_norm": critic_grad_norm.item(),
            "actor_loss": a_loss.item(),
            "policy_loss": policy_loss_base.item(),
            "entropy": entropy_mean.item(),
            "entropy_bonus": entropy_bonus.item(),
            "critic_loss": v_loss.item(),
            "actor_grad_norm_before_clip": actor_grad_norm_before_clip,
            "critic_grad_norm_before_clip": critic_grad_norm_before_clip,
            "advantage_mean": raw_adv_mean,
            "advantage_std": raw_adv_std,
            "value_mean": value_mean,
            "value_std": value_std,
            "log_prob_mean": log_prob_mean,
            "log_prob_std": log_prob_std,
            "return_mean": raw_returns_mean,
            "return_std": raw_returns_std,
            # Add concentration parameter statistics
            "concentration_max": concentration_max,
            "concentration_min": concentration_min,
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
    