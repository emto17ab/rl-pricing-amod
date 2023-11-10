import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Dirichlet, Beta
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv
from torch_geometric.utils import grid
from src.algos.summarizer import Summarizer
from src.algos.reb_flow_solver import solveRebFlow
from src.misc.utils import dictsum, create_target
from collections import namedtuple
import random


RecurrentBatch = namedtuple('RecurrentBatch', 'o a r d m edge_index')
class RecurrentReplyData:

    def __init__(self, o_dim, a_dim, max_steps, sample_steps, device, capacity=1000, batch_size=32):
        """
        max_steps: Number of time step in one episode
        capacity: Buffer capacity
        """
        
        # placeholders
        self.o = np.zeros((capacity, max_steps, *o_dim))
        self.a = np.zeros((capacity, max_steps-1, *a_dim))
        self.r = np.zeros((capacity, max_steps-1, 1))
        self.d = np.zeros((capacity, max_steps-1, 1))
        self.m = np.zeros((capacity, max_steps-1, 1))
        self.edge_index = None
        
        # pointers
        self.episode_ptr = 0
        self.time_ptr = 0

        # trackers
        self.starting_new_episode = True
        self.num_episodes = 0

        # hyper-parameters
        self.capacity = capacity
        self.o_dim = o_dim
        self.a_dim = a_dim
        self.max_steps = max_steps
        self.sample_steps = sample_steps
        self.batch_size = batch_size  
        self.device = device            

    def store(self, o, a, r, no, d, edge_index: torch.tensor):

        self.o[self.episode_ptr, self.time_ptr] = o
        if isinstance(a[0], list):
            self.a[self.episode_ptr, self.time_ptr] = a
        else:
            self.a[self.episode_ptr, self.time_ptr] = np.array(a).unsqueeze(-1)
        self.r[self.episode_ptr, self.time_ptr] = r
        self.d[self.episode_ptr, self.time_ptr] = d
        self.m[self.episode_ptr, self.time_ptr] = 1    
        if self.edge_index is None:
            self.edge_index = edge_index

        if d:

            # fill placeholders
            self.o[self.episode_ptr, self.time_ptr+1] = no

            # reset pointers
            if self.episode_ptr < self.capacity -1:
                self.episode_ptr += 1
            else:
                self.episode_ptr = ((self.episode_ptr + 1) % self.capacity)
            self.time_ptr = 0

            # update trackers
            self.num_episodes += 1

        else:
            # update pointers
            self.time_ptr += 1        

    def sample_batch(self):

        # assert self.num_episodes >= self.batch_size

        o_reshape = []
        a_reshape = []
        r_reshape = []
        d_reshape = []
        m_reshape = []
        current_buffer = min(self.num_episodes,self.capacity)
        # Previous episodes
        for i in range(0, self.max_steps - 1 - (self.sample_steps - 1)):
            o_reshape.append(self.o[:current_buffer,i:i+self.sample_steps+1,:,:]) 
            a_reshape.append(self.a[:current_buffer,i:i+self.sample_steps,:])
            r_reshape.append(self.r[:current_buffer,i:i+self.sample_steps,:])
            d_reshape.append(self.d[:current_buffer,i:i+self.sample_steps,:])
            m_reshape.append(self.m[:current_buffer,i:i+self.sample_steps,:])
        # Current epsiode
        if self.time_ptr >= self.max_steps:
            for i in range(0, self.time_ptr - (self.sample_steps - 1)):
                o_reshape.append(self.o[current_buffer,i:i+self.sample_steps+1,:,:][np.newaxis,:,:,:]) 
                a_reshape.append(self.a[current_buffer,i:i+self.sample_steps,:][np.newaxis,:,:])
                r_reshape.append(self.r[current_buffer,i:i+self.sample_steps,:][np.newaxis,:,:])
                d_reshape.append(self.d[current_buffer,i:i+self.sample_steps,:][np.newaxis,:,:])
                m_reshape.append(self.m[current_buffer,i:i+self.sample_steps,:][np.newaxis,:,:])            
        o_reshape = np.concatenate(o_reshape, axis=0)
        a_reshape = np.concatenate(a_reshape, axis=0)
        r_reshape = np.concatenate(r_reshape, axis=0)
        d_reshape = np.concatenate(d_reshape, axis=0)
        m_reshape = np.concatenate(m_reshape, axis=0)

        choices = random.sample(range(o_reshape.shape[0]), self.batch_size)

        o = o_reshape[choices]
        a = a_reshape[choices]
        r = r_reshape[choices]
        d = d_reshape[choices]
        m = m_reshape[choices]        

        o = torch.tensor(o).float().to(self.device)
        a = torch.tensor(a).float().to(self.device)
        r = torch.tensor(r).float().to(self.device)
        d = torch.tensor(d).float().to(self.device)
        m = torch.tensor(m).float().to(self.device)
        edge_index = self.edge_index.to(self.device)

        return RecurrentBatch(o, a, r, d, m, edge_index)
    
    def get_latest(self):
        o = self.o[self.episode_ptr - 1]
        a = self.a[self.episode_ptr - 1]
        r = self.r[self.episode_ptr - 1]
        d = self.d[self.episode_ptr - 1]
        m = self.m[self.episode_ptr - 1]        

        o = torch.tensor(o).unsqueeze(0).float().to(self.device)
        a = torch.tensor(a).unsqueeze(0).float().to(self.device)
        r = torch.tensor(r).unsqueeze(0).float().to(self.device)
        d = torch.tensor(d).unsqueeze(0).float().to(self.device)
        m = torch.tensor(m).unsqueeze(0).float().to(self.device)
        edge_index = self.edge_index.to(self.device)

        return RecurrentBatch(o, a, r, d, m, edge_index)       

class Scalar(nn.Module):
    def __init__(self, init_value):
        super().__init__()
        self.constant = nn.Parameter(torch.tensor(init_value, dtype=torch.float32))

    def forward(self):
        return self.constant


#########################################
############## ACTOR ####################
#########################################
class GNNActor(nn.Module):
    """
    Actor \pi(a_t | s_t) parametrizing the concentration parameters of a Dirichlet Policy.
    """

    def __init__(self, in_channels, hidden_size=32, act_dim=6, mode=0):
        super().__init__()
        self.mode = mode
        self.in_channels = in_channels
        self.act_dim = act_dim
        self.conv1 = GCNConv(in_channels, in_channels)
        self.lin1 = nn.Linear(in_channels, hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        if self.mode == 0:
            self.lin3 = nn.Linear(hidden_size, 1)
        elif self.mode == 1:
            self.lin3 = nn.Linear(hidden_size, 4)

    def forward(self, state, edge_index, deterministic=False):
        out = F.relu(self.conv1(state, edge_index))
        x = out + state
        x = x.reshape(-1, self.act_dim, self.in_channels)
        x = F.leaky_relu(self.lin1(x))
        x = F.leaky_relu(self.lin2(x))
        x = F.softplus(self.lin3(x))
        concentration = x.squeeze(-1)
        if deterministic:
            action = (concentration) / (concentration.sum() + 1e-20)
            log_prob = None
        else:
            if self.mode == 0:
                m = Dirichlet(concentration + 1e-20)
                action = m.rsample()
                log_prob = m.log_prob(action)
            elif self.mode == 1:
                m_o = Beta(concentration[:,:,0], concentration[:,:,1])
                m_d = Beta(concentration[:,:,2], concentration[:,:,3])
                action_o = m_o.rsample()
                action_d = m_d.rsample()
                log_prob = m_o.log_prob(action_o).sum(dim=-1) + m_d.log_prob(action_d).sum(dim=-1)
                action = torch.cat((action_o.squeeze(0).unsqueeze(-1), action_d.squeeze(0).unsqueeze(-1)),-1)     
            else:
                pass                
        return action, log_prob


#########################################
############## CRITIC ###################
#########################################


class GNNCritic1(nn.Module):
    """
    Architecture 1, GNN, Pointwise Multiplication, Readout, FC
    """

    def __init__(self, in_channels, hidden_size=256, act_dim=6):
        super().__init__()
        self.act_dim = act_dim
        self.in_channels = in_channels
        self.conv1 = GCNConv(in_channels, in_channels)
        self.lin1 = nn.Linear(in_channels, hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        self.lin3 = nn.Linear(hidden_size, 1)

    def forward(self, state, edge_index, action):
        out = F.relu(self.conv1(state, edge_index))
        x = out + state
        x = x.reshape(-1, self.act_dim, self.in_channels)  # (B,N,21)
        action = action * 10
        action = action.unsqueeze(-1)  # (B,N,1)
        x = x * action  # pointwise multiplication (B,N,21)
        x = x.sum(dim=1)  # (B,21)
        x = F.relu(self.lin1(x))  # (B,H)
        x = F.relu(self.lin2(x))  # (B,H)
        x = self.lin3(x).squeeze(-1)  # (B)
        return x


class GNNCritic2(nn.Module):
    """
    Architecture 2, GNN, Readout, Concatenation, FC
    """

    def __init__(self, in_channels, hidden_size=256, act_dim=6):
        super().__init__()
        self.act_dim = act_dim
        self.conv1 = GCNConv(in_channels, in_channels)
        self.lin1 = nn.Linear(in_channels + act_dim, hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        self.lin3 = nn.Linear(hidden_size, 1)

    def forward(self, state, edge_index, action):
        out = F.relu(self.conv1(state, edge_index))
        x = out + state
        x = x.reshape(-1, self.act_dim, 21)  # (B,N,21)
        x = torch.sum(x, dim=1)  # (B, 21)
        concat = torch.cat([x, action], dim=-1)  # (B, 21+N)
        x = F.relu(self.lin1(concat))  # (B,H)
        x = F.relu(self.lin2(x))  # (B,H)
        x = self.lin3(x).squeeze(-1)  # B
        return x


class GNNCritic3(nn.Module):
    """
    Architecture 3: Concatenation, GNN, Readout, FC
    """

    def __init__(self, in_channels, hidden_size=32, act_dim=6):
        super().__init__()
        self.act_dim = act_dim
        self.conv1 = GCNConv(22, 22)
        self.lin1 = nn.Linear(22, hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        self.lin3 = nn.Linear(hidden_size, 1)

    def forward(self, state, edge_index, action):
        cat = torch.cat([state, action.unsqueeze(-1)], dim=-1)  # (B,N,22)
        out = F.relu(self.conv1(cat, edge_index))
        x = out + cat
        x = x.reshape(-1, self.act_dim, 22)  # (B,N,22)
        x = F.relu(self.lin1(x))  # (B, H)
        x = F.relu(self.lin2(x))  # (B, H)
        x = torch.sum(x, dim=1)  # (B, 22)
        x = self.lin3(x).squeeze(-1)  # (B)
        return x


class GNNCritic4(nn.Module):
    """
    Architecture 4: GNN, Concatenation, FC, Readout
    """

    def __init__(self, in_channels, hidden_size=32, act_dim=6, mode=1):
        super().__init__()
        self.act_dim = act_dim
        self.mode = mode
        self.conv1 = GCNConv(in_channels, in_channels)
        self.lin1 = nn.Linear(in_channels + self.mode + 1, hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        self.lin3 = nn.Linear(hidden_size, 1)
        self.in_channels = in_channels

    def forward(self, state, edge_index, action):
        out = F.relu(self.conv1(state, edge_index))
        x = out + state
        x = x.reshape(-1, self.act_dim, self.in_channels)  # (B,N,21)
        # batch_size, timesteps, regions = action.size()
        # action = action.view(batch_size*timesteps,regions,-1)
        concat = torch.cat([x, action], dim=-1)  # (B,N,22)
        x = F.relu(self.lin1(concat))
        x = F.relu(self.lin2(x))  # (B, N, H)
        x = torch.sum(x, dim=1)  # (B, H)
        x = self.lin3(x).squeeze(-1)  # (B)
        return x


class GNNCritic5(nn.Module):
    """
    Architecture 5, GNN, Pointwise Multiplication, FC, Readout
    """

    def __init__(self, in_channels, hidden_size=256, act_dim=6):
        super().__init__()
        self.act_dim = act_dim
        self.in_channels = in_channels
        self.conv1 = GCNConv(in_channels, in_channels)
        self.lin1 = nn.Linear(in_channels, hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        self.lin3 = nn.Linear(hidden_size, 1)

    def forward(self, state, edge_index, action):
        out = F.relu(self.conv1(state, edge_index))
        x = out + state
        x = x.reshape(-1, self.act_dim, self.in_channels)  # (B,N,21)
        action = action + 1
        action = action.unsqueeze(-1)  # (B,N,1)
        x = x * action  # pointwise multiplication (B,N,21)
        x = F.relu(self.lin1(x))  # (B,N,H)
        x = F.relu(self.lin2(x))  # (B,N,H)
        x = x.sum(dim=1)  # (B,H)
        x = self.lin3(x).squeeze(-1)  # (B)
        return x


#########################################
############## A2C AGENT ################
#########################################


class RSAC(nn.Module):
    """
    Advantage Actor Critic algorithm for the AMoD control problem.
    """

    def __init__(
        self,
        env,
        recurrent_input_size,
        recurrent_hidden_size,
        other_input_size,
        hidden_size=32,
        alpha=0.2,
        gamma=0.99,
        polyak=0.995,
        batch_size=128,
        buffer_cap=1000,
        sample_steps=4,
        p_lr=3e-4,
        q_lr=1e-3,
        use_automatic_entropy_tuning=False,
        lagrange_thresh=-1,
        min_q_weight=1,
        deterministic_backup=False,
        eps=np.finfo(np.float32).eps.item(),
        device=torch.device("cpu"),
        min_q_version=3,
        clip=200,
        critic_version=4,
        mode = 1,
        env_baseline = None
    ):
        super(RSAC, self).__init__()
        self.env = env
        self.eps = eps
        self.recurrent_input_size = recurrent_input_size
        self.recurrent_hidden_size = recurrent_hidden_size
        self.other_input_size = other_input_size
        self.input_size = recurrent_input_size + other_input_size
        self.hidden_size = hidden_size
        self.device = device
        self.path = None
        self.act_dim = env.nregion
        self.mode = mode
        self.env_baseline = env_baseline

        # SAC parameters
        self.alpha = alpha
        self.polyak = polyak
        self.env = env
        self.BATCH_SIZE = batch_size
        self.p_lr = p_lr
        self.q_lr = q_lr
        self.gamma = gamma
        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        self.min_q_version = min_q_version
        self.clip = clip

        # conservative Q learning parameters
        self.num_random = 10
        self.temp = 1.0
        self.min_q_weight = min_q_weight
        if lagrange_thresh == -1:
            self.with_lagrange = False
        else:
            print("using lagrange")
            self.with_lagrange = True
        self.deterministic_backup = deterministic_backup
        self.step = 0
        self.nodes = env.nregion

        self.replay_buffer = RecurrentReplyData((env.nregion, self.input_size), (self.act_dim,self.mode+1), env.tf, sample_steps, device, capacity=buffer_cap, batch_size=self.BATCH_SIZE)

        # trackers
        self.hidden = None

        # nnets

        self.actor_summarizer =Summarizer(self.recurrent_input_size, self.recurrent_hidden_size, batch_size)

        self.critic1_summarizer = Summarizer(self.recurrent_input_size, self.recurrent_hidden_size, batch_size)
        self.critic1_summarizer_target = create_target(self.critic1_summarizer)

        self.critic2_summarizer = Summarizer(self.recurrent_input_size, self.recurrent_hidden_size, batch_size)
        self.critic2_summarizer_target = create_target(self.critic2_summarizer)

        self.actor = GNNActor(self.recurrent_hidden_size+self.other_input_size, self.hidden_size, act_dim=self.act_dim, mode=mode)
    
        if critic_version == 1:
            GNNCritic = GNNCritic1
        if critic_version == 2:
            GNNCritic = GNNCritic2
        if critic_version == 3:
            GNNCritic = GNNCritic3
        if critic_version == 4:
            GNNCritic = GNNCritic4
        if critic_version == 5:
            GNNCritic = GNNCritic5
        
        self.critic1 = GNNCritic(
            self.recurrent_hidden_size+self.other_input_size, self.hidden_size, act_dim=self.act_dim, mode=mode
        )
        self.critic2 = GNNCritic(
            self.recurrent_hidden_size+self.other_input_size, self.hidden_size, act_dim=self.act_dim, mode=mode
        )
        assert self.critic1.parameters() != self.critic2.parameters()   

        self.critic1_target = create_target(self.critic1)
        self.critic2_target = create_target(self.critic2)

        self.optimizers = self.configure_optimizers()

        # action & reward buffer
        self.saved_actions = []
        self.rewards = []
        self.to(self.device)

        if self.with_lagrange:
            self.target_action_gap = lagrange_thresh  # lagrange treshhold
            self.log_alpha_prime = Scalar(1.0)
            self.alpha_prime_optimizer = torch.optim.Adam(
                self.log_alpha_prime.parameters(),
                lr=self.p_lr,
            )

        if self.use_automatic_entropy_tuning:
            self.target_entropy = -np.prod(self.act_dim).item()
            self.log_alpha = Scalar(0.0)
            self.alpha_optimizer = torch.optim.Adam(
                self.log_alpha.parameters(), lr=1e-3
            )

    def reinitialize_hidden(self):
        self.hidden = None

    def parse_obs(self, obs):
        state = self.obs_parser.parse_obs(obs)
        return state

    def select_action(self, data, deterministic=False):

        x = torch.from_numpy(data.x).float()
        with torch.no_grad():
            summary, self.hidden = self.actor_summarizer(x, data.edge_index, self.hidden, return_hidden=True)
            a, _ = self.actor(summary, data.edge_index, deterministic)
        a = a.squeeze(-1)
        a = a.detach().cpu().numpy().tolist()
        return list(a)

    def update(self, b: RecurrentBatch):

        bs, num_time = b.r.shape[0], b.r.shape[1]

        # Action and reward
        a1 = b.a[:,-1,:]
        r = b.r[:,-1,:].squeeze(-1)

        # Summary
        actor_summary = self.actor_summarizer(b.o, b.edge_index)
        critic1_summary = self.critic1_summarizer(b.o, b.edge_index)
        critic2_summary = self.critic2_summarizer(b.o, b.edge_index)

        critic1_summary_target = self.critic1_summarizer_target(b.o, b.edge_index)
        critic2_summary_target = self.critic2_summarizer_target(b.o, b.edge_index)

        actor_summary_1_T, actor_summary_2_Tplus1 = actor_summary[:, -2, :, :], actor_summary[:, -1, :, :]
        critic1_summary_1_T, critic1_summary_2_Tplus1 = critic1_summary[:, -2, :, :], critic1_summary_target[:, -1, :, :]
        critic2_summary_1_T, critic2_summary_2_Tplus1 = critic2_summary[:, -2, :, :], critic2_summary_target[:, -1, :, :]      

        # Compute loss for critic
        q1 = self.critic1(critic1_summary_1_T, b.edge_index, a1)
        q2 = self.critic2(critic2_summary_1_T, b.edge_index, a1)

        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = self.actor(actor_summary_2_Tplus1, b.edge_index)
            q1_pi_targ = self.critic1_target(critic1_summary_2_Tplus1, b.edge_index, a2)
            q2_pi_targ = self.critic2_target(critic2_summary_2_Tplus1, b.edge_index, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)

            # backup = (r - np.mean(self.env_baseline)) / (np.std(self.env_baseline) + self.eps) + self.gamma * (q_pi_targ - self.alpha * logp_a2)
            backup = r + self.gamma * (q_pi_targ - self.alpha * logp_a2)

        loss_q1 = F.mse_loss(q1, backup)
        loss_q2 = F.mse_loss(q2, backup)

        self.optimizers["c1_summarizer_optimizer"].zero_grad()
        self.optimizers["c1_optimizer"].zero_grad()
        loss_q1.backward()
        critic1_grad_norm = nn.utils.clip_grad_norm_(self.critic1.parameters(), self.clip)
        self.optimizers["c1_optimizer"].step()
        _ = nn.utils.clip_grad_norm_(self.critic1_summarizer.parameters(), self.clip)
        self.optimizers["c1_summarizer_optimizer"].step()
        
        self.optimizers["c2_summarizer_optimizer"].zero_grad()
        self.optimizers["c2_optimizer"].zero_grad()
        loss_q2.backward()
        critic2_grad_norm = nn.utils.clip_grad_norm_(self.critic2.parameters(), self.clip)
        self.optimizers["c2_optimizer"].step()
        _ = nn.utils.clip_grad_norm_(self.critic2_summarizer.parameters(), self.clip)
        self.optimizers["c2_summarizer_optimizer"].step()

        # Update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(
                self.critic1_summarizer.parameters(), self.critic1_summarizer_target.parameters()
            ):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)
            for p, p_targ in zip(
                self.critic2_summarizer.parameters(), self.critic2_summarizer_target.parameters()
            ):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

            for p, p_targ in zip(
                self.critic1.parameters(), self.critic1_target.parameters()
            ):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)
            for p, p_targ in zip(
                self.critic2.parameters(), self.critic2_target.parameters()
            ):
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        # for p in self.critic1_summarizer.parameters():
        #     p.requires_grad = False
        # for p in self.critic2_summarizer.parameters():
        #     p.requires_grad = False
        # for p in self.critic1.parameters():
        #     p.requires_grad = False
        # for p in self.critic2.parameters():
        #     p.requires_grad = False

        # Compute loss for actor
        actions, logp_a = self.actor(actor_summary_1_T,b.edge_index)
        q1_1 = self.critic1(critic1_summary_1_T.detach(), b.edge_index, actions)
        q2_a = self.critic2(critic2_summary_1_T.detach(), b.edge_index, actions)
        q_a = torch.min(q1_1, q2_a)

        if self.use_automatic_entropy_tuning:
            alpha_loss = -(
                self.log_alpha() * (logp_a + self.target_entropy).detach()
            ).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha().exp()

        loss_pi = (self.alpha * logp_a - q_a).mean()
        # one gradient descent step for policy network
        self.optimizers["a_summarizer_optimizer"].zero_grad()
        self.optimizers["a_optimizer"].zero_grad()
        loss_pi.backward()
        actor_grad_norm = nn.utils.clip_grad_norm_(self.actor.parameters(), 10)
        self.optimizers["a_optimizer"].step()
        _ = nn.utils.clip_grad_norm_(self.actor_summarizer.parameters(), 10)
        self.optimizers["a_summarizer_optimizer"].step()

        # Unfreeze Q-networks
        # for p in self.critic1_summarizer.parameters():
        #     p.requires_grad = True
        # for p in self.critic2_summarizer.parameters():
        #     p.requires_grad = True
        # for p in self.critic1.parameters():
        #     p.requires_grad = True
        # for p in self.critic2.parameters():
        #     p.requires_grad = True

        return {"actor_grad_norm":actor_grad_norm, "critic1_grad_norm":critic1_grad_norm, "critic2_grad_norm":critic2_grad_norm,\
                "actor_loss":loss_pi.item(), "critic1_loss":loss_q1.item(), "critic2_loss":loss_q2.item()}

    def configure_optimizers(self):
        optimizers = dict()

        optimizers["a_summarizer_optimizer"] = torch.optim.Adam(self.actor_summarizer.parameters(), lr=self.p_lr)
        optimizers["c1_summarizer_optimizer"] = torch.optim.Adam(self.critic1_summarizer.parameters(), lr=self.q_lr)
        optimizers["c2_summarizer_optimizer"] = torch.optim.Adam(self.critic2_summarizer.parameters(), lr=self.q_lr)

        optimizers["a_optimizer"] = torch.optim.Adam(self.actor.parameters(), lr=self.p_lr)
        optimizers["c1_optimizer"] = torch.optim.Adam(self.critic1.parameters(), lr=self.q_lr)
        optimizers["c2_optimizer"] = torch.optim.Adam(self.critic2.parameters(), lr=self.q_lr)

        return optimizers

    def save_checkpoint(self, path="ckpt.pth"):
        checkpoint = dict()
        checkpoint["model"] = self.state_dict()
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

    def log(self, log_dict, path="log.pth"):
        torch.save(log_dict, path)
