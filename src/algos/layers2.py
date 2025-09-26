import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Dirichlet, Beta
from torch_geometric.nn import GCNConv

class GNNCritic(nn.Module):
    """
    Critic parametrizing the value function estimator V(s_t).
    """
    def __init__(self, in_channels, hidden_size, act_dim):
        super().__init__()
        self.act_dim = act_dim
        self.in_channels = in_channels
        self.conv1 = GCNConv(in_channels, in_channels)
        self.lin1 = nn.Linear(in_channels, hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        self.lin3 = nn.Linear(hidden_size, 1)
    
    def forward(self, data):
        out = F.relu(self.conv1(data.x, data.edge_index))
        x = out + data.x
        x = x.reshape(-1, self.act_dim, self.in_channels)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = torch.sum(x, dim=1)
        x = self.lin3(x).squeeze(-1)
        return x

class GNNActor(nn.Module):
    def __init__(self, in_channels, hidden_size, act_dim, mode):
        super().__init__()
        self.mode = mode
        self.in_channels = in_channels
        self.act_dim = act_dim
        self.conv1 = GCNConv(in_channels, in_channels)
        self.lin1 = nn.Linear(in_channels, hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        if mode == 0:
            self.lin3 = nn.Linear(hidden_size, 1)
        elif mode == 1:
            self.lin3 = nn.Linear(hidden_size, 2)
        else:
            self.lin3 = nn.Linear(hidden_size, 3)

    def forward(self, data):
        out = F.relu(self.conv1(data.x, data.edge_index))
        x = out + data.x
        x = x.reshape(-1, self.act_dim, self.in_channels)
        x = F.leaky_relu(self.lin1(x))
        x = F.leaky_relu(self.lin2(x))
        x = F.softplus(self.lin3(x))
        concentration = x.squeeze(-1)
      
        if self.mode == 0:
            m = Dirichlet(concentration + 1e-20)
            action = m.rsample()
            log_prob = m.log_prob(action)
            action = action.squeeze(0).unsqueeze(-1)
        elif self.mode == 1:
            m_o = Beta(concentration[:,:,0] + 1e-10, concentration[:,:,1] + 1e-10)
            action_o = m_o.rsample()
            log_prob = m_o.log_prob(action_o).sum(dim=-1)
            action = action_o.squeeze(0).unsqueeze(-1)       
        else:        
            m_o = Beta(concentration[:,:,0] + 1e-10, concentration[:,:,1] + 1e-10)
            action_o = m_o.rsample()
            # Rebalancing desired distribution
            m_reb = Dirichlet(concentration[:,:,-1] + 1e-10)
            action_reb = m_reb.rsample()              
            log_prob = m_o.log_prob(action_o).sum(dim=-1) + m_reb.log_prob(action_reb)
            action = torch.cat((action_o.squeeze(0).unsqueeze(-1), action_reb.squeeze(0).unsqueeze(-1)),-1)       
        return action, log_prob