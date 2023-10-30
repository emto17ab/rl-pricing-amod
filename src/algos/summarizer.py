import torch
import torch.nn as nn
import numpy as np
from torch_geometric.nn import GCNConv
from torch_geometric_temporal import TGCN2
import torch.nn.functional as F


class Summarizer(nn.Module):

    def __init__(self, recurrent_input_dim, recurrent_hidden_dim, batch_size):

        super().__init__()
        self.rnn = TGCN2(in_channels=recurrent_input_dim, out_channels=recurrent_hidden_dim, batch_size=batch_size)

    def forward(self, observations, edge_index, hidden=None, return_hidden=False):
        
        if len(observations.shape) > 3:
            # batch training
            batch_size, timesteps, regions, _ = observations.size()
            x_recurrent = observations[:,:,:,-1].unsqueeze(-1)
            x_other = observations[:,:,:,:-1]
            
            outputs = []
            for t in range(timesteps):
                hidden = self.rnn(x_recurrent[:, t, :, :], edge_index, H=hidden)
                outputs.append(hidden.unsqueeze(1))
            summary =  torch.cat([x_other, torch.cat(outputs, dim=1)],-1)
        else:
            # select action
            x_recurrent = observations[:,-1].unsqueeze(-1).unsqueeze(0)
            x_other = observations[:,:-1].unsqueeze(0)
            hidden = self.rnn(x_recurrent, edge_index, H=hidden)
            summary =  torch.cat([x_other, hidden],-1)

        if return_hidden:
            return summary, hidden
        else:
            return summary