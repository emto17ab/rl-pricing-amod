import torch
import torch.nn as nn
import numpy as np
from torch_geometric.nn import GCNConv
from torch_geometric_temporal import TGCN2
import torch.nn.functional as F


class Summarizer(nn.Module):

    def __init__(self, recurrent_input_dim, recurrent_hidden_dim, batch_size):

        super().__init__()
        # self.rnn = TGCN2(in_channels=recurrent_input_dim, out_channels=recurrent_hidden_dim, batch_size=batch_size)
        self.rnn = nn.GRU(input_size = recurrent_input_dim, hidden_size = recurrent_hidden_dim, batch_first=True)

    def forward(self, observations, edge_index, hidden=None, return_hidden=False):
        
        if len(observations.shape) > 3:
            # batch training
            batch_size, timesteps, regions, _ = observations.size()
            x_recurrent = observations[:,:,:,-2:].unsqueeze(-1)
            x_other = observations[:,:,:,:-2]
            
            x_recurrent = x_recurrent.view(batch_size*regions,timesteps,2)
            outputs,_ = self.rnn(x_recurrent)
            outputs = outputs.reshape(batch_size,timesteps,regions,-1)
            summary =  torch.cat([x_other, outputs],-1)
        else:
            # select action
            regions = observations.shape[0]
            x_other = observations[:,:-2].unsqueeze(0)
            x_recurrent = observations[:,-2:].unsqueeze(-2)
            _, hidden = self.rnn(x_recurrent, hidden)
            summary =  torch.cat([x_other, hidden],-1)

        if return_hidden:
            return summary, hidden
        else:
            return summary