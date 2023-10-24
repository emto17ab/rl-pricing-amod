import torch
import torch.nn as nn
import numpy as np
from torch_geometric.nn import GCNConv
import torch.nn.functional as F


class Summarizer(nn.Module):

    def __init__(self, input_dim, nregion, hidden_dim, num_layers=2, recurrent_type='lstm'):

        super().__init__()
        self.conv1 = GCNConv(input_dim, input_dim)

        if recurrent_type == 'lstm':
            self.rnn = nn.LSTM(input_dim*nregion, hidden_dim, batch_first=True, num_layers=num_layers)
        elif recurrent_type == 'rnn':
            self.rnn = nn.RNN(input_dim*nregion, hidden_dim, batch_first=True, num_layers=num_layers)
        elif recurrent_type == 'gru':
            self.rnn = nn.GRU(input_dim*nregion, hidden_dim, batch_first=True, num_layers=num_layers)
        else:
            raise ValueError(f"{recurrent_type} not recognized")

    def forward(self, observations, edge_index, hidden=None, return_hidden=False):
        
        if len(observations.shape) > 3:
            batch_size, timesteps, regions, features = observations.size()

            x = observations.view(batch_size*timesteps, regions, features)
            out = F.relu(self.conv1(x, edge_index))
            out = out.view(batch_size, timesteps, -1)
        else:       
            out = F.relu(self.conv1(observations, edge_index))
            out = out.view(1, 1,-1)

        self.rnn.flatten_parameters()
        summary, hidden = self.rnn(out, hidden)
        if return_hidden:
            return summary, hidden
        else:
            return summary