import torch
import torch.nn as nn
import torch.nn.functional as F
from Codes.gnn_models.mlp import MLP


class GraphCNN(nn.Module):
    def __init__(self,
                 num_layers,
                 num_mlp_layers,
                 input_dim,
                 hidden_dim,
                 learn_eps,
                 neighbor_pooling_type,
                 device):

        super(GraphCNN, self).__init__()

        self.device = device
        self.num_layers = num_layers
        self.neighbor_pooling_type = neighbor_pooling_type
        self.learn_eps = learn_eps
        self.mlps = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        self.f1 = nn.Linear(26, 13)

        for layer in range(self.num_layers-1):
            if layer == 0:
                self.mlps.append(MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim))
            else:
                self.mlps.append(MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim))

            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

    def next_layer_eps(self, h, layer, padded_neighbor_list = None, Adj_block = None):
        if self.neighbor_pooling_type == "max":
            pooled = self.maxpool(h, padded_neighbor_list)
        else:
            pooled = torch.mm(Adj_block, h)
            if self.neighbor_pooling_type == "average":
                degree = torch.mm(Adj_block, torch.ones((Adj_block.shape[0], 1)).to(self.device))
                pooled = pooled/degree
        pooled = pooled + (1 + self.eps[layer])*h
        pooled_rep = self.mlps[layer](pooled)
        h = self.batch_norms[layer](pooled_rep)
        h = F.relu(h)
        return h

    def next_layer(self, h, layer, padded_neighbor_list = None, Adj_block = None):
        if self.neighbor_pooling_type == "max":
            pooled = self.maxpool(h, padded_neighbor_list)
        else:
            pooled = torch.sparse.mm(Adj_block, h)
            if self.neighbor_pooling_type == "average":
                degree = torch.mm(Adj_block, torch.ones((Adj_block.shape[0], 1)).to(self.device))
                pooled = pooled/degree
        pooled_rep = self.mlps[layer](pooled)
        h = self.batch_norms[layer](pooled_rep)
        h = F.relu(h)
        return h

    def forward(self,
                x,
                graph_pool,
                padded_nei,
                adj):

        x_concat = x # self.f1(x.float())
        graph_pool = graph_pool

        if self.neighbor_pooling_type == "max":
            padded_neighbor_list = padded_nei
        else:
            Adj_block = adj

        h = x_concat

        for layer in range(self.num_layers-1):
            if self.neighbor_pooling_type == "max" and self.learn_eps:
                h = self.next_layer_eps(h, layer, padded_neighbor_list=padded_neighbor_list)
            elif not self.neighbor_pooling_type == "max" and self.learn_eps:
                h = self.next_layer_eps(h, layer, Adj_block=Adj_block)
            elif self.neighbor_pooling_type == "max" and not self.learn_eps:
                h = self.next_layer(h, layer, padded_neighbor_list=padded_neighbor_list)
            elif not self.neighbor_pooling_type == "max" and not self.learn_eps:
                h = self.next_layer(h, layer, Adj_block=Adj_block)

        h_nodes = h.clone()
        pooled_h = torch.sparse.mm(graph_pool, h)
        return pooled_h, h_nodes
