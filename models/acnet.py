import torch.nn as nn
from models.mlp import MLPActor
from models.mlp import MLPCritic
import torch.nn.functional as F
from models.graph import GraphCNN
import torch
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self,
                 num_layers,
                 learn_eps,
                 neighbor_pooling_type,
                 input_dim,
                 hidden_dim,
                 num_mlp_layers_feature_extract,
                 num_mlp_layers_actor,
                 hidden_dim_actor,
                 num_mlp_layers_critic,
                 hidden_dim_critic,
                 device
                 ):
        super(ActorCritic, self).__init__()
        # self.n_j = n_j
        # self.n_m = n_m
        # self.n_ops_perjob = n_m
        self.device = device

        self.feature_extract = GraphCNN(num_layers=num_layers,
                                        num_mlp_layers=num_mlp_layers_feature_extract,
                                        input_dim=input_dim,
                                        hidden_dim=hidden_dim,
                                        learn_eps=learn_eps,
                                        neighbor_pooling_type=neighbor_pooling_type,
                                        device=device).to(device)
        self.actor = MLPActor(num_mlp_layers_actor, hidden_dim*2, hidden_dim_actor, 1).to(device)
        self.critic = MLPCritic(num_mlp_layers_critic, hidden_dim, hidden_dim_critic, 1).to(device)

    def forward(self,
                x,
                graph_pool,
                padded_nei,
                adj,
                candidate,
                mask):

        h_pooled, h_nodes = self.feature_extract(x=x,
                                                 graph_pool=graph_pool,
                                                 padded_nei=padded_nei,
                                                 adj=adj)
        dummy = candidate.unsqueeze(-1).expand(-1, candidate.shape[1], h_nodes.size(-1)).long()
        candidate_feature = torch.gather(h_nodes.reshape(dummy.size(0), -1, dummy.size(-1)), 1, dummy)
        h_pooled_repeated = h_pooled.unsqueeze(1).expand_as(candidate_feature)
        concateFea = torch.cat((candidate_feature, h_pooled_repeated), dim=-1)
        candidate_scores = self.actor(concateFea)
        mask_reshape = mask.reshape(candidate_scores.size())
        candidate_scores[mask_reshape] = float('-inf')
        pi = F.softmax(candidate_scores, dim=1)
        v = self.critic(h_pooled)
        return pi, v


