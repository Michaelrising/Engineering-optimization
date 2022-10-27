import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F


class MLP_Actor(nn.Module):
    def __init__(self, state_dim, action_dim, device):
        super(MLP_Actor, self).__init__()

        self.device = device
        self.actor = nn.Sequential(
                nn.Linear(state_dim, 128),
                nn.LeakyReLU(),
                nn.Linear(128, 256),
                nn.LeakyReLU(),
                nn.Linear(256, 64),
                nn.LeakyReLU(),
                nn.Linear(64, action_dim),
            )

    def forward(self, state, weights):
        candidate_scores = self.actor(state)
        # weights_reshape = weights.reshape(candidate_scores.size())
        # candidate_scores = torch.mul(candidate_scores, weights_reshape)

        return candidate_scores


class MLP_Critic(nn.Module):
    def __init__(self, state_dim, action_dim, device):
        super(MLP_Critic, self).__init__()

        self.device = device
        # critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, state):
        score = self.critic(state)
        return score


class CNN_Actor(nn.Module):
    def __init__(self, state_dim, action_dim, device):
        super(CNN_Actor, self).__init__()

        self.device = device
        self.actor = nn.Sequential(
                nn.Conv1d(in_channels=1, out_channels=1, kernel_size=(1, 3)),
                nn.LeakyReLU(),
                nn.Linear(1, 64),
                nn.LeakyReLU(),
                nn.Linear(64, 1),
            )

    def forward(self, state, weights):
        candidate_scores = self.actor(state)
        # weights_reshape = weights.reshape(candidate_scores.size())
        # candidate_scores = torch.mul(candidate_scores, weights_reshape)

        return candidate_scores.squeeze()


class CNN_Critic(nn.Module):
    def __init__(self, state_dim, action_dim, device):
        super(CNN_Critic, self).__init__()

        self.device = device
        # critic
        self.critic0 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=(1, 3)),
            nn.LeakyReLU(),
            nn.Linear(1, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1),
            nn.LeakyReLU(),
        )
        self.critic1 = nn.Linear(action_dim, 1)

    def forward(self, state):
        score = self.critic0(state)
        score = self.critic1(score.squeeze())
        return score.squeeze()


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, device, acnet='mlp'):
        super(ActorCritic, self).__init__()

        self.device = device
        if acnet == 'mlp':
            self.actor = MLP_Actor(state_dim, action_dim, device)
            self.critic = MLP_Critic(state_dim, action_dim, device)
        elif acnet == 'cnn':
            self.actor = CNN_Actor(state_dim, action_dim, device)
            self.critic = CNN_Critic(state_dim, action_dim, device)

    def forward(self):
        raise NotImplementedError

    def act(self, state, mask, weights):
        candidate_scores = self.actor(state, weights)
        mask_reshape = mask.reshape(candidate_scores.size())
        candidate_scores[mask_reshape] = float('-inf')
        action_probs = F.softmax(candidate_scores.reshape(1, -1), dim=1)
        dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action.detach(), action_logprob.detach()

    def act_exploit(self, state, mask, weights):
        with torch.no_grad():
            candidate_scores = self.actor(state, weights)
            mask_reshape = mask.reshape(candidate_scores.size())
            candidate_scores[mask_reshape] = float('-inf')
            action_probs = F.softmax(candidate_scores.reshape(1, -1), dim=1)
            dist = Categorical(action_probs)
            greedy_action = torch.argmax(
                action_probs, dim=1, keepdim=False)
            action_logprob = dist.log_prob(greedy_action)
        return greedy_action.detach(), action_logprob.detach()

    def evaluate(self, state, action, mask, weights):
        candidate_scores = self.actor(state, weights)
        mask_reshape = mask.reshape(candidate_scores.size())
        candidate_scores[mask_reshape] = float('-inf')
        action_probs = F.softmax(candidate_scores, dim=1)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy


