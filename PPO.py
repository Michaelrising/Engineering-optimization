from utils import *
from models.acnet import ActorCritic
from ProgramEnv import ProgEnv
from copy import deepcopy
import torch
import time
import torch.nn as nn
import numpy as np
from params import configs
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
from scipy.stats import bernoulli
import torchcontrib
device = torch.device(configs.device)


def permute_rows(x):
    ix_i = np.tile(np.arange(x.shape[0]), (x.shape[1], 1)).T
    ix_j = np.random.sample(x.shape).argsort(axis=1)
    return x[ix_i, ix_j]


def uni_instance_gen(n_j, n_m, low, high): # data generator including the duration and the modes
    times = np.random.randint(low=low, high=high, size=(n_j, n_m))
    modes = np.expand_dims(np.arange(1, n_m+1), axis=0).repeat(repeats=n_j, axis=0)
    modes = permute_rows(modes)
    return times, modes


class Memory:
    def __init__(self):
        self.adj_mb = []
        self.fea_mb = []
        self.candidate_mb = []
        self.mask_mb = []
        self.a_mb = []
        self.r_mb = []
        self.done_mb = []
        self.logprobs = []

    def clear_memory(self):
        del self.adj_mb[:]
        del self.fea_mb[:]
        del self.candidate_mb[:]
        del self.mask_mb[:]
        del self.a_mb[:]
        del self.r_mb[:]
        del self.done_mb[:]
        del self.logprobs[:]


class PPO:
    def __init__(self,
                 lr,
                 gamma,
                 k_epochs,
                 eps_clip,
                 num_layers,
                 neighbor_pooling_type,
                 input_dim,
                 hidden_dim,
                 num_mlp_layers_feature_extract,
                 num_mlp_layers_actor,
                 hidden_dim_actor,
                 num_mlp_layers_critic,
                 hidden_dim_critic,
                 ):
        self.lr = lr
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs

        self.policy = ActorCritic(
                                  num_layers=num_layers,
                                  learn_eps=False,
                                  neighbor_pooling_type=neighbor_pooling_type,
                                  input_dim=input_dim,
                                  hidden_dim=hidden_dim,
                                  num_mlp_layers_feature_extract=num_mlp_layers_feature_extract,
                                  num_mlp_layers_actor=num_mlp_layers_actor,
                                  hidden_dim_actor=hidden_dim_actor,
                                  num_mlp_layers_critic=num_mlp_layers_critic,
                                  hidden_dim_critic=hidden_dim_critic,
                                  device=device)
        self.policy_old = deepcopy(self.policy)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.policy_opt = deepcopy(self.policy)
        self.policy_opt.load_state_dict(self.policy.state_dict())
        # self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': 3 * lr},
            {'params': self.policy.critic.parameters(), 'lr': lr},
            {'params': self.policy.feature_extract.parameters(), 'lr': lr}
        ])
        self.swa = torchcontrib.optim.SWA(self.optimizer, swa_start=100, swa_freq=5)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                         step_size=configs.decay_step_size,
                                                         gamma=configs.decay_ratio)

        self.V_loss_2 = nn.MSELoss()

    def Swap_swa_sgd(self, steps, swa_start = 100, swa_freq = 5):
        if steps > swa_start + swa_freq:
            self.swa.swap_swa_sgd()
            return True
        return False

    def update(self, memories, n_tasks, g_pool, n_activities):

        vloss_coef = configs.vloss_coef
        ploss_coef = configs.ploss_coef
        entloss_coef = configs.entloss_coef

        rewards_all_env = []
        adj_mb_t_all_env = []
        fea_mb_t_all_env = []
        candidate_mb_t_all_env = []
        mask_mb_t_all_env = []
        a_mb_t_all_env = []
        old_logprobs_mb_t_all_env = []
        for i in range(len(memories)):
            rewards = []
            discounted_reward = 0
            for reward, is_terminal in zip(reversed(memories[i].r_mb), reversed(memories[i].done_mb)):
                if is_terminal:
                    discounted_reward = 0
                discounted_reward = reward + (self.gamma * discounted_reward)
                rewards.insert(0, discounted_reward)
            rewards = torch.tensor(rewards, dtype=torch.float).to(device)
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
            rewards_all_env.append(rewards)
            adj_mb_t_all_env.append(aggr_obs(torch.stack(memories[i].adj_mb).to(device), n_tasks))
            fea_mb_t = torch.stack(memories[i].fea_mb).to(device)
            fea_mb_t = fea_mb_t.reshape(-1, fea_mb_t.size(-1))
            fea_mb_t_all_env.append(fea_mb_t)
            candidate_mb_t_all_env.append(torch.stack(memories[i].candidate_mb).to(device))
            mask_mb_t_all_env.append(torch.stack(memories[i].mask_mb).to(device).squeeze())
            a_mb_t_all_env.append(torch.stack(memories[i].a_mb).to(device).squeeze())
            old_logprobs_mb_t_all_env.append(torch.stack(memories[i].logprobs).to(device).squeeze().detach())

        mb_g_pool_all_env = [g_pool_cal(g_pool, torch.stack(memories[i].adj_mb).to(device).shape, n_tasks, device) for i in range(len(memories))]

        for _ in range(self.k_epochs):
            loss_sum = 0
            vloss_sum = 0
            for i in range(len(memories)):
                pis, vals = self.policy(x=fea_mb_t_all_env[i],
                                        graph_pool=mb_g_pool_all_env[i],
                                        adj=adj_mb_t_all_env[i],
                                        candidate=candidate_mb_t_all_env[i],
                                        mask=mask_mb_t_all_env[i],
                                        padded_nei=None)
                logprobs, ent_loss = eval_actions(pis.squeeze(), a_mb_t_all_env[i]) ####!!!!
                ratios = torch.exp(logprobs - old_logprobs_mb_t_all_env[i].detach())
                advantages = rewards_all_env[i] - vals.detach()
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
                v_loss = self.V_loss_2(vals.squeeze(), rewards_all_env[i])
                p_loss = - torch.min(surr1, surr2)
                ent_loss = - ent_loss.clone()
                loss = vloss_coef * v_loss + ploss_coef * p_loss + entloss_coef * ent_loss
                loss_sum += loss.mean()
                vloss_sum += v_loss
            self.swa.zero_grad()
            loss_sum.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.swa.step()

        self.policy_old.load_state_dict(self.policy.state_dict())
        if configs.decayflag:
            self.scheduler.step()
        return loss_sum.mean().item(), vloss_sum.mean().item()

def aggr_obs(obs_mb, n_node):
    idxs = obs_mb.coalesce().indices()
    vals = obs_mb.coalesce().values()
    new_idx_row = idxs[1] + idxs[0] * n_node
    new_idx_col = idxs[2] + idxs[0] * n_node
    idx_mb = torch.stack((new_idx_row, new_idx_col))
    adj_batch = torch.sparse.FloatTensor(indices=idx_mb,
                                         values=vals,
                                         size=torch.Size([obs_mb.shape[0] * n_node,
                                                          obs_mb.shape[0] * n_node]),
                                         ).to(obs_mb.device)
    return adj_batch


def g_pool_cal(graph_pool_type, batch_size, n_nodes, device):
    if graph_pool_type == 'average':
        elem = torch.full(size=(batch_size[0]*n_nodes, 1),
                          fill_value=1 / n_nodes,
                          dtype=torch.float32,
                          device=device).view(-1)
    else:
        elem = torch.full(size=(batch_size[0] * n_nodes, 1),
                          fill_value=1,
                          dtype=torch.float32,
                          device=device).view(-1)
    idx_0 = torch.arange(start=0, end=batch_size[0],
                         device=device,
                         dtype=torch.long)
    idx_0 = idx_0.repeat(n_nodes, 1).t().reshape((batch_size[0]*n_nodes, 1)).squeeze()

    idx_1 = torch.arange(start=0, end=n_nodes*batch_size[0],
                         device=device,
                         dtype=torch.long)
    idx = torch.stack((idx_0, idx_1))
    graph_pool = torch.sparse.FloatTensor(idx, elem,
                                          torch.Size([batch_size[0],
                                                      n_nodes*batch_size[0]])
                                          ).to(device)
    return graph_pool


def validate(model, pars):
    feasible_printf = False
    env = ProgEnv(*pars)
    device = torch.device(configs.device)
    g_pool_step = g_pool_cal(graph_pool_type=configs.graph_pool_type,
                             batch_size=torch.Size([1, env.action_space.n, env.action_space.n]),
                             n_nodes=env.action_space.n,
                             device=device)
    adj, fea, candidate, mask , _= env.reset()
    rewards = []
    epi_rewards = 0
    actions = []
    times = []
    step = 0
    while True:
        fea_tensor = torch.from_numpy(np.copy(fea)).to(device).float()
        adj_tensor = torch.from_numpy(np.copy(adj)).to(device).to_sparse().float()
        candidate_tensor = torch.from_numpy(np.copy(candidate)).to(device).float()
        mask_tensor = torch.from_numpy(np.copy(mask)).to(device)
        with torch.no_grad():
            pi, _ = model(x=fea_tensor,
                       graph_pool=g_pool_step,
                       padded_nei=None,
                       adj=adj_tensor,
                       candidate=candidate_tensor.unsqueeze(0),
                       mask=mask_tensor.unsqueeze(0))
            action = greedy_select_action(pi, candidate)
        adj, fea, reward, done, candidate, mask,_, startTime, feasible_info = env.step(action)
        time_fsb, act_fsb, renew_fsb, non_renew_fsb = feasible_info['time'], feasible_info['activity'], feasible_info['renew'], feasible_info['nonrenew']
        epi_rewards += reward
        rewards.append(int(reward))
        actions.append(action)
        times.append(startTime)
        if not feasible_printf:
            if not time_fsb:
                print("Time Feasible error at step {}".format(step))
                feasible_printf = True
            if not act_fsb:
                print("Activity Feasible error at step {}".format(step))
                feasible_printf = True
            if not renew_fsb:
                print("Renew Resource Feasible error at step {}".format(step))
                feasible_printf = True
            if not non_renew_fsb:
                print("Non Renew Resource Feasible error at step {}".format(step))
                feasible_printf = True
        step += 1
        if done:
            break
        ActSeq = []
        ModeSeq = []
        TimeSeq = times
        mode_Number = env.Activity_mode_Num
        cum_sum = np.cumsum(mode_Number) - 1
        for act in actions:
            activity = np.where(cum_sum >= act)[0][0].item()
            mode = max(act - cum_sum[max(int(activity) - 1, 0)], 1)
            ActSeq.append(activity)
            ModeSeq.append(mode)
    return int(epi_rewards), rewards, actions, ActSeq, ModeSeq, TimeSeq


def memory_append(memory,device, adj, fea, candidate, mask, action = None, reward=None, done = None):
    memory.adj_mb.append(torch.from_numpy(np.copy(adj)).to(device).to_sparse().float())
    memory.fea_mb.append(torch.from_numpy(np.copy(fea)).to(device).float())
    memory.candidate_mb.append(torch.from_numpy(np.copy(candidate)).to(device).float())
    # memory.mask_mb.append(torch.from_numpy(np.copy(mask)).to(device))
    if action is not None:
        memory.a_mb.append(torch.from_numpy(np.copy(action)).to(device))
    if reward is not None:
        memory.r_mb.append(torch.from_numpy(np.copy(reward)).to(device))
    if done is not None:
        memory.done_mb.append(done)


def greedy(steps,max_steps, eps):
    # reward_array_mask = sum(np.array(self.eval_reward, dtype=np.float) >= self.max_episode_steps)
    # eps = 1 / self.eval_reward.maxlen * reward_array_mask
    # mask = bool(reward_array_mask > 2 and (self.steps > self.eval_reward.maxlen * self.eval_interval))
    # the probability of explore goes from 0.5 to 0.01 from step starts to num_steps/2 steps and keep in 0.1 after that
    lowest_eps = 0.1
    epsilon = eps - (eps - lowest_eps) * steps / (max_steps * 0.1)
    epsilon = epsilon if epsilon >= lowest_eps else lowest_eps
    decision = bernoulli.rvs(1 - epsilon, size=1).item()
    return decision


