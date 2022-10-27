import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F
from actor_critic import ActorCritic


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.masks = []
        self.weights = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.masks[:]
        del self.weights[:]

################################## PPO Policy ##################################

class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                  num_env, device, acnet, decay_step_size=1000, decay_ratio=0.5):


        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.device = device

        self.buffer = RolloutBuffer()
        self.buffers = [RolloutBuffer() for _ in range(num_env)]

        self.policy = ActorCritic(state_dim, action_dim, self.device, acnet).to(
            self.device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic},
            # {'params': self.policy.conv1.parameters(), 'lr': lr_actor}
        ])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                         step_size=decay_step_size,
                                                         gamma=decay_ratio)

        self.policy_old = ActorCritic(state_dim, action_dim,
                                      self.device, acnet).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()


    def select_action(self, state, mask, weights):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            action, action_logprob = self.policy_old.act(state, mask, weights)

            return state, action.item(), action_logprob

    def greedy_select_action(self, state, mask, weights):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            try:
                action, action_logprob = self.policy_old.act_exploit(state, mask, weights)
            except ValueError:
                print(weights)
                print(mask)

            return state, action.item(), action_logprob

    def update(self, decayflag, grad_clamp=None):
        rewards_all_envs = []
        old_states_all_envs = []
        old_actions_all_envs = []
        old_logprobs_all_envs = []
        old_masks_all_envs = []
        old_weights_all_envs = []
        for i in range(len(self.buffers)):
            # Monte Carlo estimate of returns
            rewards = []
            discounted_reward = 0
            for reward, is_terminal in zip(reversed(self.buffers[i].rewards), reversed(self.buffers[i].is_terminals)):
                if is_terminal:
                    discounted_reward = 0
                discounted_reward = reward + (self.gamma * discounted_reward)
                rewards.insert(0, discounted_reward)

            # Normalizing the rewards
            rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

            # convert list to tensor
            old_states = torch.squeeze(torch.stack(self.buffers[i].states, dim=0), 1).detach().to(self.device)
            old_actions = torch.squeeze(torch.tensor(self.buffers[i].actions)).detach().to(self.device)
            old_logprobs = torch.squeeze(torch.stack(self.buffers[i].logprobs, dim=0)).detach().to(self.device)
            old_masks = torch.squeeze(torch.stack(self.buffers[i].masks, dim=0)).detach().to(self.device)
            old_weights = torch.squeeze(torch.stack(self.buffers[i].weights, dim=0)).detach().to(self.device)

            rewards_all_envs.append(rewards)
            old_states_all_envs.append(old_states)
            old_actions_all_envs.append(old_actions)
            old_logprobs_all_envs.append(old_logprobs)
            old_masks_all_envs.append(old_masks)
            old_weights_all_envs.append(old_weights)

        # Optimize policy for K epochs
        VLoss = 0
        for _ in range(self.K_epochs):
            # for index in BatchSampler(SubsetRandomSampler(range(self.mini_batch), self.batch_size, True)):
            loss_sum = 0
            vloss_sum = 0
            for i in range(len(self.buffers)):
                # Evaluating old actions and values
                logprobs, state_values, dist_entropy = self.policy.evaluate(old_states_all_envs[i],
                                                                            old_actions_all_envs[i],
                                                                            old_masks_all_envs[i],
                                                                            old_weights_all_envs[i])

                # match state_values tensor dimensions with rewards tensor
                state_values = torch.squeeze(state_values)

                # Finding the ratio (pi_theta / pi_theta__old)
                ratios = torch.exp(logprobs - old_logprobs_all_envs[i].detach())

                # Finding Surrogate Loss
                advantages = rewards_all_envs[i] - state_values.detach()
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

                # final loss of clipped objective PPO
                v_loss = self.MseLoss(state_values, rewards_all_envs[i])
                loss = -torch.min(surr1, surr2) + 0.5 * v_loss - 0.01 * dist_entropy
                loss_sum += loss.mean()
                vloss_sum += v_loss.mean()

            # take gradient step
            self.optimizer.zero_grad()
            loss_sum.backward()
            if grad_clamp is not None:
                nn.utils.clip_grad_norm_(self.policy.parameters(), grad_clamp)
            self.optimizer.step()
            VLoss += vloss_sum
        if decayflag:
            self.scheduler.step()
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        for i in range(len(self.buffers)):
            self.buffers[i].clear()
        return VLoss.cpu().detach() / self.K_epochs

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))





