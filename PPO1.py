import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import torch.nn.functional as F

################################## PPO Policy ##################################


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

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, device):
        super(Actor, self).__init__()

        self.device = device
        self.actor = nn.Sequential(
                nn.Linear(state_dim, 128),
                nn.LeakyReLU(),
                nn.Linear(128, 256),
                nn.LeakyReLU(),
                nn.Linear(256, 64),
                nn.LeakyReLU(),
                nn.Linear(64, action_dim)
            )

    def forward(self, state, weights):
        candidate_scores = self.actor(state)
        # weights_reshape = weights.reshape(candidate_scores.size())
        # candidate_scores = torch.mul(candidate_scores, weights_reshape)

        return candidate_scores


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, device):
        super(Critic, self).__init__()

        self.device = device
        # critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state):
        score = self.critic(state)
        return score


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, device):
        super(ActorCritic, self).__init__()

        self.device = device

        self.actor = Actor(state_dim, action_dim, device)
        self.critic = Critic(state_dim, action_dim, device)
        # self.actor = nn.Sequential(
        #         nn.Linear(state_dim, 128),
        #         nn.LeakyReLU(),
        #         nn.Linear(128, 256),
        #         nn.LeakyReLU(),
        #         nn.Linear(256, 64),
        #         nn.LeakyReLU(),
        #         nn.Linear(64, action_dim)
        #     )
        #
        # # critic
        # self.critic = nn.Sequential(
        #     nn.Linear(state_dim, 128),
        #     nn.LeakyReLU(),
        #     nn.Linear(128, 256),
        #     nn.LeakyReLU(),
        #     nn.Linear(256, 128),
        #     nn.LeakyReLU(),
        #     nn.Linear(128, 1)
        # )

    def forward(self):
        raise NotImplementedError

    def act(self, state, mask, weights):
        candidate_scores = self.actor(state, weights)
        # weights_reshape = weights.reshape(candidate_scores.size())
        # candidate_scores = candidate_scores * weights_reshape
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
            # weights_reshape = weights.reshape(candidate_scores.size())
            # candidate_scores = candidate_scores * weights_reshape
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
        # weights_reshape = weights.reshape(candidate_scores.size())
        # candidate_scores = candidate_scores * weights_reshape
        mask_reshape = mask.reshape(candidate_scores.size())
        candidate_scores[mask_reshape] = float('-inf')
        action_probs = F.softmax(candidate_scores, dim=1)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy

class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip,
                 has_continuous_action_space, num_env, device, decay_step_size=1000, decay_ratio=0.5,
                 action_std_init=0.6):

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.device = device

        self.buffer = RolloutBuffer()
        self.buffers = [RolloutBuffer() for _ in range(num_env)]

        self.policy = ActorCritic(state_dim, action_dim, self.device).to(
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
                                      self.device).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def set_action_std(self, new_action_std):

        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)

        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")

        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")

        print("--------------------------------------------------------------------------------------------")

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
            old_states = torch.squeeze(torch.stack(self.buffers[i].states, dim=0)).detach().to(self.device)
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





