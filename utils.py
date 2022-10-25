from torch.distributions.categorical import Categorical


def select_action(p, cadidate, memory):
    dist = Categorical(p.squeeze())
    s = dist.sample()
    if memory is not None :
        memory.logprobs.append(dist.log_prob(s))
    return cadidate[s]


def eval_actions(p, actions):

    softmax_dist = Categorical(p)
    ret = softmax_dist.log_prob(actions).reshape(-1)
    entropy = softmax_dist.entropy().mean()
    return ret, entropy


def greedy_select_action(p, candidate,  memory=None):
    _, index = p.squeeze().max(0)
    action = candidate[index]
    dist = Categorical(p.squeeze())
    if memory is not None :
        memory.logprobs.append(dist.log_prob(index))
    return action


def sample_select_action(p, candidate):
    dist = Categorical(p.squeeze())
    s = dist.sample()
    return candidate[s]
