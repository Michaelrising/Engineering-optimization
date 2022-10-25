from PPO1 import *
from utils import *
import torch
import argparse
from params import configs
import time
import numpy as np
from envs import Env
from ProgramEnv import ProgEnv


def test(modelPath, pars, device):
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(configs.torch_seed)
    env = ProgEnv(*pars)
    state_dim = env.action_space.n * 2
    action_dim = env.action_space.n
    policy = ActorCritic(state_dim, action_dim, device).to(device)
    policy.load_state_dict(torch.load(modelPath))
    policy.eval()
    _, fea, _, mask = env.reset()
    rewards = 0
    actions = []
    times = []
    while True:
        fea_tensor = torch.from_numpy(np.copy(fea)).to(device).float()
        with torch.no_grad():
            action, _ = policy.act_exploit(fea_tensor, mask)
        _, fea, reward, done, _, mask, time = env.step(action.item())
        rewards += reward
        actions.append(action.item())
        times.append(time)
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
    return rewards, ActSeq, ModeSeq, TimeSeq


if __name__ == '__main__':
    modelpath = './log/summary/20220319-0945/PPO-ProgramEnv-converge-model.pth'
    pars = (
        configs.filepath, configs.Target_T, configs.price_renew, configs.price_non, configs.penalty0, configs.penalty1, configs.mode, configs.ppo)
    # state_dim = configs.input_dim
    # action_dim = configs.action_dim
    device = configs.device
    rewards, ActSeq, ModeSeq, TimeSeq = test(modelpath, pars, device)
    print('================================================')
    print('============== The Final Reward is: ============')
    print('{}'.format(np.round(rewards, 3)))
    print('======= The Final Activity Sequence is: ========')
    print(ActSeq)
    print('========= The Final Mode Sequence is: ==========')
    print(ModeSeq)
    print('========= The Final Time Sequence is: ==========')
    print(TimeSeq)
    print('================================================')



