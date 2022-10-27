from PPO import *
import torch
from params import configs
import numpy as np
from ProgramEnv import ProgEnv

def greedy_test(modelPath, pars, device):
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(configs.torch_seed)
    test_env = ProgEnv(*pars)
    state_dim = test_env.action_space.n * 2
    action_dim = test_env.action_space.n
    # upload policy
    policy = ActorCritic(state_dim, action_dim, device).to(device)
    policy.load_state_dict(torch.load(modelPath))
    policy.eval()

    _, fea, _, mask, weights  = test_env.reset()
    epi_rewards = 0
    actions = []
    times = []
    rewards = []
    time_feasible = []
    activity_feasible = []
    renew_feasible = []
    non_renew_feasible = []
    feasible_printf = False
    step = 0
    while True:
        fea_tensor = torch.from_numpy(np.copy(fea)).to(device).float()
        weights_tensor = torch.from_numpy(np.copy(weights)).to(device).float()
        with torch.no_grad():
            action, _ = policy.act_exploit(fea_tensor, mask, weights_tensor)
        _, fea, reward, done, _, mask, weights, time, feasible_info = test_env.step(action.item())
        epi_rewards += reward
        rewards.append(int(reward))
        actions.append(action.item())
        time_fsb, act_fsb, renew_fsb, non_renew_fsb = feasible_info['time'], feasible_info['activity'], feasible_info['renew'], feasible_info['nonrenew']
        time_feasible.append(time_fsb)
        activity_feasible.append(act_fsb)
        renew_feasible.append(renew_fsb)
        non_renew_feasible.append(non_renew_fsb)
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

        times.append(time)
        if done:
            break

    ActSeq = []
    ModeSeq = []
    TimeSeq = times
    mode_Number = test_env.Activity_mode_Num
    cum_sum = np.cumsum(mode_Number) - 1
    for act in actions:
        activity = np.where(cum_sum >= act)[0][0].item()
        mode = max(act - cum_sum[max(int(activity)-1, 0)], 1)
        ActSeq.append(activity)
        ModeSeq.append(mode)
    return epi_rewards, rewards, actions, ActSeq, ModeSeq, TimeSeq


if __name__ == '__main__':
    modelpath = ''
    pars = (
        configs.filepath, configs.Target_T, configs.price_renew, configs.price_non, configs.penalty0, configs.penalty1, configs.mode, configs.ppo)
    device = configs.device
    epi_rewards, rewards, actions, ActSeq, ModeSeq, TimeSeq = greedy_test(modelpath, pars, device)
    print('================================================')
    print('============== The Final Reward is: ============')
    print('{}'.format(np.round(epi_rewards, 3)))
    print('======= The Final Activity Sequence is: ========')
    print(ActSeq)
    print('========= The Final Mode Sequence is: ==========')
    print(ModeSeq)
    print('========= The Final Time Sequence is: ==========')
    print(TimeSeq)
    print('================================================')



