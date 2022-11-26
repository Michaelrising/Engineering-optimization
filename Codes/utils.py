import torch
import numpy as np

def set_device(cuda=None):
    print("============================================================================================")

    # set device to cpu or cuda
    device = torch.device('cpu')

    if torch.cuda.is_available() and cuda is not None:
        device = torch.device('cuda:' + str(cuda))
        torch.cuda.empty_cache()
        print("Device set to : " + str(torch.cuda.get_device_name(device)))
    else:
        print("Device set to : cpu")

    print("============================================================================================")
    return device


def greedy_evaluate(test_env, model):
    device = model.device
    _, fea, _, mask  = test_env.reset()
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
        # weights_tensor = torch.from_numpy(np.copy(weights)).to(device).float()
        with torch.no_grad():
            action, _ = model.act_exploit(fea_tensor, mask)
        _, fea, reward, done, _, mask, time, feasible_info = test_env.step(action.item())
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
