import numpy as np
from PPO1 import *
from utils import *
from ProgramEnv import ProgEnv
import torch
import time
from params import configs
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
device = torch.device(configs.device)



################################## set device ##################################

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
    _, fea, _, mask,weights  = test_env.reset()
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
            action, _ = model.act_exploit(fea_tensor, mask, weights_tensor)
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


################################### Training ###################################

def train(summary_dir, pars):
    ################## set device ##################
    device = 'cuda:0' #set_device() if configs.cuda_cpu == "cpu" else set_device(configs.cuda)

    ####### initialize environment hyperparameters ######

    num_env = configs.num_envs
    max_updates = configs.max_updates
    eval_interval = 50
    has_continuous_action_space = False  # continuous action space; else discrete

    max_ep_len = 120  # max timesteps in one episode

    print_freq = 2  # print avg reward in the interval (in num updating steps)
    log_freq = 2  # log avg reward in the interval (in num updating steps)
    action_std = 0.6  # starting std for action distribution (Multivariate Normal)
    explore_eps = 0.8

    ####################################################
    ################ PPO hyperparameters ################

    decay_step_size = 1000
    decay_ratio = 0.8
    grad_clamp = 0.2
    update_timestep = 2  # update policy every n epoches
    K_epochs = 2  # update policy for K epochs in one PPO update

    eps_clip = 0.2  # clip parameter for PPO
    gamma = 0.99  # discount factor

    lr_actor = 0.0001*3  # learning rate for actor network
    lr_critic = 0.00005*3  # learning rate for critic network

    ########################### Env Parameters ##########################

    envs = [ProgEnv(*pars) for _ in range(configs.num_envs)]
    torch.manual_seed(configs.torch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(configs.torch_seed)
    np.random.seed(configs.np_seed_train)

    batch_size = envs[0].action_space.n


    test_env = ProgEnv(*pars) # gym.make(configs.env_id, patient=patient).unwrapped

    # state space dimension
    state_dim = envs[0].action_space.n * 3

    # action space dimension
    action_dim = envs[0].action_space.n
    env_name = 'ProgramEnv'
    print("training environment name : " + env_name)
    ###################### logging ######################

    #### log files for multiple runs are NOT overwritten

    log_dir = "log/summary/"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    t = datetime.now().strftime("%Y%m%d-%H%M")
    # summary_dir = log_dir + '/' + str(t) + "-num_env-" + str(num_env)
    writer = SummaryWriter(log_dir=summary_dir)

    #####################################################



    #####################################################

    ############# print all hyperparameters #############

    print("--------------------------------------------------------------------------------------------")
    print("Reward mode:" + configs.mode)
    print("num of envs : " + str(num_env))
    print("max training updating times : ", max_updates)
    print("max timesteps per episode : ", max_ep_len)

    # print("model saving frequency : " + str(save_model_freq) + " episodes")
    print("log frequency : " + str(log_freq) + " episodes")
    print("printing average reward over episodes in last : " + str(print_freq) + " episodes")

    print("--------------------------------------------------------------------------------------------")

    print("state space dimension : ", state_dim)
    print("action space dimension : ", action_dim)

    print("--------------------------------------------------------------------------------------------")

    print("Initializing a discrete action space policy")

    print("--------------------------------------------------------------------------------------------")
    print("The initial explore rate : " + str(explore_eps) + " and initial exploit rate is : 1- " + str(explore_eps))

    print("PPO update frequency : " + str(update_timestep) + " episodes")
    print("PPO K epochs : ", K_epochs)
    print("PPO epsilon clip : ", eps_clip)
    print("discount factor (gamma) : ", gamma)

    print("--------------------------------------------------------------------------------------------")
    if configs.decayflag:
        print("decaying optimizer with step size : ", decay_step_size, " decay ratio : ", decay_ratio)
    print("optimizer learning rate actor : ", lr_actor)
    print("optimizer learning rate critic : ", lr_critic)

    #####################################################

    print("============================================================================================")

    ################# training procedure ################

    # initialize a PPO agent
    ppo_agent = PPO(
        state_dim,
        action_dim,
        lr_actor,
        lr_critic,
        gamma,
        K_epochs,
        eps_clip,
        has_continuous_action_space,
        num_env,
        device,
        decay_step_size,
        decay_ratio,
        action_std)

    # ppo_agent.load(
    #     "PPO_pretrained/analysis/patient025_20220121-1603-m1-0.5-AI-0.8_PPO_gym_cancerCancerControl-v0_0_0.pth")
    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    print("============================================================================================")

    # logging file
    # log_f = open(act_log_f_name, "w+")
    # log_f.write('Sample Actions List, Greedy Actions List\n')

    # ppo_agent.buffers = [ppo_agent.buffer for _ in range(configs.num_env)]
    ep_rewards = [[] for _ in range(num_env)]
    ep_dones = [[] for _ in range(num_env)]
    num_episods = [0 for _ in range(num_env)]
    flag_step = [0 for _ in range(num_env)]
    # training loop
    reward_record = -1000000
    for i_update in range(max_updates):
        for i, env in enumerate(envs):
            i_step = 1
            num_episods[i] = 0
            ep_rewards[i] = []
            ep_dones[i] = []
            flag_step[i] = 0
            # actlist =[ 0,  1,  5,  8, 57, 33, 21, 27, 51, 39, 45, 13,  3, 29, 53, 35,  6, 47,
            #  41, 23, 59, 16,  4, 34, 46, 40, 10, 58, 52, 14, 22, 28, 31, 55, 37, 49,
            #  43, 61, 19, 25, 62, 63, 65, 64, 66]
            # actlist = [0, 1, 8, 56, 26, 45, 51, 21, 13, 33, 39, 5, 41, 29, 6, 47, 53, 35, 59, 16, 23, 3,
                       # 34, 46, 40, 11, 49, 58, 61, 43, 55, 52, 28, 15, 22, 37, 31, 19, 4, 25, 62, 63, 65, 64, 66]
            while i_step < batch_size:
                num_episods[i] += 1
                eps = max(- max(i_update - 10000, 0) * (explore_eps - 0.5) / 10000 + explore_eps, 0.5)
                determine = np.random.choice(2, p=[1 - eps, eps])  # explore epsilon
                _, fea, _, mask, weights = env.reset()
                while True:
                    # select action with policy, with torch.no_grad()
                    mask_tensor = torch.from_numpy(np.copy(mask)).to(device)
                    weights_tensor = torch.from_numpy(np.copy(weights)).to(device)
                    ppo_agent.buffers[i].masks.append(mask_tensor)
                    ppo_agent.buffers[i].weights.append(weights_tensor)
                    # print(weights_tensor)
                    state_tensor, action, action_logprob = ppo_agent.select_action(fea, mask, weights_tensor) \
                        if determine else ppo_agent.greedy_select_action(fea, mask, weights_tensor)  # state_tensor is the tensor of current state
                    ppo_agent.buffers[i].states.append(state_tensor)
                    ppo_agent.buffers[i].actions.append(action)
                    ppo_agent.buffers[i].logprobs.append(action_logprob)
                    # action = actlist[i_step-1]
                    # print(i_step-1)
                    _, fea, reward, done, _, mask, weights, time, _ = env.step(action)
                    # print("Reward: {} \t Time: {}".format(reward, time))

                    # saving reward and is_terminals
                    ppo_agent.buffers[i].rewards.append(reward)
                    ppo_agent.buffers[i].is_terminals.append(done)

                    # print(action)
                    ep_rewards[i].append(reward)
                    ep_dones[i].append(done)
                    i_step += 1
                    # break; if the episode is over
                    if done:
                        flag_step[i] = i_step
                        break

        mean_rewards_all_env = sum([sum(ep_rewards[i][: flag_step[i]])/num_episods[i] for i in range(num_env)]) / num_env

        # update PPO agent
        if i_update % update_timestep == 0:
            loss = ppo_agent.update(decayflag=configs.decayflag, grad_clamp=grad_clamp)

            # log in logging file
            # if i_update % log_freq == 0:
            print("steps:{} \t\t rewards:{}".format(i_update, np.round(mean_rewards_all_env, 3)))
            writer.add_scalar('VLoss', loss, i_update)
        writer.add_scalar("Reward/train", mean_rewards_all_env, i_update)

        if i_update % eval_interval == 0:
            # rewards, survivalMonth, actions, states, colors = evaluate(test_env, ppo_agent.policy_old, eval_times)
            g_rewards, rewardSeq, ActionSeq, ActSeq, ModeSeq, TimeSeq = greedy_evaluate(test_env, ppo_agent.policy_old)
            # writer.add_scalar("Reward/evaluate", rewards, i_update)
            writer.add_scalar("Reward/greedy_evaluate", g_rewards, i_update)
            torch.save(ppo_agent.policy_old.state_dict(),summary_dir + '/PPO-ProgramEnv-converge-model-{}.pth'.format(configs.filepath[2:-3]))
            print(" Test Total reward:{} \n Test rewards List:{} \n Test Actions:{} \n Test Acts:{} \n Test Modes: {} \n Test Times:{}".format(np.round(g_rewards, 3),rewardSeq, ActionSeq, ActSeq,ModeSeq, TimeSeq))
        #
    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")


if __name__ == '__main__':
    crt_time = datetime.now().strftime("%Y%m%d-%H%M")
    summary_dir = os.path.join("log", 'summary', str(crt_time))
    if not os.path.exists(summary_dir):
        os.makedirs(summary_dir)
    total1 = time.time()
    pars = (
        configs.filepath, configs.Target_T, configs.price_renew, configs.price_non, configs.penalty0, configs.penalty1, configs.mode, configs.ppo)
    train(summary_dir, pars)








