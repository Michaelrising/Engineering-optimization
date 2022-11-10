from PPO import *
from utils import *
from Codes.ProgramEnv import ProgEnv
import torch
import time
from params import configs
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime


def train(summary_dir, pars):

    ################## set device ##################
    device = configs.device
    print("============================================================================================")
    # set device to cpu or cuda
    if torch.cuda.is_available() and device == 'cuda:0':
        torch.cuda.empty_cache()
        print("Device set to : " + str(torch.cuda.get_device_name(device)))
    else:
        print("Device set to : cpu")

    print("============================================================================================")

    ####### initialize environment hyperparameters ######
    num_env = configs.num_envs
    max_updates = configs.max_updates
    eval_interval = configs.eval_interval
    has_continuous_action_space = False  # continuous action space; else discrete

    print_freq = configs.print_freq  # print avg reward in the interval (in num updating steps)
    log_freq = configs.log_freq  # log avg reward in the interval (in num updating steps)
    action_std = 0.6  # starting std for action distribution (Multivariate Normal)
    explore_upper_eps = configs.explore_upper_eps
    explore_lower_eps = configs.explore_lower_eps
    exploit_init_step = configs.exploit_init_step

    ####################################################
    ################ PPO hyperparameters ################

    decay_step_size = configs.decay_step_size
    decay_ratio = configs.decay_ratio
    grad_clamp = configs.grad_clamp
    update_timestep = configs.update_freq  # update policy every n epoches
    K_epochs = configs.k_epochs  # update policy for K epochs in one PPO update

    eps_clip = configs.eps_clip  # clip parameter for PPO
    gamma = configs.gamma  # discount factor
    acnet = configs.acnet

    lr_actor = configs.lr_actor  # learning rate for actor network
    lr_critic = configs.lr_critic  # learning rate for critic network

    ########################### Env Parameters ##########################

    envs = [ProgEnv(*pars) for _ in range(configs.num_envs)]

    batch_size = envs[0].action_space.n

    test_env = ProgEnv(*pars)

    # state space dimension
    if configs.acnet == 'mlp':
        state_dim = envs[0].action_space.n * 3
        # action space dimension
        action_dim = envs[0].action_space.n
    else:
        state_dim = envs[0].action_space.n
        # action space dimension
        action_dim = envs[0].action_space.n
    env_name = 'ProgramEnv'
    print("training environment name : " + env_name)

    print("--------------------------------------------------------------------------------------------")

    print("state space dimension : ", state_dim)
    print("action space dimension : ", action_dim)
    print("penalty mode:" + configs.penalty_mode)
    print("num of envs : " + str(num_env))
    ###################### logging ######################

    #### log files for multiple runs are NOT overwritten
    writer = SummaryWriter(log_dir=summary_dir)
    checkpoint_format = summary_dir + '/PPO-ProgramEnv-converge-model-{}.pth'.format(configs.filepath[8:-4])

    print('summary save at: ', summary_dir)
    print('model save as: ', checkpoint_format)

    ############# print all hyperparameters #############

    print("--------------------------------------------------------------------------------------------")
    print('the actor and critic network: ', configs.acnet)

    print("max training updating times : ", max_updates)
    print("log frequency : " + str(log_freq) + " episodes")
    print("printing average reward over episodes in last : " + str(print_freq) + " episodes")


    print("--------------------------------------------------------------------------------------------")

    print("Initializing a discrete action space policy")

    print("--------------------------------------------------------------------------------------------")
    print("The upper limit for exploring rate : " + str(explore_upper_eps) + " and lower limir for exploit rate is : " + str(explore_lower_eps))

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
        num_env,
        device,
        acnet,
        decay_step_size,
        decay_ratio)

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
            while i_step < batch_size:
                num_episods[i] += 1
                eps = max(- max(i_update - exploit_init_step, 0) * (explore_upper_eps - explore_lower_eps) / exploit_init_step + explore_upper_eps, explore_lower_eps)
                determine = np.random.choice(2, p=[1 - eps, eps])  # explore epsilon
                _, fea, _, mask, weights = env.reset()
                while True:
                    # select action with policy, with torch.no_grad()
                    mask_tensor = torch.from_numpy(np.copy(mask)).to(device)
                    weights_tensor = torch.from_numpy(np.copy(weights)).to(device)
                    ppo_agent.buffers[i].masks.append(mask_tensor)
                    ppo_agent.buffers[i].weights.append(weights_tensor)
                    state_tensor, action, action_logprob = ppo_agent.select_action(fea, mask, weights_tensor) \
                        if determine else ppo_agent.greedy_select_action(fea, mask, weights_tensor)
                    ppo_agent.buffers[i].states.append(state_tensor)
                    ppo_agent.buffers[i].actions.append(action)
                    ppo_agent.buffers[i].logprobs.append(action_logprob)
                    _, fea, reward, done, _, mask, weights, time, _ = env.step(action)

                    # saving reward and is_terminals
                    ppo_agent.buffers[i].rewards.append(reward)
                    ppo_agent.buffers[i].is_terminals.append(done)

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

            print("Epochs:{} \t\t rewards:{}".format(i_update, np.round(mean_rewards_all_env, 3)))
            writer.add_scalar('VLoss', loss, i_update)
        writer.add_scalar("Reward/train", mean_rewards_all_env, i_update)

        if i_update % eval_interval == 0:
            g_rewards, rewardSeq, ActionSeq, ActSeq, ModeSeq, TimeSeq = greedy_evaluate(test_env, ppo_agent.policy_old)
            writer.add_scalar("Reward/greedy_evaluate", g_rewards, i_update)
            torch.save(ppo_agent.policy_old.state_dict(), checkpoint_format)
            print(" Test Total reward:{} \n Test rewards List:{} \n Test Actions:{} \n Test Acts:{} \n Test Modes: {} \n Test Times:{}".format(np.round(g_rewards, 3), rewardSeq, ActionSeq, ActSeq,ModeSeq, TimeSeq))

    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")


if __name__ == '__main__':
    crt_time = datetime.now().strftime("%Y%m%d-%H%M")
    summary_dir = os.path.join("../log", configs.acnet + '_summary', str(crt_time))
    if not os.path.exists(summary_dir):
        os.makedirs(summary_dir)
    total1 = time.time()
    pars = (
        configs.filepath, configs.Target_T, configs.price_renew, configs.price_non, configs.penalty0, configs.penalty1, configs.penalty_mode, configs.acnet)
    train(summary_dir, pars)








