from PPO import *
from utils import *
from Codes.ProgramEnv import ProgEnv
import torch
import time
import numpy as np
from gnn_params import configs
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime


def main(summary_dir, pars):
    device = torch.device(configs.device)
    print("============================================================================================")
    # set device to cpu or cuda
    if torch.cuda.is_available() and device == 'cuda:0':
        torch.cuda.empty_cache()
        print("Device set to : " + str(torch.cuda.get_device_name(device)))
    else:
        print("Device set to : cpu")

    print("============================================================================================")
    envs = [ProgEnv(*pars) for _ in range(configs.num_envs)]
    num_activities=len(envs[0].activities)

    memories = [Memory() for _ in range(configs.num_envs)]

    env_name = 'ProgramEnv'
    print("training environment name : " + env_name)

    print("--------------------------------------------------------------------------------------------")

    print("state space dimension : ({}, {})".format(envs[0].action_space.n, configs.input_dim_gnn))
    print("action space dimension : ", envs[0].action_space.n)
    print("penalty mode:" + configs.penalty_mode)
    print("num of envs : " + str(configs.num_envs))
    ###################### logging ######################

    #### log files for multiple runs are NOT overwritten
    writer = SummaryWriter(log_dir=summary_dir)
    checkpoint_format = summary_dir + '/PPO-ProgramEnv-converge-model-{}.pth'.format(configs.filepath[11:-4])

    print('summary save at: ', summary_dir)
    print('model save as: ', checkpoint_format)
    print("--------------------------------------------------------------------------------------------")

    ############# print all hyperparameters #############

    print("--------------------------------------------------------------------------------------------")
    print('the actor and critic network: ', configs.acnet)

    print("max training updating times : ", configs.max_updates)
    print("log frequency : " + str(configs.log_freq) + " episodes")
    print("printing average reward over episodes in last : " + str(configs.print_freq) + " episodes")


    print("--------------------------------------------------------------------------------------------")

    print("Initializing a discrete action space policy")

    print("--------------------------------------------------------------------------------------------")
    print("PPO update frequency : " + str(configs.update_freq) + " episodes")
    print("PPO K epochs : ", configs.k_epochs)
    print("PPO epsilon clip : ", configs.eps_clip)
    print("discount factor (gamma) : ", configs.gamma)

    print("--------------------------------------------------------------------------------------------")
    if configs.decayflag:
        print("decaying optimizer with step size : ", configs.decay_step_size, " decay ratio : ", configs.decay_ratio)
    print("optimizer learning rate actor : ", configs.lr_actor)
    print("optimizer learning rate critic : ", configs.lr_critic)

    #####################################################

    print("============================================================================================")

    ppo = PPO(configs.lr_actor, configs.lr_critic, configs.gamma, configs.k_epochs, configs.eps_clip,
              num_layers=configs.num_layers,
              neighbor_pooling_type=configs.neighbor_pooling_type,
              input_dim=configs.input_dim_gnn,
              hidden_dim=configs.hidden_dim,
              num_mlp_layers_feature_extract=configs.num_mlp_layers_feature_extract,
              num_mlp_layers_actor=configs.num_mlp_layers_actor,
              hidden_dim_actor=configs.hidden_dim_actor,
              num_mlp_layers_critic=configs.num_mlp_layers_critic,
              hidden_dim_critic=configs.hidden_dim_critic)

    g_pool_step = g_pool_cal(graph_pool_type=configs.graph_pool_type,
                             batch_size=torch.Size([1,  envs[0].action_space.n,  envs[0].action_space.n]),
                             n_nodes=envs[0].action_space.n,
                             device=device)

    for i_update in range(configs.max_updates):
        action_choice = greedy(i_update, configs.max_updates,  configs.explore_upper_eps)
        ep_rewards = [0 for _ in range(configs.num_envs)]
        for i, env in enumerate(envs):
            def ACT(pi, candidate, action_choice, memory):
                if action_choice:
                    return greedy_select_action(pi, candidate,  memory)
                else:
                    return select_action(pi, candidate,  memory)
            adj, fea, candidate, mask, _ = env.reset()
            D_t = np.diag(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0)
            adj_norm = adj.dot(D_t).transpose().dot(D_t).todense()
            ep_rewards[i] = 0
            while True:
                fea_tensor = torch.from_numpy(np.copy(fea)).to(device).float()
                adj_tensor = torch.from_numpy(np.copy(adj_norm)).to(device).to_sparse().float()
                candidate_tensor = torch.from_numpy(np.copy(candidate)).to(device).float()
                mask_tensor = torch.from_numpy(np.copy(mask)).to(device)
                memories[i].mask_mb.append(mask_tensor)
                with torch.no_grad():
                    pi, _ = ppo.policy_old(x=fea_tensor,
                                           graph_pool=g_pool_step,
                                           padded_nei=None,
                                           adj=adj_tensor,
                                           candidate=candidate_tensor.unsqueeze(0),
                                           mask=mask_tensor.unsqueeze(0))

                    action = ACT(pi, candidate, action_choice, memories[i])
                _, fea, reward, done, candidate, mask, _, _, _ = env.step(action)
                ep_rewards[i] += reward
                memory_append(memories[i], device, adj, fea, candidate, mask, action, reward, done)
                if env.done:
                    break

        mean_rewards_all_env = sum(ep_rewards) / len(ep_rewards)
        if i_update % configs.update_freq == 0:
            loss, v_loss = ppo.update(memories, envs[0].action_space.n, configs.graph_pool_type, len(envs[0].activities))
            writer.add_scalar('VLoss', v_loss, i_update)
            writer.add_scalar("Reward/train", mean_rewards_all_env, i_update)
            print('Episode {}\t Last reward: {:.2f}\t Mean_Vloss: {:.8f}'.format(
                i_update, mean_rewards_all_env, v_loss))
        for memory in memories:
            memory.clear_memory()

        if i_update % configs.eval_interval == 0:
            epi_rewards, rewards, actions, ActSeq, ModeSeq, TimeSeq = validate(ppo.policy_old, pars)
            torch.save(ppo.policy.state_dict(), checkpoint_format)
            ppo.policy_opt.load_state_dict(ppo.policy.state_dict())
            print('The validation quality is:', epi_rewards)
            print("The reward sequence is: ", rewards)
            print("The action sequence is:", actions)
            print('The activity sequence is: ', ActSeq)
            print('The mode sequence is: ', ModeSeq)
            print("The time sequence is:", TimeSeq)
            writer.add_scalar("Reward/Test", epi_rewards, i_update)

if __name__ == '__main__':
    crt_time = datetime.now().strftime("%Y%m%d-%H%M")
    summary_dir = os.path.join("../../log", configs.acnet+'_summary', str(crt_time))
    if not os.path.exists(summary_dir):
        os.makedirs(summary_dir)
    total1 = time.time()
    pars = (configs.filepath, configs.Target_T, configs.price_renew,
            configs.price_non, configs.penalty0, configs.penalty1,
            configs.penalty_mode, configs.acnet)
    main(summary_dir, pars)
    total2 = time.time()