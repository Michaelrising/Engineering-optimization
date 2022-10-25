from PPO import *
from utils import *
from ProgramEnv import ProgEnv
import torch
import time
import numpy as np
from params import configs
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime
device = torch.device(configs.device)


def main(summary_dir, pars):
    writer = SummaryWriter(log_dir=summary_dir)
    envs = [ProgEnv(*pars) for _ in range(configs.num_envs)]
    num_activities=len(envs[0].activities)
    torch.manual_seed(configs.torch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(configs.torch_seed)
    np.random.seed(configs.np_seed_train)

    memories = [Memory() for _ in range(configs.num_envs)]

    ppo = PPO(configs.lr, configs.gamma, configs.k_epochs, configs.eps_clip,
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
    log = []
    validation_log = []
    optimal_gaps = []
    optimal_gap = 1
    record = -100000
    record_time = 1000
    for i_update in range(configs.max_updates):
        # utilize swa parameters to generate training data
        # ppo.Swap_swa_sgd(i_update);
        action_choice = greedy(i_update, configs.max_updates,  0.9)
        ep_rewards = [0 for _ in range(configs.num_envs)]
        for i, env in enumerate(envs):
            def ACT(pi, candidate, action_choice, memory):
                if action_choice:
                    return greedy_select_action(pi, candidate,  memory)
                else:
                    return select_action(pi, candidate,  memory)
            adj, fea, candidate, mask, _ = env.reset()
            ep_rewards[i] = 0
            while True:
                fea_tensor = torch.from_numpy(np.copy(fea)).to(device).float()
                adj_tensor = torch.from_numpy(np.copy(adj)).to(device).to_sparse().float()
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

                    action = ACT(pi, candidate, action_choice, memories[i])# greedy_select_action(pi, candidate, memories[i]) if greedy(i_update,configs.max_updates,  0.5) else select_action(pi, candidate, memories[i])
                adj, fea, reward, done, candidate, mask, _, _, _ = env.step(action)
                ep_rewards[i] += reward
                memory_append(memories[i], device, adj, fea, candidate, mask, action, reward, done)
                if env.done:
                    break
        # ppo.Swap_swa_sgd(i_update);
        loss, v_loss = ppo.update(memories, envs[0].action_space.n, configs.graph_pool_type, len(envs[0].activities))
        for memory in memories:
            memory.clear_memory()
        mean_rewards_all_env = sum(ep_rewards) / len(ep_rewards)

        writer.add_scalar('VLoss', v_loss, i_update)
        writer.add_scalar("Reward/train", mean_rewards_all_env, i_update)

        print('Episode {}\t Last reward: {:.2f}\t Mean_Vloss: {:.8f}'.format(
            i_update, mean_rewards_all_env, v_loss))

        if i_update % 49 == 0:
            # ppo.Swap_swa_sgd(i_update);
            epi_rewards, rewards, actions, ActSeq, ModeSeq, TimeSeq = validate(ppo.policy_old, pars)
            # ppo.Swap_swa_sgd(i_update);
            if epi_rewards > record:
                torch.save(ppo.policy.state_dict(), summary_dir + '/{}.pth'.format("PPO-ProgramEnv-"+"best_reward-"+"seed-" + str(configs.np_seed_train)))
                ppo.policy_opt.load_state_dict(ppo.policy.state_dict())
                record = epi_rewards
                print(epi_rewards)
            if record_time > TimeSeq[-1] and len(actions) >= num_activities:
                record_time = TimeSeq[-1]
                print(record_time)
                torch.save(ppo.policy.state_dict(),
                           summary_dir + '/{}.pth'.format("PPO-ProgramEnv-" + "best_time-" +"seed-" + str(configs.np_seed_train)))

            print('The validation quality is:', epi_rewards)
            print("The reward sequence is: ", rewards)
            print("The action sequence is:", actions)
            print('The activity sequence is: ', ActSeq)
            print('The mode sequence is: ', ModeSeq)
            print("The time sequence is:", TimeSeq)
            writer.add_scalar("Reward/Test", epi_rewards, i_update)

if __name__ == '__main__':
    crt_time = datetime.now().strftime("%Y%m%d-%H%M")
    summary_dir = os.path.join("log", 'summary', str(crt_time))
    if not os.path.exists(summary_dir):
        os.makedirs(summary_dir)
    total1 = time.time()
    pars = (configs.filepath, configs.Target_T, configs.price_renew,
            configs.price_non, configs.penalty0, configs.penalty1,
            configs.mode, configs.ppo)
    main(summary_dir, pars)
    total2 = time.time()