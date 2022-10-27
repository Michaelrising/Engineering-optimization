import argparse

parser = argparse.ArgumentParser(description='Arguments for ppo_jssp')
# args for device
parser.add_argument('--device', type=str, default="cuda:0")
# args for env
# Remenber if you wanna change the value of this setting, make sure you hv prior knowledge of the program
parser.add_argument('--filepath', type=str, default=r'../../Data/Lot1.sch', help='file path for the rules')
parser.add_argument('--Target_T', type=int, default=400, help='Set the target total time of project')
parser.add_argument('--price_renew', type=float, default=[2, 3, 2, 3, 2], help='Set the price per unit of renewable resource')
parser.add_argument('--price_non', type=float, default=[4], help='Set the price per unit for nonrenewable resource')
parser.add_argument('--penalty0', type=float, default=10, help='Reward coefficient0 for not exceeding target T')
parser.add_argument('--penalty1', type=float, default=10, help='Penalty coefficient1 for exceeding target T')
# parser.add_argument('--action_dim', type=int, default=14)
parser.add_argument('--penalty_mode', type=str, default='all', help='modes calculating the reward, can be: early, resource0/1/2, each, if not specified then use all the penalties ')
# early: each activity starts greedily(most early)
# resource0/1/2: consider both renew and nonrenew/only consider renew/only consider nonrenew
# each: each activity's duration greedily(smallest)
# early+each: consider start time and duration
# if not specified or set as None or blank or any other strings, consider all the penalties
parser.add_argument('--acnet', type=str, default='gnn') # mlp, cnn or gnn
# args for network
# parser.add_argument('--input_dim_linear', type=int, default=28, help='number of dimension of raw node features')
parser.add_argument('--input_dim_gnn', type=int, default=3)
## For GNN ONLY, useless for main1
parser.add_argument('--num_layers', type=int, default=4, help='No. of layers of feature extraction GNN including input layer')
parser.add_argument('--neighbor_pooling_type', type=str, default='average', help='neighbour pooling type')
parser.add_argument('--graph_pool_type', type=str, default='average', help='graph pooling type')
parser.add_argument('--hidden_dim', type=int, default=128, help='hidden dim of MLP in fea extract GNN')
parser.add_argument('--num_mlp_layers_feature_extract', type=int, default=2, help='No. of layers of MLP in fea extract GNN')
parser.add_argument('--num_mlp_layers_actor', type=int, default=2, help='No. of layers in actor MLP')
parser.add_argument('--hidden_dim_actor', type=int, default=32, help='hidden dim of MLP in actor')
parser.add_argument('--num_mlp_layers_critic', type=int, default=2, help='No. of layers in critic MLP')
parser.add_argument('--hidden_dim_critic', type=int, default=32, help='hidden dim of MLP in critic')

# args for PPO mostly paras set in main1, and here the paras are useless for main1
parser.add_argument('--explore_upper_eps', type=float, default=0.8, help='The upper limit for exploring rate')
parser.add_argument('--explore_lower_eps', type=float, default=0.5, help='The lower limit for exploring rate')

parser.add_argument('--exploit_init_step', type=int, default=10000, help='the steps start decay explore rate')
parser.add_argument('--num_envs', type=int, default=2, help='No. of envs for training') # original is 4
parser.add_argument('--max_updates', type=int, default=100000, help='No. of episodes of each env for training')
parser.add_argument('--update_freq', type=int, default=2, help='No. of epoch to update ppo')
parser.add_argument('--log_freq', type=int, default=2, help='Log frequency')
parser.add_argument('--print_freq', type=int, default=2, help='Print results frequency')
parser.add_argument('--eval_interval', type=int, default=50, help='Evaluation frequency (steps)')

parser.add_argument('--grad_clamp', type=float, default=0.2, help='The clamp of gradient')
parser.add_argument('--lr_actor', type=float, default=0.0001*3, help='lr for actor net')
parser.add_argument('--lr_critic', type=float, default=0.00005*3, help='lr for critic net')
parser.add_argument('--decayflag', type=bool, default=False, help='lr decayflag')
parser.add_argument('--decay_step_size', type=int, default=1000, help='decay_step_size')
parser.add_argument('--decay_ratio', type=float, default=0.5, help='decay_ratio, e.g. 0.9, 0.95')
parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
parser.add_argument('--k_epochs', type=int, default=2, help='update policy for K epochs')
parser.add_argument('--eps_clip', type=float, default=0.2, help='clip parameter for PPO')
parser.add_argument('--vloss_coef', type=float, default=1, help='critic loss coefficient')
parser.add_argument('--ploss_coef', type=float, default=0.5, help='policy loss coefficient')
parser.add_argument('--entloss_coef', type=float, default=0.01, help='entropy loss coefficient')

configs = parser.parse_args()