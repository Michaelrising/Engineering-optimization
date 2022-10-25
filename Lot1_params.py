import argparse

parser = argparse.ArgumentParser(description='Arguments for ppo_jssp')
# args for device
parser.add_argument('--device', type=str, default="cpu")
# args for env
# Remenber if you wanna change the value of this setting, make sure you hv prior knowledge of the program
parser.add_argument('--filepath', type=str, default=r'./Lot3003.sch', help='file path for the rules')
parser.add_argument('--Target_T', type=int, default=300, help='Set the target total time of project')
parser.add_argument('--price_renew', type=float, default=[2, 3], help='Set the price per unit of renewable resource')
parser.add_argument('--price_non', type=float, default=[4, 5], help='Set the price per unit for nonrenewable resource')
parser.add_argument('--penalty0', type=float, default=20, help='Penalty coefficient0 for exceeding target T')
parser.add_argument('--penalty1', type=float, default=50, help='Penalty coefficient1 for exceeding target T')


parser.add_argument('--np_seed_train', type=int, default=199, help='Seed for numpy for training')
parser.add_argument('--np_seed_validation', type=int, default=199, help='Seed for numpy for validation')
parser.add_argument('--torch_seed', type=int, default=599, help='Seed for torch')
parser.add_argument('--et_normalize_coef', type=int, default=1000, help='Normalizing constant for feature LBs (end time), normalization way: fea/constant')
parser.add_argument('--wkr_normalize_coef', type=int, default=100, help='Normalizing constant for wkr, normalization way: fea/constant')
# args for network
parser.add_argument('--num_layers', type=int, default=3, help='No. of layers of feature extraction GNN including input layer')
parser.add_argument('--neighbor_pooling_type', type=str, default='sum', help='neighbour pooling type')
parser.add_argument('--graph_pool_type', type=str, default='average', help='graph pooling type')
parser.add_argument('--hidden_dim', type=int, default=64, help='hidden dim of MLP in fea extract GNN')
parser.add_argument('--num_mlp_layers_feature_extract', type=int, default=2, help='No. of layers of MLP in fea extract GNN')
parser.add_argument('--num_mlp_layers_actor', type=int, default=2, help='No. of layers in actor MLP')
parser.add_argument('--hidden_dim_actor', type=int, default=32, help='hidden dim of MLP in actor')
parser.add_argument('--num_mlp_layers_critic', type=int, default=2, help='No. of layers in critic MLP')
parser.add_argument('--hidden_dim_critic', type=int, default=32, help='hidden dim of MLP in critic')
# args for PPO
parser.add_argument('--num_envs', type=int, default=4, help='No. of envs for training') # original is 4
parser.add_argument('--max_updates', type=int, default=100000, help='No. of episodes of each env for training')
parser.add_argument('--lr', type=float, default=1e-5, help='lr')
parser.add_argument('--decayflag', type=bool, default=False, help='lr decayflag')
parser.add_argument('--decay_step_size', type=int, default=2000, help='decay_step_size')
parser.add_argument('--decay_ratio', type=float, default=0.5, help='decay_ratio, e.g. 0.9, 0.95')
parser.add_argument('--gamma', type=float, default=0.9, help='discount factor')
parser.add_argument('--k_epochs', type=int, default=4, help='update policy for K epochs')
parser.add_argument('--eps_clip', type=float, default=0.2, help='clip parameter for PPO')
parser.add_argument('--vloss_coef', type=float, default=1, help='critic loss coefficient')
parser.add_argument('--ploss_coef', type=float, default=0.5, help='policy loss coefficient')
parser.add_argument('--entloss_coef', type=float, default=0.01, help='entropy loss coefficient')

configs = parser.parse_args()