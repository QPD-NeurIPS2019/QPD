# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 16:29:14 2018

@author: 83828
"""
import argparse

parser = argparse.ArgumentParser(description='SC2@dong')
parser.add_argument('-s', '--seed', default=0, type=int, help='Seed to set')
parser.add_argument('-mn', '--map_name', default='3s5z_vs_3s6z', type=str, help='Map name to set')
parser.add_argument('-ig', '--ig_step', default=5, type=int, help='Each ig step number to set')
parser.add_argument('-a_lr', '--actor_learning_rate', default=5E-4, type=float, help='Actor lr to set')
parser.add_argument('-c_lr', '--critic_learning_rate', default=5E-4, type=float, help='Critic lr to set')
parser.add_argument('-pr', '--is_positive_reward', default=True, type=bool, help='Is positive reward')  # always True!
parser.add_argument('-bn', '--battle_num', default=5000, type=int, help='Training episode number')
parser.add_argument('-en', '--explore_num', default=0, type=int, help='Explore episode number')
parser.add_argument('-rb', '--replay_buffer', default=1000, type=int, help='Replay buffer size')
parser.add_argument('-a_l2', '--actor_l2', default=1E-2, type=float, help='Actor L2 loss ratio')
parser.add_argument('-c_l2', '--critic_l2', default=1E-2, type=float, help='Critic L2 loss ratio')
parser.add_argument('-cm', '--clip_method', default='value', type=str, choices=['global_norm', 'norm', 'value', 'no'], help='Clip method')
parser.add_argument('-cv', '--clip_value', default=0.1, type=float, help='Clip value. Effect when clip method exits')
parser.add_argument('-chm', '--channel_merge', default='concat', type=str, choices=['concat', 'add'], help='How to merge channel hidden states')
parser.add_argument('-mc', '--multi_channel', default='on', type=str, choices=['on', 'off'], help='Turn on or off multi-channel structure')
parser.add_argument('-du', '--dense_unit', default=64, type=int, help='Dense layer unit number')
parser.add_argument('-il', '--input_length', default=12, type=int, help='Sequential length for lstm input')
parser.add_argument('-ee', '--end_epsilon', default=0.0, type=float, help='End epsilon value for exploration')
parser.add_argument('-load', '--is_load_checkpoints', default=False, action='store_true', help='Log parameters')
parser.add_argument('-cp', '--chpt_path', default='./checkpoints/', type=str, help='Log parameter path')
parser.add_argument('-lp', '--load_path', default='./checkpoints/xxxx.ckpt', type=str, help='Load parameter path')
parser.add_argument('-wn', '--worker_num', default=1, type=int, help='Worker number')

args = parser.parse_args()

# Note that, for accurate decomposition, we need to set total ig steps cover the all path (>battle_step * each_ig_step)
each_ig_step_num = args.ig_step  # default is 10
map_name = args.map_name  # '3s5z'
if map_name == '2s3z':
    agent_group = [2, 3]
    ig_num_step = each_ig_step_num * 120
elif map_name == '3m':
    agent_group = [3]
    ig_num_step = each_ig_step_num * 60
elif map_name == '8m':
    agent_group = [8]
    ig_num_step = each_ig_step_num * 120
elif map_name == '3s5z':
    agent_group = [3, 5]
    ig_num_step = each_ig_step_num * 150
elif map_name == 'MMM2':
    agent_group = [2, 7, 1]
    ig_num_step = each_ig_step_num * 180
elif map_name == '3s5z_vs_3s6z':
    agent_group = [3, 5]
    ig_num_step = each_ig_step_num * 170


is_positive_reward = args.is_positive_reward
seed = args.seed
actor_lr = args.actor_learning_rate
critic_lr = args.critic_learning_rate
clip_method = args.clip_method
clip_size = args.clip_value

battle_num = args.battle_num
train_battle_sep = 100
test_battle_num = 100
enable_test = True
is_log = True

enable_double_q = True
enable_qpd = True
enable_critic_global_obs = True
enable_train_critic_last_state = True
set_critic_last_state_zero_vector = False  # set it False to keep original ending state
if args.multi_channel == 'on':
    enable_critic_multi_channel = True
elif args.multi_channel == 'off':
    enable_critic_multi_channel = False
else:
    raise RuntimeError('Multi-channel configure is not correct.@dong')
# assert enable_critic_multi_channel is False
channel_merge = args.channel_merge

enable_critic_max_a_op = False
enable_qdqn_target = False  # this is used for calculating integrated gradients
enable_l2_loss = True
l2_loss_scale_critic = args.actor_l2  # default is 0.01 (But DQN prefers 0.001)
l2_loss_scale_actor = args.critic_l2  # default is 0.001

is_debug_policy = False
is_debug_double_q = False
is_debug_dqn = False
is_debug_mixer = False

LAMBDA = 0.8
ACTOR_GAMMA = 0.99  # default is 0.99
CRITIC_GAMMA = 0.99
EXPLORATION_EPISODE_NUMBER = args.explore_num
START_EPSILON = 1.0
END_EPSILON = args.end_epsilon

TARGET_UPDATE_STEPS = 200
BUFFER_SIZE = args.replay_buffer
BATCH_SIZE = 32

DENSE_UNIT = args.dense_unit
INPUT_LENGTH = args.input_length
chpt_path = args.chpt_path
is_load_checkpoints = args.is_load_checkpoints
load_path = args.load_path
worker_num = args.worker_num
