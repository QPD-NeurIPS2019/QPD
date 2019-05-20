from smac.env import StarCraft2Env
import numpy as np
from replay_buffer import EpisodeReplayBuffer
import qpd_utils
from agentGRU import Agent
import tensorflow as tf
import datetime
import random
from tensorboard_easy import Logger
import os
import time
from multiprocessing import Pipe, Process

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"


def unroll_s_a_path(path):
    """
    Now the full path covers from S_0 to S_T, and ig_step_num = battle_limit * each_ig_step_num
    full path = S_0, P_01, P_02, ..., P_0(n-1), S_1, P_11, ..., S_(T-1), P(T-1)1, ..., P(T-1)(n-1), S_T
    :param path:
    :return:
    """
    unrolled_path = []
    for pos, next_pos in zip(path[:-1], path[1:]):
        pos, next_pos = np.asarray(pos), np.asarray(next_pos)
        step_sizes = (next_pos - pos) / qpd_utils.each_ig_step_num
        each_unrolled_s_a_path = [step_sizes * i_step + pos for i_step in range(qpd_utils.each_ig_step_num)]
        unrolled_path += each_unrolled_s_a_path
    unrolled_path.append(path[-1])  # Next pos here means the terminated state
    step_sizes = np.asarray(unrolled_path[0:-1]) - np.asarray(unrolled_path[1:])
    # print('---step size shape is {}---'.format(step_sizes.shape))
    unrolled_path = unrolled_path[0:-1]
    return unrolled_path, step_sizes


def env_worker(remote, env_seed):
    # Make environment
    # set map configuration, the last action of each agent is not appended
    env = StarCraft2Env(map_name=qpd_utils.map_name, step_mul=8, seed=env_seed,
                        # obs_last_action=True, obs_pathing_grid=True, obs_terrain_height=True,
                        reward_only_positive=qpd_utils.is_positive_reward)
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            sub_actions = data
            # Take a step in the environment
            sub_reward, sub_terminated, sub_env_info = env.step(sub_actions)
            remote.send((sub_reward, sub_terminated, sub_env_info))
        elif cmd == "reset":
            env.reset()
            remote.send({
                "state": env.get_state(),
                "avail_actions": env.get_avail_actions(),
                "obs": env.get_obs()
            })
        elif cmd == "close":
            env.close()
            remote.close()
            break
        elif cmd == "get_env_info":
            remote.send(env.get_env_info())
        elif cmd == "get_stats":
            remote.send(env.get_stats())
        elif cmd == 'get_obs_without_view_restriction':
            remote.send(env.get_obs_without_view_restriction())
        elif cmd == 'get_obs':
            remote.send(env.get_obs())
        elif cmd == 'get_avail_agent_actions':
            sub_agent_id = data
            remote.send(env.get_avail_agent_actions(sub_agent_id))
        elif cmd == 'battles_won':
            remote.send(env.battles_won)
        else:
            raise NotImplementedError


def mainprocess():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    seed = qpd_utils.seed
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)

    parent_conns, worker_conns = zip(*[Pipe() for _ in range(qpd_utils.worker_num)])
    ps = [Process(target=env_worker, args=(worker_conn, seed,)) for worker_conn in worker_conns]
    for p in ps:
        p.daemon = True
        p.start()

    # get map scenario information for constructing networks
    parent_conns[0].send(("get_env_info", None))
    env_info = parent_conns[0].recv()
    n_actions = env_info["n_actions"]
    n_agents = env_info["n_agents"]
    n_obs = env_info['obs_shape']
    n_state = env_info['state_shape']
    n_episode_max_length = env_info['episode_limit']
    input_length = qpd_utils.INPUT_LENGTH

    print('---Map {} has {} actions, {} agents, obs dim is {}, state dim is {}, battle limit is {}---'
          .format(qpd_utils.map_name, n_actions, n_agents, n_obs, n_state, n_episode_max_length))

    log = None
    if qpd_utils.is_log:
        log = Logger('logs/{}_lr_{}_{}_l2_{}_{}_seed_{}_clip_{}_{}'
                     '_gamma_{}_{}_buffer_{}_ig_{}_{}_end_e_{}_mc_{}_{}_bn_{}_en_{}'.
                     format(qpd_utils.map_name, qpd_utils.actor_lr, qpd_utils.critic_lr,
                            qpd_utils.l2_loss_scale_critic, qpd_utils.l2_loss_scale_actor,
                            seed, qpd_utils.clip_method, qpd_utils.clip_size,
                            qpd_utils.ACTOR_GAMMA, qpd_utils.CRITIC_GAMMA,
                            qpd_utils.BUFFER_SIZE,
                            # qpd_utils.enable_train_critic_last_state,
                            qpd_utils.ig_num_step, qpd_utils.each_ig_step_num,
                            qpd_utils.END_EPSILON,
                            # str(datetime.datetime.now().strftime("%H-%M-%S"))
                            qpd_utils.enable_critic_multi_channel,
                            qpd_utils.channel_merge,
                            qpd_utils.battle_num, qpd_utils.EXPLORATION_EPISODE_NUMBER
                            ))

    agent = Agent(sess, n_actions, n_obs, input_length, qpd_utils.actor_lr, qpd_utils.critic_lr, n_agents, log)
    buff = EpisodeReplayBuffer(buffer_size=qpd_utils.BUFFER_SIZE)

    battle_win_record = []
    n_episodes = qpd_utils.battle_num

    for e in range(n_episodes):
        # init environment for this battle
        for parent_conn in parent_conns:
            parent_conn.send(("reset", None))
            parent_conn.recv()
        terminated = [False for _ in range(qpd_utils.worker_num)]
        episode_reward = [0 for _ in range(qpd_utils.worker_num)]

        # set epsilon
        if e < qpd_utils.EXPLORATION_EPISODE_NUMBER:
            epsilon = qpd_utils.START_EPSILON - (qpd_utils.START_EPSILON - qpd_utils.END_EPSILON) / qpd_utils.EXPLORATION_EPISODE_NUMBER * e
        else:
            epsilon = qpd_utils.END_EPSILON

        # print('-----current episode is {} and current epsilon is {}-----'.format(e, epsilon))
        # print('-----current buffer size is {} and each episode in buffer\'s length is {}-----'
        #        .format(buff.count(), [len(ep['agent_obs']) for ep in buff.getBatch(qpd_utils.BUFFER_SIZE)]))

        # Set initial battle recorder
        episode_recorders = [{'agent_obs': [], 'critic_state': [], 'action': [], 'reward': [],
                             'terminated': [], 'alive_mask': [], 'action_mask': []}
                             for _ in range(qpd_utils.worker_num)]

        # collect trajectory samples
        for env_id, parent_conn, episode_recorder in zip(range(qpd_utils.worker_num), parent_conns, episode_recorders):
            parent_conn.send(("get_obs", None))
            obs = parent_conn.recv()
            seq_obs = [[i_ob] * input_length for i_ob in obs]
            while not terminated[env_id]:
                parent_conn.send(("get_obs", None))
                obs = parent_conn.recv()
                if qpd_utils.enable_critic_global_obs:
                    parent_conn.send(("get_obs_without_view_restriction", None))
                    global_obs = parent_conn.recv()
                else:
                    global_obs = obs

                seq_obs = [i_seq_ob[1:] + [i_ob] for i_seq_ob, i_ob in zip(seq_obs, obs)]

                valid_actions = []
                alive_masks = []
                for agent_id in range(n_agents):
                    parent_conn.send(("get_avail_agent_actions", agent_id))
                    valid_actions.append(parent_conn.recv())
                    # if valid_actions[agent_id][0] == 1:
                    #     alive_masks.append(0)
                    # else:
                    #     alive_masks.append(1)
                    alive_masks.append(1)

                actions = agent.get_action(seq_obs, valid_actions, epsilon)
                # time.sleep(0.5)
                parent_conn.send(("step", actions))
                reward, terminated[env_id], info = parent_conn.recv()
                assert reward >= 0
                episode_reward[env_id] += reward

                critic_s_a_concat = []
                for s, a in zip(global_obs, actions):
                    label_a = [0] * n_actions
                    label_a[a] = 1
                    critic_s_a_concat += list(s) + label_a

                episode_recorder['agent_obs'].append(obs)
                episode_recorder['critic_state'].append(critic_s_a_concat)
                episode_recorder['action'].append(actions)
                episode_recorder['reward'].append(reward)
                episode_recorder['terminated'].append(terminated[env_id])
                episode_recorder['alive_mask'].append(alive_masks)
                episode_recorder['action_mask'].append(valid_actions)

                # FIXME 0121: sometimes the game would be crashed, and the info becomes a empty dict and raise an error
                try:
                    if info['battle_won']:
                        battle_win_record.append(e)
                except:
                    print('Something wrong here')

                # Add the final state info, to complete the trajectory transition
                if terminated[env_id]:
                    parent_conn.send(("get_obs", None))
                    final_agent_obs = parent_conn.recv()
                    if qpd_utils.enable_critic_global_obs:
                        parent_conn.send(("get_obs_without_view_restriction", None))
                        final_agent_global_obs = parent_conn.recv()
                    else:
                        final_agent_global_obs = final_agent_obs
                    # print('-----episode step is {}-----'.format(len(episode_recorder['agent_obs'])))
                    # print('-----last agent obs is {}-----'.format(episode_recorder['agent_obs'][-1]))
                    # print('-----final agent obs is {}-----'.format(final_agent_obs))
                    # print('-----my final agent obs is {}-----'.format(info['final_agent_obs']))
                    final_valid_actions = []
                    final_alive_masks = []
                    for agent_id in range(n_agents):
                        parent_conn.send(("get_avail_agent_actions", agent_id))
                        final_valid_actions.append(parent_conn.recv())
                        # if final_valid_actions[agent_id][0] == 1:
                        #     final_alive_masks.append(0)
                        # else:
                        #     final_alive_masks.append(1)
                        final_alive_masks.append(1)
                        final_valid_actions[agent_id] = [1] + [0] * (n_actions - 1)  # No action will be performed
                    final_critic_s_a_concat = []
                    for each_obs in final_agent_global_obs:
                        label_a = [0] * n_actions
                        label_a[0] = 1  # only noop action is allowed
                        final_critic_s_a_concat += list(each_obs) + label_a
                    episode_recorder['agent_obs'].append(final_agent_obs)
                    episode_recorder['critic_state'].append(final_critic_s_a_concat)
                    if qpd_utils.set_critic_last_state_zero_vector:
                        episode_recorder['critic_state'][-1] = np.zeros_like(final_critic_s_a_concat)  # replace last state
                    # FIXME 0123: test the final state all 0
                    # episode_recorder['critic_state'].append(np.zeros_like(final_critic_s_a_concat))
                    episode_recorder['alive_mask'].append(final_alive_masks)
                    episode_recorder['action_mask'].append(final_valid_actions)
                    # episode_recorder['unrolled_s_a_path'], episode_recorder['full_step_size'] = unroll_s_a_path(episode_recorder['critic_state'])

            buff.add(episode_recorder)

        # process info for logging
        recent_win_rate = sum(np.array(battle_win_record) >= e - 100) / (100 * qpd_utils.worker_num)
        total_battles_won = 0
        for parent_conn in parent_conns:
            parent_conn.send(("battles_won", None))
            total_battles_won += parent_conn.recv()
        print("Total reward in episode {} = {} and win rate of recent 100 episodes is {} and total battle win is {}"
              .format(e, sum(episode_reward) / qpd_utils.worker_num, recent_win_rate, total_battles_won))
        if qpd_utils.is_log:
            log.log_scalar('episode_reward', sum(episode_reward) / qpd_utils.worker_num, e)
            log.log_scalar('recent_win_rate', recent_win_rate, e)

        # train
        # if (e + 1) % 8 == 0:
        # if buff.count() >= qpd_utils.BATCH_SIZE:  # train when buffer sample number >= batch size
        if True:  # train from the beginning
            episode_batches = buff.getBatch(qpd_utils.BATCH_SIZE)
            seq_states, critic_state, actions, rewards, alive_mask = [], [], [], [], []
            seq_next_states, next_critic_state, next_action_mask, next_alive_mask, terminated = [], [], [], [], []
            target_q_val = []
            for episode_track in episode_batches:
                # unrolled_full_path = episode_track['unrolled_s_a_path']
                # full_step_size = episode_track['full_step_size']
                unrolled_full_path, full_step_size = unroll_s_a_path(episode_track['critic_state'])
                ex = agent.get_integrated_gradients(unrolled_full_path)
                ex = np.reshape(ex, (-1, (n_obs + n_actions) * n_agents))
                ex = ex * full_step_size
                full_step = len(episode_track['critic_state']) - 1
                for loc in range(full_step):
                    agent_ex = ex[loc * qpd_utils.each_ig_step_num:]
                    agent_ex = np.sum(agent_ex, axis=0)
                    agent_ex = np.reshape(agent_ex, (n_agents, -1))
                    target_q_val += list(np.sum(agent_ex, axis=1))

                full_step = len(episode_track['critic_state']) - 1
                # Current step
                states = episode_track['agent_obs'][:-1]
                init_state = states[0]
                seq_state = [[i_s] * input_length for i_s in init_state]
                for i in range(len(states)):
                    seq_state = [i_seq_s[1:] + [i_s] for i_seq_s, i_s in zip(seq_state, states[i])]
                    seq_states.append(seq_state)

                critic_state += episode_track['critic_state'][:-1]
                actions += episode_track['action']
                rewards += list(episode_track['reward'])
                alive_mask += episode_track['alive_mask'][:-1]
                # Next step
                next_states = episode_track['agent_obs'][1:]
                init_next_state = next_states[0]
                seq_next_state = [[i_n_s] * input_length for i_n_s in init_next_state]
                for i in range(len(next_states)):
                    seq_next_state = [i_seq_n_s[1:] + [i_n_s] for i_seq_n_s, i_n_s in zip(seq_next_state, next_states[i])]
                    seq_next_states.append(seq_next_state)
                next_critic_state += episode_track['critic_state'][1:]
                next_action_mask += episode_track['action_mask'][1:]
                next_alive_mask += episode_track['alive_mask'][1:]
                terminated += episode_track['terminated']

            # print('---here is agent obs {}---'.format(states[0]))
            # print('---here is agent action {}---'.format(actions[0]))
            # print('---here is critic obs {}---'.format(critic_state[0]))

            agent.train_policy(seq_states, critic_state, actions, rewards, alive_mask,
                               seq_next_states, next_critic_state, next_action_mask, next_alive_mask, terminated,
                               target_q_val, e)

            if qpd_utils.enable_train_critic_last_state:
                episode_batches = buff.getBatch(qpd_utils.BATCH_SIZE)
                critic_last_state = [episode['critic_state'][-1] for episode in episode_batches]
                final_Q, end_c_loss = agent.train_critic_end(critic_last_state)
                if qpd_utils.is_log:
                    log.log_scalar('Final Q Prediction', float(final_Q), e)
                    log.log_scalar('End critic loss', float(end_c_loss), e)

            # test
            t_parent_conn = parent_conns[0]  # use the first worker as the testing environment
            if qpd_utils.enable_test:
                if e % qpd_utils.train_battle_sep == 0:  # training 100 episodes and testing 100 episodes
                    t_battle_win_num = 0
                    action_stats = []
                    for te in range(qpd_utils.test_battle_num):
                        t_parent_conn.send(("reset", None))
                        t_parent_conn.recv()
                        t_terminated = False
                        t_parent_conn.send(("get_obs", None))
                        t_obs = t_parent_conn.recv()
                        t_seq_obs = [[t_i_ob] * input_length for t_i_ob in t_obs]
                        while not t_terminated:
                            t_parent_conn.send(("get_obs", None))
                            t_obs = t_parent_conn.recv()
                            t_seq_obs = [t_i_seq_ob[1:] + [t_i_ob] for t_i_seq_ob, t_i_ob in zip(t_seq_obs, t_obs)]
                            t_valid_actions = []
                            for agent_id in range(n_agents):
                                t_parent_conn.send(("get_avail_agent_actions", agent_id))
                                t_valid_actions.append(t_parent_conn.recv())
                            t_actions = agent.get_action(t_seq_obs, t_valid_actions, epsilon=0)
                            t_parent_conn.send(("step", t_actions))
                            t_reward, t_terminated, t_info = t_parent_conn.recv()
                            action_stats += t_actions
                        try:
                            if t_info['battle_won']:
                                t_battle_win_num += 1
                        except:
                            print('Something wrong here')
                    if qpd_utils.is_log:
                        log.log_scalar('test_win_rate', t_battle_win_num / qpd_utils.test_battle_num, e // qpd_utils.train_battle_sep)
                        for i in range(n_actions):
                            log.log_scalar('action_{}_test_frequency'.format(i),
                                           action_stats.count(i) / len(action_stats), e // qpd_utils.train_battle_sep)
                    if t_battle_win_num >= 5:  # test win rate >= 5%
                        agent.save_check_points()

    agent.save_check_points()
    for parent_conn in parent_conns:
        parent_conn.send(("close", None))
        parent_conn.close()


if __name__ == '__main__':
    # main function
    mainprocess()

