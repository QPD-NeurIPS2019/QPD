# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 20:11:35 2018

@author: 83828
"""

import tensorflow as tf
import numpy as np
import qpd_utils
import datetime
import matplotlib.pyplot as plt

DENSE_UNIT_NUMBER = qpd_utils.DENSE_UNIT


class Agent(object):
    def __init__(self, sess, action_dim, state_dim, input_length, a_lr, c_lr, agent_num, log):
        self.sess = sess
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.input_length = input_length
        self.agent_num = agent_num
        self.log = log
        self.online_inputs = tf.placeholder(tf.float32, (None, self.input_length, state_dim))
        self.target_inputs = tf.placeholder(tf.float32, (None, self.input_length, state_dim))
        self.q = self.create_policy_network(self.online_inputs, 'online')
        self.target_q = self.create_policy_network(self.target_inputs, 'target')
        self.actions = tf.placeholder(tf.int32, None)
        self.onehot_action = tf.one_hot(self.actions, depth=self.action_dim)
        self.alive_label = tf.placeholder(tf.float32, (None, 1))
        self.next_alive_label = tf.placeholder(tf.float32, (None, 1))
        self.actor_target_values = tf.placeholder(tf.float32, None)
        self.mixer_target_values = tf.placeholder(tf.float32, None)
        self.update_step = 0

        params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='online_network')
        target_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target_network')
        self.copy_ops = []
        for (online_var, target_var) in zip(params, target_params):
            assign_op = tf.assign(ref=target_var, value=online_var)
            self.copy_ops.append(assign_op)
        if qpd_utils.enable_qpd:
            self.mixer_online_s_a = tf.placeholder(tf.float32, (None, (state_dim + action_dim) * self.agent_num))
            self.mixer_target_s_a = tf.placeholder(tf.float32, (None, (state_dim + action_dim) * self.agent_num))
            self.online_q_tot = self.create_mixer(self.mixer_online_s_a, 'online_mixer')
            self.target_q_tot = self.create_mixer(self.mixer_target_s_a, 'target_mixer')
            if qpd_utils.enable_qdqn_target:
                self.mixer_grads = tf.gradients(self.target_q_tot, self.mixer_target_s_a)
            else:
                self.mixer_grads = tf.gradients(self.online_q_tot, self.mixer_online_s_a)

        if qpd_utils.enable_qpd:
            mixer_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='online_mixer_network')
            mixer_target_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='target_mixer_network')
            for (online_var, target_var) in zip(mixer_params, mixer_target_params):
                assign_op = tf.assign(ref=target_var, value=online_var)
                self.copy_ops.append(assign_op)

        with tf.name_scope('loss'):
            if qpd_utils.enable_qpd:
                self.mixer_loss = tf.reduce_mean(tf.squared_difference(self.online_q_tot, self.mixer_target_values))

            # if the unit is dead now, its q = 0 and target q = 0
            target_values = tf.reshape(self.actor_target_values, (-1, 1)) * self.alive_label
            # self.oa = self.q * self.onehot_action
            self.act_q = tf.reshape(tf.reduce_sum(self.q * self.onehot_action, axis=1), (-1, 1))
            self.sd_op = tf.squared_difference(self.act_q, target_values)
            self.actor_loss = tf.reduce_sum(self.sd_op) / tf.reduce_sum(self.alive_label)
            if qpd_utils.enable_l2_loss:
                if qpd_utils.enable_qpd:
                    for par in params:
                        self.actor_loss += qpd_utils.l2_loss_scale_actor * tf.nn.l2_loss(par)
                    for par in mixer_params:
                        self.mixer_loss += qpd_utils.l2_loss_scale_critic * tf.nn.l2_loss(par)
                else:
                    for par in params:
                        self.actor_loss += qpd_utils.l2_loss_scale_dqn * tf.nn.l2_loss(par)

        # actor_trainer = tf.train.RMSPropOptimizer(learning_rate=a_lr, decay=0.99, epsilon=1e-5)
        actor_trainer = tf.train.RMSPropOptimizer(learning_rate=a_lr)
        # actor_trainer = tf.train.AdamOptimizer(learning_rate=a_lr)
        # critic_trainer = tf.train.RMSPropOptimizer(learning_rate=c_lr, decay=0.99, epsilon=1e-5)
        critic_trainer = tf.train.AdamOptimizer(learning_rate=c_lr)
        # actor_trainer = tf.train.GradientDescentOptimizer(learning_rate=a_lr)
        # critic_trainer = tf.train.GradientDescentOptimizer(learning_rate=c_lr)
        # Clip the gradients (normalize)
        if qpd_utils.clip_method == 'global_norm':
            agrads = tf.gradients(self.actor_loss, params)
            cgrads = tf.gradients(self.mixer_loss, mixer_params)
            agrads, self.a_g_norm = tf.clip_by_global_norm(agrads, qpd_utils.clip_size)
            cgrads, self.c_g_norm = tf.clip_by_global_norm(cgrads, qpd_utils.clip_size)
            self.mean_grads = [tf.reduce_mean(grad) for grad in cgrads if grad is not None]
            agrads = list(zip(agrads, params))
            cgrads = list(zip(cgrads, mixer_params))
        elif qpd_utils.clip_method == 'norm':
            agrads = actor_trainer.compute_gradients(self.actor_loss, params)
            cgrads = critic_trainer.compute_gradients(self.mixer_loss, mixer_params)
            self.mean_grads = []
            for i, (g, v) in enumerate(agrads):
                if g is not None:
                    agrads[i] = (tf.clip_by_norm(g, qpd_utils.clip_size), v)  # clip gradients
                    self.mean_grads.append(tf.reduce_mean(agrads[i]))
            for i, (g, v) in enumerate(cgrads):
                if g is not None:
                    cgrads[i] = (tf.clip_by_norm(g, qpd_utils.clip_size), v)  # clip gradients
                    self.mean_grads.append(tf.reduce_mean(cgrads[i]))
        elif qpd_utils.clip_method == 'value':
            agrads = actor_trainer.compute_gradients(self.actor_loss, params)
            cgrads = critic_trainer.compute_gradients(self.mixer_loss, mixer_params)
            self.mean_grads = []
            for i, (g, v) in enumerate(agrads):
                if g is not None:
                    agrads[i] = (tf.clip_by_value(g, -qpd_utils.clip_size, qpd_utils.clip_size), v)  # clip gradients
                    self.mean_grads.append(tf.reduce_mean(agrads[i]))
            for i, (g, v) in enumerate(cgrads):
                if g is not None:
                    cgrads[i] = (tf.clip_by_value(g, -qpd_utils.clip_size, qpd_utils.clip_size), v)  # clip gradients
                    self.mean_grads.append(tf.reduce_mean(cgrads[i]))
        else:  # no clip
            agrads = tf.gradients(self.actor_loss, params)
            cgrads = tf.gradients(self.mixer_loss, mixer_params)
            # self.mean_grads = [tf.reduce_mean(grad) for grad in cgrads if grad is not None]
            agrads = list(zip(agrads, params))
            cgrads = list(zip(cgrads, mixer_params))
        self.train_op_a = actor_trainer.apply_gradients(agrads)
        self.train_op_c = critic_trainer.apply_gradients(cgrads)
        # self.train_op = tf.train.RMSPropOptimizer(learning_rate).minimize(self.loss) # min(v) = max(-v)
        # self.sess.run(tf.global_variables_initializer())

        # SAVE/LOAD
        self.saver = tf.train.Saver(max_to_keep=10)

        if qpd_utils.is_load_checkpoints:
            print('Loading checkpoint ...')
            self.saver.restore(self.sess, qpd_utils.load_path)
            print('Checkpoingt: ' + qpd_utils.load_path + ' is loaded.')
        else:
            self.sess.run(tf.global_variables_initializer())
            print('Parameters initialized.')

        self.sess.graph.finalize()

    def get_action(self, states, valid_actions, epsilon):
        qs = self.sess.run(self.q, feed_dict={self.online_inputs: states})
        actions = []
        for val_act, q in zip(valid_actions, qs):
            val_act = np.asarray(val_act)
            if np.random.random() < epsilon:
                candi_actions = [i for i in range(self.action_dim) if val_act[i] > 0.5]
                action = np.random.choice(candi_actions)
                # print('choose random: valid action {} and chosen action {}'.format(valid_actions, action))
            else:
                valid_q = q.ravel()
                valid_q[val_act < 0.5] = -9999
                action = valid_q.argmax()
                # print('choose random: valid action {} and chosen action {}'.format(valid_actions, action))
            # print('-----agent available action is {} and selected action is {}-----'.format(val_act, action))
            actions.append(action)
        return actions

    def train_policy(self, states, critic_s_a, actions, rewards, alive_mask,
                     next_states, next_critic_s_a, next_action_mask, next_alive_mask, terminated,
                     ig_target_q_val, e):
        if qpd_utils.enable_double_q:
            states = np.reshape(states, (-1, self.input_length, self.state_dim))
            actions = np.hstack(actions)
            next_states = np.reshape(next_states, (-1, self.input_length, self.state_dim))
            next_action_mask = np.reshape(next_action_mask, (-1, self.action_dim))
            next_q_val = self.sess.run(self.q, feed_dict={self.online_inputs: next_states})
            next_q_val[next_action_mask < 0.5] = -9999
            next_max_a = np.argmax(next_q_val, axis=1)
            next_max_a = np.eye(self.action_dim)[next_max_a]
            next_target_q_val = self.sess.run(self.target_q, feed_dict={self.target_inputs: next_states})
            next_target_max_q_val = next_target_q_val[next_max_a > 0.5]
            if not qpd_utils.enable_qpd:
                # Expand reward to respond to each agent
                rewards = [[r] * self.agent_num for r in rewards]
                rewards = np.reshape(rewards, (-1, 1))
                alive_mask = np.reshape(alive_mask, (-1, 1))
                next_target_max_q_val = np.reshape(next_target_max_q_val, (-1, 1))
                next_alive_mask = np.reshape(next_alive_mask, (-1, 1))
                terminated = [[t] * self.agent_num for t in terminated]
                terminated = np.reshape(terminated, (-1, 1))
                # for an independent agent, if he is dead, then this episode terminates for him
                target_q_val = rewards + qpd_utils.ACTOR_GAMMA * next_target_max_q_val * next_alive_mask * (1 - terminated)
        else:
            next_states = np.reshape(next_states, (-1, self.state_dim))
            next_action_mask = np.reshape(next_action_mask, (-1, self.action_dim))
            next_target_q_val = self.sess.run(self.target_q, feed_dict={self.target_inputs: next_states})
            next_target_q_val[next_action_mask < 0.5] = -9999
            next_target_max_q_val = np.max(next_target_q_val, axis=1)
            if not qpd_utils.enable_qpd:
                # rewards = np.reshape(np.hstack(([rewards] * qpd_utils.agent_num)), (-1, 1))
                next_target_max_q_val = np.reshape(next_target_max_q_val, (-1, 1))
                rewards = [[r] * self.agent_num for r in rewards]
                rewards = np.reshape(rewards, (-1, 1))
                next_alive_mask = np.reshape(next_alive_mask, (-1, 1))
                terminated = [[t] * self.agent_num for t in terminated]
                terminated = np.reshape(terminated, (-1, 1))
                target_q_val = rewards + qpd_utils.ACTOR_GAMMA * next_target_max_q_val * next_alive_mask * (1 - terminated)

        if qpd_utils.enable_qpd:
            if qpd_utils.enable_critic_max_a_op:
                # use the next max action to reconstruct next_critic_s_a
                target_next_q_val = self.sess.run(self.target_q, feed_dict={self.target_inputs: next_states})  # target
                # target_next_q_val = self.sess.run(self.q, feed_dict={self.online_inputs: next_states})  # online
                target_next_q_val[next_action_mask < 0.5] = -9999
                next_max_a = np.argmax(target_next_q_val, axis=1)
                next_max_a = np.eye(self.action_dim)[next_max_a]
                next_s_a_concat = np.hstack((next_states, next_max_a))
                next_s_a_concat = np.reshape(next_s_a_concat, (-1, len(next_critic_s_a[0])))
                next_target_q_tot = self.sess.run(self.target_q_tot, feed_dict={self.mixer_target_s_a: next_s_a_concat})
            else:
                next_target_q_tot = self.sess.run(self.target_q_tot, feed_dict={self.mixer_target_s_a: next_critic_s_a})
            critic_target_q_val = np.reshape(rewards, (-1, 1)) + qpd_utils.CRITIC_GAMMA * next_target_q_tot * (1 - np.reshape(terminated, (-1, 1)))
            # TODO: update the target Q-value from Integrated Gradients
            target_q_val = ig_target_q_val

        # test decomposition
        # final_q = self.sess.run(self.online_q_tot, feed_dict={self.mixer_online_s_a: next_critic_s_a})[-1]
        # a, b = [], []
        # for i in range(len(critic_s_a)):
        #
        #     a.append(sum(target_q_val[i * self.agent_num:(i + 1) * self.agent_num]))
        #     b.append(self.sess.run(self.online_q_tot, feed_dict={self.mixer_online_s_a: critic_s_a})[i] - final_q)
        # # print(sum(a))
        # # print(sum(b))
        # plt.plot(a)
        # plt.plot(b)

        alive_mask = np.reshape(alive_mask, (-1, 1))
        feed_dict = {self.online_inputs: states,
                     self.actions: actions,
                     self.actor_target_values: target_q_val,
                     self.alive_label: alive_mask}
        if qpd_utils.enable_qpd:
            feed_dict[self.mixer_online_s_a] = critic_s_a
            feed_dict[self.mixer_target_values] = critic_target_q_val

        if qpd_utils.clip_method == 'global_norm':
            a_g_norm, c_g_norm = self.sess.run([self.a_g_norm, self.c_g_norm], feed_dict=feed_dict)
            if qpd_utils.is_log:
                self.log.log_scalar('actor_global_norm', a_g_norm, e)
                self.log.log_scalar('critic_global_norm', c_g_norm, e)

        _, _, c_loss, a_loss = self.sess.run([self.train_op_a, self.train_op_c, self.mixer_loss, self.actor_loss], feed_dict=feed_dict)
        if qpd_utils.is_log:
            self.log.log_scalar('actor_loss', a_loss, e)
            self.log.log_scalar('critic_loss', c_loss, e)

        self.update_step += 1
        if self.update_step == qpd_utils.TARGET_UPDATE_STEPS:
            self.update_target_network()
            self.update_step = 0
        # print('actor mean gradient is {}'.format(mean_grads))

    def get_integrated_gradients(self, s_a_path):
        if qpd_utils.enable_qdqn_target:
            ex = self.sess.run(self.mixer_grads, feed_dict={self.mixer_target_s_a: s_a_path})
        else:
            ex = self.sess.run(self.mixer_grads, feed_dict={self.mixer_online_s_a: s_a_path})
        return ex

    def train_critic_end(self, last_critic_s_a):
        _, final_Q, end_c_loss = self.sess.run([self.train_op_c, self.online_q_tot, self.mixer_loss],
                                               feed_dict={self.mixer_online_s_a: last_critic_s_a,
                                               self.mixer_target_values: np.zeros((len(last_critic_s_a), 1))})
        return sum(final_Q) / len(final_Q), end_c_loss

    def update_target_network(self):
        self.sess.run(self.copy_ops)
    
    def create_policy_network(self, input_state, name_scope):
        # with tf.variable_scope('policy', initializer = tf.contrib.layers.xavier_initializer()):
        with tf.variable_scope(name_scope + '_network'):
            before_layer = tf.keras.layers.LSTM(units=DENSE_UNIT_NUMBER)(input_state)
            # before_layer = tf.layers.dense(input_state, units=DENSE_UNIT_NUMBER, activation=tf.nn.relu)
            middle_dense_layer = tf.layers.dense(before_layer, units=DENSE_UNIT_NUMBER, activation=tf.nn.relu)
            q = tf.layers.dense(middle_dense_layer, self.action_dim)
        return q

    def create_mixer(self, mixer_state_with_action, name_scope):
        with tf.variable_scope(name_scope + '_network'):
            if not qpd_utils.enable_critic_multi_channel:
                layer_1 = tf.layers.dense(mixer_state_with_action, units=DENSE_UNIT_NUMBER, activation=tf.nn.relu)
                layer_2 = tf.layers.dense(layer_1, units=DENSE_UNIT_NUMBER, activation=tf.nn.relu)
                v = tf.layers.dense(layer_2, units=1)
                return v
            else:
                group = qpd_utils.agent_group
                group_num = len(group)
                layer_1 = [tf.keras.layers.Dense(units=DENSE_UNIT_NUMBER, activation=tf.nn.relu) for _ in range(len(group))]
                layer_2 = [tf.keras.layers.Dense(units=DENSE_UNIT_NUMBER, activation=tf.nn.relu) for _ in range(len(group))]
                reshaped_s_a = tf.reshape(mixer_state_with_action, (-1, self.agent_num, (self.state_dim + self.action_dim)))
                agent_s_a = [reshaped_s_a[:, i, :] for i in range(self.agent_num)]

                # group_1_hs = [layer_2[0](layer_1[0](agent_s_a[i])) for i in range(sum(group[0:0]), sum(group[0:1]))]
                # group_2_hs = [layer_2[1](layer_1[1](agent_s_a[i])) for i in range(sum(group[0:1]), sum(group[0:2]))]
                # group_hs = [layer_2(layer_1(agent_s_a[i])) for i in range(self.agent_num)]
                # group_hs = [[layer_2[j](layer_1[j](agent_s_a[i])) for i in range(sum(group[0:j]), sum(group[0:(j + 1)]))] for j in range(group_num)]

                group_hs = []
                for j in range(group_num):
                    for i in range(sum(group[0:j]), sum(group[0:(j + 1)])):
                        group_hs.append(layer_2[j](layer_1[j](agent_s_a[i])))
                if qpd_utils.channel_merge == 'concat':
                    hs = tf.concat(group_hs, 1)
                elif qpd_utils.channel_merge == 'add':
                    hs = tf.add_n(group_hs)
                else:
                    raise RuntimeError('Channel merge method is not correct.@dong')
                # hs = tf.concat(group_1_hs + group_2_hs, 1)
                # hs = tf.add(hs, 1)
                v = tf.layers.dense(hs, units=1)
                return v

    # FIXME 0127 save models
    def save_check_points(self):
        if qpd_utils.chpt_path is None:
            print('- Save path not defined.')
            return
        chpt_path = qpd_utils.chpt_path + datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S') + '.ckpt'
        self.saver.save(self.sess, chpt_path)
        print('- Checkpoint:', 'saved at', chpt_path)



