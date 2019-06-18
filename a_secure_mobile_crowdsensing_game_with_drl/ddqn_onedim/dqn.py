__author__ = 'alibaba'
__date__ = '2018/11/19'

import numpy as np
import os
import random
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
"""
使用一维数据进行训练 收敛程度不怎么好
"""


class DeepQNetwork(object):
    """
    均使用二维数据表进行表示，动作表示列数，其中的状态需要另外添加进去，表的结构均采用索引值进行访问
    """

    def __init__(self, name, learning_rate=0.1, reward_decay=0.8, e_greedy=0.9):
        """
        初始化Q-Table
        :param actions: 表示动作的大小，即动作集的大小，本身表示索引值
        :param learning_rate:  强化学习算法的学习速率
        :param reward_decay: 回报的打折率
        :param e_greedy: 选择最优解的e-greedy策略
        """
        self.DataList = []
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.name_system = name
        self.net_learn_num = 0
        # print(self.name_system)
        if self.name_system == "mcs":
            self._build_net_server()
        else:
            self._build_net()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

    def learn(self):
        state1, action1, reward1, state2 = self.gain_train_data()
        q_eva = self.sess.run(self.prediction, feed_dict={self.xs: state1})
        q_target = self.sess.run(self.prediction_target, feed_dict={self.xs1: state2})
        final_target = q_eva.copy()
        final_target[0][action1] = q_target[0][np.argmax(q_target)] + reward1
        self.sess.run(self.train_step, feed_dict={self.xs: state1, self.final_target: final_target})
        self.net_learn_num += 1

        if self.net_learn_num % 50 == 0:
            self.sess.run(self.replace_target_op)

    def store_transition(self, s, s1, s2, a, a1, a2, r, r1, r2, s_, s1_, s2_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, s1, s2, a, a1, a2, r, r1, r2, s_, s1_, s2_))
        self.DataList.append(transition)
        # self.q_table[index, :] = transition
        #  self.q_table.append(transition)
        self.memory_counter += 1

    def transform_state(self, state, state_user1, state_user2):
        A = self.transform_string(state)
        B = self.transform_string(state_user1)
        C = self.transform_string(state_user2)
        state = A + B + C
        state = np.array(state)
        return state

    def _build_net(self):
        self.xs = tf.placeholder(tf.float32, [1, None])
        self.final_target = tf.placeholder(tf.float32, [1, None])
        with tf.variable_scope('train_net'):
            c_names, w_initializer, b_initializer = ['train_net_params',
                                                     tf.GraphKeys.GLOBAL_VARIABLES], tf.random_normal_initializer(0.,
                                                                                                                  0.3), tf.constant_initializer(
                0.1)

            with tf.variable_scope('l1'):
                weights = tf.get_variable(self.name_system + 'w1', [6, 11], initializer=w_initializer,
                                          collections=c_names)
                biases = tf.get_variable(self.name_system + 'b1', [1, 11], initializer=b_initializer,
                                         collections=c_names)
                wx_plus_b = tf.matmul(self.xs, weights) + biases
                l1 = tf.nn.relu(wx_plus_b)
            with tf.variable_scope('l2'):
                weights1 = tf.get_variable(self.name_system + 'w2', [11, 11], initializer=w_initializer,
                                           collections=c_names)
                biases1 = tf.get_variable(self.name_system + 'b2', [1, 11], initializer=b_initializer,
                                          collections=c_names)
                wx_plus_b1 = tf.matmul(l1, weights1) + biases1
                self.prediction = wx_plus_b1
        self.loss = tf.reduce_mean(tf.squared_difference(self.prediction, self.final_target))
        self.train_step = tf.train.GradientDescentOptimizer(0.1).minimize(self.loss)

        self.xs1 = tf.placeholder(tf.float32, [1, None])
        with tf.variable_scope('target_net'):
            c_names, w_initializer, b_initializer = ['target_net_params',
                                                     tf.GraphKeys.GLOBAL_VARIABLES], tf.random_normal_initializer(0.,
                                                                                                                  0.3), tf.constant_initializer(
                0.1)

            with tf.variable_scope('l1'):
                Weights = tf.get_variable(self.name_system + 'w1', [6, 11], initializer=w_initializer,
                                          collections=c_names)
                biases = tf.get_variable(self.name_system + 'b1', [1, 11], initializer=b_initializer,
                                         collections=c_names)
                Wx_plus_b = tf.matmul(self.xs1, Weights) + biases
                l1 = tf.nn.relu(Wx_plus_b)
            with tf.variable_scope('l2'):
                Weights1 = tf.get_variable(self.name_system + 'w2', [11, 11], initializer=w_initializer,
                                           collections=c_names)
                biases1 = tf.get_variable(self.name_system + 'b2', [1, 11], initializer=b_initializer,
                                          collections=c_names)
                Wx_plus_b1 = tf.matmul(l1, Weights1) + biases1
                self.prediction_target = Wx_plus_b1
        if int((tf.__version__).split('.')[1]) < 12:
            init = tf.initialize_all_variables()
        else:
            init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def _build_net_server(self):
        self.xs = tf.placeholder(tf.float32, [1, None])
        self.final_target = tf.placeholder(tf.float32, [1, None])
        with tf.variable_scope('train_net'):
            c_names, w_initializer, b_initializer = ['train_net_params',
                                                     tf.GraphKeys.GLOBAL_VARIABLES], tf.random_normal_initializer(0.,
                                                                                                                  0.3), tf.constant_initializer(
                0.1)

            with tf.variable_scope('l1'):
                Weights = tf.get_variable(self.name_system + 'w1', [6, 11], initializer=w_initializer,
                                          collections=c_names)
                biases = tf.get_variable(self.name_system + 'b1', [1, 11], initializer=b_initializer,
                                         collections=c_names)
                Wx_plus_b = tf.matmul(self.xs, Weights) + biases
                l1 = tf.nn.relu(Wx_plus_b)
            with tf.variable_scope('l2'):
                Weights1 = tf.get_variable(self.name_system + 'w2', [11, 26], initializer=w_initializer,
                                           collections=c_names)
                biases1 = tf.get_variable(self.name_system + 'b2', [1, 26], initializer=b_initializer,
                                          collections=c_names)
                Wx_plus_b1 = tf.matmul(l1, Weights1) + biases1
                self.prediction = Wx_plus_b1
        self.loss = tf.reduce_mean(tf.squared_difference(self.prediction, self.final_target))
        self.train_step = tf.train.GradientDescentOptimizer(0.1).minimize(self.loss)

        self.xs1 = tf.placeholder(tf.float32, [1, None])
        with tf.variable_scope('target_net'):
            c_names, w_initializer, b_initializer = ['target_net_params',
                                                     tf.GraphKeys.GLOBAL_VARIABLES], tf.random_normal_initializer(0.,
                                                                                                                  0.3), tf.constant_initializer(
                0.1)

            with tf.variable_scope('l1'):
                Weights = tf.get_variable(self.name_system + 'w1', [6, 11], initializer=w_initializer,
                                          collections=c_names)
                biases = tf.get_variable(self.name_system + 'b1', [1, 11], initializer=b_initializer,
                                         collections=c_names)
                Wx_plus_b = tf.matmul(self.xs1, Weights) + biases
                l1 = tf.nn.relu(Wx_plus_b)
            with tf.variable_scope('l2'):
                Weights1 = tf.get_variable(self.name_system + 'w2', [11, 11], initializer=w_initializer,
                                           collections=c_names)
                biases1 = tf.get_variable(self.name_system + 'b2', [1, 11], initializer=b_initializer,
                                          collections=c_names)
                Wx_plus_b1 = tf.matmul(l1, Weights1) + biases1
                self.prediction_target = Wx_plus_b1
        if int((tf.__version__).split('.')[1]) < 12:
            init = tf.initialize_all_variables()
        else:
            init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def select_action(self, S, S1, S2):
        DX = self.transform_state(S, S1, S2)
        DX1 = []
        DX1.append(DX)
        x_data = np.array(DX1)
        stateAll = self.sess.run(self.prediction, feed_dict={self.xs: x_data})
        # self.sess.run(self.train_step, feed_dict={self.xs: x_data, self.ys: y_data})
        stateAll = np.array(stateAll)
        # self.sess.run(self.train_step, feed_dict={self.xs: x_data, self.ys: y_data})
        if np.random.uniform() < self.epsilon:
            # print("choose max")
            action = np.argmax(stateAll)

        else:
            # print("choose random")
            if self.name_system == "mcs":
                acList = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
            else:
                acList = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
            action = np.random.choice(acList)
        return action

    def gain_train_data(self):
        action_store = 0
        reward_store = 0
        A = list(random.sample(self.DataList, 1))[0]
        B = self.transform_string(A[0])
        B1 = self.transform_string(A[1])
        B2 = self.transform_string(A[2])
        D = self.transform_string(A[9])
        D1 = self.transform_string(A[10])
        D2 = self.transform_string(A[11])
        state_s = B + B1 + B2
        x_trans = np.array(state_s)
        state_s = x_trans.reshape(1, 6)
        state_s1 = D + D1 + D2
        x1_trans = np.array(state_s1)
        state_s1 = x1_trans.reshape(1, 6)
        if (self.name_system == "mcs"):
            action_store = int(A[3])
            reward_store = float(A[6])
        if (self.name_system == "user1"):
            action_store = int(A[4])
            reward_store = float(A[7])
        if (self.name_system == "user2"):
            action_store = int(A[5])
            reward_store = float(A[8])
            self.mm = 1
        return state_s, action_store, reward_store, state_s1

    def transform_string(self, A):
        B = str(A)
        C = B.split('[')
        C = str(C[1])
        C1 = C.split(']')
        C1 = str(C1[0])
        C2 = C1.split(',')
        C2[0] = int(C2[0])
        C2[1] = int(C2[1])
        return C2
