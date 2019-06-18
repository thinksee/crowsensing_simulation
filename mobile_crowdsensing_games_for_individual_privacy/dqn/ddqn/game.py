__author__ = 'alibaba'
__date__ = '2019/3/18'

from mobile_crowdsensing_games_for_individual_privacy.dqn.ddqn.ddqn import DoubleDQN
from mobile_crowdsensing_games_for_individual_privacy.agent import UserAgent, MCSAgent

import matplotlib.pyplot as plt
import numpy as np
import os

MAX_EPISODE = 100
MAX_STEP = 5000
DATA_RANGE = 10
CONFIDENCE_LEVEL = 0.95
N_USER = 2
if not os.path.exists('img'):
    os.makedirs('img')

if not os.path.exists('data'):
    os.makedirs('data')

agent_mcs = MCSAgent(range(0, 20, 2))
agent_user1 = UserAgent(np.arange(0.1, 1.1, 0.1), np.arange(0.1, 1.1, 0.1))
agent_user2 = UserAgent(np.arange(0.1, 1.1, 0.1), np.arange(0.1, 1.1, 0.1))
learn_episode = 200
memory_size = 50


def reset():
    user1_dqn = DoubleDQN(user1.n_actions,
                          user1.n_features,
                          memory_size=memory_size,
                          e_greedy_increment=0.001,
                          double_q=True,
                          sess=None
                          )
    user2_dqn = DoubleDQN(user2.n_actions,
                          user2.n_features,
                          memory_size=memory_size,
                          e_greedy_increment=0.001,
                          double_q=True,
                          sess=None)

    mcs_dqn = DoubleDQN(mcs.n_actions,
                        mcs.n_features,
                        memory_size=memory_size,
                        e_greedy_increment=0.001,
                        double_q=True,
                        sess=None)
    return mcs_dqn, user1_dqn, user2_dqn


def game():
    matrix_utility_mcs = [[0 for _ in range(MAX_EPISODE)] for _ in range(MAX_STEP)]
    matrix_utility_user1 = [[0 for _ in range(MAX_EPISODE)] for _ in range(MAX_STEP)]
    matrix_utility_user2 = [[0 for _ in range(MAX_EPISODE)] for _ in range(MAX_STEP)]

    matrix_action_mcs = [[0 for _ in range(MAX_EPISODE)] for _ in range(MAX_STEP)]
    matrix_action_user1 = [[0 for _ in range(MAX_EPISODE)] for _ in range(MAX_STEP)]
    matrix_action_user2 = [[0 for _ in range(MAX_EPISODE)] for _ in range(MAX_STEP)]

    matrix_action_mcs_index = [[0 for _ in range(MAX_EPISODE)] for _ in range(MAX_STEP)]
    matrix_action_user1_index = [[0 for _ in range(MAX_EPISODE)] for _ in range(MAX_STEP)]
    matrix_action_user2_index = [[0 for _ in range(MAX_EPISODE)] for _ in range(MAX_STEP)]

    matrix_aggregate_error = [[0 for _ in range(MAX_EPISODE)] for _ in range(MAX_STEP)]
    for episode in range(MAX_EPISODE):
        print('episode', episode)
        mcs_dqn, user1_dqn, user2_dqn = reset()
        mcs_action = mcs.init_action
        mcs_action_index = mcs.get_index(mcs_action)

        user1_action = user1.init_action
        user1_action_index = user1.get_index(user1_action)

        user2_action = user2.init_action
        user2_action_index = user2.get_index(user2_action)
        for step in range(MAX_STEP):
            mcs_state = np.array([user1_action_index, user2_action_index], dtype=float)
            user1_state = np.array([mcs_action_index, user1_action_index], dtype=float)
            user2_state = np.array([mcs_action_index, user2_action_index], dtype=float)

            user1_action_index = user1_dqn.select_action(user1_state)
            user1_action = user1.get_action(user1_action_index)

            user2_action_index = user2_dqn.select_action(user2_state)
            user2_action = user2.get_action(user2_action_index)

            mcs_action_index = mcs_dqn.select_action(mcs_state)
            mcs_action = mcs.get_action(mcs_action_index)

            payment = mcs.get_payments(mcs_action, 10, 0.5)

            r_user1 = payment[user1_action_index] - user1.get_cost(user1_action_index) * user1_action
            r_user2 = payment[user2_action_index] - user2.get_cost(user2_action_index) * user2_action
            mcs_utility_reciprocal = mcs.get_mcs_utility_reciprocal(
                user_action_list=[user1_action, user2_action],
                data_range=DATA_RANGE,
                confidence_level=CONFIDENCE_LEVEL,
                n_user=N_USER)
            r_mcs = mcs_utility_reciprocal - payment[user1_action_index] - payment[user2_action_index]

            # 获取下一状态
            _mcs_state = np.array([user1_action_index, user2_action_index], dtype=float)
            _user1_state = np.array([mcs_action_index, user1_action_index], dtype=float)
            _user2_state = np.array([mcs_action_index, user2_action_index], dtype=float)
            # 存储四元组 s, a, r, s_
            mcs_dqn.store_transition(mcs_state, mcs_action_index, r_mcs, _mcs_state)
            user1_dqn.store_transition(user1_state, user1_action_index, r_user1, _user1_state)
            user2_dqn.store_transition(user2_state, user2_action_index, r_user2, _user2_state)
            # 进行学习
            user1_dqn.learn()  # s, a, r, s_
            user2_dqn.learn()
            mcs_dqn.learn()

            # 参数存储:效用值
            matrix_utility_mcs[step][episode] = r_mcs
            matrix_utility_user1[step][episode] = r_user1
            matrix_utility_user2[step][episode] = r_user2
            # 参数存储:动作值
            matrix_action_mcs[step][episode] = mcs_action
            matrix_action_user1[step][episode] = user1_action
            matrix_action_user2[step][episode] = user2_action
            # 动作选择值
            matrix_action_mcs_index[step][episode] = mcs_action_index
            matrix_action_user1_index[step][episode] = user1_action_index
            matrix_action_user2_index[step][episode] = user2_action_index
            # 聚合错误
            matrix_aggregate_error[step][episode] = mcs_utility_reciprocal

    np.savetxt('data\\utility-e-greedy-mcs-{}.txt'.format(MAX_STEP), matrix_utility_mcs, fmt='%.3f')
    np.savetxt('data\\utility-e-greedy-user1-{}.txt'.format(MAX_STEP), matrix_utility_user1, fmt='%.3f')
    np.savetxt('data\\utility-e-greedy-user2-{}.txt'.format(MAX_STEP), matrix_utility_user2, fmt='%.3f')

    np.savetxt('data\\index-e-greedy-mcs-{}.txt'.format(MAX_STEP), matrix_action_mcs_index, fmt='%3d')
    np.savetxt('data\\index-e-greedy-user1-{}.txt'.format(MAX_STEP), matrix_action_user1_index, fmt='%3d')
    np.savetxt('data\\index-e-greedy-user2-{}.txt'.format(MAX_STEP), matrix_action_user2_index, fmt='%3d')

    np.savetxt('data\\aggregate-error-mcs-{}'.format(MAX_STEP), matrix_aggregate_error, fmt='%.3f')

    array_utility_mcs = np.sum(matrix_utility_mcs, axis=1) / MAX_EPISODE
    array_utility_mcs = array_utility_mcs.T
    array_utility_user1 = np.sum(matrix_utility_user1, axis=1) / MAX_EPISODE
    array_utility_user1 = array_utility_user1.T
    array_utility_user2 = np.sum(matrix_utility_user2, axis=1) / MAX_EPISODE
    array_utility_user2 = array_utility_user2.T

    array_aggregate_error = np.sum(matrix_aggregate_error, axis=1) / MAX_EPISODE
    array_aggregate_error = array_aggregate_error.T

    # 画图 用户和服务器的效用值
    plt.figure(1)
    plt.subplot(221)
    plt.ylabel('Utility of the User')
    plt.plot(range(MAX_STEP), array_utility_user1)
    plt.subplot(222)
    plt.plot(range(MAX_STEP), array_utility_user2)
    plt.subplot(212)
    plt.xlabel('Time Slot')
    plt.ylabel('Utility of the MCS')
    plt.plot(range(MAX_STEP), array_utility_mcs)
    plt.savefig(
        os.path.abspath(os.path.join(os.path.abspath(
            os.path.dirname(__file__)), 'img', 'utility-e-greedy-{}.png'.format(MAX_STEP))))
    plt.tight_layout()
    plt.close()

    # 画图 服务器的聚合错误
    plt.figure(2)
    plt.xlabel('Time Slot')
    plt.ylabel('Aggregate Error of MCS')
    plt.plot(range(MAX_STEP), array_aggregate_error)
    plt.close()
    # 画图 用户和服务器的动作选择
    # 基本思路就是对每一行求出现次数最多的值，然后表示当前time slot选择的值
    index_mcs = [np.argmax(np.bincount(line)) for line in matrix_action_mcs_index]
    index_user1 = [np.argmax(np.bincount(line)) for line in matrix_action_user1_index]
    index_user2 = [np.argmax(np.bincount(line)) for line in matrix_action_user2_index]
    plt.figure(3)
    plt.plot(range(MAX_STEP), index_user1)
    plt.plot(range(MAX_STEP), index_user2)
    plt.plot(range(MAX_STEP), index_mcs)
    plt.savefig(
        os.path.abspath(os.path.join(os.path.abspath(
            os.path.dirname(__file__)), 'img', 'index-e-greedy-{}.png'.format(MAX_STEP))))
    plt.tight_layout()
    plt.close()


if __name__ == '__main__':
    game()

