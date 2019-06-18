__author__ = 'alibaba'
__date__ = '2018/11/19'

from mobile_crowdsensing_games_in_vehicular_networks.model import *
from mobile_crowdsensing_games_in_vehicular_networks.q_learning import *

import matplotlib.pyplot as plt
import os
import numpy as np

# 初始化 平台和用户的基本参数
mcs = MCSAgent(range(0, 51, 2), np.arange(0, 11, 1))
user1 = UserAgent(range(0, 11, 1),
                  [1, 10],
                  [0.1, 0.9],
                  0.9,
                  5,
                  [0, 1, 2, 3, 4, 5],
                  [0.02, 0.04, 0.3, 0.3, 0.3, 0.04],
                  np.arange(0, 5.5, 0.5)
                  )
user2 = UserAgent(range(0, 11, 1),
                  [1, 10],
                  [0.1, 0.9],
                  0.9,
                  5,
                  [0, 1, 2, 3, 4, 5],
                  [0.02, 0.04, 0.3, 0.3, 0.3, 0.04],
                  np.arange(0, 5.5, 0.5)
                  )

mcs_table = QLearningTable(actions=list(range(len(mcs.actions))))  # 11*11 = 121
user1_table = QLearningTable(actions=list(range(len(user1.actions))))  # 26*2 = 52
user2_table = QLearningTable(actions=list(range(len(user2.actions))))  # 26*2 = 52

# 初始化程序变量
max_episode = 500
max_step = 1500

beta = 10  # effort贡献率


def game():
    utility_mcs = [[0 for _ in range(max_episode)]for _ in range(max_step)]
    utility_user1 = [[0 for _ in range(max_episode)]for _ in range(max_step)]
    utility_user2 = [[0 for _ in range(max_episode)]for _ in range(max_step)]
    action_mcs = [[0 for _ in range(max_episode)] for _ in range(max_step)]
    action_user1 = [[0 for _ in range(max_episode)] for _ in range(max_step)]
    action_user2 = [[0 for _ in range(max_episode)] for _ in range(max_step)]
    action_mcs_hist = np.zeros(len(mcs.actions))
    action_user1_hist = np.zeros(len(user1.actions))
    action_user2_hist = np.zeros(len(user2.actions))
    action_mcs_list = list()
    action_user1_list = list()
    action_user2_list = list()
    cost_user1 = [[0 for _ in range(max_episode)] for _ in range(max_step)]
    cost_user2 = [[0 for _ in range(max_episode)] for _ in range(max_step)]
    for episode in range(max_episode):
        mcs_table.clear()
        user1_table.clear()
        user2_table.clear()
        # 平台选择一个action即支付的基础值，用户选择一个action和snr，用作系统的初始化动作和状态
        mcs_action = mcs.select_action()
        mcs_action_index = mcs.get_action_index(mcs_action)
        user1_snr = user1.select_snr()
        user1_snr_index = user1.get_snr_index(user1_snr)
        user2_snr = user2.select_snr()
        user2_snr_index = user2.get_snr_index(user2_snr)
        user1_action = user1.select_action()
        user1_action_index = user1.get_action_index(user1_action)
        user2_action = user2.select_action()
        user2_action_index = user2.get_action_index(user2_action)
        for step in range(max_step):
            # 组合状态
            mcs_state = str([user1_action_index, user2_action_index])
            user1_state = str([mcs_action_index, user1_snr_index])
            user2_state = str([mcs_action_index, user2_snr_index])
            user1_action_index = user1_table.select_action(user1_state)
            action_user1_hist[user1_action_index] += 1
            action_user1_list.append(user1_action_index)

            user1_action = user1.get_action_by_index(user1_action_index)
            user2_action_index = user2_table.select_action(user2_state)
            action_user2_hist[user2_action_index] += 1
            action_user2_list.append(user2_action_index)

            user2_action = user2.get_action_by_index(user2_action_index)
            mcs_action_index = mcs_table.select_action(mcs_state)
            action_mcs_hist[mcs_action_index] += 1
            action_mcs_list.append(mcs_action_index)

            mcs_action = mcs.get_action_by_index(mcs_action_index)
            payment = mcs.get_payment_layer(mcs_action, 11, 0.5)
            # 获取reward

            r_user1 = payment[user1_action_index] - user1.select_cost_by_index(
                user1_action_index) * user1_action / user1.get_snr_by_index(user1_snr_index)
            r_user2 = payment[user2_action_index] - user2.select_cost_by_index(
                user2_action_index) * user2_action / user2.get_snr_by_index(user2_snr_index)
            r_mcs = mcs.beta[user2_action_index] * user2.get_action_by_index(
                user2_action_index) + mcs.beta[user1_action_index] * user1.get_action_by_index(
                user1_action_index) - payment[user1_action_index] - payment[user2_action_index]

            user1_speed = user1.select_speed()
            user2_speed = user2.select_speed()
            next_snr1, next_snr_prob1 = user1.get_snr_prob(user1_speed, user1_snr)
            next_snr2, next_snr_prob2 = user2.get_snr_prob(user2_speed, user2_snr)
            user1_snr_index = user1.get_snr_index(next_snr1)
            user2_snr_index = user2.get_snr_index(next_snr2)
            # 获取下一状态
            _mcs_state = str([user1_action_index, user2_action_index])
            _user1_state = str([mcs_action_index, user1_snr_index])
            _user2_state = str([mcs_action_index, user2_snr_index])
            # 进行学习
            user1_table.learn(user1_state, user1_action_index, r_user1, _user1_state)  # s, a, r, s_
            user2_table.learn(user2_state, user2_action_index, r_user2, _user2_state)
            mcs_table.learn(mcs_state, mcs_action_index, r_mcs, _mcs_state)
            # # PDS-learning
            # # 其主要思想是:前半部分使用Q-learning,后半部分使用PDS-learning来加速收敛过程
            # for i in range(mcs.get_action_length()):
            #     for snr1 in range(user1.get_snr_length()):
            #         for action in range(user1.get_action_length()):
            #             temp = 0
            #             for snr in range(user1.get_snr_length()):
            #                 temp += user1.select_prob_by_snr(next_snr_prob1, snr1, snr) * \
            #                         user1_table.get_table_value(str([i, snr1]), action)
            #             user1_table.set_table_value(str([i, snr1]), action, temp)
            #     for snr2 in range(user2.get_snr_length()):
            #         for action in range(user2.get_action_length()):
            #             temp = 0
            #             for snr in range(user2.get_snr_length()):
            #                 temp += user2.select_prob_by_snr(next_snr_prob2, snr2, snr) * \
            #                         user2_table.get_table_value(str([i, snr2]), action)
            #             user2_table.set_table_value(str([i, snr2]), action, temp)

            # 参数存储：效用值

            utility_mcs[step][episode] = r_mcs
            utility_user1[step][episode] = r_user1
            utility_user2[step][episode] = r_user2
            # 参数存储：动作值
            action_mcs[step][episode] = mcs_action
            action_user1[step][episode] = user1_action
            action_user2[step][episode] = user2_action

            cost_user1[step][episode] = user1.select_cost_by_index(user1_action_index)
            cost_user2[step][episode] = user1.select_cost_by_index(user2_action_index)
    #     file.write(str(action_user2))
    # utility的平均趋势
    avu_mcs = np.sum(utility_mcs, axis=1)/max_episode
    avu_mcs = avu_mcs.T
    avu_user1 = np.sum(utility_user1, axis=1)/max_episode
    avu_user1 = avu_user1.T
    avu_user2 = np.sum(utility_user2, axis=1)/max_episode
    avu_user2 = avu_user2.T
    avc_user1 = np.sum(cost_user1, axis=1)/max_episode
    avc_user1 = avc_user1.T
    avc_user2 = np.sum(cost_user2, axis=1)/max_episode
    avc_user2 = avc_user2.T
    # 画图 两个汽车的消耗值
    plt.figure(0)
    plt.plot(range(max_step), avc_user1)
    plt.plot(range(max_step), avc_user2)
    plt.xlabel('Time slot')
    plt.ylabel('Energy consumption of the vehicle')
    plt.savefig(os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'img', 'consumption.png')))
    plt.close()
    # 画图 汽车和服务器的效用值
    plt.figure(1)
    plt.subplot(221)
    plt.ylabel('Utility of the vehicle')
    plt.plot(range(max_step), avu_user1)
    plt.subplot(222)
    plt.ylabel('Utility of the vehicle')
    plt.plot(range(max_step), avu_user2)
    plt.subplot(212)
    plt.xlabel('Time Slot')
    plt.ylabel('Utility of the mcs')
    plt.plot(range(max_step), avu_mcs)
    plt.savefig(os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'img', 'utility.png')))
    plt.tight_layout()
    plt.close()
    # action的平均趋势
    ava_mcs = np.sum(action_mcs, axis=1)/max_episode
    ava_mcs = ava_mcs.T
    ava_user1 = np.sum(action_user1, axis=1)/max_episode
    ava_user1 = ava_user1.T
    ava_user2 = np.sum(action_user2, axis=1)/max_episode
    ava_user2 = ava_user2.T
    np.savetxt('xlang_lua_data\\aca_mcs.txt', ava_mcs)
    np.savetxt('xlang_lua_data\\ava_user1.txt', ava_user1)
    np.savetxt('xlang_lua_data\\ava_user2.txt', ava_user2)
    # 画图 汽车和服务器的消耗值
    plt.figure(2)
    plt.subplot(221)
    plt.ylabel('Sense Effort of the vehicle')
    plt.plot(range(max_step), ava_user1)
    plt.subplot(222)
    plt.ylabel('Sense Effort of the vehicle')
    plt.plot(range(max_step), ava_user2)
    plt.subplot(212)
    plt.xlabel('Time Slot')
    plt.ylabel('Payment of the vehicle')
    plt.plot(range(max_step), ava_mcs)
    plt.savefig(os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'img', 'sense-effort.png')))
    plt.tight_layout()
    plt.close()

    # 画图 汽车和服务器的动作选择直方图
    # import seaborn as sns
    plt.figure(3)
    plt.hist(action_mcs_list, bins=26, align='mid', facecolor='yellow', edgecolor='black', normed=True)
    plt.title('The Histogram of mcs\'s action')
    plt.xlabel('The actions')
    plt.ylabel('The number of actions')
    plt.savefig(os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'img', 'mcs_action_hist.png')))
    np.savetxt('xlang_lua_data\\action_mcs_hist.txt', action_mcs_hist)
    plt.close()
    plt.figure(4)
    plt.hist(action_user1_list, bins=11, align='mid', facecolor='yellow', edgecolor='black', normed=True)
    plt.title('The Histogram of user1\'s action')
    # plt.plot(action_user1_hist)
    plt.xlabel('The actions')
    plt.ylabel('The number of actions')
    plt.savefig(os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'img', 'user1_action_hist.png')))
    np.savetxt('xlang_lua_data\\action_user1_hist.txt', action_user1_hist)
    plt.close()
    plt.figure(5)
    plt.hist(action_user2_list, bins=11, align='mid', facecolor='yellow', edgecolor='black', normed=True)
    plt.title('The Histogram of user2\'s action')
    plt.xlabel('The actions')
    plt.ylabel('The number of actions')
    plt.savefig(os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'img', 'user2_action_hist.png')))
    np.savetxt('xlang_lua_data\\action_user2_hist.txt', action_user2_hist)
    plt.close()


if __name__ == '__main__':
    game()
