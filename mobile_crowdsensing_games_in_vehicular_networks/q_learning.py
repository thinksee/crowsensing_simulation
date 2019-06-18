__author__ = 'alibaba'
__date__ = '2018/11/19'

import numpy as np
import pandas as pd


class QLearningTable(object):
    """
    均使用二维数据表进行表示，动作表示列数，其中的状态需要另外添加进去，表的结构均采用索引值进行访问
    """
    def __init__(self, actions, learning_rate=0.1, reward_decay=0.8, e_greedy=0.9):
        """
        初始化Q-Table
        :param actions: 表示动作的大小，即动作集的大小，本身表示索引值
        :param learning_rate:  强化学习算法的学习速率
        :param reward_decay: 回报的打折率
        :param e_greedy: 选择最优解的e-greedy策略
        """
        self.actions = actions
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy

    def select_action(self, observation):
        """
        根据observation选择相应的
        :param observation: 观测当前的状态 list 存储索引值
        :return: action,索引值
        """
        self.is_exist_state(observation)
        if np.random.uniform() < self.epsilon:
            state_action = self.q_table.loc[observation, :]
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            action = np.random.choice(self.actions)
        return action

    def select_action_random(self, observation):
        """
        随机选择特征\action
        :param observation:观测当前的状态 list 存储索引值
        :return:
        """
        self.is_exist_state(observation)
        return np.random.choice(self.actions)

    def learn(self, s, a, r, s_):
        """
        对于r进行学习，修改Q-table的值
        :param s: 当前状态
        :param a: 当前选择的动作
        :param r: 当前的奖励值，即各自的效用值
        :param s_: 观测到的下一状态
        :return: null
        """
        self.is_exist_state(s_)
        q_target = r + self.gamma * self.q_table.loc[s_, :].max()
        self.q_table.loc[s, a] = (1 - self.lr) * self.q_table.loc[s, a] + self.lr * q_target

    def is_exist_state(self, state):
        """
        检查状态的存在性，若该状态不在DataFrame中，利用Series把它加入进去
        :param state:
        :return:
        """
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )
            return False
        return True

    def clear(self):
        """
        Q-table数据清洗
        :return:
        """
        self.q_table[self.q_table != 0] = 0

    def get_table_value(self, state, action):
        """
        返回table所对应的value
        :param state:
        :param action:
        :return:
        """
        if self.is_exist_state(state):
            return self.q_table.loc[state, action]
        else:
            return 0.0

    def set_table_value(self, state, action, value):
        """
        为相应的状态-action集合设置value
        :param state:
        :param action:
        :param value:
        :return:
        """
        self.is_exist_state(state)
        self.q_table.loc[state, action] = value









