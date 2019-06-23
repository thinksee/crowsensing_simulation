__author__ = 'think see'
__date__ = '2018/11/19'

import numpy as np
import pandas as pd


class QLearningTable(object):

    def __init__(self, actions, learning_rate=0.1, reward_decay=0.8, e_greedy=0.9):
        """
        init q table
        :param actions: the action of the agent, is a index, not value.
        :param learning_rate:  the learning of q learning, [] can be the best.
        :param reward_decay: the reward decay of q learning, while is a number between 0 and 1.
        :param e_greedy: the e greedy value that can solve the trap of the Local Optimal Solution.
        """
        self.__actions = actions
        self.__q_table = pd.DataFrame(columns=self.__actions, dtype=np.float64)
        self.__lr = learning_rate
        self.__gamma = reward_decay
        self.__epsilon = e_greedy

    def select_action(self, observation, policy):
        """
        through q learning select action
        :param observation:
        :param policy: policy=1, e-greedy; policy=2, random; policy=3, greedy
        :return:
        """
        self.is_exist_state(observation)
        if policy == 1:
            if np.random.uniform() < self.__epsilon:
                state_action = self.__q_table.loc[observation, :]
                action = np.random.choice(state_action[state_action == np.max(state_action)].index)
            else:
                action = np.random.choice(self.__actions)
        elif policy == 2:
            action = np.random.choice(self.__actions)
        elif policy == 3:
            state_action = self.__q_table.loc[observation, :]
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            raise NameError('the policy is\'t exist.')

        return action

    def learn(self, s, a, r, s_):
        self.is_exist_state(s_)
        q_target = r + self.__gamma * self.__q_table.loc[s_, :].max()
        self.__q_table.loc[s, a] = (1 - self.__lr) * self.__q_table.loc[s, a] + self.__lr * q_target

    def is_exist_state(self, state):

        if state not in self.__q_table.index:
            self.__q_table = self.__q_table.append(
                pd.Series(
                    [0]*len(self.__actions),
                    index=self.__q_table.columns,
                    name=state,
                )
            )
            return False
        return True

    def clear(self):
        for idx in self.__q_table.index:
            self.__q_table.drop(idx, inplace=True)

    def get_table_point_value(self, state, action):
        """
        get the table[state, action] of the value
        :param state:
        :param action:
        :return:
        """
        if self.is_exist_state(state):
            return self.__q_table.loc[state, action]
        else:
            return 0.0

    def set_table_point_value(self, state, action, value):
        """
        set the value of the [state, action]
        :param state:
        :param action:
        :param value:
        :return:
        """
        self.is_exist_state(state)
        self.__q_table.loc[state, action] = value









