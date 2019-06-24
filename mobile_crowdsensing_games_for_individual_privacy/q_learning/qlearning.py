__author__ = 'think see'
__date__ = '2019/3/15'
import pandas as pd
import numpy as np
# These codes reference from https://morvanzhou.github.io/


class QLearningSingleUser:
    """
    single user Q-learning
    """
    def __init__(self, actions, learning_rate=0.1, reward_decay=0.9, e_greedy=0.9):
        """
        The class that
        :param actions: the action of the agent, is a index, not value.
        :param learning_rate: the learning of q learning, [] can be the best.
        :param reward_decay: the reward decay of q learning, while is a number between 0 and 1.
        :param e_greedy: the e greedy value that can solve the trap of the Local Optimal Solution.
        """
        self.__actions = actions
        # This is q table which the index can uniquely identify a row of data
        self.__q_table = pd.DataFrame(columns=self.__actions, dtype=np.float64)
        self.__lr = learning_rate
        self.__gamma = reward_decay
        self.__epsilon = e_greedy

    def select_action(self, observation, mode):
        self.is_exist_state(observation)
        if mode == 1:
            if np.random.uniform() < self.__epsilon:
                state_action = self.__q_table.loc[observation, :]
                return np.random.choice(state_action[state_action == np.max(state_action)].index)
            else:
                return np.random.choice(self.__actions)
        elif mode == 2:
            return np.random.choice(self.__actions)
        elif mode == 3:
            state_action = self.__q_table.loc[observation, :]
            return np.random.choice(state_action[state_action == np.max(state_action)].index),

    def learn(self, s, a, r, _s):
        self.is_exist_state(_s)
        q_target = r + self.__gamma * self.__q_table.loc[_s, :].max()
        # This is function you can see
        self.__q_table.loc[s, a] = (1 - self.__lr) * self.__q_table.loc[s, a] + self.__lr * q_target

    def is_exist_state(self, observation):
        if observation not in self.__q_table.index:
            self.__q_table = self.__q_table.append(
                pd.Series(
                    [0] * len(self.__actions),
                    index=self.__q_table.columns,
                    name=observation,
                )
            )
            return False
        return True

    def clear(self,):
        # not really delete the elements from q table
        # self.__q_table[self.__q_table != 0] = 0
        for idx in self.__q_table.index:
            self.__q_table.drop(idx, inplace=True)

    def get_table_point_value(self, state, action):
        if self.is_exist_state(state):
            return self.__q_table.loc[state, action]
        else:
            return 0.0

    def set_table__point_value(self, state, action, value):
        self.is_exist_state(state)
        self.__q_table.loc[state, action] = value

    def print_terminal_q_table(self):
        print(self.__q_table)

    def save_csv_q_table(self, out_path):
        self.__q_table.to_csv(out_path, index=False)

    def print_table(self):
        print('The q table of current user is {}'.format(self.__q_table))


class QLearningMultiUser:
    """
    multi users Q-learning
    """
    def __init__(self, n_user, actions, learning_rate=0.1, reward_decay=0.9, e_greedy=0.9):
        self.__actions = actions
        self.__q_table = [pd.DataFrame(columns=self.__actions, dtype=np.float) for _ in range(n_user)]
        self.__lr = learning_rate
        self.__gamma = reward_decay
        self.__epsilon = e_greedy
        self.__n_user = n_user

    def select_action(self, user_idx, observation, mode):
        self.is_exist_state(user_idx, observation)
        """
        The three ways about Q-learning are random, e-greedy and greed.
        And the e-greedy can jump out of local optimal solution.
        """
        if mode == 'random':
            return np.random.choice(self.__actions)
        elif mode == 'e-greedy':
            if np.random.uniform() < self.__epsilon:
                state_action = self.__q_table[user_idx].loc[observation, :]
                return np.random.choice(state_action[state_action == np.max(state_action)].index)
            else:
                return np.random.choice(self.__actions)
        elif mode == 'greedy':
            state_action = self.__q_table[user_idx].loc[observation, :]
            return np.random.choice(state_action[state_action == np.max(state_action)].index)

    def learn(self, user_idx, s, a, r, _s):
        self.is_exist_state(user_idx, _s)
        q_target = r + self.__gamma * self.__q_table[user_idx].loc[_s, :].max()
        self.__q_table[user_idx].loc[s, a] = (1 - self.__lr) * self.__q_table[user_idx].loc[s, a] + self.__lr * q_target

    def is_exist_state(self, user_idx, observation):
        if observation not in (self.__q_table[user_idx]).index:
            self.__q_table[user_idx] = (self.__q_table[user_idx]).append(
                pd.Series(
                    [0] * len(self.__actions),
                    index=(self.__q_table[user_idx]).columns,
                    name=observation,
                )
            )
            return False
        return True

    def clear(self, user_idx):
        # not really delete the rows of the q-table
        # self.__q_table[user_idx][self.__q_table[user_idx] != 0] = 0
        for idx in self.__q_table[user_idx].index:
            self.__q_table[user_idx].drop(idx, inplace=True)

    def clear_all(self):
        for user_idx in range(self.__n_user):
            for idx in self.__q_table[user_idx].index:
                self.__q_table[user_idx].drop(idx, inplace=True)

    def get_table_value(self, user_idx, state, action):
        if self.is_exist_state(user_idx, state):
            return self.__q_table[user_idx].loc[state, action]
        else:
            return 0.0

    def set_table_value(self, user_idx, state, action, value):
        self.is_exist_state(user_idx, state)
        self.__q_table[user_idx].loc[state, action] = value

    def print_terminal_q_table(self):
        for user_idx in range(self.__n_user):
            print("The %dth user's q-table is ".format(user_idx + 1), self.__q_table[user_idx])

    def save_csv_q_table(self, out_path):

        for user_idx in range(self.__n_user):
            out_path = out_path + '/q_table_{}'.format(user_idx + 1)
            self.__q_table[user_idx].to_csv(out_path, index=False)

    def get_action_len(self):
        return len(self.__actions)

    def print_table(self):
        for usr in range(self.__n_user):
            print('The q-table of user {} is {}'.format(usr, self.__q_table[usr]))

