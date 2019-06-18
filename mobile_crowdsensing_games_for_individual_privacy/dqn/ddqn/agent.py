__author__ = 'alibaba'
__date__ = '2019/3/18'

import numpy as np


class MCSAgent(object):
    def __init__(self, actions,):
        self.actions = actions
        self.n_actions = len(actions)
        self.init_action = np.random.choice(self.actions)
        self.n_features = 2

    @staticmethod
    def get_payments(action, len, acc):
        payments = list()
        payments.append(0)
        for idx in range(len):
            payments.append(action + acc)
        return payments

    def get_index(self, action):
        return self.actions.index(action)

    def get_action(self, idx):
        return self.actions[idx]

    @staticmethod
    def get_mcs_utility_reciprocal(user_action_list, data_range, confidence_level, n_user):
        if n_user > len(user_action_list):
            raise RuntimeError('the user action list not enough.')
        privacy_parameter_square_sum = 0
        for user_action in user_action_list:
            privacy_parameter_square_sum += np.square(user_action)
        if privacy_parameter_square_sum == 0:
            return 0.0
        else:
            return 1 / (np.sqrt(2) * data_range / (2 * np.sqrt(n_user - confidence_level)) * np.sqrt(
                1 / privacy_parameter_square_sum))

    @staticmethod
    def get_mcs_utility_percentage(user_action_list, data_range, confidence_level, n_user):
        if n_user > len(user_action_list):
            raise RuntimeError('the user action list not enough.')
        privacy_parameter_square_sum = 0
        for user_action in user_action_list:
            privacy_parameter_square_sum += np.square(user_action)
        return data_range * (1 - privacy_parameter_square_sum / data_range) / (n_user * (1 - confidence_level))


class UserAgent(object):
    def __init__(self, actions, costs):
        self.actions = actions
        self.n_actions = len(actions)
        self.init_action = np.random.choice(self.actions)
        self.costs = costs
        self.n_features = 2

    def get_index(self, action):
        return self.actions.tolist().index(action)

    def get_action(self, idx):
        return self.actions[idx]

    def get_cost(self, idx):
        return self.costs[idx]
