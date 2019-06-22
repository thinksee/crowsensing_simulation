__author__ = 'think see'
__date__ = '2019/3/14'
import numpy as np
# These codes which need match your model.


class MCSAgent(object):
    # all return types must be numpy type
    def __init__(self, actions,):
        # privacy
        self.__actions = actions
        # public
        self.n_actions = len(actions)

    def get_action_by_random(self):
        return np.random.choice(self.__actions)

    def get_actions_index(self):
        return np.arange(0, self.n_actions)

    def get_action_index(self, action):
        return int(np.argwhere(self.__actions == action))

    def init_action_and_index_by_random(self):
        action = self.get_action_by_random()
        return self.get_action_index(action), action

    def get_action_value(self, idx):
        return self.__actions[idx]

    @staticmethod
    def get_mcs_reward(u_s, user_action_index, payment):
        pay = 0
        for idx in user_action_index:
            pay += payment[idx]
        return u_s - pay

    @staticmethod
    def init_action_and_index_by_zero():
        return 0, 0

    @staticmethod
    def get_payments(action, len, acc):
        payments = list()
        payments.append(0)
        for _ in range(len - 1):
            payments.append(action)
            action += acc
        return payments

    @staticmethod
    def get_mcs_utility_reciprocal(user_action_list, data_range, confidence_level, n_user):
        """
        These codes mainly relate your model.
        :param user_action_list: the actions which users can choose on their own initiative,
            which between 0 and 1. not including 0 and 1. Because 0 and 1 can be viewed
            special examples, Not participating in the experiment.
        :param data_range: a constant that relevant to the actual scenario(your model).
        :param confidence_level: a constant that relevant the model.
        :param n_user: the number of user, which can exam the right of the user_action_list.
        :return:
        """
        if n_user > len(user_action_list):
            raise RuntimeError('the user action list not enough.')
        privacy_parameter_square_reci_sum = 0
        for user_action in user_action_list:
            privacy_parameter_square_reci_sum += (1 / np.square(user_action))

        utility = 10e4
        aggregate_error = np.sqrt(2) * data_range / (n_user * np.sqrt(1 - confidence_level)) * \
                          np.sqrt(privacy_parameter_square_reci_sum)
        utility = 1.0 / aggregate_error * utility
        return aggregate_error, utility

    @staticmethod
    def get_mcs_utility_percentage(user_action_list, data_range, confidence_level, n_user):
        """
        This function which i will to do.
        :param user_action_list:
        :param data_range:
        :param confidence_level:
        :param n_user:
        :return:
        """
        if n_user > len(user_action_list):
            raise RuntimeError('the user action list not enough.')
        privacy_parameter_square_perc_sum = 0
        for user_action in user_action_list:
            privacy_parameter_square_perc_sum += (1 / np.square(user_action))
        utility = 1500
        aggregate_error = np.sqrt(2) * data_range / (n_user * np.sqrt(1 - confidence_level)) * \
                          np.sqrt(privacy_parameter_square_perc_sum)
        utility = (1.0 - aggregate_error / data_range) * utility
        return aggregate_error, utility


class UserAgent(object):
    def __init__(self, actions, costs, n_user=2):
        if len(actions) != len(costs):
            raise RuntimeError('The action and cost of User Agent is\'t the same length.')
        self.__actions = actions
        self.__n_actions = len(actions)
        self.__costs = costs
        self.__n_user = n_user

    def get_action_index(self, action):
        return int(np.argwhere(self.__actions == action))

    def get_action_by_random(self):
        return np.random.choice(self.__actions)

    def single_init_action_and_index_by_random(self):
        action = self.get_action_by_random()
        return self.get_action_index(action), action

    def get_action_value(self, idx):
        return self.__actions[idx]

    def get_cost_value(self, idx):
        return self.__costs[idx]

    def get_actions_index(self):
        return np.arange(0, self.__n_actions)

    def multi_init_action_and_index_by_random(self):
        action_value = np.zeros(self.__n_user)
        action_index = np.zeros(self.__n_user)
        for idx in range(self.__n_user):
            action = self.get_action_by_random()
            action_value[idx] = action
            action_index[idx] = self.get_action_index(action)
        return action_index, action_value

    def zero_actions_len(self):
        return np.zeros(self.__n_actions, dtype=np.int32)

    def zero_user_state_len(self):
        return ['' for _ in range(self.__n_user)]

    def zero_user_len(self, tt):
        return np.zeros(self.__n_user, dtype=tt)

    def get_actions_len(self):
        return self.__n_actions
