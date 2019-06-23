__author__ = 'think see'
__date__ = '2018/11/19'
import numpy as np


class MCSAgent(object):

    def __init__(self, actions, contribution_factor):
        # value, not index
        self.__actions = actions
        self.n_actions = self.get_action_length()
        self.__beta = contribution_factor

    def get_action_index(self, action):
        return int(np.argwhere(self.__actions == action))

    def get_action_by_random(self):
        return np.random.choice(self.__actions)

    def init_action_and_index_by_random(self):
        action = self.get_action_by_random()
        return self.get_action_index(action), action

    def get_action_value(self, idx):
        return self.__actions[idx]

    def get_actions_index(self):
        return np.arange(0, self.n_actions)

    def get_action_length(self):
        return len(self.__actions)

    def get_contribution_value(self, idx):
        return self.__beta[idx]

    def get_utility_value(self, payments, val1, idx1, val2, idx2):
        return self.get_contribution_value(idx1) * val1 + self.get_contribution_value(idx2) * val2 - payments[idx1] - payments[idx2]

    @staticmethod
    def get_payments(action, length, acc_value):
        payments = np.zeros(length)
        payments.put(0, 0)
        for idx in range(length - 1):
            payments.put(idx, action)
            action += acc_value
        return payments


class UserAgent(object):

    def __init__(self, actions, snr_set, snr_prob_set,
                 gamma, max_speed, speed_set, speed_prob_set, cost_set):
        """
        init user agent
        :param actions: sense effort
        :param snr_set: signal noise rate
        :param snr_prob_set: the same length as {snr_set}, you have snr_prob_set[i] to select snr_set[i]
        :param gamma: the param relate to user state
        :param max_speed: the max speed. relate to user state
        :param speed_set: the vehicular speed
        :param speed_prob_set: the same length as {speed_set}, you have speed_prob_set[i] to select speed_set[i]
        :param cost_set: the set of the user's cost
        """
        if len(actions) != len(cost_set) or len(snr_prob_set) != len(snr_set) or len(speed_prob_set) != len(speed_set):
            raise Exception('The actions and cost_set is\'t same length.')
        self.__actions = actions
        self.__snr_set = snr_set
        self.__snr_prob_set = snr_prob_set
        self.__gamma = gamma
        self.__max_speed = max_speed
        self.__speed_set = speed_set
        self.__speed_prob_set = speed_prob_set
        self.__cost_set = cost_set
        self.n_actions = self.get_action_length()

    def get_action_index(self, action):
        return int(np.argwhere(self.__actions == action))

    def get_actions_index(self):
        return np.arange(0, self.n_actions)

    def get_snr_index(self, snr):
        return int(np.argwhere(self.__snr_set == snr))

    def get_speed_index(self, speed):
        return int(np.argwhere(self.__speed_set == speed))

    def get_speed_value(self):
        return np.random.choice(self.__speed_set, 1, self.__speed_prob_set)

    def init_snr_and_index_by_random(self):
        snr = np.random.choice(self.__snr_set, 1, self.__snr_prob_set)
        return self.get_snr_index(snr), snr

    def init_speed_and_index_by_random(self):
        speed = np.random.choice(self.__speed_set, 1, self.__speed_prob_set)
        return self.get_speed_index(speed), speed

    def get_cost_value(self, idx):
        return self.__cost_set[idx]

    def get_snr_and_prob(self, pre_speed, pre_snr):
        """
        dynamic tuning the vehicular speed and vehicular speed prob

        :param pre_speed:
        :param pre_snr:
        :return: next_snr,next_snr_prob
        """
        p1 = float(1 - 1. * self.__gamma * pre_speed / self.__max_speed)  # m=n
        p2 = float(self.__gamma * pre_speed / (2 * self.__max_speed))  # []
        p3 = float(self.__gamma * pre_speed / self.__max_speed)  # [0,1] & [N,N-1]
        snr_index = self.get_snr_index(pre_snr)
        snr_length = self.get_snr_length()
        cur_snr_prob = list()
        if snr_length > 2:  # the size of the signal noise rate more than 3
            cur_snr_prob.append(p1)
            cur_snr_prob.append(p2)
            cur_snr_prob.append(p3)
        elif snr_length == 2:  # the size of the snr is 2
            cur_snr_prob.append(p1)
            cur_snr_prob.append(p3)
        else:  # the size of the snr is less than 2
            raise RuntimeError('the signal noise rate is too small.')
        snr_prob = list()
        snr = list()
        if snr_length >= 2:  #
            if snr_index == 0 or snr_index == self.get_snr_length() - 1:
                snr_prob.append(p1)
                snr_prob.append(p3)
                snr.append(pre_snr)
                if snr_index == 0:
                    snr.append(self.get_snr_value(snr_index + 1))
                else:
                    snr.append(self.get_snr_value(snr_index - 1))
            else:
                snr_prob.append(p2)
                snr_prob.append(p1)
                snr_prob.append(p2)
                snr.append(self.get_snr_value(snr_index - 1))
                snr.append(pre_snr)
                snr.append(self.get_snr_value(snr_index + 1))
        else:
            raise RuntimeError('the size of prob set is too small.')

        cur_snr = np.random.choice(snr, 1, snr_prob)

        return cur_snr, cur_snr_prob

    def select_prob_by_snr(self, snr_prob, snr_idx1, snr_idx2):
        if abs(snr_idx1 - snr_idx2) == 1:
            if (snr_idx1 == self.get_snr_length() - 1 or snr_idx1 == 0 or snr_idx2 == self.get_snr_length() - 1or snr_idx2 == 0) \
                    and self.get_snr_length() > 2:
                return snr_prob[2]
            else:
                return snr_prob[1]
        elif snr_idx1 == snr_idx2:
            return snr_prob[0]
        else:
            return .0

    def get_action_by_random(self):
        return np.random.choice(self.__actions)

    def init_action_and_index_by_random(self):
        action = self.get_action_by_random()
        return self.get_action_index(action), action

    def get_action_value(self, idx):
        return self.__actions[idx]

    def get_snr_value(self, idx):
        return self.__snr_set[idx]

    def get_action_length(self):
        return len(self.__actions)

    def get_snr_length(self):
        return len(self.__snr_set)

