__author__ = 'alibaba'
__date__ = '2018/11/19'
import numpy as np


class MCSAgent(object):
    """
    没有索引值一说，均采用值访问的形式
    """
    def __init__(self, actions, contribution_factor):
        """
        初始化平台
        :param actions: 平台可选择的支付actions，不表示索引值，而表示真实值
        """
        self.actions = actions
        self.beta = contribution_factor

    def get_payment_layer(self, action, length, acc_value):
        """
        根据
        :param: action: 选择出来的基础支付，即平台选择的action，不表示索引值，表示真实值
        :return: set,一个关于effort layer的payment layer
        """
        payment = list()
        payment.append(0)
        for index in range(length):
            payment.append(action + acc_value)
        return payment

    def get_action_index(self, action):
        """
        根据实际的值获取相应的索引值，主要是为了满足对Q-table的操作
        :param action:
        :return:
        """
        return self.actions.index(action)

    def select_action(self):
        """
        随机返回一个初始值
        :return:
        """
        return np.random.choice(self.actions)

    def init_action(self):
        return self.actions[1]


    def get_action_by_index(self, index):
        """
        按照索引值返回相应的action
        :param index:
        :return:
        """
        return self.actions[index]

    def get_action_length(self):
        """
        返回支付动作长度
        :return:
        """
        return len(self.actions)


class UserAgent(object):

    def __init__(self, actions, snr_set, snr_prob_set,
                 gamma, max_speed, speed_set, speed_prob_set, cost_set):
        """
        初始化用户Agent
        :param actions: 用户可选的的action集合，表示其effort layer
        :param snr_set: 信噪比，表示可选的信道集合
        :param snr_prob_set: 与set_set对应的概率值，即snr_set[0]有snr_prob_set[0]的概率被选中
        :param gamma: 与用户状态相关的系数
        :param max_speed: 最大的速度，与用户状态相关的系数
        :param speed_set: 可选的用户系数
        :param speed_prob_set: 选择的概率，和snr_set\snr_prob_set相似
        :param cost_set: 用户所花费的代价值集合
        """
        self.actions = actions
        self.snr_set = snr_set
        self.snr_prob_set = snr_prob_set
        self.gamma = gamma
        self.max_speed = max_speed
        self.speed_set = speed_set
        self.speed_prob_set = speed_prob_set
        self.cost_set = cost_set

    def get_action_index(self, action):
        """
        获得动作的索引值
        :param action:
        :return:
        """
        return self.actions.index(action)

    def get_snr_index(self, snr):
        """
        获得信噪比的索引值
        :param snr:
        :return:
        """
        return self.snr_set.index(snr)

    def get_speed_index(self, speed):
        """
        获得速度的索引值
        :param speed:
        :return:
        """
        return self.speed_set.index(speed)

    def select_snr(self):
        """
        选择信噪比
        :return:
        """
        return np.random.choice(list(self.snr_set), 1, list(self.snr_prob_set))

    def init_snr(self):
        return self.snr_set[0]

    def select_speed(self):
        """
        选择速度值
        :return:
        """
        return np.random.choice(list(self.speed_set), 1, list(self.speed_prob_set))

    def select_cost_by_index(self, index):
        """
        选择用户花费的代价
        :return:
        """
        return self.cost_set[index]

    def get_snr_prob(self, cur_speed, cur_snr):
        """
        考虑PDS的使用，速度和速度概率会动态变化
        :param cur_speed: 当前的速度值
        :param cur_snr: 当前的信噪比
        :return: 下一信噪比next_snr,下一状态的概率转化next_snr_prob
        """
        p1 = float(1 - 1. * self.gamma * cur_speed / self.max_speed)  # m=n
        p2 = float(self.gamma * cur_speed / (2 * self.max_speed))  # []
        p3 = float(self.gamma * cur_speed / self.max_speed)  # [0,1] & [N,N-1]
        snr_index = self.get_snr_index(cur_snr)
        snr_length = len(self.snr_set)
        next_snr_prob = list()
        if snr_length > 2:  # 信噪比大小为3个以上
            next_snr_prob.append(p1)
            next_snr_prob.append(p2)
            next_snr_prob.append(p3)
        elif snr_length == 2:  # 信噪比大小为2个
            next_snr_prob.append(p1)
            next_snr_prob.append(p3)
        else:  # 若信噪比大小小于2个
            print("可选信噪比集合长度过小\n")
        ch_snr_prob = list()
        ch_snr = list()
        if snr_length >= 2:  #
            if snr_index == 0 or snr_index == len(self.snr_set) - 1:
                ch_snr_prob.append(p1)
                ch_snr_prob.append(p3)
                ch_snr.append(cur_snr)
                if snr_index == 0:
                    ch_snr.append(self.snr_set[snr_index + 1])
                else:
                    ch_snr.append(self.snr_set[snr_index - 1])
            else:
                ch_snr_prob.append(p2)
                ch_snr_prob.append(p1)
                ch_snr_prob.append(p2)
                ch_snr.append(self.snr_set[snr_index - 1])
                ch_snr.append(cur_snr)
                ch_snr.append(self.snr_set[snr_index + 1])
        else:
            print("可选择的概率集合过小\n")
        # 对ch_snr_prob 进行归一化操作使之和为1
        next_snr = np.random.choice(ch_snr, 1, ch_snr_prob)

        return next_snr, next_snr_prob

    def select_prob_by_snr(self, snr_prob, index_snr1, index_snr2):
        """
        按照不同的信噪比转换来获取相应的概率值
        :param snr_prob: 转换概率
        :param index_snr1: 前一个状态
        :param index_snr2: 后一个状态
        :return: snr1和snr2之间转换的概率值
        """
        if abs(index_snr1 - index_snr2) == 1:
            if (index_snr1 == len(self.snr_set) - 1 or \
                    index_snr1 == 0 or \
                    index_snr2 == len(self.snr_set) - 1 or \
                    index_snr2 == 0) and \
                    len(self.snr_set) > 2:
                return snr_prob[2]
            else:
                return snr_prob[1]
        elif index_snr2 == index_snr1:
            return snr_prob[0]
        else:
            return .0

    def select_action(self):
        """
        随机选择当前的动作值
        :return:
        """
        return np.random.choice(self.actions)

    def init_action(self):
        return self.actions[1]

    def get_action_by_index(self, index):
        """
        根据索引值获取相应的动作，即sense effort
        :return:
        """
        return self.actions[index]

    def get_snr_by_index(self, index):
        """
        根据相应的索引值获取相应的snr
        :param index:
        :return:
        """
        return self.snr_set[index]

    def get_action_length(self):
        """
        返回action即effort的长度
        :return:
        """
        return len(self.actions)

    def get_snr_length(self):
        """
        返回信噪比长度
        :return:
        """
        return len(self.snr_set)

