__author__ = 'alibaba'
__date__ = '2018/12/27'
import random
import numpy as np


class Exponential:
    @staticmethod
    def sample(epsilon):
        # print(int(-(np.log(-random.uniform(0., 1.0) + 1.))/epsilon/2.))
        return int(-(np.log(-random.uniform(0., 1.0) + 1.))/epsilon/2.)

    @staticmethod
    def categorical_sample(data, epsilon):
        if not type(data) == list:
            raise TypeError("Return Type of scoring function must be a list")
        index = Exponential.sample(epsilon)

        if index > len(data) - 1:
            index = len(data) - 1

        return data[index]
