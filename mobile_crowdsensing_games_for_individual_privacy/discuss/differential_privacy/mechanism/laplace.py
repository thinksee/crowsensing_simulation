__author__ = 'alibaba'
__date__ = '2018/12/27'
import numpy as np


class Laplace(object):
    @staticmethod
    def sample(epsilon):
        return int(abs(np.random.laplace(0, scale=1./epsilon)))
