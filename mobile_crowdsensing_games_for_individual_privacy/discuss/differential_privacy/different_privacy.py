__author__ = 'alibaba'
__date__ = '2018/12/27'

from mobile_crowdsensing_games_for_individual_privacy.discuss.differential_privacy.mechanism.laplace import Laplace as laplace
from mobile_crowdsensing_games_for_individual_privacy.discuss.differential_privacy.mechanism.exponential import Exponential as exponential
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as ss
import os
from collections import Counter
import math
from matplotlib.backends.backend_pdf import PdfPages
np.random.seed(2019)


class DP(object):

    def __init__(self, ):
        if type(self) == DP:
            raise Exception("<DP> must not be instantiated. May lead to unexpected behaviour")

    @staticmethod
    def noise(data, epsilon, mechanism="exponential", delta=1.):

        if type(data) is dict:
            for key in data:
                if mechanism == "laplace":
                    data[key] += laplace.sample(epsilon)
                    print(laplace.sample(epsilon))
                else:
                    data[key] += exponential.sample(epsilon)
                    print(exponential.sample(epsilon))

        elif type(data) is list:
            for index, value in enumerate(data):
                if mechanism == "laplace":
                    data[index] += laplace.sample(epsilon)
                    # print(laplace.sample(epsilon))
                else:
                    data[index] += exponential.sample(epsilon)
                    # print(exponential.sample(epsilon))
        else:
            data += laplace.sample(epsilon)
            if np.random.rand(1) > 0.5:
                # data = math.ceil(data)
                data = math.floor(data)
                # data = math.floor(data)
            else:
                data = math.floor(data)
        return data


def get_tuple_list2(tuple_list):
    return [x[1] for x in tuple_list]


if __name__ == "__main__":
    x = np.arange(-10, 11)
    xU, xL = x + 0.5, x - 0.5
    prob = ss.norm.cdf(xU, scale=3) - ss.norm.cdf(xL, scale=3)  # 累计分布函数
    prob = prob / prob.sum()  # normalize the probabilities so their sum is 1
    nums = np.random.choice(x, size=20000, p=prob)
    res = list()
    for num in nums:
        res.append(DP.noise(num, np.random.uniform(1, 2, 1), mechanism="laplace"))
    raw_pri = PdfPages(os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'figure', 'raw-privacy.pdf')))
    plt.figure()
    private_data = sorted(Counter(res).items(), key=lambda item: item[0])
    raw_data = sorted(Counter(nums).items(), key=lambda item: item[0])
    plt.xlabel('Range of data')
    plt.ylabel('Number of data')
    plt.title('Raw and private data histogram comparison distribution')
    plt.bar(x - 0.2, get_tuple_list2(raw_data), alpha=0.9, width=0.4, facecolor='b', edgecolor='white', label='Raw Data', lw=1)
    plt.bar(x + 0.2, get_tuple_list2(private_data)[: len(x)], alpha=0.9, width=0.4, facecolor='r', edgecolor='white', label='Privacy Data', lw=1)
    plt.legend(loc="best")  # label的位置在左上，没有这句会找不到label去哪了
    raw_pri.savefig()
    raw_pri.close()

