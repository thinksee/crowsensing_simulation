__author__ = 'alibaba'
__date__ = '2018/11/19'
import numpy as np
from collections import namedtuple
import torch
# history = np.zeros(
# #             [3, 2, 5], dtype=np.float32)
# # print(history[:-1])
# # print('--------------------')
# # print(history[1:])
# # print('--------------------')
# # print(history[-1])

# a = np.empty([2, 4] + [3, 5])
# print(a)
# print((-2) % 100)
# print(np.random.randint(1, 2))
# print(np.random.randint(2, 4))

# Transition_chain = namedtuple('Transition_chain',
#                               ('net_input', 'next_net_input', 'action', 'reward'))
#
# list1 = [(11, 12, 13, 14), (21, 22, 23, 24), (31, 32, 33, 34), (41, 42, 43, 44), (51, 52, 53, 54)]
# print(list(zip(*list1)))
# tuple1 = Transition_chain(*zip(*list1))
# # 打印zip函数的返回类型
# print("zip()函数的返回类型：\n", type(tuple1))
# # 将zip对象转化为列表
# print("zip对象转化为列表：\n", list(tuple1))

b = torch.Tensor([[1, 2, 3], [4, 5, 6]])
print(b)
index_1 = torch.LongTensor([[0, 1], [2, 0]])
index_2 = torch.LongTensor([[0, 1, 1], [0, 0, 0]])
print(torch.gather(b, dim=1, index=index_1))
print(torch.gather(b, dim=0, index=index_2))
