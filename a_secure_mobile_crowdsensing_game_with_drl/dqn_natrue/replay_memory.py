__author__ = 'alibaba'
__date__ = '2019/1/2'

import os
import random
import numpy as np

from utils import save_npy, load_npy
"""
 Attention机制：神经网络里的Attention机制是松散地基于人类视觉注意机制。
    所有模型归根结底都是按照”高分辨率“聚焦在图片的某个特定区域并以”低分辨率“
    感知图像的周边区域的模式，然后不断地调整聚焦点。
"""


class ReplayMemory:
    # 参数初始化
    """
    batch size: 选择n个样本组成一个batch,然后将batch丢进神经网络，得到输出结果。
    memory size:
        experience replay: 是因为样本是从游戏中的连续帧获得的，这与简单的reinforcement learning问题相比，样本的关联性大了很多，
          若没有experience replay，算法在连续一段时间内基本朝着同一个方向做gradient descent, 同样的步长下，直接计算gradient就有可能
          不收敛。因此experience replay是从一个memory pool 中随机选取了一些experience, 然后再求梯度，从而避免了这个问题。
    history size: 表示用多对数据表示组合成当前状态
    """
    def __init__(self, config, model_dir):
        self.model_dir = model_dir

        self.cnn_format = config.cnn_format
        self.memory_size = config.memory_size
        self.actions = np.empty(self.memory_size, dtype=np.uint8)
        self.rewards = np.empty(self.memory_size, dtype=np.integer)
        self.screens = np.empty((self.memory_size, config.screen_height, config.screen_width), dtype=np.float16)
        self.terminals = np.empty(self.memory_size, dtype=np.bool)
        self.history_length = config.history_length
        self.dims = (config.screen_height, config.screen_width)
        self.batch_size = config.batch_size
        self.count = 0
        self.current = 0

        # pre-allocate prestates and poststates for minibatch
        self.prestates = np.empty((self.batch_size, self.history_length) + self.dims, dtype=np.float16)   # 变成4维的数据
        self.poststates = np.empty((self.batch_size, self.history_length) + self.dims, dtype=np.float16)  # 变成4维的数据

    # 向经验回放池中存放screen, reward, action, terminal等数据
    def add(self, screen, reward, action, terminal):
        assert screen.shape == self.dims
        # NB! screen is post-state, after action and reward
        self.actions[self.current] = action
        self.rewards[self.current] = reward
        # a[:,:,None]和a[...,None]是一样的，表示数据切片
        self.screens[self.current, ...] = screen
        self.terminals[self.current] = terminal
        self.count = max(self.count, self.current + 1)         # count最大值不会超过self.memory_size
        self.current = (self.current + 1) % self.memory_size   # 记录在循环数组中的history size的索引

    # 根据索引index获取状态
    def getState(self, index):
        assert self.count > 0, "replay memory is empy, use at least --random_steps 1"
        # normalize index to expected range, allows negative indexes
        index = index % self.count
        # if is not in the beginning of matrix
        if index >= self.history_length - 1:
            # use faster slicing 从当前元素向后计数history_length大小
            return self.screens[(index - (self.history_length - 1)):(index + 1), ...]
        else:
            # otherwise normalize indexes and use slower list based access
            # reversed: 返回一个反转的迭代器
            # seqString = 'Runoob'
            # print(list(reversed(seqString)))  ---> ['b', 'o', 'o', 'n', 'u', 'R']
            indexes = [(index - i) % self.count for i in reversed(range(self.history_length))]
            return self.screens[indexes, ...]

    def sample(self):
        # memory must include poststate, prestate and history
        assert self.count > self.history_length
        # sample random indexes
        indexes = []
        while len(indexes) < self.batch_size:
            # find random index
            while True:
                # sample one index (ignore states wraping over
                index = random.randint(self.history_length, self.count - 1)
                # if wraps over current pointer, then get new one
                # 整个memory_size被填满，然后当前current正在覆盖之前的内容
                if index >= self.current and index - self.history_length < self.current:
                    continue
                # if wraps over episode end, then get new one
                # NB! poststate (last screen) can be terminal state!
                # np.array.any(): 是或操作，任意一个元素作为True,输出True
                # np.array.all(): 是与操作，所有元素作为True,输出True
                """
                    arr1 = np.array([0,1,2,3])
                    print(arr1.any()) ---> True
                    print(arr1.all()) ---> False
                """
                if self.terminals[(index - self.history_length):index].any():
                    continue
                # otherwise use this index
                break

            # NB! having index first is fastest in C-order matrices
            self.prestates[len(indexes), ...] = self.getState(index - 1)
            self.poststates[len(indexes), ...] = self.getState(index)
            indexes.append(index)

        actions = self.actions[indexes]
        rewards = self.rewards[indexes]
        terminals = self.terminals[indexes]

        # data_format设置为
        #       'NHWC'时，排列顺序为[batch, height, width, channels]
        #       'NCHW'时，排列顺序为[batch, channels, height, width]
        if self.cnn_format == 'NHWC':
            """
                arr = np.arange(12).reshape(2,2,3)
                --> array([[[0, 1, 2],
                            [3, 4, 5]],
                           [[6, 7, 8],
                            [9, 10, 11]]])
                2 --- 0
                2 --- 1
                3 --- 2
                
            """
            return np.transpose(self.prestates, (0, 2, 3, 1)), actions, \
                   rewards, np.transpose(self.poststates, (0, 2, 3, 1)), terminals
        else:
            return self.prestates, actions, rewards, self.poststates, terminals

    """
    zip: 用于将可迭代的对象作为参数，将对象对应的元素打包成一个个元组，然后返回由这些元组组成的列表
        a = [1,2,3]
        b = [4,5,6]
        c = [4,5,6,7,8]
        zip(a, b) = [(1,4),(2,5),(3,6)]
        zip(a, c) = [(1,4),(2,5),(3,6)]
        zip(*zipped)与zip相反,*zipped可理解为解压
    enumerate: 用于将一个可遍历的数据对象组合成一个索引序列，同时列出数据和数据下标。
        seq = ['one', 'two', 'three']
        for i, element in enumerate(seq):
            print i, element
    """
    def save(self):

        for idx, (name, array) in enumerate(
                zip(['actions', 'rewards', 'screens', 'terminals', 'prestates', 'poststates'],
                    [self.actions, self.rewards, self.screens, self.terminals, self.prestates, self.poststates])):
            save_npy(array, os.path.join(self.model_dir, name))

    def load(self):
        for idx, (name, array) in enumerate(
                zip(['actions', 'rewards', 'screens', 'terminals', 'prestates', 'poststates'],
                    [self.actions, self.rewards, self.screens, self.terminals, self.prestates, self.poststates])):
            array = load_npy(os.path.join(self.model_dir, name))
