__author__ = 'alibaba'
__date__ = '2019/1/2'
import numpy as np


class History:
    def __init__(self, config):
        self.cnn_format = config.cnn_format

        batch_size, history_length, screen_height, screen_width = \
            config.batch_size, config.history_length, config.screen_height, config.screen_width

        self.history = np.zeros(
            [history_length, screen_height, screen_width], dtype=np.float32)

    def add(self, screen):
        self.history[:-1] = self.history[1:]
        self.history[-1] = screen

    def reset(self):
        self.history *= 0

    def get(self):
        if self.cnn_format == 'NHWC':
            #    [history_length, screen_height, screen_width]
            # => [screen_height, screen_width, history_length]
            return np.transpose(self.history, (1, 2, 0))  # 矩阵转置，按照维度优先级进行操作
        else:
            return self.history
