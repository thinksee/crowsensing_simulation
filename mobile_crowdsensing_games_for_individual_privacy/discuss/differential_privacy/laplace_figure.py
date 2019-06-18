__author__ = 'alibaba'
__date__ = '2018/12/27'

import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.backends.backend_pdf import PdfPages


def laplace(data, epsilon, mean=0):
    result = (1 / (2 * epsilon)) * np.e ** (-1 * (np.abs(data - mean)/epsilon))
    return result


if __name__ == '__main__':
    mean = 0
    epsilon1 = 1
    epsilon2 = 2
    epsilon3 = 1
    mean3 = 4
    data = np.linspace(-10, 11, 20000)

    y1 = [laplace(_data, epsilon1, mean) for _data in data]
    y2 = [laplace(_data, epsilon2, mean) for _data in data]
    y3 = [laplace(_data, epsilon3, mean3) for _data in data]
    plt.figure(0)
    # 设置坐标轴宽度
    plt.margins(1)
    plt.xlim((-10, 10))
    plt.ylim((0.0, 0.02 + max(laplace(0, epsilon1, mean), laplace(0, epsilon2, mean))))
    # 设置坐标轴刻度
    x_ticks = np.arange(-10, 10, 2)
    y_ticks = np.arange(0, 0.02 + max(laplace(0, epsilon1, mean), laplace(0, epsilon2, mean)), 0.05)
    plt.xticks(x_ticks)
    plt.yticks(y_ticks)
    # 绘制laplace曲线
    plt.plot(data, y1, color='g', alpha=0.7, label='$laplace(\epsilon_{i}=1,)$')
    plt.plot(data, y2, color='b', alpha=0.7, label='$PPL\ \epsilon_{i}^{\prime}=2$')
    left = -4
    right = 4
    mid = 2
    y_max = max(laplace(left, epsilon1, mean), laplace(right, epsilon2, mean))
    y_min = min(laplace(left, epsilon1, mean), laplace(right, epsilon2, mean))
    plt.plot([left, left], [0, y_min], alpha=0.7, color='r', linestyle='--', linewidth=1.0)
    plt.scatter([left, ], [0, ], s=10, color='b', marker='x')
    plt.text(left + 0.05, 0, s=r'$d_{i}$')
    plt.plot([right, right], [0, y_max], alpha=0.7, color='r', linestyle='--', linewidth=1.0)
    plt.scatter([right, ], [0, ], s=10, color='g', marker='x')
    plt.text(right + 0.05, 0, s=r'$d_{i}^{\prime}$')

    y_max = max(laplace(mid, epsilon1, mean), laplace(mid, epsilon2, mean))
    plt.plot([mid, mid], [0, y_max], alpha=0.7, color='y', linestyle='--', linewidth=1.0)
    plt.plot([-10, mid], [laplace(mid, epsilon1, mean), laplace(mid, epsilon1, mean)], alpha=0.7, color='#000000',
             linestyle='--', linewidth=1.0)
    plt.plot([-10, mid], [laplace(mid, epsilon2, mean), laplace(mid, epsilon2, mean)], alpha=0.7, color='#000000',
             linestyle='--', linewidth=1.0)
    plt.text(mid + 0.1, 0, s='$d^{obs}$')
    plt.text(mid + 0.1, laplace(mid, epsilon1, mean) - 0.001, s='$p_{i}$')

    plt.text(mid + 0.15, laplace(mid, epsilon2, mean), s='$p_{i}^{\prime}$')

    plt.scatter([mid, ], [laplace(mid, epsilon2, mean), ], s=10, color='#000000', marker='x')
    plt.scatter([mid, ], [laplace(mid, epsilon1, mean), ], s=10, color='#000000', marker='x')

    plt.ylabel('Probability')
    plt.xlabel('Data')
    plt.title("Illustration of Private protection model")
    plt.legend(loc='best')

    raw_pri = PdfPages(
        os.path.abspath(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'figure', 'Illustration-of-different-privacy.pdf')))
    raw_pri.savefig()
    raw_pri.close()
