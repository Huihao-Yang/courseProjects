import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
plt.rcParams["axes.unicode_minus"] = False  # 该语句解决图像中的“-”负号的乱码问题


def sigmoid(x):
    y = 1 / (1 + np.exp(-x))
    return y * (1 - y)


def tanh(x):
    y = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    return 1 - y * y


if __name__ == '__main__':
    # x = np.arange(-8, 8, 0.2)
    # dy1 = sigmoid(x)
    # dy2 = tanh(x)
    # fig, ax = plt.subplots(1)
    # ax.plot(x, dy2, label='tanh的一阶导数')
    # ax.plot([0, 0], [-0.1, 1.1], ls='--', c='black')
    x = np.arange(0, 7)
    y = [0,2, -3, -2, -6, -3, -5]
    plt.plot(x, y,c='black')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()
