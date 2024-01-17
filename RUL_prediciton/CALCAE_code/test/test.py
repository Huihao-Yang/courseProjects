import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
plt.rcParams["axes.unicode_minus"] = False  # 该语句解决图像中的“-”负号的乱码问题

import pandas as pd

if __name__ == '__main__':
    # df = pd.read_excel('迭代数据.xlsx')
    # len = df.shape[0]
    # x = np.arange(1, len + 1)
    # fig, ax = plt.subplots(1)
    # ax.plot(x, df.iloc[:, 0], label='P')
    # ax.plot(x, df.iloc[:, 1], label='recall')
    # ax.plot(x, df.iloc[:, 2], label='F')
    # ax.plot([1, len // 10 * 10 + 10], [1, 1], ls='--',color='black')
    # ax.set_xlabel('迭代次数', fontsize=14)
    # plt.legend(fontsize=14)
    # plt.show()

    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.svm import SVR

    df = pd.read_excel('迭代数据.xlsx')
    len = df.shape[0]
    x = np.arange(1, len + 1).reshape(-1, 1)

    fig, ax = plt.subplots(1, figsize=(12, 8))
    labels = ['P', 'R', 'F']
    for i in range(3):
        svr = SVR(kernel='rbf')
        svr.fit(x, df.iloc[:, i])
        pred = svr.predict(x)
        ax.plot(x, pred, label=labels[i])
    ax.set_xlabel('迭代次数', fontsize=14)
    plt.legend(fontsize=14)
    plt.show()
