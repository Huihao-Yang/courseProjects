import glob
import math

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from math import sqrt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
plt.rcParams["axes.unicode_minus"] = False  # 该语句解决图像中的“-”负号的乱码问题

dir_path = '.././dataset/params_data'
Battery_list = glob.glob(dir_path + '/*.csv')
datas = pd.DataFrame()
for i in range(3):
    datas = pd.concat([datas.copy(), pd.read_csv(Battery_list[i], sep=',')])  # 合并所有数据
test_data = pd.read_csv(Battery_list[-1], sep=',')


def evaluation(y_test, y_predict):
    rmse = sqrt(mean_squared_error(y_test, y_predict))
    mae = mean_squared_error(y_test, y_predict)
    return mae, rmse


def get_train_test():
    train_x = datas.iloc[:, 0].values
    train_y = datas.iloc[:, 1].values
    test_x = test_data.iloc[:, 0].values
    test_y = test_data.iloc[:, 1].values

    return train_x.reshape(-1, 1), train_y, test_x.reshape(-1, 1), test_y


x_train, y_train, x_test, y_test = get_train_test()


def try_different_method(clf, name):
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.plot(x_test, y_test, color='red', label='实际容量')
    ax.plot(x_test, pred, color='blue', label='预测值')

    plt.plot([-1, 1000], [0.7 * 1.1, 0.7 * 1.1], c='black', lw=1, ls='--', label='失效阈值')  # 临界点直线
    ax.set_xlabel('循环次数', fontsize=14)
    ax.set_ylabel('电池容量/Ah', fontsize=14)
    plt.legend()
    plt.savefig('./' + name + '.png')
    plt.show()

    mae, rmse = evaluation(y_test, pred)
    print(
        'RMSE:{:<6.4f} | MAE:{:<6.4f}'.format(rmse, mae))


if __name__ == '__main__':
    # 分割测试集和训练集,并且是随机抽样

    # # 用随机森林分类器进行训练
    # clf = RandomForestClassifier()
    # clf.fit(train_x, train_y)
    # pred = clf.predict(test_x)
    # x = np.arange(1, len(pred) + 1)
    # plt.plot(x, pred)
    # plt.show()

    # svr = SVR(kernel='linear')
    # svr.fit(train_x, train_y)
    # pred = svr.predict(test_x)

    clf = SVR(kernel='rbf')
    try_different_method(clf,'SVR')

    # from sklearn.tree import DecisionTreeRegressor
    #
    # tree = DecisionTreeRegressor()
    # try_different_method(tree,'dt')

    # from sklearn import ensemble
    #
    # ada = ensemble.AdaBoostRegressor(n_estimators=50)
    # try_different_method(ada, 'ada')
