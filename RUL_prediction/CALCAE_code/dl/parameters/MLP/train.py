import numpy as np
import random
import math
import os
import pandas as pd
import glob
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from math import sqrt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

dir_path = '../../../dataset/params_data'
Battery_list = glob.glob(dir_path + '/*.csv')
datas = pd.DataFrame()
for i in range(len(Battery_list) - 1):
    datas = pd.concat([datas.copy(), pd.read_csv(Battery_list[i], sep=',')])  # 合并所有数据
test_data = pd.read_csv(Battery_list[-1], sep=',')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
EPOCH = 7000
LR = 0.01  # learning rate
feature_size = 5
hidden_size = [32, 16]
weight_decay = 0.0
Rated_Capacity = 1.1


def setup_seed(seed):
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现。
    torch.manual_seed(seed)  # 为CPU设置随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU，为所有GPU设置随机种子
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


# 随机选取20%数据为测试集，其他所有数据全部拿来训练
def get_train_test():
    train_x = datas.iloc[:, 2:].values
    train_y = datas.iloc[:, 1].values
    test_x = test_data.iloc[:, 2:].values
    test_y = test_data.iloc[:, 1].values
    return train_x, train_y, test_x, test_y


def relative_error(y_test, y_predict, threshold):
    true_re, pred_re = len(y_test), 0
    for i in range(len(y_test) - 1):
        if y_test[i] <= threshold >= y_test[i + 1]:
            true_re = i - 1
            break
    for i in range(len(y_predict) - 1):
        if y_predict[i] <= threshold:
            pred_re = i - 1
            break
    return abs(true_re - pred_re) / true_re if abs(true_re - pred_re) / true_re <= 1 else 1


def RE(y_test, y_predict):
    error = 0
    n = len(y_predict)
    for i in range(n):
        if y_test[i] == 0:
            error += 1
        else:
            error += abs(y_test[i] - y_predict[i][0]) / y_test[i]
    return error / n


def evaluation(y_test, y_predict):
    mae = mean_absolute_error(y_test, y_predict)
    mse = mean_squared_error(y_test, y_predict)
    rmse = sqrt(mean_squared_error(y_test, y_predict))

    return mae, rmse


class Net(nn.Module):
    def __init__(self, feature_size=8, hidden_size=[16, 8]):
        super(Net, self).__init__()
        self.feature_size, self.hidden_size = feature_size, hidden_size
        self.layer0 = nn.Linear(self.feature_size, self.hidden_size[0])
        self.layers = [nn.Sequential(nn.Linear(self.hidden_size[i], self.hidden_size[i + 1]), nn.ReLU())
                       for i in range(len(self.hidden_size) - 1)]
        self.linear = nn.Linear(self.hidden_size[-1], 1)

    def forward(self, x):
        out = self.layer0(x)
        for layer in self.layers:
            out = layer(out)
        out = self.linear(out)
        return out


def tain(LR=0.001, feature_size=feature_size, hidden_size=[16, 8], weight_decay=0.0, EPOCH=1000, seed=0):
    score_list, result_list = [], []
    train_x, train_y, test_x, test_y = get_train_test()
    train_size = len(train_x)
    print('sample size: {}'.format(train_size))

    setup_seed(seed)
    model = Net(feature_size=feature_size, hidden_size=hidden_size)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    loss_list = [0]
    mae, rmse, re = 1, 1, 1
    score_, score = 1, 1
    X = np.reshape(train_x, (-1, feature_size)).astype(
        np.float32)
    y = np.reshape(train_y, (-1, 1)).astype(np.float32)  # shape 为 (batch_size, 1)
    X, y = torch.from_numpy(X).to(device), torch.from_numpy(y).to(device)

    for epoch in range(EPOCH):
        output = model(X)
        loss = criterion(output, y)
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients

        if (epoch + 1) % 100 == 0:
            loss_list.append(loss)
            pred = output.detach().numpy()
            mae, rmse = evaluation(y_test=train_y, y_predict=pred)
            # re = relative_error(y_test=train_y, y_predict=pred, threshold=Rated_Capacity * 0.7)\
            re = 0
            print(
                'epoch:{:<2d} | loss:{:<6.4f} | MAE:{:<6.4f} | RMSE:{:<6.4f} | RE:{:<6.4f}'.format(epoch, loss, mae,
                                                                                                   rmse, re))
        score = [re, mae, rmse]
        if (loss < 1e-3) and (score_[0] < score[0]):
            break
        score_ = score.copy()

    score_list.append(score_)
    test_X = np.reshape(test_x, (-1, feature_size)).astype(
        np.float32)
    test_X = torch.from_numpy(test_X).to(device)
    pred_y = model(test_X).detach().numpy()
    MAE, RMSE = evaluation(test_y, pred_y)
    result_list.append(pred_y)

    return score_list, result_list, MAE, RMSE


if __name__ == '__main__':
    mode = 'LSTM'  # RNN, LSTM, GRU

    SCORE = []

    for seed in range(10):
        print('seed: ', seed)
        score_list, _ = tain(LR=LR, feature_size=feature_size, hidden_size=hidden_size,
                             weight_decay=weight_decay, EPOCH=EPOCH, seed=seed)
        print('------------------------------------------------------------------')
        for s in score_list:
            SCORE.append(s)

    mlist = ['re', 'mae', 'rmse']
    for i in range(3):
        s = [line[i] for line in SCORE]
        print(mlist[i] + ' mean: {:<6.4f}'.format(np.mean(np.array(s))))
    print('------------------------------------------------------------------')
    print('------------------------------------------------------------------')
