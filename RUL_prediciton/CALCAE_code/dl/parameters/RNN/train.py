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
from sklearn.metrics import accuracy_score

dir_path = '../../../dataset/params_data'
Battery_list = glob.glob(dir_path + '/*.csv')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
Rated_Capacity = 1.1

feature_size = 5
window_size = 128
EPOCH = 1000
lr = 0.001  # learning rate  0.01 epoch 10
hidden_dim = 256
num_layers = 2
weight_decay = 0.0


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


def get_train_test(index):
    datas = pd.DataFrame()
    for i in range(len(Battery_list)):
        if index == i:
            continue
        datas = pd.concat([datas.copy(), pd.read_csv(Battery_list[i], sep=',')])  # 合并所有数据
    test_data = pd.read_csv(Battery_list[index], sep=',')

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


def evaluation(y_test, y_predict):
    mae = mean_absolute_error(y_test, y_predict)
    mse = mean_squared_error(y_test, y_predict)
    rmse = sqrt(mean_squared_error(y_test, y_predict))
    return mae, rmse


class Net(nn.Module):
    def __init__(self, input_size, hidden_dim, num_layers, n_class=1, mode='LSTM'):
        super(Net, self).__init__()
        self.hidden_dim = hidden_dim
        self.cell = nn.LSTM(input_size=input_size, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        if mode == 'GRU':
            self.cell = nn.GRU(input_size=input_size, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        elif mode == 'RNN':
            self.cell = nn.RNN(input_size=input_size, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, n_class)

    def forward(self, x):  # x shape: (batch_size, seq_len, input_size)
        out, _ = self.cell(x)
        out = out.reshape(-1, self.hidden_dim)
        out = self.linear(out)  # out shape: (batch_size, n_class=1)
        return out


def tain(lr=0.001, feature_size=feature_size, hidden_dim=128, num_layers=2, weight_decay=0.0, mode='LSTM', EPOCH=1000,
         seed=0, index=3, start=0):
    score_list, result_list = [], []
    train_x, train_y, test_x, test_y = get_train_test(index)
    train_size = len(train_x)
    print('sample size: {}'.format(train_size))

    setup_seed(seed)
    model = Net(input_size=feature_size, hidden_dim=hidden_dim, num_layers=num_layers, mode=mode)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
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
            re = relative_error(y_test=train_y, y_predict=pred, threshold=Rated_Capacity * 0.7)
            # with open('seeds.txt', 'a') as file:
            #     file.write(str(seed) + ':' +
            #                'epoch:{:<2d} | loss:{:<6.4f} | MAE:{:<6.4f} | RMSE:{:<6.4f} | RE:{:<6.4f}\n'.
            #                format(epoch, loss, mae, rmse, re))
        score = [re, mae, rmse]
        if (loss < 1e-3) and (score_[0] < score[0]):
            break
        score_ = score.copy()

    score_list.append(score_)
    test_X = np.reshape(test_x[start:], (-1, feature_size)).astype(
        np.float32)
    test_X = torch.from_numpy(test_X).to(device)
    pred_y = model(test_X).detach().numpy()
    MAE, RMSE = evaluation(test_y[start:], pred_y)
    result_list.append(pred_y)
    return score_list, result_list, MAE, RMSE


if __name__ == '__main__':
    modes = ['LSTM', 'RNN']  # RNN, LSTM, GRU

    for mode in modes:
        SCORE = []
        for seed in range(10):
            print('seed: ', seed)
            score_list, _, = tain(lr=lr, feature_size=feature_size, hidden_dim=hidden_dim, num_layers=num_layers,
                                  weight_decay=weight_decay, mode=mode, EPOCH=EPOCH, seed=seed)
            print('------------------------------------------------------------------')
            for s in score_list:
                SCORE.append(s)

        mlist = ['re', 'mae', 'rmse']
        # with open('seeds.txt', 'a') as file:
        #     for i in range(3):
        #         s = [line[i] for line in SCORE]
        #         file.write(mlist[i] + ' mean: {:<6.4f}'.format(np.mean(np.array(s))))
        #     file.write('\n')
        for i in range(3):
            s = [line[i] for line in SCORE]
            print(mlist[i] + ' mean: {:<6.4f}'.format(np.mean(np.array(s))))
