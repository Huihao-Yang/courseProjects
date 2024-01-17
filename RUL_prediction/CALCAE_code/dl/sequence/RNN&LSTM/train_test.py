import torch

from dl.sequence.function import *

window_size = 128
EPOCH = 1000
lr = 0.001  # learning rate  0.01 epoch 10
hidden_dim = 256
num_layers = 2
weight_decay = 0.0


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


def tain(lr=0.001, feature_size=16, hidden_dim=128, num_layers=2, weight_decay=0.0, mode='LSTM', EPOCH=1000, seed=0):
    score_list, result_list = [], []
    name = Battery_list[0]
    train_x, train_y, train_data, test_data = get_train_test(Battery, name, window_size=feature_size)
    train_size = len(train_x)
    print('sample size: {}'.format(train_size))

    setup_seed(seed)
    model = Net(input_size=feature_size, hidden_dim=hidden_dim, num_layers=num_layers, mode=mode)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    loss_list, y_ = [0], []
    mae, rmse, re = 1, 1, 1
    score_, score = 1, 1
    for epoch in range(EPOCH):
        X = np.reshape(train_x / Rated_Capacity, (-1, 1, feature_size)).astype(
            np.float32)  # (batch_size, seq_len, input_size)
        y = np.reshape(train_y[:, -1] / Rated_Capacity, (-1, 1)).astype(np.float32)  # shape 为 (batch_size, 1)

        X, y = torch.from_numpy(X).to(device), torch.from_numpy(y).to(device)
        output = model(X)
        output = output.reshape(-1, 1)


        loss = criterion(output, y)
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients

        if (epoch + 1) % 100 == 0:
            test_x = train_data.copy()  # 每100次重新预测一次
            point_list = []
            while (len(test_x) - len(train_data)) < len(test_data):
                x = np.reshape(np.array(test_x[-feature_size:]) / Rated_Capacity, (-1, 1, feature_size)).astype(
                    np.float32)
                x = torch.from_numpy(x).to(device)  # shape: (batch_size, 1, input_size)
                pred = model(x)
                next_point = pred.data.numpy()[0, 0] * Rated_Capacity
                test_x.append(next_point)  # 测试值加入原来序列用来继续预测下一个点
                point_list.append(next_point)  # 保存输出序列最后一个点的预测值
            y_.append(point_list)  # 保存本次预测所有的预测值
            loss_list.append(loss)
            mae, rmse = evaluation(y_test=test_data, y_predict=y_[-1])
            re = relative_error(y_test=test_data, y_predict=y_[-1], threshold=Rated_Capacity * 0.7)
            print(
                'epoch:{:<2d} | loss:{:<6.4f} | MAE:{:<6.4f} | RMSE:{:<6.4f} | RE:{:<6.4f}'.format(epoch, loss, mae,
                                                                                                   rmse, re))
        score = [re, mae, rmse]
        if (loss < 1e-3) and (score_[0] < score[0]):
            break
        score_ = score.copy()

    score_list.append(score_)
    result_list.append(y_[-1])
    return score_list, result_list


if __name__ == '__main__':
    mode = 'LSTM'  # RNN, LSTM, GRU

    SCORE = []
    for seed in range(10):
        print('seed: ', seed)
        score_list, _ = tain(lr=lr, feature_size=window_size, hidden_dim=hidden_dim, num_layers=num_layers,
                             weight_decay=weight_decay, mode=mode, EPOCH=EPOCH, seed=seed)
        print('------------------------------------------------------------------')
        for s in score_list:
            SCORE.append(s)

    mlist = ['re', 'mae', 'rmse']
    for i in range(3):
        s = [line[i] for line in SCORE]
        print(mlist[i] + ' mean: {:<6.4f}'.format(np.mean(np.array(s))))
    print('------------------------------------------------------------------')
    print('------------------------------------------------------------------')
