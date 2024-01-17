from dl.sequence.function import *

window_size = 8
EPOCH = 1000
LR = 0.01  # learning rate
feature_size = window_size
hidden_size = [32, 16]
weight_decay = 0.0
Rated_Capacity = 1.1


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


def tain(LR=0.01, feature_size=8, hidden_size=[16, 8], weight_decay=0.0, window_size=8, EPOCH=1000, seed=0):
    mae_list, rmse_list, re_list = [], [], []
    result_list = []
    for i in range(4):
        name = Battery_list[i]
        train_x, train_y, train_data, test_data = get_train_test(Battery, name, window_size)
        train_size = len(train_x)
        print('sample size: {}'.format(train_size))

        setup_seed(seed)
        model = Net(feature_size=feature_size, hidden_size=hidden_size)
        model = model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=weight_decay)
        criterion = nn.MSELoss()

        test_x = train_data.copy()
        loss_list, y_ = [0], []
        for epoch in range(EPOCH):
            X = np.reshape(train_x / Rated_Capacity, (-1, feature_size)).astype(np.float32)
            y = np.reshape(train_y[:, -1] / Rated_Capacity, (-1, 1)).astype(np.float32)

            X, y = torch.from_numpy(X).to(device), torch.from_numpy(y).to(device)
            output = model(X)
            loss = criterion(output, y)
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

            if (epoch + 1) % 100 == 0:
                test_x = train_data.copy()  # 每100次重新预测一次
                point_list = []
                while (len(test_x) - len(train_data)) < len(test_data):
                    x = np.reshape(np.array(test_x[-feature_size:]) / Rated_Capacity, (-1, feature_size)).astype(
                        np.float32)
                    x = torch.from_numpy(x).to(device)
                    pred = model(x)  # 测试集 模型预测#pred shape为(batch_size=1, feature_size=1)
                    next_point = pred.data.numpy()[0, 0] * Rated_Capacity
                    test_x.append(next_point)  # 测试值加入原来序列用来继续预测下一个点
                    point_list.append(next_point)  # 保存输出序列最后一个点的预测值
                y_.append(point_list)  # 保存本次预测所有的预测值
                loss_list.append(loss)
                mae, rmse = evaluation(y_test=test_data, y_predict=y_[-1])
                re = relative_error(
                    y_test=test_data, y_predict=y_[-1], threshold=Rated_Capacity * 0.7)
                print(
                    'epoch:{:<2d} | loss:{:<6.4f} | MAE:{:<6.4f} | RMSE:{:<6.4f} | RE:{:<6.4f}'.format(epoch, loss, mae,
                                                                                                       rmse, re))
            if (len(loss_list) > 1) and (abs(loss_list[-2] - loss_list[-1]) < 1e-6):
                break

        mae, rmse = evaluation(y_test=test_data, y_predict=y_[-1])
        re = relative_error(y_test=test_data, y_predict=y_[-1], threshold=Rated_Capacity * 0.7)
        mae_list.append(mae)
        rmse_list.append(rmse)
        re_list.append(re)
        result_list.append(y_[-1])
    return re_list, mae_list, rmse_list, result_list


if __name__ == '__main__':
    MAE, RMSE, RE = [], [], []
    for seed in range(10):
        re_list, mae_list, rmse_list, _ = tain(LR=LR, feature_size=feature_size, hidden_size=hidden_size,
                                               weight_decay=weight_decay,
                                               window_size=window_size, EPOCH=EPOCH, seed=seed)
        RE.append(np.mean(np.array(re_list)))
        MAE.append(np.mean(np.array(mae_list)))
        RMSE.append(np.mean(np.array(rmse_list)))

        print('------------------------------------------------------------------')

    for seed in range(10):
        print('seed:{:<6.0f} RE:{:<6.4f} MAE:{:<6.4f} RMSE:{:<6.4f}'.format(seed, RE[seed], MAE[seed], RMSE[seed]))

    print('RE: mean: {:<6.4f} | std: {:<6.4f}'.format(np.mean(np.array(RE)), np.std(np.array(RE))))
    print('MAE: mean: {:<6.4f} | std: {:<6.4f}'.format(np.mean(np.array(MAE)), np.std(np.array(MAE))))
    print('RMSE: mean: {:<6.4f} | std: {:<6.4f}'.format(np.mean(np.array(RMSE)), np.std(np.array(RMSE))))
    print('------------------------------------------------------------------')
    print('------------------------------------------------------------------')
