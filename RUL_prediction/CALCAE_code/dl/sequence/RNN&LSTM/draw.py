from train import *

if __name__ == '__main__':
    modes = {'LSTM', 'RNN', 'GRU'}
    seed = 6
    for mode in modes:
        print('-' * 23 + 'mode in use: ' + mode + '-' * 23)
        _, result_list = tain(lr=lr, feature_size=window_size, hidden_dim=hidden_dim, num_layers=num_layers,
                              weight_decay=weight_decay, mode=mode, EPOCH=EPOCH, seed=seed)

        for i in range(4):
            name = Battery_list[i]
            train_x, train_y, train_data, test_data = get_train_test(Battery, name, window_size)

            aa = train_data[:window_size + 1].copy()  # 第一个输入序列
            [aa.append(a) for a in result_list[i]]  # 测试集预测结果

            battery = Battery[name]
            fig, ax = plt.subplots(1, figsize=(12, 8))
            ax.plot(battery['cycle'], battery['capacity'], 'b.', label=name)
            ax.plot(battery['cycle'], aa, 'r.', label='Prediction')
            plt.plot([-1, 1000], [Rated_Capacity * 0.7, Rated_Capacity * 0.7], c='black', lw=1, ls='--')  # 临界点直线
            ax.set(xlabel='Discharge cycles', ylabel='Capacity (Ah)',
                   title='Capacity degradation at ambient temperature of 1°C')
            plt.legend()
            # plt.savefig('.././img/' + name + '-' + mode + '.png')
            plt.show()
