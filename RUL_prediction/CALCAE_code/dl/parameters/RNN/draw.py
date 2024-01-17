import pandas as pd

from train import *

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
plt.rcParams["axes.unicode_minus"] = False  # 该语句解决图像中的“-”负号的乱码问题

if __name__ == '__main__':
    modes = {'LSTM', 'RNN', 'GRU'}
    seed = 6
    epoch = 2000
    name = Battery_list[3]
    n = name[name.rfind('\\') + 1:name.rfind('.')]
    count = pd.read_csv('../../.././dataset/params_data/CS2_38.csv').shape[0]

    for j in [0, 0.2, 0.4]:
        start = int(count * j)
        for mode in modes:
            print('-' * 23 + n + ' mode in use: ' + mode + '-' * 23)
            _, result_list, MAE, RMSE = tain(lr=lr, feature_size=feature_size, hidden_dim=hidden_dim,
                                             num_layers=num_layers,
                                             weight_decay=weight_decay, mode=mode, EPOCH=epoch, seed=seed,
                                             start=start)
            train_x, train_y, test_x, test_y = get_train_test(3)

            aa = []
            for a in result_list[0]:  # 测试集预测结果
                if a > 1.1 * Rated_Capacity:
                    aa.append(0.7 * Rated_Capacity)
                else:
                    aa.append(a)
            idx = np.arange(1, count + 1, 1).reshape(-1, 1)

            fig, ax = plt.subplots(1, figsize=(12, 8))
            ax.plot(idx, test_y, color='red', label='实际容量')
            ax.plot(idx[start:], aa, color='blue', label=mode + '预测值')
            plt.plot([-1, 1000], [0.7, 0.7], c='black', lw=1, ls='--', label='失效阈值')  # 临界点直线
            ax.set_xlabel('循环次数', fontsize=14)
            ax.set_ylabel('电池容量/Ah', fontsize=14)
            if start != 0:
                plt.plot([start, start], [0, Rated_Capacity], c='m', lw=1, ls='--', label='预测起点')
            plt.legend(fontsize=14)
            plt.savefig('.././img/params' + '-' + mode + '-RUL-' + n + '-' + str(j) + '.png')
            # plt.show()

            mlist = ['mae', 'rmse']
            SCORE = [MAE, RMSE]
            with open('seeds.txt', 'a') as file:
                for i in range(2):
                    file.write(mode + str(epoch) +'-' + str(j) + ':' + mlist[i] + ' mean: {:<6.4f}'.format(SCORE[i]))
                file.write('\n')
