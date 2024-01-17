from train import *

if __name__ == '__main__':
    seed = 6
    mode = 'MLP'
    _, _, _, result_list = tain(LR=LR, feature_size=feature_size, hidden_size=hidden_size, weight_decay=weight_decay,
                                window_size=window_size, EPOCH=EPOCH, seed=seed)
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
        plt.savefig('.././img/' + name + '-' + mode + '.png')
