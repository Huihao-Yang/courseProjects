from train import *

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
plt.rcParams["axes.unicode_minus"] = False  # 该语句解决图像中的“-”负号的乱码问题

if __name__ == '__main__':
    seed = 0
    mode = 'MLP'
    print('-' * 23 + 'mode in use: ' + mode + '-' * 23)
    _, result_list, MAE, RMSE = tain(LR=LR, feature_size=feature_size, hidden_size=hidden_size,
                                     weight_decay=weight_decay, EPOCH=8000, seed=seed)

    train_x, train_y, test_x, test_y = get_train_test()

    aa = []
    count = 0
    for a in result_list[0]:  # 测试集预测结果
        if a > Rated_Capacity or a < 0.2:
            aa.append(Rated_Capacity*0.7)
        # elif len(aa) > 0 and (a > aa[-1] * 1.5 or a < aa[-1] / 1.5):
        #     aa.append(aa[-1])
        else:
            aa.append(a)

        count += 1
    idx = np.arange(1, count + 1, 1).reshape(-1, 1)

    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.plot(idx, test_y, color='red', label='实际容量')
    ax.plot(idx, aa, color='blue', label=mode + '预测值',ls='--')
    plt.plot([-1, 1000], [Rated_Capacity * 0.7, Rated_Capacity * 0.7], c='black', lw=1, ls='--',
             label='失效阈值')  # 临界点直线

    ax.set_xlabel('循环次数', fontsize=14)
    ax.set_ylabel('电池容量/Ah', fontsize=14)

    plt.legend()
    plt.savefig('.././img/params' + '-' + mode + '.png')
    plt.show()

    mlist = ['mae', 'rmse']
    SCORE = [MAE, RMSE]
    print(MAE)
    with open('seeds.txt', 'a') as file:
        for i in range(2):
            file.write(mode + ':' + mlist[i] + ' mean: {:<6.4f}'.format(SCORE[i]))
        file.write('\n')
