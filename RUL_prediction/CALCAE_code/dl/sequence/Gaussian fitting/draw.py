import matplotlib.pyplot as plt

from train import *

if __name__ == '__main__':
    seed = 0
    setup_seed(seed=seed)
    mode = 'Gaussian fitting'
    for i in range(4):
        name = Battery_list[i]
        df = Battery[name]
        x, y = np.array(df['cycle']), np.array(df['capacity'])
        model, loss_list = train(data=np.c_[x, y], k=k, lr=lr, stop=stop, epochs=epochs, device=device)

        log = open('./img/' + name + '-' + mode + '.txt', mode='a', encoding='utf-8')
        print(model.w, file=log)
        log.close()

        # # 误差曲线
        # fig, ax = plt.subplots(1, figsize=(12, 8))
        # ax.plot([i for i in range(len(loss_list))], loss_list, 'b-', label='loss')
        # ax.set(xlabel='epochs', ylabel='loss', title=name + 'Loss Curve')
        # plt.legend()
        # plt.show()

        # 拟合曲线
        y_ = predict(model, x, device=device)
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.plot(x, y, 'b-', label='True Value')
        ax.plot(x, y_, 'r-', label='Prediction Value')
        ax.set(xlabel='Discharge cycles', ylabel='Capacity (Ah)', title=name + ': True vs. Prediction')
        plt.legend()
        plt.savefig('./img/' + name + '-' + mode + '.png')
