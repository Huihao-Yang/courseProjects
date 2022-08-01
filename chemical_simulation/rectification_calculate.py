import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize
import math
from scipy.interpolate import splev, splrep

R = 3.2  # 回流比
xF = 0.5  # 进料组成
xD = 0.95  # 馏出液组成
xW = 0.05  # 釜液组成
q = 0.6  # 进料热状况参数
a = 2.4  # 相对挥发度


def cross_poin():
    a_D = R / (R + 1)
    b_D = xD / (R + 1)
    if q != 1.0 and q != 0.0:
        a_q = q / (q - 1)
        b_q = -xF / (q - 1)
        A = np.array([[-a_q, 1], [-a_D, 1]])
        B = np.array([b_q, b_D])
        return np.linalg.solve(A, B)
    elif q == 0.0:
        return [(xF - b_D) / a_D, xF]
    else:
        return [xF, a_D * xF + b_D]


pos = cross_poin()  # 精馏操作线和q线交点坐标


def vapor_liquid_equilibrium_equation(x):
    return a * x / (1 + (a - 1) * x)


def get_x(y):
    # 利用苯-甲苯.xlsx进行插值计算yn对应的xn
    Y = [0, 0.025, 0.0711, 0.112, 0.208, 0.294, 0.372, 0.442, 0.507, 0.566, 0.619, 0.667, 0.713, 0.755, 0.791, 0.825,
         0.857, 0.885, 0.912, 0.936, 0.959, 0.98, 0.988, 0.9961, 1]  # yn
    X = [0, 0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85,
         0.9, 0.95, 0.97, 0.99, 1]  # xn
    spl = splrep(Y, X)  # 获取插值系数
    return splev(y, spl)  # 进行插值


def rectification_equation(x):  # 精馏操作线方程
    return (R * x + xD) / (R + 1)


def q_equation(x):  # q线方程
    return q * x / (q - 1) - xF / (q - 1)


def stripping_equation(x):  # 提馏线方程
    return (pos[1] - xW) * (x - xW) / (pos[0] - xW) + xW


def calculate(xop):  # 计算理论塔板数
    '''
    :param xop : 更换操作线点的横坐标
    :return: 用于绘图的x,y坐标、每一块塔板的组成(x,y)、理论塔板数
    '''
    N = 0
    x_points = []
    y_points = []
    X = []
    Y = []
    # 添加精馏操作线上的点
    x_points.append(xD)
    y_points.append(xD)
    y = xD
    x = get_x(y)
    # 添加相平衡线上的点
    x_points.append(x)
    y_points.append(y)
    X.append(x)
    Y.append(y)
    while x >= xW:
        if x > xop:
            # 添加精馏操作线上的点
            y = rectification_equation(x)
            x_points.append(x)
            y_points.append(y)
            x = get_x(y)
            # 添加相平衡线上的点
            x_points.append(x)
            y_points.append(y)
            X.append(x)
            Y.append(y)
        else:
            y = stripping_equation(x)
            # 添加提馏操作线上的点
            x_points.append(x)
            y_points.append(y)
            x = get_x(y)
            # 添加相平衡线上的点
            X.append(x)
            Y.append(y)
            if x >= xW:
                x_points.append(x)
                y_points.append(y)
        N += 1
    return x_points, y_points, X, Y, N


def get_Rmin():  # 计算最小回流比
    def f(x):
        return q_equation(x) - vapor_liquid_equilibrium_equation(x)

    x = optimize.root(f, x0=xW)['x'][0]
    y = q_equation(x)
    return (xD - y) / (y - x), x, y


def get_t(y):
    '''
    :param y: 轻组分的摩尔分数
    :return: 根据安度因方程计算所得的塔板温度
    '''
    p = 101.325
    t1 = 80  # 苯沸点
    t2 = 110.6  # 甲苯沸点

    def pA0(t):  # 苯的安托因方程
        return math.pow(10, 6.031 - 1211 / (t + 220.8))

    def pB0(t):  # 甲苯的安托因方程
        return math.pow(10, 6.08 - 1345 / (t + 219.5))

    def f(t, y):
        return y * p * (pA0(t) - pB0(t)) - pA0(t) * (p - pB0(t))

    return round(optimize.root(f, x0=max(t1, t2), args=(y))['x'][0], 1)


if __name__ == '__main__':
    fig, axes = plt.subplots(1, 2)  # 创建两个子图,分别用于展示逐板计算结果、组分和温度的关系

    # 逐板计算
    ax1 = axes[0]
    xop = 0.3  # 更换操作线点的横坐标

    x1 = np.linspace(0, 1, 100)
    ax1.plot(x1, vapor_liquid_equilibrium_equation(
        x1), color='skyblue', label='相平衡')
    x2 = np.linspace(pos[0], xF, 10)
    ax1.plot(x2, q_equation(x2), color='green', label='q线')
    x3 = np.linspace(xop, xD, 10)
    ax1.plot(x3, rectification_equation(x3), color='red', label='精馏段操作线')
    x4 = np.linspace(xW, pos[0], 10)
    ax1.plot(x4, stripping_equation(x4), color='purple', label='提馏段操作线')

    x, y, X, Y, Nt = calculate(xop)  # Nt:理论塔板数
    Rmin, x_, y_ = get_Rmin()  # 最小回流比
    Rmin = round(Rmin, 3)
    # 在图像上显示Rmin,Nt
    ax1.plot(x_, y_, marker='o', color='pink')  # q线与相平衡线的交点
    ax1.text(0.1, 0.6, 'Rmin=' + str(Rmin), fontsize=15)
    ax1.text(0.1, 0.5, 'Nt=' + str(Nt), fontsize=15)
    ax1.plot(x, y, linestyle='--', color='black')

    ax1.plot([0, 1], [0, 1], color='orange', linestyle='--')  # 绘制y = x图像
    ax1.plot(pos[0], pos[1], marker='>')  # q线和精馏线交点
    ax1.legend()
    ax1.set_title('最小回流比和逐板计算结果', fontsize=18)
    ax1.set_xlabel('x', fontsize=15)
    ax1.set_ylabel('y', fontsize=15, rotation=0)

    # 组分组成与温度的关系
    x = np.arange(1, len(X) + 1)
    ax2 = axes[1]
    ax3 = ax2.twinx()
    ax2.plot(x, X, color='blue', label='x', marker='o', mfc='g', mec='g')
    ax2.plot(x, Y, color='orange', label='y', marker='o', mfc='red', mec='red')
    ax2.set_ylim(0, 1)
    ax2.set_xlim(0, len(X) + 2)
    ax2.legend(loc=2)  # 将图例放在左上角
    ax2.set_xlabel('塔板', fontsize=15)
    ax2.set_ylabel('组成', fontsize=15, rotation=0)

    t = [get_t(y) for y in Y]  # 根据轻组分的气相摩尔分数和安托因方程计算塔板温度
    ax3.plot(x, t, color='blue', label='t',
             marker='>', mfc='orange', mec='orange')
    ax3.legend(loc=1)  # 将图例放在右上角
    ax3.set_ylim(t[0] - t[0] % 10, t[len(t) - 1] - t[len(t) - 1] % 10 + 10)
    ax3.set_title('组成与温度分布情况', fontsize=18)
    ax3.set_ylabel('温度/℃', fontsize=15, rotation=0, labelpad=18)

    # 解决中文无法正常显示的问题
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 指定字体,正常显示中文,解决中文乱码
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显⽰为⽅块的问题

    plt.show()
