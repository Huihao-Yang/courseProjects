import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root

d = 4e-2  # 管道直径(单位:m)
e = 2e-4  # 绝对粗糙度(单位:m)
U = 1.24e-3  # 粘度(单位:Pa*s)
p = 950  # 密度(单位:kg/m3)
g = 9.8  # 重力加速度(单位:m2/s)
z = 4.5  # 高度差(单位:m)
P = -3.82e3  # 压强差(单位:Pa)
L = 35  # 管路当量长度(单位:m)


def laminar_flow_equation(Re):  # 层流公式
    '''
    :param Re: 雷诺准数
    :return: 摩擦系数
    '''
    return 64 / Re


def turbulent_flow_equation(r, Re):  # 湍流公式
    '''
    :param Re: 雷诺数
    :param r: 摩擦系数
    :return: 湍流方程(Colebrook公式)
    '''
    return -1 / np.sqrt(r) + 2 * np.log10(d / e) + 1.14 - 2 * np.log10(1 + 9.35 * (d / e) / Re / np.sqrt(r))


def u(r):  # 流速计算公式
    '''
    :param r: 摩擦系数
    :return: 根据伯努利方程计算所得的流速
    '''
    # 式子中1.5(0.5+1)为管路进出口局部阻力损失系数之和
    return np.sqrt(2 * (g * z + P / p) / (r * L / d + 1.5))


def Re_equation(u):  # 雷诺准数
    return d * u * p / U


def draw_lines():  # 绘制摩擦系数和雷诺数的关系曲线
    # 绘制层流曲线
    x1 = np.linspace(100, 2300, 20)  # 在100-2300范围内选取20个点
    plt.plot(x1, laminar_flow_equation(x1), ls='--', color='c')

    # 绘制湍流曲线
    x = np.linspace(2300, 1e6)  # 雷诺数范围
    y = np.linspace(1e-4, 1)  # 摩擦系数范围
    x, y = np.meshgrid(x, y)  # 构造网格
    f = turbulent_flow_equation(y, x)
    plt.contour(x, y, f, 0, linestyles='--', colors='orange')  # 用等高线绘制隐函数图像
    # 设置对数坐标
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Re', fontsize=15)
    plt.ylabel('λ', fontsize=15, rotation=0)


def trial_error(eps, v):  # 试差法求解
    '''
    :param eps: 收敛时的误差判据
    :param v: 管内流速初值(迭代初值)
    '''

    def f1(X):
        return 1 / (2 * np.log10(d / e) + 1.14 - 2 * np.log10(1 + 9.35 * (d / e) / X[1] / np.sqrt(X[0]))) ** 2

    def f2(X):
        return np.sqrt(2 * (g * z + P / p) / (X[0] * L / d + 1.5)) * d * p / U

    def f3(X):
        return 64 / X[1]

    x_points = []
    y_points = []
    X = [1e-3, Re_equation(10)]
    x = f1(X)
    y = f2(X)
    Xk = [x, y]
    x_points.append(x)
    y_points.append(y)
    while abs(Xk[0] - X[0]) >= eps or abs(Xk[1] - X[1]) >= eps:
        X = Xk
        x = f1(X)
        y = f2(X)
        Xk = [x, y]
        x_points.append(x)
        y_points.append(y)
    count = len(x_points)
    UV = u(x_points[count - 1]) * 3600 * 3.14 / 4 * d * d
    print('试差法求解结果:')
    print("湍流假定下流量为:{:.2f}m3/h,雷诺数Re为{:.0f},迭代次数为{}".format(UV, y_points[count - 1], count))
    plt.plot(y_points, x_points, marker='^', label='turbulent flow')

    x_points = []
    y_points = []
    X = [1e-3, Re_equation(10)]
    x = f3(X)
    y = f2(X)
    Xk = [x, y]
    x_points.append(x)
    y_points.append(y)
    while abs(Xk[0] - X[0]) >= eps or abs(Xk[1] - X[1]) >= eps:
        X = Xk
        x = f3(X)
        y = f2(X)
        Xk = [x, y]
        x_points.append(x)
        y_points.append(y)
    count = len(x_points)
    UV = u(x_points[count - 1]) * 3600 * 3.14 / 4 * d * d
    print("层流假定下流量为:{:.2f}m3/h,雷诺数Re为{:.0f},迭代次数为{}".format(UV, y_points[count - 1], count))
    plt.plot(y_points, x_points, marker='v', label='laminar flow')

    plt.legend()  # 显示图标
    plt.title('试差法:Re-λ', fontsize=15)


def newton(eps, v):  # 牛顿迭代法
    '''
    :param eps: 收敛时的误差判据
    :param v: 管内流速初值(迭代初值)
    '''

    # 联立雷诺数方程Re_equation(u)和流速方程u(r),得到一个摩擦系数r和雷诺数Re的等式f1(x)
    def f1(X):
        return U * X[1] / (d * p) - np.sqrt(2 * (g * z + P / p) / (X[0] * L / d + 1.5))

    def f11(X):  # 偏导数
        return np.sqrt(2 * (g * z + P / p) / (X[0] * L / d + 1.5)) / 2 * L / d / (X[0] * L / d + 1.5)

    def f12(X):
        return U / d / p

    def f2(X):  # Colebrook公式
        return -1 / np.sqrt(X[0]) + 2 * np.log10(d / e) + 1.14 - 2 * np.log10(1 + 9.35 * (d / e) / X[1] / np.sqrt(X[0]))

    def f21(X):
        return 1 / (2 * X[0] * np.sqrt(X[0])) + 2 / (1 + 9.35 * (d / e) / X[1] / np.sqrt(X[0]) * np.log(10)) * 1 / (
                2 * X[0] * np.sqrt(X[0])) * 9.35 * (d / e) / X[1]

    def f22(X):
        return 2 / (1 + 9.35 * (d / e) / X[1] / np.sqrt(X[0]) * np.log(10)) * 9.35 * (d / e) / X[1] / X[1] / np.sqrt(
            X[0])

    def f3(X):  # 层流条件下,雷诺数Re和摩擦系数r的关系式子
        return X[0] - 64 / X[1]

    def f31(X):
        return 1

    def f32(X):
        return 64 / X[1] / X[1]

    # 在湍流假定下迭代
    x_points = []
    y_points = []
    X = [1e-3, Re_equation(v)]  # 迭代的初值
    x = X[0] + (f1(X) * f22(X) - f2(X) * f12(X)) / (f21(X) * f12(X) - f11(X) * f22(X))
    y = X[1] + (f2(X) * f11(X) - f1(X) * f21(X)) / (f21(X) * f12(X) - f11(X) * f22(X))
    Xk = [x, y]
    x_points.append(x)
    y_points.append(y)
    while abs(Xk[0] - X[0]) >= eps or abs(Xk[1] - X[1]) >= eps:  # 迭代
        X = Xk
        x = X[0] + (f1(X) * f22(X) - f2(X) * f12(X)) / (f21(X) * f12(X) - f11(X) * f22(X))
        y = X[1] + (f2(X) * f11(X) - f1(X) * f21(X)) / (f21(X) * f12(X) - f11(X) * f22(X))
        Xk = [x, y]
        x_points.append(x)
        y_points.append(y)
    count = len(x_points)  # 迭代次数
    UV = u(x_points[count - 1]) * 3600 * 3.14 / 4 * d * d  # 计算体积流量
    print('牛顿迭代法求解结果:')
    print("湍流假定下流量为:{:.2f}m3/h,雷诺数Re为{:.0f},迭代次数为{}".format(UV, y_points[count - 1], count))
    plt.plot(y_points, x_points, marker='^', label='turbulent flow')

    # 层流假定下迭代
    x_points = []
    y_points = []
    X = [1e-3, Re_equation(10)]  # 迭代的初值
    x = X[0] + (f1(X) * f32(X) - f3(X) * f12(X)) / (f31(X) * f12(X) - f11(X) * f32(X))
    y = X[1] + (f3(X) * f11(X) - f1(X) * f31(X)) / (f31(X) * f12(X) - f11(X) * f32(X))
    Xk = [x, y]
    x_points.append(x)
    y_points.append(y)
    while abs(Xk[0] - X[0]) >= eps or abs(Xk[1] - X[1]) >= eps:
        X = Xk
        x = X[0] + (f1(X) * f32(X) - f3(X) * f12(X)) / (f31(X) * f12(X) - f11(X) * f32(X))
        y = X[1] + (f3(X) * f11(X) - f1(X) * f31(X)) / (f31(X) * f12(X) - f11(X) * f32(X))
        Xk = [x, y]
        x_points.append(x)
        y_points.append(y)
    count = len(x_points)  # 迭代次数
    UV = u(x_points[count - 1]) * 3600 * 3.14 / 4 * d * d  # 计算体积流量
    print("层流假定下流量为:{:.2f}m3/h,雷诺数Re为{:.0f},迭代次数为{}".format(UV, y_points[count - 1], count))
    plt.plot(y_points, x_points, marker='v', label='laminar flow')

    plt.legend()  # 显示图标
    plt.title('牛顿迭代法:Re-λ', fontsize=15)


def relaxation(eps, v, w):  # 松弛迭代法
    '''
    :param eps: 收敛时的误差判据
    :param v: 管内流速初值(迭代初值)
    :param w:  松弛因子
    '''

    def f1(X):
        return 1 / (2 * np.log10(d / e) + 1.14 - 2 * np.log10(1 + 9.35 * (d / e) / X[1] / np.sqrt(X[0]))) ** 2

    def f2(X):
        return np.sqrt(2 * (g * z + P / p) / (X[0] * L / d + 1.5)) * d * p / U

    def f3(X):
        return 64 / X[1]

    x_points = []
    y_points = []
    X = [1e-3, Re_equation(10)]
    x = X[0] + w * (f1(X) - X[0])
    y = X[1] + w * (f2(X) - X[1])
    Xk = [x, y]
    x_points.append(x)
    y_points.append(y)
    while abs(Xk[0] - X[0]) >= eps or abs(Xk[1] - X[1]) >= eps:
        X = Xk
        x = X[0] + w * (f1(X) - X[0])
        y = X[1] + w * (f2(X) - X[1])
        Xk = [x, y]
        x_points.append(x)
        y_points.append(y)
    count = len(x_points)
    UV = u(x_points[count - 1]) * 3600 * 3.14 / 4 * d * d
    print('松弛迭代法求解结果:')
    print("湍流假定下流量为:{:.2f}m3/h,雷诺数Re为{:.0f},迭代次数为{}".format(UV, y_points[count - 1], count))
    plt.plot(y_points, x_points, marker='^', label='turbulent flow')

    x_points = []
    y_points = []
    X = [1e-3, Re_equation(10)]
    x = X[0] + w * (f3(X) - X[0])
    y = X[1] + w * (f2(X) - X[1])
    Xk = [x, y]
    x_points.append(x)
    y_points.append(y)
    while abs(Xk[0] - X[0]) >= eps or abs(Xk[1] - X[1]) >= eps:
        X = Xk
        x = X[0] + w * (f3(X) - X[0])
        y = X[1] + w * (f2(X) - X[1])
        Xk = [x, y]
        x_points.append(x)
        y_points.append(y)
    count = len(x_points)
    UV = u(x_points[count - 1]) * 3600 * 3.14 / 4 * d * d
    print("层流假定下流量为:{:.2f}m3/h,雷诺数Re为{:.0f},迭代次数为{}".format(UV, y_points[count - 1], count))
    plt.plot(y_points, x_points, marker='v', label='laminar flow')

    plt.legend()  # 显示图标
    plt.title('松弛迭代法:Re-λ', fontsize=15)


if __name__ == '__main__':
    # 绘制试差法的图像
    ax1 = plt.subplot(1, 3, 1)
    draw_lines()
    trial_error(0.001, 10)
    # 绘制牛顿迭代法的图像
    ax2 = plt.subplot(1, 3, 2)
    draw_lines()
    newton(0.001, 10)
    # 绘制松弛迭代法的图像
    ax3 = plt.subplot(1, 3, 3)
    draw_lines()
    relaxation(0.001, 10, 0.5)
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 指定字体,正常显示中文,解决中文乱码
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显⽰为⽅块的问题
    plt.show()
