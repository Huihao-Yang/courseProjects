'''
测试内容
温度：测试温度 1 度
充电：以的恒定电流（CC）模式进行充电，直到电池电压达到 4.2V，然后以恒定电压（CV）模式充电，直到充电电流降至 20mA。
放电：以恒定电流（CC）模式进行放电，直到电池电压降到 2.7V。
终止条件：当电池达到寿命终止（End Of Life, EOF）标准——额定容量下降到它的30%，即电池的额定容量从 1.1Ahr 到 0.77Ahr。
'''

import matplotlib.pyplot as plt
import pandas as pd
import glob
import seaborn as sns
import numpy as np

Rated_Capacity = 1.1

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
plt.rcParams["axes.unicode_minus"] = False  # 该语句解决图像中的“-”负号的乱码问题


def capacities_cycles(dir_path):  # 绘制每组数据中电池容量与充放电循环次数的趋势图
    Battery_list = glob.glob(dir_path + '/*.xlsx')
    fig, ax = plt.subplots(1, figsize=(12, 8))
    color_list = ['b:', 'g--', 'r-.', 'c:']

    for name, color in zip(Battery_list, color_list):
        battery = pd.read_excel(name)
        ax.plot(battery['cycle'], battery['capacity'], color,
                label='Battery_' + name[name.rfind('\\') + 1:name.rfind('.')])
    ax.plot([-1, 1000], [Rated_Capacity * 0.7, Rated_Capacity * 0.7], c='black', lw=1, ls='--',
            label='失效阈值')  # 临界点直线
    ax.set_xlabel('循环次数', fontsize='14')
    ax.set_ylabel('电池容量/Ah', fontsize='14')
    plt.legend(fontsize='14')
    plt.savefig('./capacities.png')
    plt.show()


def capacity_cycles(dir_path):  # 单独绘制每组数据中电池容量与充放电循环次数
    Battery_list = glob.glob(dir_path + '/*.xlsx')
    color_list = ['b:', 'g--', 'r-.', 'c:']
    Rated_Capacity = 1.1
    for name, color in zip(Battery_list, color_list):
        battery = pd.read_excel(name)
        label = name[name.rfind('\\') + 1:name.rfind('.')]
        plt.plot(battery['cycle'], battery['capacity'], color,
                 label='Battery_' + label)
        plt.plot([-1, 1000], [Rated_Capacity * 0.7, Rated_Capacity * 0.7], c='black', lw=1, ls='--',
                 label='critical state line')  # 临界点直线
        plt.xlabel('Discharge cycles')
        plt.ylabel('Capacity (Ah)')
        plt.title('Capacity degradation at ambient temperature of 1°C')
        plt.legend()
        plt.savefig('./' + label + '-capacities.png')
        plt.show()


def resistance_cycles(data_path):  # 绘制单组数据中电池的电阻、电池健康指标与充放电循环次数的趋势图
    battery = pd.read_excel(data_path)
    plt.figure(figsize=(9, 6))
    plt.scatter(battery['cycle'], battery['SoH'], c=battery['resistance'], s=10)
    cbar = plt.colorbar()
    cbar.set_label('Internal Resistance (Ohm)', fontsize=14, rotation=-90, labelpad=20)
    plt.xlabel('Number of Cycles', fontsize=14)
    plt.ylabel('State of Health', fontsize=14)
    plt.show()


def resistances_cycles(dir_path):
    Battery_list = glob.glob(dir_path + '/*.xlsx')
    plt.figure(figsize=(12, 9))
    for i in range(4):
        battery = pd.read_excel(Battery_list[i])
        plt.subplot(2, 2, i + 1)
        plt.scatter(battery['cycle'], battery['SoH'], c=battery['resistance'], s=10)
        cbar = plt.colorbar()
        cbar.set_label('Internal Resistance (Ohm)', fontsize=14, rotation=-90, labelpad=20)
        plt.xlabel('Number of Cycles', fontsize=14)
        plt.ylabel('State of Health', fontsize=14)
    plt.show()


def params_cycles(datapath):  # 绘制各个指标和充放电循环次数的趋势图
    battery = pd.read_excel(datapath)
    plt.figure(figsize=(12, 9))
    names = ['capacity', 'resistance', 'CCCT', 'CVCT']
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.scatter(battery['cycle'], battery[names[i]], s=10)
        plt.xlabel('Number of Cycles', fontsize=14)
        plt.ylabel(names[i], fontsize=14)
    plt.show()


def param_cycle(datapath, line, param, unity):
    battery = pd.read_excel(datapath)
    plt.figure(figsize=(12, 9))
    plt.scatter(battery['cycle'], battery[param], s=10)
    plt.xlabel('Number of Cycles', fontsize=14)
    if unity == '':
        plt.ylabel(param, fontsize=14)
    else:
        plt.ylabel(param + '(' + unity + ')', fontsize=14)
    if line:
        plt.plot([-1, 1000], [Rated_Capacity * 0.7, Rated_Capacity * 0.7], c='black', lw=1, ls='--'
                 , label='critical state line')  # 临界点直线
        plt.legend()
    plt.savefig('./' + datapath[datapath.rfind('\\') + 1:datapath.rfind('.')] + '_' + param + '.png')
    plt.clf()


def params_heatmap(dir_path):  # 绘制参数的热力图
    Battery_list = glob.glob(dir_path + '*.csv')
    datas = pd.DataFrame()
    for name in Battery_list:
        datas = pd.concat([datas.copy(), pd.read_csv(name, sep=',')])  # 合并所有数据

    # 热力图
    corr_matrix = datas.corr(method='spearman')
    plt.figure(figsize=(15, 10))
    sns.heatmap(corr_matrix,
                annot=True, annot_kws={'size': 14},
                fmt='.2f', cmap='Pastel1'
                # ,mask=np.triu(corr_matrix)
                )
    plt.show()


if __name__ == '__main__':
    # dir_path = './dataset/cp_data/'
    dir_path = './dataset/params_data/'
    # Battery_list = glob.glob(dir_path + '/*.xlsx')
    # capacity_cycles(dir_path)
    # params_heatmap(dir_path)
    # capacities_cycles('./dataset/cp_data/')

    name = './dataset/params_data/CS2_35.csv'
    battery = pd.read_csv(name)
    plt.plot(battery['cycle'], battery['capacity'], color='red')
    plt.plot([-1, 1000], [Rated_Capacity * 0.7, Rated_Capacity * 0.7], c='black', lw=1, ls='--',
             label='失效阈值')  # 临界点直线
    plt.plot([100, 100],[0, Rated_Capacity], c='m', lw=1, ls='--', label='失效阈值')
    plt.xlabel('循环次数', fontsize=14)
    plt.ylabel('电池容量/Ah', fontsize=14)
    plt.legend(fontsize=14)
    plt.show()
