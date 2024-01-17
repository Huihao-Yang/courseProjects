import matplotlib.pyplot as plt
import pandas as pd
import glob
import seaborn as sns
import numpy as np

Rated_Capacity = 1.1
Battery_name = 'CS2_35'

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
plt.rcParams["axes.unicode_minus"] = False  # 该语句解决图像中的“-”负号的乱码问题


def draw_lines(discharge, params, chinese):
    Battery_list = glob.glob('./' + Battery_name + '-*.csv')
    fig, ax = plt.subplots(1, figsize=(12, 8))
    color_list = ['b:', 'g--', 'r-.', 'c:']

    for name, color in zip(Battery_list, color_list):
        df_lim = pd.read_csv(name)
        if discharge:
            data = df_lim[(df_lim['Step_Index'] >= 2) & (df_lim['Step_Index'] <= 4)]
        else:
            data = df_lim[df_lim['Step_Index'] > 4]

        time = data['Test_Time(s)'].copy()
        time = list(time)
        start = time[0]
        for i in range(len(time)):
            time[i] = time[i] - start
        ax.plot(time, data[params], color,
                label='Cycle-' + name[name.rfind('-') + 1:name.rfind('.')])

    ax.set_xlabel('时间/s', fontsize='14')
    ax.set_ylabel(chinese + '/' + params[-2], fontsize='14')
    plt.legend(fontsize='14')
    plt.savefig('./' + Battery_name + '-' + chinese + '.png')


def draw_cycle():
    data_path = './' + Battery_name + '-100.csv'
    data = pd.read_csv(data_path)
    time = data['Test_Time(s)'].copy()
    time = list(time)
    start = time[0]
    for i in range(len(time)):
        time[i] = time[i] - start
    fig, ax = plt.subplots(1, figsize=(12, 8))
    plt.xticks(rotation=45)
    ax.plot(time, data['Voltage(V)'], 'b:',
            label='电压')
    ax.set_xlabel('时间/s', fontsize='14')
    ax.set_ylabel('电压/V', fontsize='14')

    ax2 = ax.twinx()
    ax2.plot(time, data['Current(A)'], 'r-.',
             label='电流')
    ax2.set_ylabel('电流/A', fontsize='14')
    fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes, fontsize='14')
    plt.savefig('./' + Battery_name + '-cycle.png')
    plt.show()


if __name__ == '__main__':
    names = ['充电电压', '充电电流', '放电电压', '放电电流']
    params = ['Voltage(V)', 'Current(A)', 'Voltage(V)', 'Current(A)']
    for i in range(4):
        if i < 2:
            draw_lines(True, params[i], names[i])
        else:
            draw_lines(False, params[i], names[i])
    # draw_cycle()
