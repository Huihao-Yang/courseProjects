'''
温度：测试温度 1 度

充电：以的恒定电流（CC）模式进行充电，直到电池电压达到 4.2V，
然后以恒定电压（CV）模式充电，直到充电电流降至 20mA。

放电：以恒定电流（CC）模式进行放电，直到电池电压降到 2.7V。

终止条件：当电池达到寿命终止（End Of Life, EOF）标准——额定容量下降到它的30%，
即电池的额定容量从 1.1Ahr 到 0.77Ahr。
'''
import math

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob

Battery_list = ['CS2_35', 'CS2_36', 'CS2_37', 'CS2_38']
dir_path = 'dataset/'
des_dir = dir_path + 'params_data/'


def load_data(dir_path):
    """
    :param Battary_list: 存放各组电池数据的文件夹名称列表
    :param dir_path: 数据文件所在的根目录
    :return: 处理后每组电池数据
    """
    Battery = {}

    for name in Battery_list:  # 遍历每个数据文件夹
        print('Load Dataset ' + name + ' ...')
        '''
        glob通配符匹配-正则表达式，获取Battery_list每个文件夹下每个数据文件的名称
        返回一个包含该文件夹所有以下.xlsx结尾的文件名称 的列表(list) 
        '''
        path = glob.glob(dir_path + name + '/*.xlsx')
        dates = []

        '''
        提取每个数据表中的第一个测试时间(每组数据中的测试时间是从小到大排列)
        '''
        for p in path:
            '读取名称为p的文件下，索引为1的表'
            df = pd.read_excel(p, sheet_name=1)
            print('Load ' + str(p) + ' ...')
            dates.append(df['Date_Time'][0])

        '''
        根据实验数据的实验时间从小到大排列
        np.argsort(dates) 将dates中的元素从小到大排列,输出其在原来集合中的索引
        '''
        idx = np.argsort(dates)
        path_sorted = np.array(path)[idx]

        count = 0  # 统计电池充放电的总次数
        discharge_capacities = []
        health_indicator = []
        internal_resistance = []
        CCCT = []
        CCDT = []
        CVCT = []
        for p in path_sorted:
            df = pd.read_excel(p, sheet_name=1)
            print('Load ' + str(p) + ' ...')
            # 获取充放电次数的数据的列表(先用set去重再转为list)
            cycles = list(set(df['Cycle_Index']))
            for c in cycles:
                # 获取循环次数为第c次的全部实验数据
                df_lim = df[df['Cycle_Index'] == c]
                # Charging,获取充电的数据
                '''
                Step_Index 1-7为一次充放电的循环,Step_Index=7时为放电状态
                '''
                df_c = df_lim[(df_lim['Step_Index'] == 2) | (df_lim['Step_Index'] == 4)]
                c_v = df_c['Voltage(V)']
                c_c = df_c['Current(A)']
                c_t = df_c['Test_Time(s)']
                # CC or CV
                df_cc = df_lim[df_lim['Step_Index'] == 2]  # 充电时的恒定电流
                df_cv = df_lim[df_lim['Step_Index'] == 4]  # 充电时的恒定电压
                ccct = np.max(df_cc['Test_Time(s)']) - np.min(df_cc['Test_Time(s)'])
                cvct = np.max(df_cv['Test_Time(s)']) - np.min(df_cv['Test_Time(s)'])

                if math.isnan(ccct) | (ccct == 0) | math.isnan(cvct) | (cvct == 0):
                    continue

                CCCT.append(ccct)  # 恒定电流充电时间
                CVCT.append(cvct)  # 恒定电压充电时间

                # Discharging
                df_d = df_lim[df_lim['Step_Index'] == 7]
                d_v = df_d['Voltage(V)']
                d_c = df_d['Current(A)']
                d_t = df_d['Test_Time(s)']
                d_im = df_d['Internal_Resistance(Ohm)']
                ccdt = np.max(df_d['Test_Time(s)']) - np.max(df_cv['Test_Time(s)'])
                if math.isnan(ccdt) | (ccdt == 0):
                    continue
                CCDT.append(ccdt)  # 电流放电时间

                if (len(list(d_c)) != 0):  # 如果存在放电数据
                    time_diff = np.diff(list(d_t))  # np.diff函数——数组中a[n]-a[n-1],即放电时间
                    d_c = np.array(list(d_c))[1:]  # 以每个时间段结束时电流大小作为计算Q的电流
                    discharge_capacity = time_diff * d_c / 3600  # Q = A*h
                    # 将电荷量Q的结果进行累加,最后discharge_capacity[-1]即为本次放电释放的总电荷量
                    discharge_capacity = [np.sum(discharge_capacity[:n]) for n in range(discharge_capacity.shape[0])]
                    discharge_capacities.append(-1 * discharge_capacity[-1])  # 放电时电流数据为负,表示放电,因此Q的结果需要✖-1

                    '''
                    numpy.argmin(a, axis=None, out=None)[source] 给出axis方向最小值的下标.
                    就是取放电电压在 [3.4, 3.8] 之间的容量作为 电池的 SOH。因为现在 SOH 还没有稳定的定义，
                    所以这个区间的数值不一定就是这两个，你可以选择放电电压在 [3.3, 3.8], [3.5, 3.8] 之间的容量作为 SOH 也没问题。
                    因为容量预测的时候可能不太准确，不可能满充满放，所以选择电池在中间这段放电的时候的电容量来作为 SOH
                    '''
                    dec = np.abs(np.array(d_v) - 3.8)[1:]
                    start = np.array(discharge_capacity)[np.argmin(dec)]
                    dec = np.abs(np.array(d_v) - 3.4)[1:]
                    end = np.array(discharge_capacity)[np.argmin(dec)]
                    health_indicator.append(-1 * (end - start))

                    internal_resistance.append(np.mean(np.array(d_im)))  # 取放电时Ohm的平均值作为该次充放电的internal_resisitance
                    count += 1
        discharge_capacities = np.array(discharge_capacities)
        health_indicator = np.array(health_indicator)
        internal_resistance = np.array(internal_resistance)
        CCCT = np.array(CCCT)
        CVCT = np.array(CVCT)
        CCDT = np.array(CCDT)

        idx = drop_outlier(discharge_capacities, count, 40)
        df_result = pd.DataFrame({'cycle': np.linspace(1, idx.shape[0], idx.shape[0]),
                                  'capacity': discharge_capacities[idx],
                                  'SoH': health_indicator[idx],
                                  'resistance': internal_resistance[idx],
                                  'CCCT': CCCT[idx],
                                  'CVCT': CVCT[idx],
                                  'CCDT': CCDT[idx]
                                  })
        Battery[name] = df_result
    return Battery


# 去除异常点
def drop_outlier(array, count, bins):
    '''
    :param array: 存放电池容量数据的列表
    :param count: 电池充放电总次数
    :param bins: 筛选数据的步长
    :return: 处理后的合理电池容量数据对应的下标
    '''
    index = []
    range_ = np.arange(1, count, bins)
    for i in range_[:-1]:
        array_lim = array[i:i + bins]
        sigma = np.std(array_lim)  # 数据的标准差
        mean = np.mean(array_lim)
        # 将数据近似正态分布处理,剔除5%的数据 P(u−2σ≤x≤u+2σ) = 95%
        th_max, th_min = mean + sigma * 2, mean - sigma * 2
        idx = np.where((array_lim < th_max) & (array_lim > th_min))
        '''
        np.where返回的结果是一个元组,因此需要用idx[0]获取符合要求的下标
        e.g.
            a = np.array([1, 2, 3, 4, 5])
            ls = np.where(a > 2) (array([2, 3, 4], dtype=int64),)
        '''
        idx = idx[0] + i  # 合理数据的所有下标
        index.extend(list(idx))
    return np.array(index)


if __name__ == '__main__':
    data = load_data(dir_path)
    for name in Battery_list:
        # data[name].to_excel(des_dir + name + '.xlsx', index=False)
        data[name].to_csv(des_dir + name + '.csv', index=False)
