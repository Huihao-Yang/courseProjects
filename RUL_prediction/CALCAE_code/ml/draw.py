import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

if __name__ == '__main__':
    dir_path = '../dataset/params_data/'
    Battery_list = glob.glob(dir_path + '*.csv')
    battery = pd.read_csv(Battery_list[0], sep=',')
    labels = battery.columns.values

    # 热力图
    corr_matrix = battery.corr(method='spearman')
    plt.figure(figsize=(15, 10))
    sns.heatmap(corr_matrix,
                annot=True, annot_kws={'size': 14},
                fmt='.2f', cmap='Pastel1',
                mask=np.triu(corr_matrix))
    plt.show()
