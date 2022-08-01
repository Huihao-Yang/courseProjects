import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    df = pd.read_csv('winequality-white.csv', sep=';')
    req_cols = df.columns.values

    # 绘制每个变量对应的直方图
    f, axes = plt.subplots(4, 3, figsize=(14, 9))
    counter = 0
    for r in range(4):
        for c in range(3):
            if counter > 11:
                continue
            sns.histplot(data=df, x=req_cols[counter], ax=axes[r][c])  # 柱状图
            counter += 1
    f.tight_layout()

    f, axes = plt.subplots(4, 3, figsize=(14, 9))
    counter = 0
    for r in range(4):
        for c in range(3):
            if counter > 11:
                continue
            sns.boxplot(data=df, x=req_cols[counter], ax=axes[r][c])  # 箱型图
            counter += 1
    f.tight_layout()

    # 热力图
    corr_matrix = df.corr(method='spearman')
    plt.figure(figsize=(15, 10))
    sns.heatmap(corr_matrix,
                annot=True, annot_kws={'size': 14},
                fmt='.2f', cmap='Pastel1',
                mask=np.triu(corr_matrix))

    plt.show()
