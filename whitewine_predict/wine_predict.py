import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

if __name__ == '__main__':

    df = pd.read_csv('winequality-white.csv', sep=';')
    df['quality'] = ['Good' if i >= 7 else 'Bad' for i in df['quality']]
    X = df.iloc[:, :-1].values  # 特征变量
    y = df.iloc[:, -1].values  # 输出结果(真实值)

    ax = df['quality'].value_counts().plot(kind='bar', figsize=(10, 6), fontsize=13, color='#087E8B')
    ax.set_title('Counts of Bad and Good wines', size=20, pad=30)
    ax.set_ylabel('Count', fontsize=14)

    for i in ax.patches:
        ax.text(i.get_x() + 0.19, i.get_height() + 40, str(round(i.get_height(), 2)), fontsize=15)
    plt.show()

    # 分割测试集和训练集,并且是随机抽样
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    # 用随机森林分类器进行训练
    model_RF = RandomForestClassifier(n_estimators=100, random_state=1)
    model_RF.fit(X_train, y_train)
    prob_RF = model_RF.predict_proba(X_test)[:, 1]

    y_test = pd.Series(y_test).replace({'Good': 1, 'Bad': 0})
    roc_score_RF = roc_auc_score(y_test, prob_RF)
    print("准确率:" + str(roc_score_RF))

    # 绘制重要度直方图
    importances = model_RF.feature_importances_
    features = df.columns.values[:-1]
    d = {}
    for i in range(len(features)):
        d[features[i]] = importances[i]
    sorted_items = dict(sorted(d.items(), key=lambda x: -x[1]))
    data = pd.DataFrame()
    data['importances'] = sorted_items.values()
    data['features'] = sorted_items.keys()
    g = sns.barplot(data=data, x='features', y='importances')
    for index, row in data.iterrows():
        g.text(row.name, row.importances, round(row.importances, 3), color='black', ha="center")
    plt.show()
