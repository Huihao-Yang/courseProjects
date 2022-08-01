import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

if __name__ == '__main__':
    df = pd.read_csv('winequality-white.csv', sep=';')
    df['quality'] = ['Good' if i >= 7 else 'Bed' for i in df['quality']]
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    model_RF = RandomForestClassifier()

    parameter_forest = {  # 参数列表
        'n_estimators': range(100, 1100, 100),
        'max_depth': range(1, 13)
    }

    grid = GridSearchCV(model_RF, parameter_forest)  # 网格搜索法
    grid.fit(X_train, y_train)

    print(grid.best_params_,  # 相关参数
          grid.best_score_,  # 评分
          grid.best_estimator_,  # 估计器
          grid.best_index_)  # 索引
