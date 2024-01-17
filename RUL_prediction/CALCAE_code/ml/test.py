if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt

    x = np.random.uniform(-3, 3, size=100)
    X = x.reshape(-1, 1)
    y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, size=100)

    # 多项式回归最主要的事情是数据预处理
    from sklearn.preprocessing import PolynomialFeatures

    poly = PolynomialFeatures(degree=2)  # 添加二次幂特征
    poly.fit(X)
    X2 = poly.transform(X)

    from sklearn.linear_model import LinearRegression

    lin_reg2 = LinearRegression()
    lin_reg2.fit(X2, y)
    y_predict2 = lin_reg2.predict(X2)

    plt.scatter(x, y)
    plt.plot(np.sort(x), y_predict2[np.argsort(x)], color='r')
    plt.show()
