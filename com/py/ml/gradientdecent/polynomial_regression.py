# -*- coding: UTF-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X ** 2 + X + 2 + np.random.randn(m, 1)
# [1.99075453] [[0.90664707 0.50156886]]
plt.plot(X, y, 'b.')

d = {1: 'g-', 2: 'r+', 10: 'y*'}
for i in d:
    '''
    include_bias:
    If True (default), then include a bias column, the feature in which
        all polynomial powers are zero (i.e. a column of ones - acts as an
        intercept term in a linear model).
        
    所以多了一条轴 x=1
    '''
    poly_features = PolynomialFeatures(degree=i, include_bias=False)#不要w0
    X_poly = poly_features.fit_transform(X)
    print(X[0])
    print(X_poly[0])
    print(X_poly[:, 0])

    lin_reg = LinearRegression(fit_intercept=True)#当上面的include_bias为true时，变成垂直于平面的维度，视图为x=1，当fit_intercept=false，接触y=0
    lin_reg.fit(X_poly, y)
    print(lin_reg.intercept_, lin_reg.coef_)

    y_predict = lin_reg.predict(X_poly)
    plt.plot(X_poly[:, 0], y_predict, d[i])#只算x轴
plt.show()

plt.plot(X, y, 'b.')

d = {1: 'g-', 2: 'r+', 10: 'y*'}
for i in d:
    '''
    include_bias:
    If True (default), then include a bias column, the feature in which
        all polynomial powers are zero (i.e. a column of ones - acts as an
        intercept term in a linear model).

    所以多了一条轴 x=1
    '''
    poly_features = PolynomialFeatures(degree=i, include_bias=True)  # 不要w0
    X_poly = poly_features.fit_transform(X)
    print(X[0])
    print(X_poly[0])
    print(X_poly[:, 0])

    lin_reg = LinearRegression(
        fit_intercept=True)  # 当上面的include_bias为true时，变成垂直于平面的维度，视图为x=1，当fit_intercept=false，接触y=0
    lin_reg.fit(X_poly, y)
    print(lin_reg.intercept_, lin_reg.coef_)

    y_predict = lin_reg.predict(X_poly)
    plt.plot(X_poly[:, 0], y_predict, d[i])  # 只算x轴
plt.show()

plt.plot(X, y, 'b.')
d = {1: 'g-', 2: 'r+', 10: 'y*'}
for i in d:
    '''
    include_bias:
    If True (default), then include a bias column, the feature in which
        all polynomial powers are zero (i.e. a column of ones - acts as an
        intercept term in a linear model).

    所以多了一条轴 x=1
    '''
    poly_features = PolynomialFeatures(degree=i, include_bias=True)
    X_poly = poly_features.fit_transform(X)
    print(X[0])
    print(X_poly[0])
    print(X_poly[:, 0])

    lin_reg = LinearRegression(
        fit_intercept=True)  # 当上面的include_bias为true时，变成垂直于平面的维度，视图为x=1，当fit_intercept=false，接触y=0
    lin_reg.fit(X_poly, y)
    print(lin_reg.intercept_, lin_reg.coef_)

    y_predict = lin_reg.predict(X_poly)
    plt.plot(X_poly[:, 1], y_predict, d[i])  # 只算x轴
plt.show()