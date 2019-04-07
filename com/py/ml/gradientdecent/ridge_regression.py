# -*- coding: UTF-8 -*-

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import ElasticNet

X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

ridge_reg = Ridge(alpha=1, solver="auto")
ridge_reg.fit(X, y)

print(ridge_reg)

"""
ridge_reg = Ridge(alpha=1, solver='sag')
ridge_reg.fit(X, y)
# print(ridge_reg.predict(1.5))
print(ridge_reg.intercept_)
print(ridge_reg.coef_)
"""



sgd_reg = SGDRegressor(penalty='l2', n_iter=1000)
sgd_reg.fit(X, y.ravel())
#print(sgd_reg.predict(1.5))
print("W0=", sgd_reg.intercept_)
print("W1=", sgd_reg.coef_)

lasso_reg = Lasso(alpha=0.15)
lasso_reg.fit(X, y)
print(lasso_reg.predict(1.5))
print(lasso_reg.coef_)

sgd_reg = SGDRegressor(penalty='l1', max_iter=1000)
sgd_reg.fit(X, y.ravel())
print(sgd_reg.predict(1.5))
print(sgd_reg.coef_)

elastic_net = ElasticNet(alpha=0.0001, l1_ratio=0.15)
elastic_net.fit(X, y)
print(elastic_net.predict(1.5))

sgd_reg = SGDRegressor(penalty='elasticnet', max_iter=1000)
sgd_reg.fit(X, y.ravel())
print(sgd_reg.predict(1.5))