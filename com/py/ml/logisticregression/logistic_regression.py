# -*- coding: UTF-8 -*-

import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from time import time

__author__ = 'yasaka'

iris = datasets.load_iris()
print(list(iris.keys()))# cause iris is a dict
print(iris['DESCR']) # Min  Max   Mean    SD (Standard Deviation)  Class Correlation
print(iris['feature_names'])

X = iris['data'][:, 3:]
print(X)

print(iris['target'])
y = iris['target']
# y = (iris['target'] == 2).astype(np.int)
print(y)


# Utility function to report best scores

def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


start = time()
param_grid = {"tol": [1e-4, 1e-3, 1e-2],
              "C": [0.4, 0.6, 0.8]}
# 'ovr', 'multinomial'
# ovr -> 多分类 LogisticRegression => 多个二分类
# multinomial -> 多分类 => softmax

log_reg = LogisticRegression(multi_class='ovr', solver='sag')

grid_search = GridSearchCV(log_reg, param_grid=param_grid, cv=3) # 3折交叉验证
grid_search.fit(X, y) # 下面可用grid_search替换log_reg
# log_reg.fit(X, y) #对所有数据训练，容易过拟合，超参数也不好改
# print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
#       % (time() - start, len(grid_search.cv_results_['params'])))
# report(grid_search.cv_results_)


X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
print(X_new)

y_proba = log_reg.predict_proba(X_new) # 预测分类概率
y_hat = log_reg.predict(X_new) # 预测分类号
print(y_proba)
print(y_hat)

# 有三个斜率（3,1）、截距(1, 3)；三分类
# 如果把四个特征都算进去，（3，4）（1， 3）3个分类，每个分类四个特征
print("w1", log_reg.coef_)
print("w1", grid_search.best_estimator_) #打印选好超参数的log_reg
print("w0", log_reg.intercept_)

plt.plot(X_new, y_proba[:, 2], 'g-', label='Iris-Virginica')
plt.plot(X_new, y_proba[:, 1], 'r-', label='Iris-Versicolour')
plt.plot(X_new, y_proba[:, 0], 'b--', label='Iris-Setosa')
plt.show()

print(log_reg.predict([[1.7], [1.5]]))


