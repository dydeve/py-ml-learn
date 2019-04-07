# coding=utf-8

import numpy as np

X = 2 * np.random.randn(100, 1)
Y = 4 + 3 * X + np.random.randn(100, 1)
X_b = np.c_[np.ones((100, 1)), X]

n_epochs = 500

t0, t1 = 1, 10  # 超参数
#t0, t1 = 0, 50  # 超参数

m = 100


def learning_schedule(t):
    return t0 / (t + t1)


theta = np.random.randn(2, 1)

# for ecoph in range(n_epochs):#轮次
#     for i in range(m):
#         random_index = np.random.randint(m)
#         xi = X_b[random_index:random_index + 1]
#         yi = Y[random_index:random_index + 1]
#         gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
#         learning_rate = learning_schedule(ecoph * m + i)
#         theta = theta - learning_rate * gradients

for ecoph in range(n_epochs):#轮次
    for i in range(m):
        random_index = np.random.randint(m)
        xi = X_b[random_index:random_index + 1]
        yi = Y[random_index:random_index + 1]
        gradients = xi.T.dot(xi.dot(theta) - yi)
        learning_rate = learning_schedule(ecoph * m + i)
        theta = theta - learning_rate * gradients
print(theta)


