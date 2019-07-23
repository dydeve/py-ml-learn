#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from keras.datasets import boston_housing

"""
13个特征
Per capita crime rate.
Proportion of residential land zoned for lots over 25,000 square feet.
Proportion of non-retail business acres per town.
Charles River dummy variable (= 1 if tract bounds river; 0 otherwise).
Nitric oxides concentration (parts per 10 million).
Average number of rooms per dwelling.
Proportion of owner-occupied units built prior to 1940.
Weighted distances to five Boston employment centres.
Index of accessibility to radial highways.
Full-value property-tax rate per $10,000.
Pupil-teacher ratio by town.
1000 * (Bk - 0.63) ** 2 where Bk is the proportion of Black people by town.
% lower status of the population.
"""
(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

print(train_data.shape, test_data.shape)  # (404, 13) (102, 13)

# 数据标准化：直接将数据输入神经网络不太好。减去mean，除以std，得到mean-0，dtd-1
mean = train_data.mean(axis=0)
train_data -= mean
# 算的是减去平均数后的标准差
std = train_data.std(axis=0)
train_data /= std  # float32

# 使用训练集上的mean与std
test_data -= mean
test_data /= std

from keras import models
from keras import layers


# 通常训练数据少，过拟合会严重，使用较小的网络可降低过拟合

def build_model():
    # 因为会多次实例化模型，用一个函数来构建
    model = models.Sequential()
    # 特征的长度
    model.add(layers.Dense(64, activation="relu", input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(1))  # 没有激活函数，是个线性层
    # 若mae=0.5 则预测价与实际价相差500$ 单位：千美元
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model


# 因为样本少，验证集小，不同的验证集之间验证分数会有很大方差，无法对模型做可靠评估
# k-fold validation
import numpy as np

k = 4
# /得到浮点数，//为地板除，得到浮点数
num_val_samples = len(train_data) // k
num_epochs = 100
all_scores = []

for i in range(k):
    # print('processing fold %d' % i)
    print("processing fold #", i)
    # 获取第k个分区的验证数据
    val_data = train_data[i * num_val_samples:(i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples:(i + 1) * num_val_samples]

    # 训练集，验证集之外的数据
    partial_train_data = np.concatenate([train_data[:i * num_val_samples], train_data[(i + 1) * num_val_samples:]],
                                        axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples], train_targets[(i + 1) * num_val_samples:]],
        axis=0
    )

    model = build_model()  # 已编译的keras模型
    # 静默模式 0 = silent, 1 = progress bar, 2 = one line per epoch.
    model.fit(partial_train_data, partial_train_targets, epochs=num_epochs, batch_size=1, verbose=0)
    # 用验证集评估模型
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)

    all_scores.append(val_mae)

print(all_scores)
print(np.mean(all_scores))
"""
差异较大
[2.096386241440726, 2.2173714897420145, 2.948959950173255, 2.3867242082510844]
2.4123604724017698
"""

from keras import backend as K

# some memory clean-up
K.clear_session()

num_epochs = 500
all_mae_histories = []  # 保存每折的验证结果

for i in range(k):
    print("processing fold #", i)
    val_data = train_data[i * num_val_samples:(i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples:(i + 1) * num_val_samples]

    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples],
         train_data[(i + 1) * num_val_samples:]],
        axis=0)

    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples],
         train_targets[(i + 1) * num_val_samples:]],
        axis=0)

    # batch_size=16时，两张图大致都是下降
    # 书中，以及jupyter里，batch_size=1，在第二张图，大概80 epochs后，val mae 上升，过拟合
    model = build_model()
    history = model.fit(
        partial_train_data,
        partial_train_targets,
        validation_data=(val_data, val_targets),
        epochs=num_epochs,
        batch_size=1,
        verbose=0)
    # size 500(与num_epochs相同)
    mae_history = history.history['val_mean_absolute_error']
    all_mae_histories.append(mae_history)  # 4 * 500

# average_mae_history = [
#     np.mean(all_mae_histories[i]) for i in range(num_epochs)
# ]
# 以epoch编号为横轴，k为纵轴，沿着epoch，计算k的均值。np.mean(axis=0)
average_mae_history = [
    np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]

# 等价于
all_mean=[]
for i in range(num_epochs):
    all=[]
    for x in all_mae_histories:
        all.append(x[i])
    all_mean.append(np.mean(all))

import matplotlib.pyplot as plt

plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()

"""
因为纵轴跨度大，数据方差大，难以看清规律，重绘一张
1. 删除前10个数据，因为和其他点取值范围不同
2. 将每个点替换为前面数据点的指数移动平均数，让曲线光滑
"""


def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


smooth_mae_history = smooth_curve(average_mae_history[10:])

plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()


# Get a fresh, compiled model.
model = build_model()
# Train it on the entirety of the data.
model.fit(train_data, train_targets,
          epochs=80, batch_size=16, verbose=0)
test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)

print(test_mae_score)

