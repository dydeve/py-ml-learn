#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from keras.datasets import reuters

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

print(len(train_data))
print(len(test_data))

word_index = reuters.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
# Note that our indices were offset by 3
# because 0, 1 and 2 are reserved indices for "padding", "start of sequence", and "unknown".
decoded_newswire = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])
print(decoded_newswire)


# prepare data
import numpy as np

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

# Our vectorized training data 训练数据向量化
x_train = vectorize_sequences(train_data)
# Our vectorized test data
x_test = vectorize_sequences(test_data)

def to_one_hot(labels, dimension=46):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results

# Our vectorized training labels 训练标签向量化
one_hot_train_labels = to_one_hot(train_labels)
# Our vectorized test labels
one_hot_test_labels = to_one_hot(test_labels)

# Note that there is a built-in way to do this in Keras, which you have already seen in action in our MNIST example:
from keras.utils.np_utils import to_categorical

# 将标签向量化
one_hot_train_labels = to_categorical(train_labels)
one_hot_test_labels = to_categorical(test_labels)


from keras import models
from keras import layers

model = models.Sequential()
# 之前16维中间层，无法学会区分46个不同类别
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
# 预测46个新闻分类, 各类别概率之和为1
"""
\begin{equation}
P(y=j | \mathbf{x})=\frac{e^{\mathbf{x}^{\top} \mathbf{w}_{j}}}{\sum_{k=1}^{K} e^{\mathbf{x}^{\top} \mathbf{w}_{k}}}
\end{equation}
"""
model.add(layers.Dense(46, activation='softmax'))

# 分类交叉熵 loss函数针对label
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


x_val = x_train[:1000]
partial_x_train = x_train[1000:]

y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))


import matplotlib.pyplot as plt

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.clf()   # clear figure

acc = history.history['acc']
val_acc = history.history['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


# It seems that the network starts overfitting after 8 epochs. Let's train a new network from scratch for 8 epochs, then let's evaluate it on the test set:
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
# 当隐藏单元数降为4时，精度下降：试图将大量信息(足够将足够恢复成46个类别的分隔超平面)压缩到维度很小的中间空间。
# 网络可将大部分信息塞入四维表示中，但不是全部信息
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(partial_x_train,
          partial_y_train,
          epochs=8,
          batch_size=512,
          validation_data=(x_val, y_val))
# 评估测试集
results = model.evaluate(x_test, one_hot_test_labels)  # the loss value & metrics values

print(results)
# 两个隐藏层 原版 64*2
# 第二层4个隐藏单元     [1.53084806385363,   0.6313446126447017]
# 第二层64个隐藏单元    [0.9853645672462715, 0.7858414959928762]
# 第二层128个隐藏单元   [0.9638390003944866, 0.7960819234724885]

# 一个隐藏层 64*1      [0.8960939963163272, 0.7978628673196795]
# 三个隐藏层 64*3      [1.0453951844973746, 0.7756010685928783]

import copy

test_labels_copy = copy.copy(test_labels)
np.random.shuffle(test_labels_copy)
hits_array = np.array(test_labels) == np.array(test_labels_copy)
# 完全随机精度约为19%
print(float(np.sum(hits_array)) / len(test_labels))

predictions = model.predict(x_test)
print(predictions[0].shape)  # (46, )
print(np.sum(predictions[0]))  # ~~1
print(np.argmax(predictions[0]))  # 概率最大的分类


# A different way to handle the labels and the loss
# 将其转换成整数张量
y_train = np.array(train_labels)
y_test = np.array(test_labels)
# 对于整数标签，使用 sparse_categorical_crossentropy 稀疏
# This new loss function is still mathematically the same as categorical_crossentropy; it just has a different interface.
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['acc'])


"""
1. 分为N类，最后一层为大小为N的dense层
2. 单标签、多分类，最后一层使用softmax激活，输出N个类别的概率分布
3. 使用分类交叉熵，将网络输出的概率分布与目标的真实分布之间的距离最小化
4. 处理多分类标签
4.1 one-hot编码，并使用categorical_crossentropy做损失函数
4.2 将标签编码为整数，使用sparse_categorical_crossentropy做损失函数
"""