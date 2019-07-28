#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import tensorflow as tf

# 开始使用 TensorFlow

# 与 2_1_first_look_at_neural_network.py 相似 label没有one-hot，loss用 categorical_crossentropy，optimizer 用 rmsprop

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data(path="/opt/data/dl/mnist.npz")
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),  # model.output_shape == (None, 28*28)
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

"""
batch = 128
Epoch 1/5
60000/60000 [==============================] - 24s 393us/sample - loss: 0.2844 - acc: 0.9186
Epoch 2/5
60000/60000 [==============================] - 21s 353us/sample - loss: 0.1218 - acc: 0.9639
Epoch 3/5
60000/60000 [==============================] - 26s 436us/sample - loss: 0.0832 - acc: 0.9763
Epoch 4/5
60000/60000 [==============================] - 24s 403us/sample - loss: 0.0636 - acc: 0.9807
Epoch 5/5
60000/60000 [==============================] - 25s 423us/sample - loss: 0.0493 - acc: 0.9845
10000/10000 [==============================] - 5s 506us/sample - loss: 0.0620 - acc: 0.9807
loss: 0.06197880537600722 metrics: 0.9807

batch_size default

Epoch 1/5
60000/60000 [==============================] - 44s 737us/sample - loss: 0.2174 - acc: 0.9350
Epoch 2/5
60000/60000 [==============================] - 102s 2ms/sample - loss: 0.0973 - acc: 0.9704
Epoch 3/5
60000/60000 [==============================] - 72s 1ms/sample - loss: 0.0697 - acc: 0.9787
Epoch 4/5
60000/60000 [==============================] - 58s 960us/sample - loss: 0.0542 - acc: 0.9827
Epoch 5/5
60000/60000 [==============================] - 47s 780us/sample - loss: 0.0415 - acc: 0.9867
10000/10000 [==============================] - 2s 156us/sample - loss: 0.0682 - acc: 0.9788
loss: 0.06819601536180125 metrics: 0.9788
"""
model.compile(optimizer='adam',
              loss="sparse_categorical_crossentropy",
              metrics=['accuracy'])  # 稀疏

model.fit(x_train, y_train, epochs=5)  # If unspecified, `batch_size` will default to 32

loss, metrics = model.evaluate(x_test, y_test)
print("loss:", loss, "metrics:", metrics)

