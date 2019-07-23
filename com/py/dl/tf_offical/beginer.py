#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import tensorflow as tf

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),  # model.output_shape == (None, 28*28)
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
