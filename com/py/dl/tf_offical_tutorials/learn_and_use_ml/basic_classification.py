from __future__ import absolute_import, division, print_function, unicode_literals
# To write a Python 2/3 compatible codebase, the first step is to add this line to the top of each module:

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

"""
标签	类别
0	T-shirt/top
1	Trouser
2	Pullover
3	Dress
4	Coat
5	Sandal
6	Shirt
7	Sneaker
8	Bag
9	Ankle boot
"""
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 探索数据
print(train_images.shape)  # (60000, 28, 28) dtype=uint8
print("train_images.shape:", train_images.shape)
print(train_labels)
print("train_labels.shape:", train_labels.shape)  # (60000,)

# 数据预处理

# 显示第一个图像
plt.figure()  # Create a new figure.
plt.imshow(train_images[0])
plt.colorbar()
# plt.grid()
plt.grid(False)
plt.show()

# 归一化
train_images = train_images / 255.0
test_images = test_images / 255.0

# 显示前25个图像，显示类名.验证数据格式是否正确，我们是否已准备好构建和训练网络
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)  # subplot(nrows, ncols, index, **kwargs)  注意：nrows*ncols = 25
    plt.xticks([])
    plt.yticks([])
    """
    cmap : str or `~matplotlib.colors.Colormap`, optional
            The Colormap instance or registered colormap name used to map
            scalar data to colors. This parameter is ignored for RGB(A) data.
            Defaults to :rc:`image.cmap`.
    """
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

# plt.figure(figsize=(10, 10))
# for i in range(25):
#     plt.subplot(5, 5, i + 1)  # subplot(nrows, ncols, index, **kwargs)
#     print(plt.xticks())
#     print(plt.yticks())
#     plt.imshow(train_images[i])
#     plt.xlabel(class_names[train_labels[i]])
# plt.show()

# 设置网络层
"""
```python
      model = Sequential()
      model.add(Convolution2D(64, 3, 3,
                              border_mode='same',
                              input_shape=(3, 32, 32)))
      # now: model.output_shape == (None, 64, 32, 32)

      model.add(Flatten())
      # now: model.output_shape == (None, 65536)
```
"""
model = keras.Sequential([
    # Think of this layer as unstacking rows of pixels in the image and lining them up. This layer has no parameters to learn; it only reformats the data.
    keras.layers.Flatten(input_shape=(28, 28)),  # (28, 28) ---> 784  可以将这个网络层视为它将图像中未堆叠的像素排列在一起
    keras.layers.Dense(128, activation=tf.nn.relu),  # 全连接层  128个神经元/节点
    keras.layers.Dense(10, activation=tf.nn.softmax)  # 10个概率分布，总和为1
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 训练模型
model.fit(train_images, train_labels, epochs=5)
# trainset: loss: 0.3556 - acc: 0.8747

# 评估准确率 test_loss 0.35562063887119294 test_acc 0.8747
test_loss, test_acc = model.evaluate(test_images, test_labels)

print("test_loss", test_loss, "test_acc", test_acc)

# 进行预测
# Numpy array(s) of predictions.
predictions = model.predict(test_images)

# [2.9516755e-06 8.4146535e-08 1.4497009e-06 4.2565625e-06 5.4365495e-07
#  1.6415132e-02 1.1787711e-05 1.6207859e-01 1.7277378e-05 8.2146794e-01]
print("predictions[0]:", predictions[0])
print("np.argmax(predictions[0]):", np.argmax(predictions[0]))  # 9

print("real label for index of 0:", test_labels[0])


def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]), color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color='#777777')
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


i = 0
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions, test_labels)
plt.show()

i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions,  test_labels)
plt.show()


# 绘制前X个测试图像，预测标签和真实标签
# 以蓝色显示正确的预测，红色显示不正确的预测
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions, test_labels, test_images)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions, test_labels)
plt.show()


img = test_images[0]  # 减了一维
print(img.shape)

# tf.keras模型经过优化，可以一次性对批量,或者一个集合的数据进行预测。因此，即使我们使用单个图像，我们也需要将其添加到列表中:
# Insert a new axis that will appear at the `axis` position in the expanded array shape
img = np.expand_dims(img, 0)  # 在0axis增加一轴
print(img.shape)  # (1, 28, 28)


# 预测图像
predictions_single = model.predict(img)

print(predictions_single)  # (1, 10)


plot_value_array(0, predictions_single, test_labels)
plt.xticks(range(10), class_names, rotation=45)  # 旋转
plt.show()

prediction_result = np.argmax(predictions_single[0])
print(prediction_result)  # 9


