import keras

print("keras version:", keras.__version__)

from keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data(path="/opt/data/dl/mnist.npz")

print("train_images.shape:", train_images.shape)  # (60000, 28, 28)
print("len(train_labels):", len(train_labels))  # 60000

print(train_labels[0:10])  # ndarray 0-9
print(train_labels)

print("test_images.shape:", test_images.shape)  # (10000, 28, 28)
print("len(test_labels):", len(test_labels))

from keras import models
from keras import layers

network = models.Sequential()  # Linear stack of layers
# 基本都有值，用稠密
network.add(layers.Dense(512, activation="relu", input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))

# optimizer:网络基于数据和损失函数更新自己的机制
# loss:衡量网络在训练数据上的性能
# metrics:训练、测试过程中，评估模型的指标
network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])


# preprocess 60000,28,28 uint8 within [0,255]  -->   60000,28*28 float32 [0,1]
train_images = train_images.reshape((-1, 28 * 28))  # 可用flatten
train_images = train_images.astype('float32') / 255  # 归一化

test_images = test_images.reshape((-1, 28 * 28))
test_images = test_images.astype('float32') / 255

# notice here
from keras.utils import to_categorical

train_labels = to_categorical(train_labels)  # one hot
test_labels = to_categorical(test_labels)

network.fit(train_images, train_labels, epochs=5, batch_size=128)
# Returns the loss value & metrics values for the model in test mode.
test_loss, test_acc = network.evaluate(test_images, test_labels)

print("test_loss", test_loss, "test_acc", test_acc)  # test_loss 0.0634960279740626 test_acc 0.9809


