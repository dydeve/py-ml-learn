import keras

print(keras.__version__)

from keras.datasets import imdb

"""
num_words: max number of words to include. Words are ranked
            by how often they occur (in the training set) and only
            the most frequent words are kept
            保留前10000个高频单词，得到的向量数据不会太大
"""
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
train_data[0]
train_labels[0]

max([max(sequence) for sequence in train_data])
# word_index is a dictionary mapping words to an integer index
word_index = imdb.get_word_index()
# We reverse it, mapping integer indices to words
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
# We decode the review; note that our indices were offset by 3
# because 0, 1 and 2 are reserved indices for "padding", "start of sequence", and "unknown".
decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])

# print(decoded_review)

import numpy as np

"""
不能直接将整数序列直接输入神经网络，需转换为张量
way1:填充列表，使其具有相同长度
way2:one-hot编码
"""
def vectorize_sequences(sequences, dimension=10000):
    # Create an all-zero matrix of shape (len(sequences), dimension)
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.  # set specific indices of results[i] to 1s
    return results


# Our vectorized training data
x_train = vectorize_sequences(train_data)
# Our vectorized test data
x_test = vectorize_sequences(test_data)

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(16, activation="relu", input_shape=(10000,)))
model.add(layers.Dense(16, activation="relu"))
"""
<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

$$
S(x)=\frac{1}{1+e^{-x}}
$$

\begin{equation}
S(x)=\frac{1}{1+e^{-x}}
\end{equation}
"""
model.add(layers.Dense(1, activation="sigmoid"))

"""
output = relu(dot(W, input) + b)
16个隐藏单元对应权重矩阵W的形状为(input_dimension, 16),
与W点乘相当于将输入数据投影到16维表示空间
"""

# model.compile(optimizer='rmsprop',
#               loss='binary_crossentropy',
#               metrics=['accuracy'])

from keras import optimizers
from keras import metrics
from keras import losses

# 对于二分类  采用二元交叉熵
model.compile(optimizer=optimizers.RMSprop(lr=0.001),
              loss=losses.binary_crossentropy,
              metrics=[metrics.binary_accuracy])

# 验证集
x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]


history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))

history_dict = history.history
print("=====================")
print(history_dict.keys())
print("=====================")


import matplotlib.pyplot as plt

#acc = history.history['acc']
binary_accuracy = history.history['binary_accuracy']
val_binary_accuracy = history.history['val_binary_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']


epochs = range(1, len(binary_accuracy) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


plt.clf()   # clear figure

plt.plot(epochs, binary_accuracy, 'bo', label='Training acc')
plt.plot(epochs, val_binary_accuracy, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# 具上图所示，发生过拟合.可在3轮后停止训练
history = model.fit(x_train, y_train, epochs=4, batch_size=512)  # epochs = 4
# Returns the loss value & metrics values for the model in test mode
result = model.evaluate(x_test, y_test)
print(result)

epochs = range(1, len(history.history['binary_accuracy']) + 1)
plt.plot(epochs, history.history['binary_accuracy'], 'ro', label='accuracy')
plt.plot(epochs, history.history['loss'], 'r', label='loss')
plt.show()

#
y_test_hat = model.predict(x_test)
print(y_test_hat)

