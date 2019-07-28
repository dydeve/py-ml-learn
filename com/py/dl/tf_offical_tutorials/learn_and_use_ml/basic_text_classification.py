import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

"""
https://www.tensorflow.org/tutorials/keras/basic_text_classification
"""

imdb = keras.datasets.imdb

# 参数 num_words=10000 会保留训练数据中出现频次在前 10000 位的字词。为确保数据规模处于可管理的水平，罕见字词将被舍弃
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(path="/opt/data/dl/imdb.npz", num_words=10000)

# 探索数据
# 0 表示负面影评，1 表示正面影评
print("train set shape:{}, train set shape:{}".format(train_data.shape, train_labels.shape))
# train set shape:(25000,) --> 25000个list, train set shape:(25000,)

# 影评文本已转换为整数，其中每个整数都表示字典中的一个特定字词
print(train_data[0])  # [1, 14, 22, 16, 43, 530, 973, 1622...]

# 影评的长度可能会有所不同。以下代码显示了第一条和第二条影评中的字词数。由于神经网络的输入必须具有相同长度，因此我们稍后需要解决此问题
print("len(train_data[0]):{}, len(train_data[1]):{}".format(len(train_data[0]), len(train_data[1])))  # 218, 189

# A dictionary mapping words to an integer index
word_index = imdb.get_word_index()

# word_index = dict([(value, key) for (key, value) in word_index.items()])

word_index = {k: (v + 3) for (k, v) in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

# type set: reverse_word_index = {(v, k) for (k, v) in word_index.items()}
reverse_word_index = dict([(v, k) for (k, v) in word_index.items()])


def decode_review(text):
    return " ".join(reverse_word_index.get(i, '?') for i in text)


print(decode_review(train_data[0]))

"""
准备数据
影评（整数数组）必须转换为张量，然后才能馈送到神经网络中。我们可以通过以下两种方法实现这种转换：

对数组进行独热编码，将它们转换为由 0 和 1 构成的向量。例如，序列 [3, 5] 将变成一个 10000 维的向量，除索引 3 和 5 转换为 1 之外，其余全转换为 0。然后，将它作为网络的第一层，一个可以处理浮点向量数据的密集层。
不过，这种方法会占用大量内存，需要一个大小为 num_words * num_reviews 的矩阵。

或者，我们可以填充数组，使它们都具有相同的长度，然后创建一个形状为 max_length * num_reviews 的整数张量。我们可以使用一个能够处理这种形状的嵌入层作为网络中的第一层
"""


# one-hot
def vectorize_sequence(sequences, dimension=1000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results


from keras.utils.np_utils import to_categorical

# to_categorical(train_data)

# padding

# pad_sequences函数将长度标准化
train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        maxlen=256,
                                                        value=word_index['<PAD>'],
                                                        padding='post')  # 在后面填充

test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)

# len(train_data[0]):256, len(train_data[1]):256
print("len(train_data[0]):{}, len(train_data[1]):{}".format(len(train_data[0]), len(train_data[1])))

print(train_data[0])  # 末尾有填充 0

# 构建模型

# input shape is the vocabulary count used for the movie reviews (10,000 words)

"""
按顺序堆叠各个层以构建分类器:
1. 第一层是 Embedding 层。该层会在整数编码的词汇表中查找每个字词-索引的嵌入向量。
   模型在接受训练时会学习这些向量。这些向量会向输出数组添加一个维度。生成的维度为：(batch, sequence, embedding)
2. 接下来，一个 GlobalAveragePooling1D 层通过对序列维度求平均值，针对每个样本返回一个长度固定的输出向量。
   这样，模型便能够以尽可能简单的方式处理各种长度的输入。
3. 该长度固定的输出向量会传入一个全连接 (Dense) 层（包含 16 个隐藏单元）
4. 最后一层与单个输出节点密集连接。应用 sigmoid 激活函数后，结果是介于 0 到 1 之间的浮点值，表示概率或置信水平
"""
vocab_size = 10000
model = keras.Sequential()
# Turns positive integers (indexes) into dense vectors of fixed size.
# This layer can only be used as the first layer in a model.
#   Input shape:
#       2D tensor with shape: `(batch_size, input_length)`.
#
#   Output shape:
#       3D tensor with shape: `(batch_size, input_length, output_dim)`
model.add(keras.layers.Embedding(vocab_size, 16))  # input_dim，output_dim
model.add(keras.layers.GlobalAveragePooling1D())  # [0.328175099272728, 0.87244]下面的result
"""
model.add(keras.layers.Dense(16, activation=tf.nn.relu, input_shape=(256,)))
[7.90534213897705, 0.505]所绘图像不平滑
"""
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))

model.summary()
"""
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding (Embedding)        (None, None, 16)          160000    
_________________________________________________________________
global_average_pooling1d (Gl (None, 16)                0         
_________________________________________________________________
dense (Dense)                (None, 16)                272       
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 17        
=================================================================
Total params: 160,289
Trainable params: 160,289
Non-trainable params: 0
_________________________________________________________________
"""

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss="binary_crossentropy",
              metrics=['accuracy'])

x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]


history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)


# Returns the loss value & metrics values for the model in test mode.
results = model.evaluate(test_data, test_labels)

# [0.328175099272728, 0.87244]
print(results)

# 创建准确率和损失随时间变化的图

history_dict = history.history
print("history_dict.keys():{}".format(history_dict.keys()))

history.ep
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

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
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

# 出现过拟合
