from __future__ import absolute_import, division, print_function, unicode_literals

import os
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import contrib

tf.enable_eager_execution()

print("Tensorflow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))


# download the dataset

train_dataset_url = "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv"

train_dataset_fp = tf.keras.utils.get_file(fname=os.path.basename(train_dataset_url),
                                           origin=train_dataset_url)

print("Local copy of the dataset file: {}".format(train_dataset_fp))


# head -n5 /Users/xmly/.keras/datasets/iris_training.csv
"""
120,4,setosa,versicolor,virginica
6.4,2.8,5.6,2.2,2
5.0,2.3,3.3,1.0,1
4.9,2.5,4.5,1.7,2
4.9,3.1,1.5,0.1,0
There are 120 total examples. Each example has four features and one of three possible label names
"""
# column order in CSV file
column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
feature_names = column_names[:-1]
label_name = column_names[-1]

print("Features: {}".format(feature_names))
print("Label: {}".format(label_name))

class_names = ['Iris setosa', 'Iris versicolor', 'Iris virginica']


# Create a tf.data.Dataset
"""
TensorFlow's Dataset API handles many common cases for loading data into a model. 
This is a high-level API for reading data and transforming it into a form used for training
"""

batch_size = 32
# train_dataset = tf.contrib.data.make_csv_dataset(
#     train_dataset_fp,
#     batch_size,
#     column_names=column_names,
#     label_name=label_name,
#     num_epochs=1)
#
# shuffle=True,
#     shuffle_buffer_size=10000,
train_dataset = tf.data.experimental.make_csv_dataset(
    train_dataset_fp,
    batch_size,
    column_names=column_names,
    label_name=label_name,
    num_epochs=1)

"""
The make_csv_dataset function returns a tf.data.Dataset of (features, label) pairs, 
where features is a dictionary: {'feature_name': value}

With eager execution enabled, these Dataset objects are iterable. Let's look at a batch of features:
"""

features, labels = next(iter(train_dataset))  # dict  4 key, value shape=(32,)
print(features)
print(labels)

plt.scatter(features['petal_length'].numpy(),
            features['sepal_length'].numpy(),
            c=labels.numpy(),
            cmap='viridis')

plt.xlabel("Petal length")
plt.ylabel("Sepal length")
plt.show()

"""
<class 'list'>: [
<tf.Tensor 'arg2:0' shape=(?,) dtype=float32>, 
<tf.Tensor 'arg3:0' shape=(?,) dtype=float32>, 
<tf.Tensor 'arg0:0' shape=(?,) dtype=float32>, 
<tf.Tensor 'arg1:0' shape=(?,) dtype=float32>
]
"""


def pack_features_vector(features, labels):
    """Pack the features into a single array."""
    features = tf.stack(list(features.values()), axis=1)
    return features, labels  # features shape=(?, 4)  -> (batch_size, num_features)


train_dataset = train_dataset.map(pack_features_vector)


features, labels = next(iter(train_dataset))

print(features)


# creating model using keras

"""
The ideal number of hidden layers and neurons depends on the problem and the dataset. 
Like many aspects of machine learning, picking the best shape of the neural network 
requires a mixture of knowledge and experimentation. As a rule of thumb, 
increasing the number of hidden layers and neurons typically creates a more powerful model, 
which requires more data to train effectively.
"""
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(4,)),  # num_features
    tf.keras.layers.Dense(10, activation=tf.nn.relu),
    tf.keras.layers.Dense(3)
])

# using the model
predictions = model(features)
print(predictions[:5])
print(tf.nn.softmax(predictions[:5]))

print("Prediction: {}".format(tf.argmax(predictions, axis=1)))
print("    Labels: {}".format(labels))


# training the model

# define the loss and gradient function
def loss(model, x, y):
    y_ = model(x)
    return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)


l = loss(model, features, labels)
print("Loss test: {}".format(l))


def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)  # trainable_variables 六个变量:3个kernel 3个bias


# create an optimizer
# 优化器动图 https://cs231n.github.io/assets/nn3/opt1.gif
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
global_step = tf.Variable(0)

loss_value, grads = grad(model, features, labels)  # grads；list, len=6, 对应上面的6个变量
print("Step: {}, Initial Loss: {}".format(global_step.numpy(),
                                          loss_value.numpy()))

optimizer.apply_gradients(zip(grads, model.trainable_variables), global_step)
print("Step: {},         Loss: {}".format(global_step.numpy(),
                                          loss(model, features, labels).numpy()))  # optimizer.apply_gradients 修改了model













