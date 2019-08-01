from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

tf.enable_eager_execution()

# In the tf.keras.layers package, layers are objects. To construct a layer,
# simply construct the object. Most layers take as a first argument the number
# of output dimensions / channels.
layer = tf.keras.layers.Dense(100)
# The number of input dimensions is often unnecessary, as it can be inferred
# the first time the layer is used, but it can be provided if you want to
# specify it manually, which is useful in some complex models.
# input_shape不必要，可以推断。但有些复杂模型会需要
layer = tf.keras.layers.Dense(10, input_shape=(None, 5))  # 最后一维变为10

# layers in keras: https://www.tensorflow.org/api_docs/python/tf/keras/layers
# include Dense (a fully-connected layer), Conv2D, LSTM, BatchNormalization, Dropout, and many others

# To use a layer, simply call it.
zeros = tf.zeros([10, 5])
print(zeros)
print(layer(zeros))  # function: Dense.call     inputs = ops.convert_to_tensor(inputs)

# Layers have many useful methods. For example, you can inspect all variables
# in a layer using `layer.variables` and trainable variables using
# `layer.trainable_variables`. In this case a fully-connected layer
# will have variables for weights and biases.
print(layer.variables)  # x(10, 5)  w(5, 10) b(10)

# The variables are also accessible through nice accessors
print(layer.kernel, layer.bias)

# Implementing custom layers

print("=======", "Implementing custom layers", "=======")


class MyDenseLayer(tf.keras.layers.Layer):

    # def __init__(self, trainable=True, name=None, dtype=None, **kwargs):
    #     super().__init__(trainable, name, dtype, **kwargs)
    def __init__(self, num_outputs):  # do all input-independent initialization
        super(MyDenseLayer, self).__init__()  # 参考Dense
        self.num_outputs = num_outputs

    """
    the advantage of creating them in build is that it enables late variable creation based on the shape of the inputs the layer will operate on
    """
    def build(self, input_shape):  # you know the shapes of the input tensors and can do the rest of the initialization
        self.kernel = self.add_variable("kernel",
                                        shape=[int(input_shape[-1]),
                                               self.num_outputs])

    def call(self, input):  # do the forward computation
        return tf.matmul(input, self.kernel)


layer = MyDenseLayer(10)
print(layer(tf.zeros([10, 5])))
print(layer.trainable_variables)


# Models: composing layers
class ResnetIdentityBlock(tf.keras.Model):
    def __init__(self, kernel_size, filters):
        super(ResnetIdentityBlock, self).__init__(name='')
        filters1, filters2, filters3 = filters  # 1, 2, 3 = [1, 2, 3]

        self.conv2a = tf.keras.layers.Conv2D(filters1, (1, 1))
        self.bn2a = tf.keras.layers.BatchNormalization()

        self.conv2b = tf.keras.layers.Conv2D(filters2, kernel_size, padding="same")
        self.bn2b = tf.keras.layers.BatchNormalization()

        self.conv2c = tf.keras.layers.Conv2D(filters3, (1, 1))
        self.bn2c = tf.keras.layers.BatchNormalization()


    def call(self, input_tensor, training=False):
        x = self.conv2a(input_tensor)  # shape(1, 2, 3, 1)
        x = self.bn2a(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2b(x)  # shape(1, 2, 3, 2)
        x = self.bn2b(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2c(x)  # shape(1, 2, 3, 3)
        x = self.bn2c(x, training=training)

        x += input_tensor
        return tf.nn.relu(x)


block = ResnetIdentityBlock(1, [1, 2, 3])
print(block(tf.zeros([1, 2, 3, 3])))
print([x.name for x in block.trainable_variables])
"""
[
'resnet_identity_block/conv2d/kernel:0', 
'resnet_identity_block/conv2d/bias:0', 
'resnet_identity_block/batch_normalization_v1/gamma:0', 
'resnet_identity_block/batch_normalization_v1/beta:0', 
'resnet_identity_block/conv2d_1/kernel:0', 
'resnet_identity_block/conv2d_1/bias:0', 
'resnet_identity_block/batch_normalization_v1_1/gamma:0', 
'resnet_identity_block/batch_normalization_v1_1/beta:0', 
'resnet_identity_block/conv2d_2/kernel:0', 
'resnet_identity_block/conv2d_2/bias:0', 
'resnet_identity_block/batch_normalization_v1_2/gamma:0', 
'resnet_identity_block/batch_normalization_v1_2/beta:0'
]
"""


"""
Much of the time, however, models which compose many layers simply call one layer after the other. 
This can be done in very little code using tf.keras.Sequential
"""
my_seq = tf.keras.Sequential([tf.keras.layers.Conv2D(1, (1, 1)),
                              tf.keras.layers.BatchNormalization(),
                              tf.keras.layers.Conv2D(2, 1, padding='same'),
                              tf.keras.layers.BatchNormalization(),
                              tf.keras.layers.Conv2D(3, (1, 1)),
                              tf.keras.layers.BatchNormalization()])

print(my_seq(tf.zeros([1, 2, 3, 3])))




