from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import matplotlib.pyplot as plt

tf.enable_eager_execution()

"""
Tensors in TensorFlow are immutable stateless objects. 
Machine learning models, however, need to have changing state
you can choose to rely on the fact that Python is a stateful programming language:
"""
# Using python state
x = tf.zeros([10, 10])
x += 2  # This is equivalent to x = x + 2, which does not mutate the original value of x
print(x)

v = tf.Variable(1.0)
assert v.numpy() == 1.0

# Re-assign the value
v.assign(3.0)
assert v.numpy() == 3.0

# Use `v` in a TensorFlow operation like tf.square() and reassign
v.assign(tf.square(v))
assert v.numpy() == 9.0  # AssertionError

"""
Computations using Variables are automatically traced when computing gradients. 
For Variables representing embeddings TensorFlow will do sparse updates by default, 
which are more computation and memory efficient.

Using Variables is also a way to quickly let a reader of your code know that this piece of state is mutable.
"""


# fitting a linear regression: f(x) = x * W + b

# Define the model
class Model(object):

    def __init__(self):
        # Initialize variable to (5.0, 0.0)
        # In practice, these should be initialized to random values.
        self.W = tf.Variable(5.0)
        self.b = tf.Variable(0.0)

    def __call__(self, x):
        return self.W * x + self.b


model = Model()
assert model(3).numpy() == 15.0


# Define a loss function
# A loss function measures how well the output of a model
# for a given input matches the desired output. Let's use the standard L2 loss.
def loss(predicted_y, desired_y):
    return tf.reduce_mean(tf.square(predicted_y - desired_y))


# Obtain training data
# Let's synthesize the training data with some noise.


TRUE_W = 3.0
TRUE_b = 2.0
NUM_EXAMPLES = 1000

inputs = tf.random_normal([NUM_EXAMPLES])
noise = tf.random_normal([NUM_EXAMPLES])

outputs = TRUE_W * inputs + noise + TRUE_b

# Before we train the model let's visualize where the model stands right now.
# We'll plot the model's predictions in red and the training data in blue.


plt.scatter(inputs, outputs, c='b')  # tensor can auto convert to numpy
plt.scatter(inputs, model(inputs), c='r')  # input[i] 与 output[i] 对的上
plt.show()

print('Current loss: '),
print(loss(model(inputs), outputs).numpy())  # 8.786571


# Define a training loop


def train(current_model, current_inputs, current_outputs, learning_rate):
    with tf.GradientTape() as t:  # Variables are automatically traced when computing gradients.
        l2_loss = tf.reduce_mean(tf.square(current_model(current_inputs) - current_outputs))
    dW, db = t.gradient(l2_loss, [current_model.W, current_model.b])
    current_model.W.assign_sub(learning_rate * dW)
    current_model.b.assign_sub(learning_rate * db)


model = Model()
# Collect the history of W-values and b-values to plot later
Ws, bs = [], []
epochs = range(10)
for epoch in epochs:
    Ws.append(model.W.numpy())
    bs.append(model.b.numpy())
    current_loss = loss(model(inputs), outputs)

    train(model, inputs, outputs, learning_rate=0.1)
    print('Epoch %2d: W=%1.2f b=%1.2f loss=%2.5f' %
          (epoch, Ws[-1], bs[-1], current_loss))


# let's plot it all
plt.plot(epochs, Ws, 'r',
         epochs, bs, 'b')
plt.plot([TRUE_W] * len(epochs), 'r--',
         [TRUE_b] * len(epochs), 'b--')
plt.legend(['W', 'b', 'true W', 'true b'])
plt.show()

plt.clf()
plt.plot(epochs, Ws, 'r', label='W')
plt.plot(epochs, bs, 'b', label='b')
plt.plot([TRUE_W] * len(epochs), 'r--', label='true W')
plt.plot([TRUE_b] * len(epochs), 'b--', label='true b')
plt.legend(['W', 'b', 'true W', 'true b'])
plt.show()














