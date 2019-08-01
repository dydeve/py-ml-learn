from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

# Automatic differentiation and gradient tape
# 自动微分 https://en.wikipedia.org/wiki/Automatic_differentiation

tf.enable_eager_execution()

# Gradient tapes
x = tf.ones((2, 2))

with tf.GradientTape() as t:
    t.watch(x)  # Ensures that `tensor` is being traced by this tape
    y = tf.reduce_sum(x)
    z = tf.multiply(y, y)

# Derivative of z with respect to the original input tensor x
"""
\frac{\partial z}{\partial y}*\frac{\partial y}{\partial x}=\frac{\partial y^2}{\partial y}*\frac{\partial y}{\partial x}
=dy^2/dy * dy/dx = 2*y * 1 = 8.0
"""
dz_dx = t.gradient(z, x)
for i in [0, 1]:
    for j in [0, 1]:
        assert dz_dx[i][j].numpy() == 8.0

print(dz_dx)
"""
tf.Tensor(
[[8. 8.]
 [8. 8.]], shape=(2, 2), dtype=float32)
"""

x = tf.ones((2, 2))

with tf.GradientTape() as t:
    t.watch(x)
    y = tf.reduce_sum(x)
    z = tf.multiply(y, y)

# Use the tape to compute the derivative of z with respect to the
# intermediate value y.
dz_dy = t.gradient(z, y)
assert dz_dy.numpy() == 8.0

"""
Recording control flow

Because tapes record operations as they are executed, 
Python control flow (using ifs and whiles for example) is naturally handled
"""


def f(xx, yy):
    output = 1.0
    for ii in range(yy):
        if ii > 1 and ii < 5:
            output = tf.multiply(output, xx)
    return output


def grad(xx, yy):
    with tf.GradientTape() as tt:
        tt.watch(xx)
        out = f(xx, yy)
    return tt.gradient(out, xx)


x = tf.convert_to_tensor(2.0)

assert grad(x, 6).numpy() == 12.0  # d x^3 / d x
assert grad(x, 5).numpy() == 12.0  # d x^3 / d x
assert grad(x, 4).numpy() == 4.0  # d x^2 / d x

# Higher-order gradients
x = tf.Variable(1.0)  # Create a Tensorflow variable initialized to 1.0

with tf.GradientTape() as t:
    with tf.GradientTape() as t2:
        y = x * x * x
    # Compute the gradient inside the 't' context manager
    # which means the gradient computation is differentiable as well.
    dy_dx = t2.gradient(y, x)  # d x^3 / d x = 3*x^2
d2y_dx2 = t.gradient(dy_dx, x)  # d 3*x^2 / d x = 6*x

assert dy_dx.numpy() == 3.0
assert d2y_dx2.numpy() == 6.0
