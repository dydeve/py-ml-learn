from __future__ import absolute_import, division, print_function

import tensorflow as tf
import numpy as np
import time

"""
  Eager execution provides an imperative interface to TensorFlow. With eager
  execution enabled, TensorFlow functions execute operations immediately (as
  opposed to adding to a graph to be executed later in a `tf.Session`) and
  return concrete values (as opposed to symbolic references to a node in a
  computational graph).
"""
tf.enable_eager_execution()

"""
A Tensor is a multi-dimensional array. Similar to NumPy ndarray objects, Tensor objects have a data type and a shape. 
Additionally, Tensors can reside in accelerator (like GPU) memory. 
TensorFlow offers a rich library of operations (tf.add, tf.matmul, tf.linalg.inv etc.) that consume and produce Tensors. 
These operations automatically convert native Python types.
"""
print(tf.add(1, 2))  # tf.Tensor(3, shape=(), dtype=int32)
print(tf.add([1, 2], [3, 4]))  # tf.Tensor([4 6], shape=(2,), dtype=int32)
print(tf.square(5))  # tf.Tensor(25, shape=(), dtype=int32)
print(tf.reduce_sum([1, 2, 3]))  # tf.Tensor(6, shape=(), dtype=int32)
print(tf.encode_base64("hello world"))  # tf.Tensor(b'aGVsbG8gd29ybGQ', shape=(), dtype=string)

# Operator overloading is also supported
print(tf.square(2) + tf.square(3))  # tf.Tensor(13, shape=(), dtype=int32)

# Each Tensor has a shape and a datatype
x = tf.matmul([[1]], [[2, 3]])
print(x.shape)
print(x.dtype)

# The most obvious differences between NumPy arrays and TensorFlow Tensors are:
#   1. Tensors can be backed by accelerator memory (like GPU, TPU).
#   2. Tensors are immutable.

"""
Tensors can be explicitly converted to NumPy ndarrays by invoking the .numpy() method on them. 
These conversions are typically cheap as the array and Tensor share the underlying memory representation if possible. 
However, sharing the underlying representation isn't always possible since the Tensor may be hosted in GPU memory 
while NumPy arrays are always backed by host memory, and the conversion will thus involve a copy from GPU to host memory.
"""

ndarray = np.ones([3, 3])

print("TensorFlow operations convert numpy arrays to Tensors automatically")
tensor = tf.multiply(ndarray, 42)
print(tensor)

print("And NumPy operations convert Tensors to numpy arrays automatically")
print(np.add(tensor, 1))

print("The .numpy() method explicitly converts a Tensor to a numpy array")
print(tensor.numpy())

"""
Many TensorFlow operations can be accelerated by using the GPU for computation. 
Without any annotations, TensorFlow automatically decides whether to use the 
GPU or CPU for an operation (and copies the tensor between CPU and GPU memory if necessary). 
Tensors produced by an operation are typically backed by the memory of the device on which the operation executed.
"""

x = tf.random_uniform([3, 3])

# depends on nvidia gpu or tensorflow-gpu
print("Is there a GPU available: "),
print(tf.test.is_gpu_available())

print("Is the Tensor on GPU #0:  "),
print(x.device.endswith('GPU:0'))


"""
TensorFlow automatically decides which device to execute an operation, and copies Tensors to that device if needed. 
However, TensorFlow operations can be explicitly placed on specific devices using the tf.device context manager.
"""


def time_matmul(x):
    start = time.time()
    for loop in range(10):
        tf.matmul(x, x)

    result = time.time() - start

    print("10 loops: {:0.2f}ms".format(1000 * result))


# Force execution on CPU
print("On CPU:")
with tf.device("CPU:0"):
    x = tf.random_uniform([1000, 1000])
    assert x.device.endswith("CPU:0")
    time_matmul(x)

# Force execution on GPU #0 if available
if tf.test.is_gpu_available():
    with tf.device("GPU:0"):  # Or GPU:1 for the 2nd GPU, GPU:2 for the 3rd etc.
        x = tf.random_uniform([1000, 1000])
        assert x.device.endswith("GPU:0")
        time_matmul(x)
