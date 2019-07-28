from __future__ import absolute_import, division, print_function, unicode_literals

import os
import time
import tensorflow as tf
from tensorflow import keras

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

train_labels = train_labels[:1000]
test_labels = test_labels[:1000]

train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0


# saving and loading weights
# Returns a short sequential model
def create_model():
    network = tf.keras.models.Sequential([
        keras.layers.Dense(512, activation=tf.keras.activations.relu, input_shape=(784,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation=tf.keras.activations.softmax)
    ])

    network.compile(optimizer=tf.keras.optimizers.Adam(),
                    loss=tf.keras.losses.sparse_categorical_crossentropy,
                    metrics=['accuracy'])

    return network


# Create a basic model instance
model = create_model()
model.summary()
"""
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense (Dense)                (None, 512)               401920    
_________________________________________________________________
dropout (Dropout)            (None, 512)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 10)                5130      
=================================================================
"""

# Save checkpoints during training

checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
print("checkpoint_dir;")
print(checkpoint_dir)
# Create checkpoint callback
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

# model = create_model()
#
# model.fit(train_images, train_labels, epochs=10,
#           validation_data=(test_images, test_labels),
#           callbacks=[cp_callback])  # pass callback to training

model = create_model()

loss, acc = model.evaluate(test_images, test_labels)
print("Untrained model, accuracy: {:5.2f}%".format(100 * acc))
# Untrained model, accuracy:  8.10%
model.load_weights(checkpoint_path)
loss, acc = model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))
# Restored model, accuracy: 85.90%


# Checkpoint callback options
# include the epoch in the file name. (uses `str.format`)
checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

print("checkpoint_dir:", checkpoint_dir)  # training_2

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, verbose=1, save_weights_only=True,
    # Save weights, every 5-epochs.
    period=5)

# model = create_model()
# model.save_weights(checkpoint_path.format(epoch=0))
# model.fit(train_images, train_labels,
#           epochs=50, callbacks=[cp_callback],
#           validation_data=(test_images, test_labels),
#           verbose=0)


latest = tf.train.latest_checkpoint(checkpoint_dir)
print(latest)  # training_2/cp-0050.ckpt

model = create_model()
model.load_weights(latest)
loss, acc = model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))  # Restored model, accuracy: 87.80%

# Manually save weights
# Save the weights
model.save_weights('./checkpoints/my_checkpoint')

# Restore the weights
model = create_model()
model.load_weights('./checkpoints/my_checkpoint')

loss, acc = model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

# Save the entire model
# The entire model can be saved to a file that contains the weight values,
# the model's configuration, and even the optimizer's configuration (depends on set up).
"""
Keras saves models by inspecting the architecture. 
Currently, it is not able to save TensorFlow optimizers (from tf.train). 
When using those you will need to re-compile the model after loading, 
and you will lose the state of the optimizer.
"""
# As an HDF5 file
# model = create_model()
#
# model.fit(train_images, train_labels, epochs=5)
#
# # Save entire model to a HDF5 file
# model.save('my_model.h5')


# # Recreate the exact same model, including weights and optimizer.
# new_model = keras.models.load_model('my_model.h5')
# new_model.summary()

# loss, acc = new_model.evaluate(test_images, test_labels)
# print("Restored model, accuracy: {:5.2f}%".format(100*acc))


# As a saved_model

model = create_model()

model.fit(train_images, train_labels, epochs=5)

saved_model_path = "./saved_models/" + str(int(time.time()))
tf.contrib.saved_model.save_keras_model(model, saved_model_path)
new_model = tf.contrib.saved_model.load_keras_model(saved_model_path)
new_model.summary()

# Run the restored model.
# The model has to be compiled before evaluating.
# This step is not required if the saved model is only being deployed.

new_model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])

# Evaluate the restored model.
loss, acc = new_model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))
