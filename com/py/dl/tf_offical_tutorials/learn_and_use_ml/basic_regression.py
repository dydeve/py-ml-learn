from __future__ import absolute_import, division, print_function, unicode_literals

import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras

print(tf.__version__)

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 100)

# get data
dataset_path = keras.utils.get_file("auto-mpg.data",
                                    "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")

print(dataset_path)  # /Users/xmly/.keras/datasets/auto-mpg.data

column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']

# DataFrame
raw_dataset = pd.read_csv(dataset_path, names=column_names,
                          na_values="?", comment='\t',
                          sep=" ", skipinitialspace=True)

dataset = raw_dataset.copy()
print(dataset.tail())  # DataFrame

print("dataset.shape before clean", dataset.shape)  # (398, 8)
# clean data
print(dataset.isna().sum())
"""
MPG             0
Cylinders       0
Displacement    0
Horsepower      6
Weight          0
Acceleration    0
Model Year      0
Origin          0
dtype: int64
"""
dataset = dataset.dropna()
print("dataset.shape after clean", dataset.shape)  # (392, 8)

# The "Origin" column is really categorical, not numeric. So convert that to a one-hot:
origin = dataset.pop('Origin')  # Return item and drop from frame.
# origin is Series
dataset['USA'] = (origin == 1) * 1.0
dataset['Europe'] = (origin == 2) * 1.0
dataset['Japan'] = (origin == 3) * 1.0
print(dataset.tail())
"""
      MPG  Cylinders  Displacement  Horsepower  ...  Model Year  USA  Europe  Japan
393  27.0          4         140.0        86.0  ...          82  1.0     0.0    0.0
394  44.0          4          97.0        52.0  ...          82  0.0     1.0    0.0
395  32.0          4         135.0        84.0  ...          82  1.0     0.0    0.0
396  28.0          4         120.0        79.0  ...          82  1.0     0.0    0.0
397  31.0          4         119.0        82.0  ...          82  1.0     0.0    0.0
"""

# Split the data into train and test
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# Inspect the data 诊断数据
# Have a quick look at the joint distribution(联合分布) of a few pairs of columns from the training set.
sns.pairplot(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde")
plt.show()

# Also look at the overall statistics:
train_stats = train_dataset.describe()
train_stats.pop("MPG")
train_stats = train_stats.transpose()  # 转置
print(train_stats)

# Split features from labels

train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')

"""
Normalize the data
Look again at the train_stats block above and note how different the ranges of each feature are.

It is good practice to normalize features that use different scales and ranges. 
Although the model might converge without feature normalization, 
it makes training more difficult, and it makes the resulting model dependent on the choice of units used in the input.
"""


def norm(x):
    return (x - train_stats['mean']) / train_stats['std']  # train_stats已转置 train_stats['mean']是series


normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)


def build_model():
    model = keras.Sequential([
        keras.layers.Dense(64, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]),
        keras.layers.Dense(64, activation=tf.nn.relu),
        keras.layers.Dense(1)
    ])

    optimizer = keras.optimizers.RMSprop(0.01)  # lr
    model.compile(optimizer=optimizer,
                  loss=keras.losses.mse,
                  metrics=['mean_absolute_error', 'mean_squared_error'])

    return model


model = build_model()

# Inspect the model
print(model.summary())

example_batch = normed_train_data[:10]
example_result = model.predict(example_batch)
print(example_result)


# Train the model

# Display training progress by printing a single dot for each completed epoch


class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0:
            print('')
        print('.', end='')


EPOCHS = 1000

history = model.fit(
    normed_train_data, train_labels,
    epochs=EPOCHS, validation_split=0.2, verbose=0,
    callbacks=[PrintDot()])


# Visualize the model's training progress using the stats stored in the history object.
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
print(hist.tail())


def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [MPG]')
    plt.plot(hist['epoch'], hist['mean_absolute_error'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
             label='Val Error')
    plt.ylim([0, 5])
    plt.legend()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$MPG^2$]')
    plt.plot(hist['epoch'], hist['mean_squared_error'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'],
             label='Val Error')
    plt.ylim([0, 20])
    plt.legend()
    plt.show()


plot_history(history)


model = build_model()

# Stop training when a monitored quantity has stopped improving.
# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(normed_train_data, train_labels, epochs=EPOCHS,
                    validation_split=0.2, verbose=0, callbacks=[early_stop, PrintDot()])

plot_history(history)

loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=0)
print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))


# Make predictions
test_predictions = model.predict(normed_test_data)  # shape (78,1)
test_predictions = test_predictions.flatten()  # shape (78,)

plt.clf()
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])
plt.show()


# Let's take a look at the error distribution
error = test_predictions - test_labels
plt.clf()
plt.hist(error, bins=25)
plt.xlabel("Prediction Error [MPG]")
_ = plt.ylabel("Count")
plt.show()
# It's not quite gaussian, but we might expect that because the number of samples is very small.

print("done")
