from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import tensorflow.feature_column as fc

import os
import sys
import pandas
import inspect
import functools
import tempfile
import numpy as np

import matplotlib.pyplot as plt
from IPython.display import clear_output

# estimators是tf扩展性最强、面向生产的模型类型。https://www.tensorflow.org/guide/estimators
# 个人年龄、受教育程度、婚姻状况和职业（即特征）数据在内的普查数据，尝试预测个人年收入是否超过 5 万美元

# enable eager execution to inspect this program as we run it
tf.enable_eager_execution()

# download the official implementation
# ! pip install -q requests
# ! git clone --depth 1 https://github.com/tensorflow/models
models_path = os.path.join('/opt/data/dl', 'models-master')
sys.path.append(models_path)

from official.wide_deep import census_dataset
from official.wide_deep import census_main

census_dataset.download("/tmp/census_data/")

train_file = "/tmp/census_data/adult.data"
test_file = "/tmp/census_data/adult.test"

# read data
# census_dataset._CSV_COLUMNS: ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship', 'race', 'gender', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income_bracket']
train_df = pandas.read_csv(train_file, header=None, names=census_dataset._CSV_COLUMNS)  # (32561, 15)
test_df = pandas.read_csv(test_file, header=None, names=census_dataset._CSV_COLUMNS)  # (16281, 15)

print(train_df.head())

"""
The columns are grouped into two types: categorical and continuous columns:
    - its value can only be one of the categories in a finite set
    -  its value can be any numerical value in a continuous range
"""


# converting data into tensors

def easy_input_function(df, label_key, num_epochs, shuffle, batch_size):
    label = df[label_key]  # series
    # new dictionary initialized from a mapping object's
    #             (key, value) pairs
    # dict(df) -> key:features, value:series
    ds = tf.data.Dataset.from_tensor_slices((dict(df), label))

    if shuffle:
        ds = ds.shuffle(10000)

    # Combines consecutive elements of this dataset into batches.
    ds = ds.batch(batch_size).repeat(num_epochs)

    return ds


ds = easy_input_function(train_df, label_key="income_bracket", num_epochs=5, shuffle=True, batch_size=10)

for feature_batch, label_batch in ds.take(1):
    print('Some feature keys:', list(feature_batch.keys())[:5])
    print()
    print('A batch of Ages  :', feature_batch['age'])
    print()
    print('A batch of Labels:', label_batch)

"""
this approach has severly-limited scalability. Larger datasets should be streamed from disk. 
The census_dataset.input_fn provides an example of how to do this using tf.decode_csv and tf.data.TextLineDataset
"""

# 从磁盘流式传输
print(inspect.getsource(census_dataset.input_fn))  # Return the text of the source code for an object
# def input_fn(data_file, num_epochs, shuffle, batch_size):
#   """Generate an input function for the Estimator."""
#   assert tf.gfile.Exists(data_file), (
#       '%s not found. Please make sure you have run census_dataset.py and '
#       'set the --data_dir argument to the correct path.' % data_file)
#
#   def parse_csv(value):
#     tf.logging.info('Parsing {}'.format(data_file))
#     columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS)
#     features = dict(zip(_CSV_COLUMNS, columns))
#     labels = features.pop('income_bracket')
#     classes = tf.equal(labels, '>50K')  # binary classification
#     return features, classes
#
#   # Extract lines from input files using the Dataset API.
#   dataset = tf.data.TextLineDataset(data_file)
#
#   if shuffle:
#     dataset = dataset.shuffle(buffer_size=_NUM_EXAMPLES['train'])
#
#   dataset = dataset.map(parse_csv, num_parallel_calls=5)
#
#   # We call repeat after shuffling, rather than before, to prevent separate
#   # epochs from blending together.
#   dataset = dataset.repeat(num_epochs)
#   dataset = dataset.batch(batch_size)
#   return dataset

# Estimators 期望 input_fn 没有参数
train_inpf = functools.partial(census_dataset.input_fn, train_file, num_epochs=2, shuffle=True, batch_size=64)
test_inpf = functools.partial(census_dataset.input_fn, test_file, num_epochs=1, shuffle=False, batch_size=64)

# Selecting and Engineering Features for the Model

# Estimators use a system called feature columns to describe how the model should interpret
# each of the raw input features. An Estimator expects a vector of numeric inputs,
# and feature columns describe how the model should convert each feature.

# base feature columns

#  numeric columns
age = fc.numeric_column('age')
print(fc.input_layer(feature_batch, [age]).numpy())

# train and evaluate model using age
classifier = tf.estimator.LinearClassifier(feature_columns=[age])
classifier.train(train_inpf)
result = classifier.evaluate(test_inpf)
print(result)

education_num = tf.feature_column.numeric_column('education_num')
capital_gain = tf.feature_column.numeric_column('capital_gain')
capital_loss = tf.feature_column.numeric_column('capital_loss')
hours_per_week = tf.feature_column.numeric_column('hours_per_week')

my_numeric_columns = [age, education_num, capital_gain, capital_loss, hours_per_week]

numpy = fc.input_layer(feature_batch, my_numeric_columns).numpy()  # ndarray
print(numpy)

classifier = tf.estimator.LinearClassifier(feature_columns=my_numeric_columns)
classifier.train(train_inpf)

result = classifier.evaluate(test_inpf)  # dict

"""
accuracy: 0.7917204
accuracy_baseline: 0.76377374
auc: 0.72355855
auc_precision_recall: 0.55940807
average_loss: 0.5736052
global_step: 1018
label/mean: 0.23622628
loss: 36.623005
precision: 0.9892473
prediction/mean: 0.23960777
recall: 0.11960478
"""
for key, value in sorted(result.items()):
    print('%s: %s' % (key, value))

#  categorical columns

# 需要知道所有的值，会分配一个从0开始的id，适用于数量少的情况
# This creates a sparse one-hot vector from the raw input feature.
relationship = tf.feature_column.categorical_column_with_vocabulary_list(
    'relationship',
    ['Husband', 'Not-in-family', 'Wife', 'Own-child', 'Unmarried', 'Other-relative'])

"""
The input_layer function we're using is designed for DNN models and expects dense inputs. 
To demonstrate the categorical column we must wrap it in a tf.feature_column.indicator_column 
to create the dense one-hot output (Linear Estimators can often skip this dense-step).

Note: the other sparse-to-dense option is tf.feature_column.embedding_column.
"""
print("create the dense one-hot output")
tensor = tf.feature_column.input_layer(feature_batch, [age, tf.feature_column.indicator_column(
    relationship)])  # create the dense one-hot output
"""
tf.Tensor(
[[54.  1.  0.  0.  0.  0.  0.]
 [50.  0.  0.  0.  1.  0.  0.]
 [37.  1.  0.  0.  0.  0.  0.]
 [29.  1.  0.  0.  0.  0.  0.]
 [41.  0.  1.  0.  0.  0.  0.]
 [37.  1.  0.  0.  0.  0.  0.]
 [29.  0.  1.  0.  0.  0.  0.]
 [28.  0.  1.  0.  0.  0.  0.]
 [69.  1.  0.  0.  0.  0.  0.]
 [24.  0.  0.  0.  1.  0.  0.]], shape=(10, 7), dtype=float32)
"""

# If we don't know the set of possible values in advance, use the categorical_column_with_hash_bucket instead
occupation = tf.feature_column.categorical_column_with_hash_bucket(
    'occupation', hash_bucket_size=1000)
# each possible value in the feature column occupation is hashed to an integer ID as we encounter them in training

for item in feature_batch['occupation'].numpy():
    print(item.decode())

occupation_result = fc.input_layer(feature_batch, [fc.indicator_column(occupation)])

print(occupation_result.numpy().shape)  # (10, 1000)

print(tf.argmax(occupation_result, axis=1).numpy())  # 1
print(occupation_result.numpy().max(axis=1))

#
education = tf.feature_column.categorical_column_with_vocabulary_list(
    'education', [
        'Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college',
        'Assoc-acdm', 'Assoc-voc', '7th-8th', 'Doctorate', 'Prof-school',
        '5th-6th', '10th', '1st-4th', 'Preschool', '12th'])

marital_status = tf.feature_column.categorical_column_with_vocabulary_list(
    'marital_status', [
        'Married-civ-spouse', 'Divorced', 'Married-spouse-absent',
        'Never-married', 'Separated', 'Married-AF-spouse', 'Widowed'])

workclass = tf.feature_column.categorical_column_with_vocabulary_list(
    'workclass', [
        'Self-emp-not-inc', 'Private', 'State-gov', 'Federal-gov',
        'Local-gov', '?', 'Self-emp-inc', 'Without-pay', 'Never-worked'])

my_categorical_columns = [relationship, occupation, education, marital_status, workclass]

classifier = tf.estimator.LinearClassifier(feature_columns=my_numeric_columns + my_categorical_columns)
classifier.train(train_inpf)
result = classifier.evaluate(test_inpf)

print("base feature columns here=============")
for key, value in sorted(result.items()):
    print('%s: %s' % (key, value))

# derived feature columns
# Make Continuous Features Categorical through Bucketization
# age与income并不是线性关系，而是一条类抛物线。Bucketization可以把连续特征分成连续的桶，
# 然后把数值特征转成bucket id(像分类特征)

age_buckets = tf.feature_column.bucketized_column(
    age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])

# With bucketing, the model sees each bucket as a one-hot feature
print(tf.feature_column.input_layer(feature_batch, [age, age_buckets]).numpy())

# Learn complex relationships with crossed column
# Using each base feature column separately may not be enough to explain the data.
# For example, the correlation between education and the label (earning > 50,000 dollars)
# may be different for different occupations.

education_x_occupation = tf.feature_column.crossed_column(['education', 'occupation'], hash_bucket_size=1000)

# Each constituent column can be either a base feature column that is categorical (SparseColumn),
# a bucketized real-valued feature column, or even another CrossColumn
age_buckets_x_education_x_occupation = tf.feature_column.crossed_column(
    [age_buckets, 'education', 'occupation'], hash_bucket_size=1000)

# 这些base、derived feature_columns 都是FeatureColumn的子类

base_columns = [
    education, marital_status, relationship, workclass, occupation, age_buckets,
]

crossed_columns = [
    tf.feature_column.crossed_column(
        ['education', 'occupation'], hash_bucket_size=1000),
    tf.feature_column.crossed_column(
        [age_buckets, 'education', 'occupation'], hash_bucket_size=1000),
]

model = tf.estimator.LinearClassifier(
    model_dir=tempfile.mkdtemp(),  # learned model files are stored in model_dir
    feature_columns=base_columns + crossed_columns,
    optimizer=tf.train.FtrlOptimizer(learning_rate=0.1)
)

# train and evaluate the model

train_inpf = functools.partial(census_dataset.input_fn, train_file,
                               num_epochs=40, shuffle=True, batch_size=64)

model.train(train_inpf)  # 调用父类EstimatorV2的train方法
results = model.evaluate(test_inpf)

for key, value in sorted(results.items()):
    print('%s: %0.2f' % (key, value))

# pandas
predict_df = test_df[:20].copy()
pred_iter = model.predict(lambda: easy_input_function(predict_df, label_key='income_bracket',
                                                      num_epochs=1, shuffle=False, batch_size=10))

classes = np.array(['<=50K', '>50K'])
pred_class_id = []

for pred_dict in pred_iter:
    pred_class_id.append(pred_dict['class_ids'])

predict_df['predicted_class'] = classes[np.array(pred_class_id)]
predict_df['correct'] = predict_df['predicted_class'] == predict_df['income_bracket']

print(predict_df[['income_bracket', 'predicted_class', 'correct']])

model_l1 = tf.estimator.LinearClassifier(
    feature_columns=base_columns + crossed_columns,
    optimizer=tf.train.FtrlOptimizer(
        learning_rate=0.1,
        l1_regularization_strength=10.0,
        l2_regularization_strength=0.0))

model_l1.train(train_inpf)

results = model_l1.evaluate(test_inpf)
for key in sorted(results):
    print('%s: %0.2f' % (key, results[key]))

model_l2 = tf.estimator.LinearClassifier(
    feature_columns=base_columns + crossed_columns,
    optimizer=tf.train.FtrlOptimizer(
        learning_rate=0.1,
        l1_regularization_strength=0.0,
        l2_regularization_strength=10.0))

model_l2.train(train_inpf)

results = model_l2.evaluate(test_inpf)
clear_output()
for key in sorted(results):
    print('%s: %0.2f' % (key, results[key]))


# These regularized models don't perform much better than the base model.
# Let's look at the model's weight distributions to better see the effect of the regularization

def get_flat_weights(model):
    weight_names = [
        name for name in model.get_variable_names()
        if "linear_model" in name and "Ftrl" not in name]

    weight_values = [model.get_variable_value(name) for name in weight_names]

    weights_flat = np.concatenate([item.flatten() for item in weight_values], axis=0)

    return weights_flat


weights_flat = get_flat_weights(model)
weights_flat_l1 = get_flat_weights(model_l1)
weights_flat_l2 = get_flat_weights(model_l2)

"""
The models have many zero-valued weights caused by unused hash bins 
(there are many more hash bins than categories in some columns). 
We can mask these weights when viewing the weight distributions:
"""
weight_mask = weights_flat != 0

weights_base = weights_flat[weight_mask]
weights_l1 = weights_flat_l1[weight_mask]
weights_l2 = weights_flat_l2[weight_mask]

plt.figure()
_ = plt.hist(weights_base, bins=np.linspace(-3, 3, 30))
plt.title('Base Model')
plt.ylim([0, 500])

plt.figure()
_ = plt.hist(weights_l1, bins=np.linspace(-3, 3, 30))
plt.title('L1 - Regularization')
plt.ylim([0, 500])

plt.figure()
_ = plt.hist(weights_l2, bins=np.linspace(-3, 3, 30))
plt.title('L2 - Regularization')
_ = plt.ylim([0, 500])
