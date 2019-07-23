# print("hello py")
# print("i", "love", "you")
#
# name = input()
# print(name)
#
# name = input("please enter your name:")
# print("hello:", name)
from sklearn.pipeline import Pipeline


def fun1():
    print("i love you")
    print("fuck you")


fun1()


def fun2(a):
    b = a + 1
    c = a + 2
    return b, c


d, e = fun2(0)
print(d, e)
a = 999999999999999999999999999999999999999999999999
print(a + 1)

a = [{"1": 2, "3": 3}, {"1": 1, "3": 3}]
a.sort(key=lambda x: x["1"])
print(a)


class Singleton:
    __instance = None

    def __new__(cls):
        if not cls.__instance:
            cls.__instance = object.__new__(cls)
        return cls.__instance


a = Singleton()
b = Singleton()
print(id(a))
print(id(b))


class Person(object):
    def __init__(self, name):
        self.name = name

    def work(self, type_axe):
        print("%s 开始工作" % self.name)
        # axe = SteelAxe()
        # axe.cut_tree()
        axe = Factory.class_create_axe(type_axe)
        axe.cut_tree()


class Axe(object):
    def cut_tree(self):
        print("cut tree")


class StoneAxe(Axe):
    def cut_tree(self):
        print("cut tree by stone")


class SteelAxe(Axe):
    def cut_tree(self):
        print("cut tree by steel")


class Factory(object):

    @classmethod
    def class_create_axe(cls, type_axe):
        if type_axe == "stone":
            return StoneAxe()
        elif type_axe == "stell":
            return SteelAxe()
        else:
            print("参数")

    @staticmethod
    def create_axe(cls):
        pass


try:
    print("begin")
    # print(num)
    # open("fuck.txt", "r")
    print("end")
except (FileNotFoundError, NameError) as e:
    print("error")
    print(e)
else:
    print("no exception")
finally:
    print("finally---------")

print("here")


class ToShortInuputException(Exception):
    def __init__(self, length, asLeastLength):
        self.length = length
        self.asLeastLength = asLeastLength


def main():
    try:
        s = input("please input")
        if len(s) < 3:
            raise ToShortInuputException(len(s), 3)
    except ToShortInuputException as e:
        print("short input with length%d, at least %d, e:%s" % (e.length, e.asLeastLength, e))
    else:
        print("no exception")

# main()

import os
from os import path
#from os import *
print(os.getcwd())
print(path.isfile("/opt/fuck.txt"))

import numpy
import matplotlib as mpl
import matplotlib.pyplot as plt

#mpl.rc()

import matplotlib.pyplot as plt
import pandas as pd

def load_housing_data(housing_path=""):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

# Pipeline([
#         ('imputer', SimpleImputer(strategy="median")),
#         ('attribs_adder', FunctionTransformer(add_extra_features, validate=False)),
#         ('std_scaler', StandardScaler()),
#     ])
#
# housing = load_housing_data()
# housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
#     s=housing["population"]/100, label="population", figsize=(10,7),
#     c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
#     sharex=False)
# np.random.permutation(10)

print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

rng = np.random.RandomState(42)

# Generate train data
X = 0.3 * rng.randn(100, 2)
X_train = np.r_[X + 2, X - 2]
# Generate some regular novel observations
X = 0.3 * rng.randn(20, 2)
X_test = np.r_[X + 2, X - 2]
# Generate some abnormal novel observations
X_outliers = rng.uniform(low=-4, high=4, size=(20, 2))

# fit the model
clf = IsolationForest(behaviour='new', max_samples=100,
                      random_state=rng, contamination='auto')
clf.fit(X_train)
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)
y_pred_outliers = clf.predict(X_outliers)

# plot the line, the samples, and the nearest vectors to the plane
xx, yy = np.meshgrid(np.linspace(-5, 5, 50), np.linspace(-5, 5, 50))
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.title("IsolationForest")
plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)

b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c='white',
                 s=20, edgecolor='k')
b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c='green',
                 s=20, edgecolor='k')
c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c='red',
                s=20, edgecolor='k')
plt.axis('tight')
plt.xlim((-5, 5))
plt.ylim((-5, 5))
plt.legend([b1, b2, c],
           ["training observations",
            "new regular observations", "new abnormal observations"],
           loc="upper left")
plt.show()