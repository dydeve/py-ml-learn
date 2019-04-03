# print("hello py")
# print("i", "love", "you")
#
# name = input()
# print(name)
#
# name = input("please enter your name:")
# print("hello:", name)

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