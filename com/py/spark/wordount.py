# coding:utf-8
from pyspark import SparkConf, SparkContext
import os

PYSPARK_PYTHON = "/usr/local/bin/python3.7"
os.environ["PYSPARK_PYTHON"] = PYSPARK_PYTHON

if __name__ == '__main__':
    conf = SparkConf()
    conf.setMaster("local")
    conf.setAppName("wordCount")
    sc = SparkContext(conf=conf)
    lines = sc.textFile("file:///Users/xmly/PycharmProjects/demo/com/py/spark/word.txt")
    words = lines.flatMap(lambda line: line.split(" "))
    wordPair = words.map(lambda word: (word, 1))
    reduceResult = wordPair.reduceByKey(lambda num0, num1: num0 + num1)
    reduceResult.foreach(lambda pair: print(pair))

    sc.stop()

