# coding=utf-8
import sys

# print(sys.getdefaultencoding())
from pyspark import SparkConf, SparkContext

import com.py.spark.pys


def webUserPair(line):
    words = line.split("\t")
    userId = words[2]
    website = words[4]
    return (website, userId)


def pv(lines):
    sitePair = lines.map(lambda line: (line.split("\t")[4], 1))
    reducedRes = sitePair.reduceByKey(lambda v1, v2: v1 + v2)
    sorted = reducedRes.sortBy(lambda tuple: tuple[1], ascending=False)
    sorted.foreach(lambda one: print(one))


def uv(lines):
    uniqueUserIdWithWebsite = lines.map(lambda line: line.split("\t")[2] + "_" + line.split("\t")[4]).distinct()
    uv = uniqueUserIdWithWebsite.map(lambda userIdWebsite: (userIdWebsite.split("_")[1], 1)).reduceByKey(
        lambda v1, v2: v1 + v2)
    sortedUv = uv.sortBy(lambda t: t[1], ascending=False)  # 'NoneType' object is not callable  .foreach(print())
    sortedUv.foreach(lambda x: print(x))


# ('taobao.com', 8005)
# ('alipay.com', 8001)
# ('tmall.com', 7955)
# ('google.com', 7839)
# ('facebook.com', 7801)
# ('toutiao.com', 7762)
# ('amazon.com', 7673)
#
# **************************************************
# ('taobao.com', 5212)
# ('alipay.com', 5168)
# ('tmall.com', 5151)
# ('google.com', 5125)
# ('facebook.com', 5092)
# ('amazon.com', 5061)
# ('toutiao.com', 5033)

# 2019-03-02	192.168.39.175	69882	xianggang	tmall.com	login

def getTop2City(one):
    webSite = one[0]
    cities = one[1]
    cityNumdict = {}
    for city in cities:
        # if (cityNumdict.get(city))
        if city in cityNumdict:
            cityNumdict[city] += 1
        else:
            cityNumdict[city] = 1

    # item: kv [0, 1]
    sortedList = sorted(cityNumdict.items(), key=lambda item: item[1], reverse=True)

    resultList = []
    if len(sortedList) < 2:
        resultList = sortedList
    else:
        for i in range(0, 2):
            # IndexError: list assignment index out of range
            # resultList[i] = sortedList[i]
            resultList.append(sortedList[i])

    return (webSite, resultList)


def uvExceptBJ(lines):
    distinct = lines.filter(lambda line: line.split("\t")[3] != 'beijing').map(
        lambda line: line.split("\t")[1] + "_" + line.split("\t")[4]).distinct()
    reduceResult = distinct.map(lambda distinct: (distinct.split("_")[1], 1)).reduceByKey(lambda v1, v2: v1 + v2)
    result = reduceResult.sortBy(lambda tp: tp[1], ascending=False)
    result.foreach(lambda one: print(one))


def top2Address(lines):
    # (web_address, count)
    webAddressAndNum = lines.map(lambda line: (line.split("\t")[4] + "_" + line.split("\t")[3], 1)) \
        .reduceByKey(lambda v1, v2: v1 + v2)
    # (web, (address, count))
    webAndAddressCount = webAddressAndNum.map(
        lambda tuple: (tuple[0].split("_")[0], (tuple[0].split("_")[1], tuple[1])))
    sorted = webAndAddressCount.sortBy(lambda tuple: tuple[1][1], ascending=False)
    sorted.foreach(lambda x: print(x))
    webAddresses = lines.map(lambda line: (line.split("\t")[4], line.split("\t")[3])).groupByKey()
    list = webAddresses.map(lambda one: getTop2City(one)).collect()
    for i in list:
        print(i)


def getTop3User(lines):
    site_uid_count = lines.map(lambda line: (line.split("\t")[2], line.split("\t")[4])).groupByKey().flatMap(
        lambda one: getSiteInfo(one))
    result = site_uid_count.groupByKey().map(lambda one: getCurSiteTop3User(one)).collect()
    for elem in result:
        print(elem)


def getSiteInfo(one):
    userid = one[0]
    sites = one[1]
    dic = {}
    for site in sites:
        if site in dic:
            dic[site] += 1
        else:
            dic[site] = 1
    resultList = []
    for site, count in dic.items():
        resultList.append((site, (userid, count)))
    return resultList


def getCurSiteTop3User(one):
    site = one[0]
    userid_count_Iterable = one[1]
    top3List = ["", "", ""]
    for userid_count in userid_count_Iterable:
        for i in range(0, len(top3List)):
            if top3List[i] == "":
                top3List[i] = userid_count
                break
            else:
                if userid_count[1] > top3List[i][1]:
                    for j in range(2, i, -1):
                        top3List[j] = top3List[j - 1]
                    top3List[i] = userid_count
                break
    return site, top3List


if __name__ == '__main__':
    conf = SparkConf().setMaster("local").setAppName("pvuv")
    sc = SparkContext(conf=conf)
    lines = sc.textFile("file:///Users/xmly/PycharmProjects/demo/com/py/spark/view.txt").cache()
    # webUserPair = lines.map(lambda line: webUserPair(line)).cache()
    # pv = webUserPair.reduceByKey(lambda id0, id1: 1 + 1)
    # uv = webUserPair.mapV
    # pv(lines)
    print("*" * 50)
    # uv(lines)
    top2Address(lines)
