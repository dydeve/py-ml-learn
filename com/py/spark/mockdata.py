# coding=utf-8
import time
import random
import sys

def getUserId():
    userId = random.randint(10000, 99999)
    return str(userId)


def writeLongToPath(log, path):
    with open(path, "a+") as file:
        file.writelines(log + "\n")


def mock(path):
    date = time.strftime("%Y-%m-%d")
    list = ["192.168", str(random.randint(0, 255)), str(random.randint(0, 255))]
    ip = ".".join(list)

    userId = getUserId()
    locations = ["shanghai", "hangzhou", "beijing", "shenzhen", "xianggang"]
    location = locations[random.randint(0, 4)]

    for j in range(0, random.randint(1, 10)):
        websites = ["google.com", "alipay.com", "toutiao.com", "taobao.com", "tmall.com", "facebook.com", "amazon.com"]
        website = websites[random.randint(0, 6)]

        operations = ["register", "view", "login", "logout", "buy", "comment", "jump"]
        operation = operations[random.randint(0, 6)]

        oneInfo = date + "\t" + ip + "\t" + userId + "\t" + location + "\t" + website + "\t" + operation
        print(oneInfo)

        writeLongToPath(oneInfo, path)

if __name__ == '__main__':
    #outputPath = sys.argv[1]
    pwd = sys.argv[0]
    path = "/Users/xmly/PycharmProjects/demo/com/py/spark/view.txt"
    for i in range(1, 10000):
        mock(path)