# -*- coding: UTF-8 -*-

"""
author:dy
date:2019***
"""

import os


def get_user_click(rating_file):
    """
    get user click list
    :param rating_file: input file
    :return: dict: key:user_id value:[item0,item1...]
    """
    if not os.path.exists(rating_file):
        return {}, {}
    fp = open(rating_file)
    num = 0
    user_click = {}
    user_click_time = {}
    for line in fp:
        if num == 0:
            num += 1
            continue
        item = line.strip().split(",")
        if len(item) < 4:
            continue
        [user_id, item_id, rating, timestamp] = item
        if user_id + "_" + item_id not in user_click_time:
            user_click_time[user_id + "_" + item_id] = int(timestamp)
        if float(rating) < 3.0:  # 不够喜欢
            continue
        if user_id not in user_click:
            user_click[user_id] = []
        user_click[user_id].append(item_id)

    fp.close()
    return user_click, user_click_time


def get_graph_from_data(input_file):
    """
    dual dict
    :param input_file:
    :return: dict: {userA:{item_1:1, item_2:1}, item_1:{userA:1, ...}, ...}
    """
    if not os.path.exists(input_file):
        return {}
    graph = {}
    line_num = 0
    score_thr = 4.0
    fp = open(input_file)
    for line in fp:
        if line_num == 0:
            line_num += 1
            continue
        item = line.strip().split(",")
        if len(item) < 3:
            continue
        user_id, item_id, rating = item[0], "item_" + item[1], item[2]
        if float(rating) < score_thr:
            continue
        if user_id not in graph:
            graph[user_id] = {}
        graph[user_id][item_id] = 1
        if item_id not in graph:
            graph[item_id] = {}
        graph[item_id][user_id] = 1
    fp.close()
    return graph


def get_item_info(item_file):
    """
    get item info [title, genres]
    :param item_file: input iteminfo file
    :return: a dict, key:item_id, value:[title, genres]
    """
    if not os.path.exists(item_file):
        return {}
    num = 0
    item_info = {}
    fp = open(item_file)
    for line in fp:
        if num == 0:
            num += 1
            continue
        item = line.strip().split(',')
        if len(item) < 3:
            continue
        elif len(item) == 3:
            item_id, title, genres = item[0], item[1], item[2]
        elif len(item) > 3:
            itemid = item[0]
            genres = item[-1]
            title = ",".join(item[1:-1])
        # if item_id not in item_info:
        item_info[item_id] = [title, genres]
    fp.close()
    return item_info


if __name__ == "__main__":
    user_click = get_user_click("../data/ratings.txt")
    print(len(user_click))
    print(user_click["1"])

    item_info = get_item_info("../data/movies.txt")
    print(len(item_info))





