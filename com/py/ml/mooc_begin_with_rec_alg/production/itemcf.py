# -*- coding: UTF-8 -*-

import sys
import math
import operator
from __future__ import division

# sys.path.append("../util")
import com.py.ml.mooc_begin_with_rec_alg.util.reader as reader


def base_contribute_score():
    """
    item cf base contribution score by user
    :return:
    """
    return 1


def upgrad_one_contribute_score(user_total_click_num):
    """
    item cf update sim contribution score by user
    :param user_total_click_num:
    :return:
    """
    return 1/math.log(1 + user_total_click_num)


def upgrad_two_contribute_score(click_time_one, click_time_two):
    """
    item cf update two sim contribution score by user
    :param click_time_one:
    :param click_time_two:
    :return:
    """
    delata_time = abs(click_time_two - click_time_one)  # 基于秒，在秒的维度，score差别不大
    total_sec = 24 * 60 * 60
    delata_time = delata_time / total_sec
    return 1 / (delata_time + 1)


def cal_sim_info(user_click, user_click_time):
    """

    :param user_click:dict key:user_id, value:[item_id,...]
    :param user_click_time: dict key:user_id_item_id, value:timestamp
    :return:
    """
    co_appear = {}
    item_user_click_time = {}
    for user_id, item_ids in user_click.items():
        for index_i in range(0, len(item_ids)):
            item_id_i = item_ids[index_i]
            item_user_click_time.setdefault(item_id_i, 0)
            item_user_click_time[item_id_i] += 1

            for index_j in range(index_i + 1, len(item_ids)):
                item_id_j = item_ids[index_j]

                if user_id + "_" + item_id_i not in user_click_time:
                    click_time_one = 0
                else:
                    click_time_one = user_click_time[user_id + "_" + item_id_i]
                if user_id + "_" + item_id_j not in user_click_time:
                    click_time_two = 0
                else:
                    click_time_two = user_click_time[user_id + "_" + item_id_j]

                co_appear.setdefault(item_id_i, {})
                co_appear[item_id_i].setdefault(item_id_j, 0)
                co_appear[item_id_i][item_id_j] += upgrad_two_contribute_score(click_time_one, click_time_two)

                co_appear.setdefault(item_id_j, {})
                co_appear[item_id_j].setdefault(item_id_i, 0)
                co_appear[item_id_j][item_id_i] += upgrad_two_contribute_score(click_time_one, click_time_two)

    item_sim_score = {}  # dict key: item_id_i, value:dict: key:item_id_j, value:sim_score
    for item_id_i, relate_item in co_appear.items():
        for item_id_j, co_time in relate_item.items():
            sim_score = co_time / math.sqrt(item_user_click_time[item_id_i]*item_user_click_time[item_id_j])
            item_sim_score.setdefault(item_id_i, {})
            item_sim_score[item_id_i].setdefault(item_id_j, 0)
            item_sim_score[item_id_i][item_id_j] += sim_score

    item_sim_score_sorted = {}
    for item_id in item_sim_score:
        item_sim_score_sorted[item_id] = sorted(item_sim_score[item_id].iteritems(), key = \
                                                operator.itemetter(1), reverse=True)

    # dict: key: item_id_i, value: List(item_id_j, score)
    return item_sim_score_sorted


def cal_recom_result(sim_info, user_click):
    """
    recom by itemcf
    :param sim_info: dict: key: item_id_i, value: List(item_id_j, score)
    :param user_click: dict: key:user_id value:[item0,item1...]
    :return: dict: key:user_id, value:dict {key: item_id, value: recom_score}
    """
    recent_click_num = 3
    top_k = 5
    recom_info = {}
    for user in user_click:
        click_list = user_click[user]
        recom_info.setdefault(user, {})
        for item_id in click_list[:recent_click_num]:
            if item_id not in sim_info:
                continue
            for item_score_cp in sim_info[item_id][:top_k]:
                sim_item_id = item_score_cp[0]
                sim_item_score = item_score_cp[1]
                recom_info[user][sim_item_id] = sim_item_score

    return recom_info


def debug_itemsim(item_info, sim_info):
    """
    show itemsim info
    :param item_info:dict, key:item_id, value:[title, genres]
    :param sim_info:dict, key:item_id, value:dict
    :return:
    """
    fixed_item_id = "1"
    if fixed_item_id not in item_info:
        print("invalid item_id")
        return
    [title_fix, genres_fix] = item_info[fixed_item_id]
    for cp in sim_info[fixed_item_id]:  # [:5]:
        item_id_sim = cp[0]
        sim_score = cp[1]
        if item_id_sim not in item_info:
            continue
        [title, genres] = item_info[item_id_sim]
        print(title_fix + "\t" + genres_fix + "\tsim:" + title + "\t" + genres + "\t" + str(sim_score))


def debug_recomresult(recom_result, item_info):
    """

    :param recom_result: item_id_i, item_id_j, score
    :param item_info: item_id, [title, genres]
    :return:
    """
    user_id = "1"
    if user_id not in recom_result:
        print("invalid result")
        return
    for cp in sorted(recom_result[user_id].iteritems(), key=operator.itemgetter(1), reverse=True):
        item_id, score = cp
        if item_id not in item_info:
            continue
        print(".".join(item_info[item_id]) + "\t" + str(score))


def main_flow():
    """
    main flow of itemcf
    :return:
    """
    user_click, user_click_time = reader.get_user_click("../data/rating.txt")  # dict: key:user_id value:[item0,item1...]
    item_info = reader.get_item_info("../data/movies.txt")

    sim_info = cal_sim_info(user_click, user_click_time)
    debug_itemsim(item_info, sim_info)
    recom_result = cal_recom_result(sim_info, user_click)
    print(recom_result["1"])
    debug_recomresult(recom_result, item_info)


if __name__ == "__main__":
    main_flow()


