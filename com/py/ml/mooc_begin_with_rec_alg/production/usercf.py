# -*- coding: UTF-8 -*-

import sys
import math
import operator
from __future__ import division  # 避免整数相除
# sys.path.append("../util")
import com.py.ml.mooc_begin_with_rec_alg.util.reader as reader


def transfer_user_click(user_click):
    """
    get item by user_click
    :param user_click: dict key:user_id, value:item_list
    :return: dict: key:item_id, value:[user_id]
    """
    item_click_by_user = {}
    for user in user_click:
        item_id_list = user_click[user]
        for item_id in item_id_list:
            item_click_by_user.setdefault(item_id, [])
            item_click_by_user[item_id].append(user)
    return item_click_by_user


def base_contribution_score():
    return 1


def update_contribution_score(item_user_click_count):
    return 1 / math.log10(1 + item_user_click_count)


def update_two_contribution_score(click_time_one, click_time_two):
    delta_time = abs(click_time_one - click_time_two)
    norm_num = 24 * 60 * 60
    delta_time = delta_time / norm_num

    return 1 / (1 + delta_time)


def cal_user_sim(item_click_by_user, user_click_time):
    """
    get user sim info
    :param item_click_by_user: dict: key:item_id value:[user]
    :return: dict key:item_id_i, value: dict  key: item_id_j, value:sim_score
    """
    co_appear = {}
    user_click_count = {}
    for item_id, user_list in item_click_by_user.items():
        for index_i in range(0, len(user_list)):
            user_i = user_list[index_i]
            user_click_count.setdefault(user_i, 0)
            user_click_count[user_i] += 1

            if user_i + "_" + item_id not in user_click_time:
                click_time_one = 0
            else:
                click_time_one = user_click_time[user_i + "_" + item_id]

            for index_j in range(index_i + 1, len(user_list)):
                user_j = user_list[index_j]
                if user_j + "_" + item_id not in user_click_time:
                    click_time_two = 0
                else:
                    click_time_two = user_click_time[user_j + "_" + item_id]
                co_appear.setdefault(user_i, {})
                co_appear[user_i].setdefault(user_j, 0)
                # update_contribution_score(len(user_list))
                co_appear[user_i][user_j] += update_two_contribution_score(click_time_one, click_time_two)

                co_appear.setdefault(user_j, {})
                co_appear[user_j].setdefault(user_i, 0)
                co_appear[user_j][user_i] += update_two_contribution_score(click_time_one, click_time_two)

    user_sim_info = {}
    user_sim_info_sorted = {}
    for user_i, relate_user in co_appear.items():
        user_sim_info.setdefault(user_i, {})
        for user_j, co_time in relate_user.items():
            user_sim_info[user_i].setdefault(user_j, 0)
            user_sim_info[user_i][user_j] = co_time / math.sqrt(user_click_count[user_i] * user_click_count[user_j])

    for user in user_sim_info:
        user_sim_info_sorted[user] = sorted(user_sim_info[user].iteritems(), key=operator.itemgetter(1), reverse=True)

    return user_sim_info_sorted


def cal_recom_result(user_click, user_sim):
    """
    recom by usercf algo
    :param user_click: user_id [item_id]
    :param user_sim: user_id_i, (user_id_j, score)
    :return:
    dict, key:user_id value: dict key: item_id, value: score
    """
    recom_result = {}
    top_k_user = 3
    item_num = 5
    for user, item_list in user_click.items():
        tmp_dict = {}
        for item_id in item_list:
            tmp_dict.setdefault(item_id, 1)
        recom_result.setdefault(user, {})
        for cp in user_sim[user][:top_k_user]:
            user_id_j, sim_score = cp
            if user_id_j not in user_click:
                continue
            for item_id_j in user_click[user_id_j][:item_num]:
                recom_result[user].setdefault(item_id_j, sim_score)
    return recom_result


def debug_user_sim(user_sim):
    """
    print user sim result
    :param user_sim:key user_id vlaue[(userid, score)]
    :return:
    """
    top_k = 5
    fix_user = "1"
    if fix_user not in user_sim:
        print("invalid user")
        return
    for cp in user_sim[fix_user][:top_k]:
        user_id, score = cp
        print(fix_user + "\tsim_user " + user_id + "\t" + str(score))


def debug_recom_result(item_info, recom_result):
    """
    print recom result for user
    :param item_info: item_id -> [title, genres]
    :param recom_result: user_id -> {item_id -> recom_score}
    :return:
    """
    fix_user = "1"
    if fix_user not in recom_result:
        print("invalid user for recoming result")
        return
    for itemid in recom_result["1"]:
        if itemid not in item_info:
            continue
        recom_score = recom_result["1"][itemid]
        print("recom_result:" + ",".join(item_info[itemid]) + "\t" + str(recom_score))


def main_flow():
    user_click, user_click_time = reader.get_user_click("../data/ratings.txt")
    item_info = reader.get_item_info("../data/movies.txt")
    item_click_by_user = transfer_user_click(user_click)
    user_sim = cal_user_sim(item_click_by_user, user_click_time)
    debug_user_sim(user_sim)
    recom_result = cal_recom_result(user_click, user_sim)
    # print(recom_result["1"])
    debug_recom_result(item_info, recom_result)


if __name__ == "__main__":
    main_flow()
