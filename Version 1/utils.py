# -*- coding: utf-8 -*-


import os
import dhg


def get_user_item(user_list, data):
    '''
    get itmes of user
    :param user_list:
    :param data:
    :return:
    '''
    user_item_list = []
    # n_user = set(user_list)
    # items_list = []
    for user in user_list:
        u_items = data[user]
        # for item in u_items:
        #     # temp = (user, item)
        #     user_item_list.append((user, item))
        temp = [user] + u_items
        # items_list = items_list + u_items
        user_item_list.append(temp)
    # n_items = set(items_list)
    return user_item_list


def generate_edge_list(data):
    edge_list = []
    for user in data.keys():
        for trustee in data[user]:
            edge_list.append((user, trustee))

    return edge_list
