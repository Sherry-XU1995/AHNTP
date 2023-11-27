# -*- coding: utf-8 -*-


import pickle
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.io as scio

from pagerank import generate_pagerank_weights

# rebuild the user index
dic_user_o2i = defaultdict()
# rebuild the item index
dic_item_o2i = defaultdict()
# user ratting item history
history_u_lists = defaultdict()
history_ur_lists = defaultdict()
# item union user history
history_v_lists = defaultdict()
history_vr_lists = defaultdict()
# full adj matrix
data_adj_lists = defaultdict()
# the train adj matrix
social_adj_lists = defaultdict()
# ratting value list
ratings_list = defaultdict()


def load_trust_info(files):
    '''
    load trust network data
    :param files:
    :return:
    '''
    print("load trust data.")
    trust = scio.loadmat(files)
    trust = trust['trustnetwork']
    print("trust:", trust.shape)

    # remove self trust
    delete = []
    for i in range(len(trust)):
        if trust[i, 0] == trust[i, 1]:
            delete.append(i)
    # print("delete:", delete)
    trust = np.delete(trust, delete, 0)

    return trust


def load_rating(files):
    '''
    load the user ratting item data
    :param files:
    :return:
    '''
    print("load rating data.")
    rating = scio.loadmat(files)
    rating = rating['rating']
    print("rating:", rating.shape)
    return rating


def rebuild_data(trust, rating):
    '''
    rebuild the user item and ratting data, rebuild the index and delete the repeat value
    :param trust:
    :param rating:
    :return:
    '''
    global dic_user_o2i
    global dic_item_o2i

    i = 0
    # build user index
    for user in trust.reshape(-1, ):
        if user in dic_user_o2i.keys():
            continue
        else:
            dic_user_o2i[user] = i
            i += 1
    # user num 7317, from trust
    # build from trust
    user_sizes = len(dic_user_o2i.keys())
    print('num of users', len(dic_user_o2i))

    delete = []
    # delete the not exist
    for i in range(len(rating)):
        if rating[i, 0] not in dic_user_o2i.keys():
            delete.append(i)
    rating = np.delete(rating, delete, 0)
    print('rating_num:', len(rating))  # ciao: 283320

    i = 0
    # build item index
    for item in rating[:, 1]:
        if item in dic_item_o2i.keys():
            continue
        else:
            dic_item_o2i[item] = i
            # dic_item_o2i[item] = item
            i += 1
    items_num = len(dic_item_o2i.keys())
    print('number of items ', len(dic_item_o2i.keys()))  # ciao 104975
    # user & item index build is done.
    return user_sizes, items_num, rating


def generate_trust_pair(trust):
    '''
    generate the trust pair and trustor list
    :param trust:
    :return:
    [
        [A1, B1],
        [A2, B2]
        ...
    ]
    '''

    global dic_user_o2i
    trustor_set = []  # 存储信任人 发起者
    total_trust_pair = []  # N * 3 ( 发起者, 被信任者， 1)
    for row in trust:
        trustor = row[0]
        trustee = row[1]
        # 记录信任发起者
        trustor_set.append(dic_user_o2i[trustor])
        total_trust_pair.append([dic_user_o2i[trustor], dic_user_o2i[trustee], 1])
    np.random.shuffle(total_trust_pair)
    trustor_set = set(trustor_set)
    return trustor_set, total_trust_pair


def generate_adj_matrix(trust, train_trust):
    '''
    generate the adj matrix
    :param trust: full trust network
    :param train_trust: the train part trust network
    :return:
    '''
    global dic_user_o2i
    global data_adj_lists
    global social_adj_lists
    # social adj is adj matrix not self-loop
    print('building social_adj_lists')
    begin = time.time()

    # the adj is new index
    for user in range(len(dic_user_o2i)):
        social_adj_lists[user] = []  # (key, value)  存储训练集 用户信任邻接矩阵 80%
        data_adj_lists[user] = []  # 全局的玲接矩阵 100%

    # use the train data build the social adj
    # Full adj
    for line in trust:
        data_adj_lists[dic_user_o2i[line[0]]].append(dic_user_o2i[line[1]])
        data_adj_lists[dic_user_o2i[line[1]]].append(dic_user_o2i[line[0]])

    # train adj
    for line in train_trust:
        social_adj_lists[line[0]].append(line[1])
        social_adj_lists[line[1]].append(line[0])

    # delete the repeat value
    for user in range(len(dic_user_o2i)):
        social_adj_lists[user] = set(social_adj_lists[user])
        data_adj_lists[user] = set(data_adj_lists[user])

    print("adj Time : %.2fs" % (time.time() - begin))


def generate_train_neg_data(train_trust, total_trust, trustor_set, users_num, neg_num):
    '''
    generate the train dataset neg sample
    :param train_trust:
    :param total_trust:
    :param trustor_set:
    :param users_num:
    :param neg_num:
    :return:
    '''
    global data_adj_lists
    begin = time.time()
    for trustor in trustor_set:  # A1
        neg_list = []
        for t in range(neg_num):  # count
            j = np.random.randint(users_num)
            while j in neg_list or j in data_adj_lists[trustor]:
                j = np.random.randint(users_num)
            neg_list.append(j)
            total_trust.append([trustor, j, 0])
            train_trust.append([trustor, j, 0])
    print("train neg Time : %.2fs" % (time.time() - begin))
    return train_trust, total_trust


def generate_test_neg_data(test_trust, users_num, neg_num_test):
    '''
    generate test dataset neg sample
    :param test_trust:
    :param users_num:
    :param neg_num_test:
    :return:
    '''
    global data_adj_lists
    # process the test dateset
    test_trustor_set = []
    print("build the test trust pair")
    for row in test_trust:
        trustor = row[0]
        test_trustor_set.append(trustor)
    test_trustor_set = set(test_trustor_set)

    print("build test neg data.")
    begin = time.time()
    for test_trustor in test_trustor_set:
        test_neglist = []
        for r in range(neg_num_test):
            m = np.random.randint(users_num)
            while m in data_adj_lists[test_trustor] or m in test_neglist:
                m = np.random.randint(users_num)
            test_neglist.append(m)
            test_trust.append([test_trustor, m, 0])

    print("test neg Time : %.2fs" % (time.time() - begin))
    np.random.shuffle(test_trust)
    return test_trust


def generate_other_data(rating):
    '''
    generate other information
    :param rating:
    :return:
    '''
    global dic_user_o2i
    global dic_item_o2i
    global history_v_lists
    global history_u_lists
    global history_ur_lists
    global history_vr_lists
    global ratings_list

    print('building ratings_list')
    i = 0
    for rate in set(rating[:, 3]):
        ratings_list[rate] = i
        i += 1

    print('building other dicts')
    for user in range(len(dic_user_o2i)):
        history_u_lists[user] = []  # 用户A: 商品i 1 2 3 4 ...
        history_ur_lists[user] = []  # 用户A： 对商品的打分

    for item in range(len(dic_item_o2i)):
        history_v_lists[item] = []  # 商品i1:  用户A1 2 3 4..
        history_vr_lists[item] = []  # 商品i1： 商品被打分

    # build rating tripe
    rating = rating[:, [0, 1, 3]]

    bi_rating = []

    np.random.shuffle(rating)

    print('build other information')
    for line in rating:
        user = line[0]
        item = line[1]
        rate = ratings_list[line[2]]
        if user in dic_user_o2i.keys():
            history_u_lists[dic_user_o2i[user]].append(dic_item_o2i[item])
            history_ur_lists[dic_user_o2i[user]].append(rate)

            history_v_lists[dic_item_o2i[item]].append(user)
            history_vr_lists[dic_item_o2i[item]].append(rate)
    # edge_cnt = 0
    for user in history_u_lists.keys():
        temp = [user] + history_u_lists[user]
        # print(temp)
        bi_rating.append(temp)
        # history_u_lists[user] = set(history_u_lists[user])
        # print(user, history_u_lists[user])
        # edge_cnt += len(history_u_lists[user])
        # history_ur_lists[user] = set(history_ur_lists[user])

    # for item in history_v_lists.keys():
    #     history_v_lists[item] = set(history_v_lists[item])
    #     history_vr_lists[item] = set(history_vr_lists[item])

    # print(edge_cnt)
    return rating, bi_rating


def save_pkl(filenames, data):
    with open(filenames, 'wb') as fo:
        pickle.dump(data, fo)
    print(str(filenames), 'save done.')


def save_txt(filenames, data):
    with open(filenames, 'w') as f:
        for line in data:
            list2 = [str(i) for i in line]
            temp = " ".join(list2)
            print(temp, file=f)
    print(str(filenames), 'save done.')


def preprocess(dataset_name):
    print("The dataset is ", dataset_name)
    d_path = Path(Path(__file__).parent, 'data')
    path = Path(d_path, dataset_name)
    f_trust = Path(path, 'trustnetwork.mat')
    f_rating = Path(path, 'rating.mat')
    # 读取 trustnetwork
    trust_data = load_trust_info(f_trust)
    # 读取 rating
    rating_data = load_rating(f_rating)

    num_users, num_items, rating = rebuild_data(trust_data, rating_data)
    print('users: %.2f , items: %.2f , rating: %.2f ' % (num_users, num_items, len(rating)))
    train_trustor_set, total_trust_pair = generate_trust_pair(trust_data)

    # test_trust_pair = total_trust_pair[: int(0.2 * len(total_trust_pair))]
    # train_trust_pair = total_trust_pair[int(0.2 * len(total_trust_pair)):]
    train_size = int(0.8 * len(total_trust_pair))
    test_size = int(0.2 * len(total_trust_pair))

    train_trust_pair = total_trust_pair[:train_size]
    test_trust_pair = total_trust_pair[train_size:train_size + test_size]

    print("test trust len:", len(test_trust_pair))
    print("train trust len:", len(train_trust_pair))

    generate_adj_matrix(trust_data, train_trust_pair)

    # train dataset neg value
    neg_num = 2
    train_trust_pair, total_trust_pair = generate_train_neg_data(train_trust_pair, total_trust_pair, train_trustor_set,
                                                                 num_users, neg_num)
    # test dataset neg value
    neg_num_test = 2
    test_trust_pair = generate_test_neg_data(test_trust_pair, num_users, neg_num_test)

    rating, bi_rating = generate_other_data(rating)
    # print(len(rating), len(rating[0]))
    # print(len(train_trust_pair))
    # print(len(test_trust_pair))
    # print(len(history_u_lists.keys()))
    # print(len(history_v_lists.keys()))

    pagerank_weights = generate_pagerank_weights(train_trust_pair, len(history_u_lists))

    print('*' * 80)
    print("begin write data.")
    d_f_rating = Path(path, 'rating.pkl')
    d_f_bi_rating = Path(path, 'adj_list.txt')
    f_total_trust_pair = Path(path, 'total_trust_pair.pkl')
    f_train_pair = Path(path, 'train_mask.pkl')
    f_test_pair = Path(path, 'test_mask.pkl')
    f_full_adj_matrix = Path(path, 'full_adj.pkl')
    f_social_adj_matrix = Path(path, 'social_adj.pkl')
    f_user_item_info = Path(path, 'u_v_info.pkl')
    f_user_relation_info = Path(path, 'u_r_info.pkl')
    f_item_user_info = Path(path, 'v_u_info.pkl')
    f_item_relation_info = Path(path, 'v_r_info.pkl')
    f_pagerank_weights = Path(path, 'pr_weights.pkl')

    print('user: %d, item: %d ' % (num_users, num_items))
    print('interaction: %d ' % (len(rating)))

    save_pkl(d_f_rating, rating)
    save_txt(d_f_bi_rating, bi_rating)
    save_pkl(f_total_trust_pair, total_trust_pair)
    save_pkl(f_train_pair, train_trust_pair)
    save_pkl(f_test_pair, test_trust_pair)
    save_pkl(f_full_adj_matrix, data_adj_lists)
    save_pkl(f_social_adj_matrix, social_adj_lists)
    save_pkl(f_user_item_info, history_u_lists)
    save_pkl(f_user_relation_info, history_ur_lists)
    save_pkl(f_item_user_info, history_v_lists)
    save_pkl(f_item_relation_info, history_vr_lists)
    save_pkl(f_pagerank_weights, pagerank_weights)

    print('*' * 80)
    print('save over!!!')


if __name__ == "__main__":
    np.random.seed(0)
    preprocess('ciao')
    preprocess('epinions')
