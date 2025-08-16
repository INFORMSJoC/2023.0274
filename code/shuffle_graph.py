#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 18:34:33 2023

@author: negarmaleki
"""

import pandas as pd
import numpy as np
import networkx as nx
import json
import ast
import math
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from steem import Steem
import time
s = Steem()

for random in [0, 20, 42, 50]:

    #time.sleep(200)
    # Load Steemit Dataset
    data_steemit = pd.read_csv(".../health_data_same_num.csv")
    data_steemit.drop('Unnamed: 0', axis=1, inplace=True)
    data_steemit = data_steemit.sample(frac=1, random_state=random, ignore_index=True)

    """# Retriving posts with API
    posts = []
    for t in range(int(len(data_steemit)/1000)+1):
        print(t)
        if t == int(len(data_steemit)/1000):
            for j in range(t*1000, len(data_steemit)):
                if type(data_steemit.iloc[j,1]) != float and type(data_steemit.iloc[j,2]) != float:
                    posts.append(s.get_content_replies(data_steemit.iloc[j,1],data_steemit.iloc[j,2]))
        else:
            for j in range(t*1000, (t+1)*1000):
                if type(data_steemit.iloc[j,1]) != float and type(data_steemit.iloc[j,2]) != float:
                    posts.append(s.get_content_replies(data_steemit.iloc[j,1],data_steemit.iloc[j,2]))
        time.sleep(200)

    replies_dataset = pd.DataFrame(posts)
    replies_dataset.to_csv(".../shuffle_reply_"+str(random)+".csv")"""

    replies_dataset = pd.read_csv(".../shuffle_reply_"+str(random)+".csv")

    print("read main data")
    print('Random State '+str(random)+ ' is started!')

    data_category = data_steemit
    rate = 1627
    label_encoder = LabelEncoder()
    y_node = label_encoder.fit_transform(data_category['category'].to_list())
    y_node_org = label_encoder.fit_transform(data_category['category'].to_list())

    for i in range(rate, len(y_node)):
        y_node[i] = -1

    data_category['category_encode'] = y_node
    data_category['category_encode_org'] = y_node_org

    # create a Graph
    #G = nx.DiGraph()
    G = nx.Graph()
    for i in range(len(data_category)):
        print("First Loop, round: ", i)
        # create user nodes
        # create unique user node (if it isn't already created)
        if data_category.iloc[i,1] not in list(G.nodes):
            G.add_node(data_category.iloc[i,1])
            # adding attributes to the created user node
            G.nodes[data_category.iloc[i,1]]['username'] = data_category.iloc[i,1]
            G.nodes[data_category.iloc[i,1]]['otype'] = 'user'
        # create post nodes
        G.add_node(data_category.iloc[i,2])
        # adding attributes to the created post node
        G.nodes[data_category.iloc[i,2]]['permlink'] = data_category.iloc[i,2]
        G.nodes[data_category.iloc[i,2]]['otype'] = 'post'
        G.nodes[data_category.iloc[i,2]]['category'] = data_category.iloc[i,3]
        G.nodes[data_category.iloc[i,2]]['category_encode'] = data_category.iloc[i,46]
        G.nodes[data_category.iloc[i,2]]['category_encode_org'] = data_category.iloc[i,47]
        G.nodes[data_category.iloc[i,2]]['title'] = [data_category.iloc[i,6]]
        G.nodes[data_category.iloc[i,2]]['body'] = [data_category.iloc[i,7]]
        # some posts do not have 'tags'
        try:
            G.nodes[data_category.iloc[i,2]]['tag'] = json.loads(data_category.iloc[i,8])['tags']
        except:
            pass
        # all datetime attributes converted to the timestamp
        G.nodes[data_category.iloc[i,2]]['last_update'] = [int(datetime.timestamp(datetime.fromisoformat(data_category.iloc[i,9])))]
        G.nodes[data_category.iloc[i,2]]['created'] = int(datetime.timestamp(datetime.fromisoformat(data_category.iloc[i,10])))
        G.nodes[data_category.iloc[i,2]]['last_payout'] = int(datetime.timestamp(datetime.fromisoformat(data_category.iloc[i,12])))
        G.nodes[data_category.iloc[i,2]]['payout'] = float(data_category.iloc[i,23].split()[0])+float(data_category.iloc[i,24].split()[0])
        G.nodes[data_category.iloc[i,2]]['net_rshares'] = int(data_category.iloc[i,15])
        G.nodes[data_category.iloc[i,2]]['abs_rshares'] = int(data_category.iloc[i,16])
        G.nodes[data_category.iloc[i,2]]['vote_rshares'] = int(data_category.iloc[i,17])
        G.nodes[data_category.iloc[i,2]]['author_rewards'] = int(data_category.iloc[i,25])
        G.nodes[data_category.iloc[i,2]]['url'] = 'https://steemit.com'+data_category.iloc[i,35]
        # author reputation calculated based on the formula in "https://steemit.com/basictraining/@vaansteam/how-does-steemit-reputation-work-ultimate-guide"
        try:
            G.nodes[data_category.iloc[i,2]]['author_reputation'] = math.floor((math.log10(int(data_category.iloc[i,41]))-9)*9+25)
        except:
            pass

        # create edge between user and their posts (time attribute is in form of timestamp)
        G.add_edge(data_category.iloc[i,1], data_category.iloc[i,2], time=int(datetime.timestamp(datetime.fromisoformat(data_category.iloc[i,10]))), otype='authored')

        # create node and edge for vote
        if len(data_category.iloc[i,39]) != 0:
            for j in range(len(ast.literal_eval(data_category.iloc[i,39]))):
                try:
                    # create vote nodes by searching in active_votes column json
                    if ast.literal_eval(data_category.iloc[i,39])[j]['voter'] not in list(G.nodes):
                        G.add_node(ast.literal_eval(data_category.iloc[i,39])[j]['voter'])
                        G.nodes[ast.literal_eval(data_category.iloc[i,39])[j]['voter']]['username'] = ast.literal_eval(data_category.iloc[i,39])[j]['voter']
                        G.nodes[ast.literal_eval(data_category.iloc[i,39])[j]['voter']]['otype'] = 'user'
                    # create edge between voters and posts (time attribute is in form of timestamp)
                    G.add_edge(ast.literal_eval(data_category.iloc[i,39])[j]['voter'], data_category.iloc[i,2], time=int(datetime.timestamp(datetime.fromisoformat(ast.literal_eval(data_category.iloc[i,39])[j]['time']))), otype='vote')
                except:
                    pass


    keys = np.array(list(G.nodes))
    values = [int(i) for i in np.arange(0, len(G.nodes))]
    dic = dict(zip(keys, values))
    H = nx.relabel_nodes(G, dic)

    print("Second loop has just started")
    replies_data_category = replies_dataset
    #replies_data_category = replies_data_category.iloc[:100,:]
    print("read replies")

    for i in range(len(replies_data_category)):
        print(i)
        for j in range(replies_data_category.shape[1]):
            print(j)
            if type(replies_data_category.iloc[i,j]) == str:
                if type(ast.literal_eval(replies_data_category.iloc[i,j])) == dict:

                    print("Second Loop, round: ", i, " and ", j)
                    # create user nodes
                    # create unique user node (if it isn't already created)
                    if ast.literal_eval(replies_data_category.iloc[i,j])['author'] not in list(G.nodes):
                        G.add_node(ast.literal_eval(replies_data_category.iloc[i,j])['author'])
                        # adding attributes to the created user node
                        G.nodes[ast.literal_eval(replies_data_category.iloc[i,j])['author']]['username'] = ast.literal_eval(replies_data_category.iloc[i,j])['author']
                        G.nodes[ast.literal_eval(replies_data_category.iloc[i,j])['author']]['otype'] = 'user'
                    # create comment nodes
                    G.add_node(ast.literal_eval(replies_data_category.iloc[i,j])['permlink'])
                    # adding attributes to the created post node
                    G.nodes[ast.literal_eval(replies_data_category.iloc[i,j])['permlink']]['permlink'] = ast.literal_eval(replies_data_category.iloc[i,j])['permlink']
                    G.nodes[ast.literal_eval(replies_data_category.iloc[i,j])['permlink']]['otype'] = 'comment'
                    G.nodes[ast.literal_eval(replies_data_category.iloc[i,j])['permlink']]['category'] = ast.literal_eval(replies_data_category.iloc[i,j])['category']
                    G.nodes[ast.literal_eval(replies_data_category.iloc[i,j])['permlink']]['category_encode'] = label_encoder.transform([ast.literal_eval(replies_data_category.iloc[i,j])['category']])[0]
                    G.nodes[ast.literal_eval(replies_data_category.iloc[i,j])['permlink']]['category_encode_org'] = label_encoder.transform([ast.literal_eval(replies_data_category.iloc[i,j])['category']])[0]
                    if i>=rate:
                        G.nodes[ast.literal_eval(replies_data_category.iloc[i,j])['permlink']]['category_encode'] = -1
                    G.nodes[ast.literal_eval(replies_data_category.iloc[i,j])['permlink']]['title'] = [ast.literal_eval(replies_data_category.iloc[i,j])['title']]
                    G.nodes[ast.literal_eval(replies_data_category.iloc[i,j])['permlink']]['body'] = [ast.literal_eval(replies_data_category.iloc[i,j])['body']]
                    G.nodes[ast.literal_eval(replies_data_category.iloc[i,j])['permlink']]['parent_permlink'] = ast.literal_eval(replies_data_category.iloc[i,j])['parent_permlink']
                    # some posts do not have 'tags'
                    try:
                        G.nodes[ast.literal_eval(replies_data_category.iloc[i,j])['permlink']]['tag'] = json.loads(ast.literal_eval(replies_data_category.iloc[i,j])['json_metadata'])['tags']
                    except:
                        pass
                    # all datetime attributes converted to the timestamp
                    G.nodes[ast.literal_eval(replies_data_category.iloc[i,j])['permlink']]['last_update'] = [int(datetime.timestamp(datetime.fromisoformat(ast.literal_eval(replies_data_category.iloc[i,j])['last_update'])))]
                    G.nodes[ast.literal_eval(replies_data_category.iloc[i,j])['permlink']]['created'] = int(datetime.timestamp(datetime.fromisoformat(ast.literal_eval(replies_data_category.iloc[i,j])['created'])))
                    G.nodes[ast.literal_eval(replies_data_category.iloc[i,j])['permlink']]['last_payout'] = int(datetime.timestamp(datetime.fromisoformat(ast.literal_eval(replies_data_category.iloc[i,j])['last_payout'])))
                    G.nodes[ast.literal_eval(replies_data_category.iloc[i,j])['permlink']]['payout'] = float((ast.literal_eval(replies_data_category.iloc[i,j])['total_payout_value']).split()[0])+float((ast.literal_eval(replies_data_category.iloc[i,j])['curator_payout_value']).split()[0])
                    G.nodes[ast.literal_eval(replies_data_category.iloc[i,j])['permlink']]['net_rshares'] = int(ast.literal_eval(replies_data_category.iloc[i,j])['net_rshares'])
                    G.nodes[ast.literal_eval(replies_data_category.iloc[i,j])['permlink']]['abs_rshares'] = int(ast.literal_eval(replies_data_category.iloc[i,j])['abs_rshares'])
                    G.nodes[ast.literal_eval(replies_data_category.iloc[i,j])['permlink']]['vote_rshares'] = int(ast.literal_eval(replies_data_category.iloc[i,j])['vote_rshares'])
                    G.nodes[ast.literal_eval(replies_data_category.iloc[i,j])['permlink']]['author_rewards'] = int(ast.literal_eval(replies_data_category.iloc[i,j])['author_rewards'])
                    G.nodes[ast.literal_eval(replies_data_category.iloc[i,j])['permlink']]['url'] = 'https://steemit.com'+ast.literal_eval(replies_data_category.iloc[i,j])['url']
                    # author reputation calculated based on the formula in "https://steemit.com/basictraining/@vaansteam/how-does-steemit-reputation-work-ultimate-guide"
                    try:
                        G.nodes[ast.literal_eval(replies_data_category.iloc[i,j])['permlink']]['author_reputation'] = math.floor((math.log10(int(ast.literal_eval(replies_data_category.iloc[i,j])['author_reputation']))-9)*9+25)
                    except:
                        pass

                    # create edge between user and their comments (time attribute is in form of timestamp)
                    G.add_edge(ast.literal_eval(replies_data_category.iloc[i,j])['author'], ast.literal_eval(replies_data_category.iloc[i,j])['permlink'], time=int(datetime.timestamp(datetime.fromisoformat(ast.literal_eval(replies_data_category.iloc[i,j])['created']))), otype='authored')

                    # create node and edge for vote
                    if len(ast.literal_eval(replies_data_category.iloc[i,j])['active_votes']) != 0:
                        for k in range(len(ast.literal_eval(ast.literal_eval(replies_data_category.iloc[i,j])['active_votes']))):
                            try:
                                # create vote nodes by searching in active_votes column json
                                if ast.literal_eval(ast.literal_eval(replies_data_category.iloc[i,j])['active_votes'])[k]['voter'] not in list(G.nodes):
                                    G.add_node(ast.literal_eval(ast.literal_eval(replies_data_category.iloc[i,j])['active_votes'])[k]['voter'])
                                    G.nodes[ast.literal_eval(ast.literal_eval(replies_data_category.iloc[i,j])['active_votes'])[k]['voter']]['username'] = ast.literal_eval(ast.literal_eval(replies_data_category.iloc[i,j])['active_votes'])[k]['voter']
                                    G.nodes[ast.literal_eval(ast.literal_eval(replies_data_category.iloc[i,j])['active_votes'])[k]['voter']]['otype'] = 'user'
                                # create edge between voters and posts (time attribute is in form of timestamp)
                                G.add_edge(ast.literal_eval(ast.literal_eval(replies_data_category.iloc[i,j])['active_votes'])[k]['voter'], ast.literal_eval(replies_data_category.iloc[i,j])['permlink'], time=int(datetime.timestamp(datetime.fromisoformat(ast.literal_eval(ast.literal_eval(replies_data_category.iloc[i,j])['active_votes'])[j]['time']))), otype='vote')
                            except:
                                pass
                    # create edge between post and comment
                    try:
                        if data_category.iloc[i,2] == ast.literal_eval(replies_data_category.iloc[i,j])['parent_permlink']:
                            G.add_edge(data_category.iloc[i,2], ast.literal_eval(replies_data_category.iloc[i,j])['permlink'], time=int(datetime.timestamp(datetime.fromisoformat(ast.literal_eval(replies_data_category.iloc[i,j])['created']))), otype='reply')
                    except:
                        pass


    keys = np.array(list(G.nodes))
    values = [int(i) for i in np.arange(0, len(G.nodes))]
    dic = dict(zip(keys, values))
    H = nx.relabel_nodes(G, dic)


    def comment_add(G, replies_aug_sep_2019, aug_sep_2019):
        for i in range(len(replies_aug_sep_2019)):
            print(i)
            for j in range(replies_aug_sep_2019.shape[1]):
                print(j)
                if type(replies_aug_sep_2019.iloc[i,j]) == str:
                    if type(ast.literal_eval(replies_aug_sep_2019.iloc[i,j])) == dict:
                        print("Second Loop, round: ", i, " and ", j)
                        if ast.literal_eval(replies_aug_sep_2019.iloc[i,j])['author'] not in list(G.nodes):
                            G.add_node(ast.literal_eval(replies_aug_sep_2019.iloc[i,j])['author'])
                            # adding attributes to the created user node
                            G.nodes[ast.literal_eval(replies_aug_sep_2019.iloc[i,j])['author']]['username'] = ast.literal_eval(replies_aug_sep_2019.iloc[i,j])['author']
                            G.nodes[ast.literal_eval(replies_aug_sep_2019.iloc[i,j])['author']]['otype'] = 'user'

                        # create edge between user and their comments (time attribute is in form of timestamp)
                        G.add_edge(ast.literal_eval(replies_aug_sep_2019.iloc[i,j])['author'], ast.literal_eval(replies_aug_sep_2019.iloc[i,j])['permlink'], time=int(datetime.timestamp(datetime.fromisoformat(ast.literal_eval(replies_aug_sep_2019.iloc[i,j])['created']))), otype='authored')

                        # create edge between post and comment
                        if aug_sep_2019.iloc[i,2] == ast.literal_eval(replies_aug_sep_2019.iloc[i,j])['parent_permlink']:
                            G.add_edge(aug_sep_2019.iloc[i,2], ast.literal_eval(replies_aug_sep_2019.iloc[i,j])['permlink'], time=int(datetime.timestamp(datetime.fromisoformat(ast.literal_eval(replies_aug_sep_2019.iloc[i,j])['created']))), otype='reply')
        return G

    G_train = comment_add(G, replies_data_category, data_category)

    def author(aug_sep_2019, G):
        for i in range(len(aug_sep_2019)):
            print("First Loop, round: ", i)
            # create user nodes
            # create unique user node (if it isn't already created)
            if aug_sep_2019.iloc[i,1] not in list(G.nodes):
                G.add_node(aug_sep_2019.iloc[i,1])
                # adding attributes to the created user node
                G.nodes[aug_sep_2019.iloc[i,1]]['username'] = aug_sep_2019.iloc[i,1]
                G.nodes[aug_sep_2019.iloc[i,1]]['otype'] = 'user'

            # create edge between user and their posts (time attribute is in form of timestamp)
            G.add_edge(aug_sep_2019.iloc[i,1], aug_sep_2019.iloc[i,2], time=int(datetime.timestamp(datetime.fromisoformat(data_category.iloc[i,10]))), otype='authored')
        return G

    G_train = author(data_category, G_train)

    def relable(G):
        keys = np.array(list(G.nodes))
        values = [int(i) for i in np.arange(0, len(G.nodes))]
        dic = dict(zip(keys, values))
        return nx.relabel_nodes(G, dic)

    G_train = relable(G_train)

    nx.write_gpickle(G_train, ".../g_shuffled_"+str(random)+".gpickle")

    print('Random State '+str(random)+ ' is DONE!')
