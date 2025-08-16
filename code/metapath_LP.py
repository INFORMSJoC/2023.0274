#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 23:17:00 2023

@author: negarmaleki
"""

import sys
import time
import random
import networkx as nx
import numpy as np
from numpy.linalg import inv
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from collections import Counter

for rate in [1340, 1723]:  #1340, 1436, 1531, 1627, 1723, 1818

    Graph_labeled = nx.read_gpickle(".../g_train+test_"+str(round(rate/1914*100))+".gpickle")

    def get_key_node(val, G):
        key_indx_node = []
        for key, value in nx.get_node_attributes(G,'otype').items():
             if val == value:
                key_indx_node.append(key)
        return key_indx_node

    # indices for each type of dict
    post_dict = get_key_node('post', Graph_labeled)

    def meta_path(Graph, post_dict):
        A = np.zeros((len(post_dict),len(post_dict)))
        y_node, real_cat, y_node_org = [], [], []
        p_A_u_A_p, p_V_u_A_p, p_R_c_A_u_A_p, p_R_c_V_u_A_p, p_A_u_A_c_V_u_A_p, p_V_u_A_c_V_u_A_p = [],[],[],[],[],[]

        for i in range(len(Graph.nodes())):
            print(i)
            if Graph.nodes()[i]['otype'] == 'post':
                for j in range(len(Graph.nodes())):
                    # post_user_post
                    if i!=j and Graph.nodes()[j]['otype'] == 'user':
                        if (j,i) in Graph.edges() and (Graph.edges[j,i]['otype'] == 'authored' or Graph.edges[j,i]['otype'] == 'vote'):
                            for z in range(len(Graph.nodes())):
                                if j!=z and i!=z and Graph.nodes()[z]['otype'] == 'post':
                                    if (j,z) in Graph.edges() and Graph.edges[j,z]['otype'] == 'authored' and (Graph.edges[j,i]['time'] <= Graph.edges[j,z]['time']):
                                        A[post_dict.index(i)][post_dict.index(z)] += 1
                                        A[post_dict.index(z)][post_dict.index(i)] += 1
                                        # post --authored--> user --authored--> post
                                        if Graph.edges[j,i]['otype'] == 'authored':
                                            p_A_u_A_p.append([i,j,z,post_dict.index(i),post_dict.index(z)])
                                        # post --vote--> user --authored--> post
                                        if Graph.edges[j,i]['otype'] == 'vote':
                                            p_V_u_A_p.append([i,j,z,post_dict.index(i),post_dict.index(z)])

                    # post_comment_user_post
                    if i!=j and Graph.nodes()[j]['otype'] == 'comment':
                        if (i,j) in Graph.edges() and Graph.edges[i,j]['otype'] == 'reply':
                            for z in range(len(Graph.nodes())):
                                if j!=z and i!=z and Graph.nodes()[z]['otype'] == 'user':
                                    if (z,j) in Graph.edges() and (Graph.edges[z,j]['otype'] == 'authored' or Graph.edges[z,j]['otype'] == 'vote') and (Graph.nodes()[j]['created'] <= Graph.edges[z,j]['time']):
                                        for t in range(len(Graph.nodes())):
                                            if z!=t and i!=t and j!=t and Graph.nodes()[t]['otype'] == 'post':
                                                if (z,t) in Graph.edges() and Graph.edges[z,t]['otype'] == 'authored' and (Graph.edges[z,j]['time'] <= Graph.edges[z,t]['time']):
                                                    A[post_dict.index(i)][post_dict.index(t)] += 1
                                                    A[post_dict.index(t)][post_dict.index(i)] += 1
                                                    # post --reply--> comment --authored--> user --authored--> post
                                                    if Graph.edges[z,j]['otype'] == 'authored':
                                                        p_R_c_A_u_A_p.append([i,j,z,t,post_dict.index(i),post_dict.index(t)])
                                                    # post --reply--> comment --vote--> user --authored--> post
                                                    if Graph.edges[z,j]['otype'] == 'vote':
                                                        p_R_c_V_u_A_p.append([i,j,z,t,post_dict.index(i),post_dict.index(t)])

                    # post_user_comment_user_post
                    if i!=j and Graph.nodes()[j]['otype'] == 'user':
                        if (j,i) in Graph.edges() and (Graph.edges[j,i]['otype'] == 'authored' or Graph.edges[j,i]['otype'] == 'vote'):
                            for z in range(len(Graph.nodes())):
                                if j!=z and i!=z and Graph.nodes()[z]['otype'] == 'comment':
                                    if (j,z) in Graph.edges() and Graph.edges[j,z]['otype'] == 'authored' and (Graph.edges[j,i]['time'] <= Graph.edges[j,z]['time']):
                                        for t in range(len(Graph.nodes())):
                                            if z!=t and i!=t and j!=t and Graph.nodes()[t]['otype'] == 'user':
                                                if (t,z) in Graph.edges() and Graph.edges[t,z]['otype'] == 'vote' and (Graph.nodes()[z]['created'] <= Graph.edges[t,z]['time']):
                                                    for s in range(len(Graph.nodes())):
                                                        if z!=s and i!=s and j!=s and t!=s and Graph.nodes()[s]['otype'] == 'post':
                                                            if (t,s) in Graph.edges() and Graph.edges[t,s]['otype'] == 'authored' and (Graph.edges[t,z]['time'] <= Graph.edges[t,s]['time']):
                                                                A[post_dict.index(i)][post_dict.index(s)] += 1
                                                                A[post_dict.index(s)][post_dict.index(i)] += 1
                                                                # post --authored--> user --authored--> comment --vote--> user --authored--> post
                                                                if Graph.edges[j,i]['otype'] == 'authored':
                                                                    p_A_u_A_c_V_u_A_p.append([i,j,z,t,s,post_dict.index(i),post_dict.index(s)])
                                                                # post --vote--> user --authored--> comment --vote--> user --authored--> post
                                                                if Graph.edges[j,i]['otype'] == 'vote':
                                                                    p_V_u_A_c_V_u_A_p.append([i,j,z,t,s,post_dict.index(i),post_dict.index(s)])

                # post_label
                y_node.append(Graph.nodes[i]['category_encode'])
                real_cat.append(Graph.nodes[i]['category'])
                y_node_org.append(Graph.nodes[i]['category_encode_org'])


        return A, y_node, real_cat, y_node_org
    start1 = time.time()
    A, y_node, real_cat, y_node_org = meta_path(Graph_labeled, post_dict)
    start2 = time.time()
    diag = np.eye(len(post_dict))
    A_loop = A + diag

    matrix = pd.DataFrame(A_loop)
    #matrix.to_excel('.../'+str(round(rate/1914*100))+'-5cat.xlsx')

    # Label propagation
    y = np.array(y_node)
    shape = (len(y_node), 1)
    y = y.reshape(shape)

    data = pd.DataFrame()
    data['category'] = y_node

    one_hot_encoded_y = pd.get_dummies(data, columns = ['category'])
    #random_idx = random.sample(range(26991), 2000)

    # Create unlabeled node
    y_labeled = one_hot_encoded_y.iloc[:,1:].to_numpy()
    """for i in range(len(y_node)):
        if y_node[i] == -1:
            y_labeled[i] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]"""

    # transition matrix
    A_matrix = A_loop
    row_s = list(map(sum, A_matrix))

    D = np.zeros((A_matrix.shape[0], A_matrix.shape[0]))
    for i in range(A_matrix.shape[0]):
        D[i,i] = row_s[i] #+ 0.00001*np.random.rand()

    D_inv = inv(D)
    S = np.dot(D_inv, A_matrix)

    # label Propagation Algorithm
    def LabelPropagation(T, Y, diff, max_iter):
        # Initialize
        Y_init = Y
        Y1 = Y
        # Initialize convergence parameters
        n=0
        current_diff = sys.maxsize
        # Iterate till difference reduces below diff or till the maximum number of iterations is reached
        while current_diff > diff and n < max_iter:
            print('current_diff: ', current_diff)
            print('iteration: ', n)
            current_diff = 0.0
            # Set Y(t)
            Y0 = Y1
            # Calculate Y(t+1)
            Y1 = np.dot(T,Y0)
            # Clamp labelled data
            """for i in range(26991):
                if i not in random_idx:
                    Y1[i] = Y_init[i]"""

            for i in range(Y_init.shape[0]):
                if Y_init[i].any() != np.zeros((len(np.unique(y_node))-1,), dtype=int).any():
                    Y1[i] = Y_init[i]

            """for i in range(Y_init.shape[0]):
                if i in labelled:
                    for j in range(Y_init.shape[1]):
                        if i!=j:
                            Y1[i][j] = Y_init[i][j]"""
            # Get difference between values of Y(t+1) and Y(t)
            for i in range(Y1.shape[0]):
                if Y_init[i].any() == np.zeros((len(np.unique(y_node))-1,), dtype=int).any():
                    for j in range(Y1.shape[1]):
                        current_diff += abs(Y1[i][j] - Y0[i][j])
            n += 1
        return Y1

    start3 = time.time()
    L = LabelPropagation(S, y_labeled, 0, 5000) #
    res = L.argmax(1)
    df = pd.DataFrame()
    df['LP_label'] = res
    df['labeled_unlabeled'] = y_node
    df['Original_label'] = y_node_org
    df['Original_cat'] = real_cat
    end = time.time()
    print('First Meta-path: ', end - start1)
    print('Second matrix calc.: ', end - start2)
    print('Third LP: ', end - start3)
    #df.to_csv(".../result_metapath_"+str(round(rate/1914*100))+".csv")
    """acc = 0
    for i in range(len(res)):
        if res[i] != y_node[i]:
            acc += 1
    print(acc)"""
    #one_hot_encoded_y.to_numpy()
