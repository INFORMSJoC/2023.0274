#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 16 21:19:56 2023

@author: negarmaleki
"""

import networkx as nx
from torch_geometric.data import HeteroData
import dill
import torch_geometric.transforms as T
import pandas as pd
import numpy as np
import torch
from warnings import filterwarnings
filterwarnings("ignore")

G_train = nx.read_gpickle(".../g_train+test.gpickle")

# function to get node index
def get_key_node(val, G):
    key_indx_node = []
    for key, value in nx.get_node_attributes(G,'otype').items():
         if val == value:
            key_indx_node.append(key)
    return key_indx_node

# function to get edge index
def get_key_edge(val, G):
    key_indx_edge = []
    for key, value in nx.get_edge_attributes(G,'otype').items():
         if val == value:
            key_indx_edge.append(key)
    return key_indx_edge

# indices for each type of dict
post_train = get_key_node('post', G_train)
comment_train = get_key_node('comment', G_train)
user_train = get_key_node('user', G_train)
vote_train = get_key_edge('vote', G_train)
authored_train = get_key_edge('authored', G_train)
reply_train = get_key_edge('reply', G_train)

"""# nodes and edges within 7-day period
def time_vote(vote, reply, post, comment, user, authored, G):

    user_in, post_in, cm_in  = [], [], []
    print(len(G.nodes()), len(G.edges()))

    subG_reply = []
    for i in reply:
        if G.edges()[i]['time'] >= G.nodes()[i[0]]['created'] and G.edges()[i]['time'] <= G.nodes()[i[0]]['last_payout']:
            subG_reply.append(i)
            cm_in.append(i[1])
            post_in.append(i[0])

    subG_vote = []
    for i in vote:
        if G.edges()[i]['time'] >= G.nodes()[i[1]]['created'] and G.edges()[i]['time'] <= G.nodes()[i[1]]['last_payout']:
            if G.nodes()[i[1]]['otype'] == 'post':
                subG_vote.append(i)
                user_in.append(i[0])
                post_in.append(i[1])
            elif G.nodes()[i[1]]['otype'] == 'comment':
                if i[1] in cm_in:
                    subG_vote.append(i)
                    user_in.append(i[0])

    subG_authored = []
    for i in authored:
        if G.nodes()[i[1]]['otype'] == 'post': #i[1] in post_in:
            subG_authored.append(i)
            user_in.append(i[0])
            if i[1] not in post_in:
                post_in.append(i[1])
        if i[1] in cm_in:
            user_in.append(i[0])
            subG_authored.append(i)

    #subG_authored = []
    #for i in authored:
    #    if i[1] in post_in:
    #        subG_authored.append(i)
    #        user_in.append(i[0])
    #    if i[1] in cm_in:
    #        subG_authored.append(i)
    #        user_in.append(i[0])

    G_node = G.subgraph(post_in + cm_in + user_in)
    Graph = G_node.edge_subgraph(subG_vote + subG_authored + subG_reply)

    g = nx.Graph(Graph)
    print(len(g.nodes()), len(g.edges()))
    return Graph #nx.Graph(Graph)

G_train = time_vote(vote_train, reply_train, post_train, comment_train, user_train, authored_train, G_train)"""

"""# counting engagement
def engagement(G):
    vote_count = []
    comment_count = []
    for i in G.nodes():
        vote = 0
        comment = 0
        if G.nodes()[i]['otype'] == 'post':
            for j in G.out_edges(i):
                if G.edges()[i,j[1]]['otype'] == 'reply':
                    comment += 1
            comment_count.append(comment)
            for j in G.in_edges(i):
                if G.edges()[j[0],i]['otype'] == 'vote':
                    vote += 1
            vote_count.append(vote)
    return vote_count, comment_count

vote_count_train, comment_count_train = engagement(G_train)
print(len(vote_count_train))
print(len(comment_count_train))"""

"""# relabel the graph
import numpy as np
def relable(G):
    keys = np.array(list(G.nodes))
    values = [int(i) for i in np.arange(0, len(G.nodes))]
    dic = dict(zip(keys, values))
    return nx.relabel_nodes(G, dic)

G_train = relable(G_train)"""

# getting graph info
p_category_train = []
created = []
last_payout = []
p_payout_train = []
p_body_train = []
#username = []

u_otype_train = []

c_otype_train = []
c_category_train = []

for i in range(len(G_train.nodes())):
    if G_train.nodes()[i]['otype']=='post':
        p_category_train.append(G_train.nodes[i]['category'])
        created.append(G_train.nodes[i]['created'])
        last_payout.append(G_train.nodes[i]['last_payout'])
        p_payout_train.append(G_train.nodes[i]['payout'])
        p_body_train.append(G_train.nodes[i]['body'][0])
        #username.append(G_train.nodes[i]['username'])

    elif G_train.nodes()[i]['otype']=='user':
        u_otype_train.append(0)

    else:
        c_otype_train.append(2)
        c_category_train.append(G_train.nodes[i]['category'])

# getting number of posts in each class
def countFunction(elements):
    e_c = {}
    for element in elements:
        e_c[element] = elements.count(element)
    return e_c

counting_train = countFunction(p_category_train)
print(counting_train)
#username_train = countFunction(username)
#print(username_train)
df = pd.DataFrame(p_category_train)

# encoding labels
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
p_category_train = label_encoder.fit_transform(p_category_train)
print(p_category_train)

df['label'] = p_category_train
#df.to_csv(".../label.csv")

import preprocessor as p
import pandas as pd
from bs4 import BeautifulSoup
import re
import demoji
import string
import tensorflow as tf
import tensorflow_hub as hub

# preprocess text
print("Text pre-processing start...")
CLEANR = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')

def cleanhtml(raw_html):
    cleantext = re.sub(CLEANR, '', raw_html)
    cleantext = re.sub("[\(\[].*?[\)\]]", "", cleantext)
    return cleantext

def preprocessing(data):
    new_p_body_train, banned_list = [], []
    print(len(data))
    for i in range(len(data)):
        if i in []:
            new_p_body_train.append('')
            banned_list.append(i)
        elif type(data[i]) != str:
            new_p_body_train.append('')
        else:
            #print("Text No.: ",i)
            new_p_body_train.append(demoji.replace("".join([j for j in " ".join(p.clean(cleanhtml(BeautifulSoup(data[i], "lxml").text)).lower().translate(str.maketrans('', '', string.punctuation)).split()) if not j.isdigit()]), ""))
            #print(new_p_body_train[i])

    return new_p_body_train

p_body_train = preprocessing(p_body_train)
#pd.DataFrame(p_body_train).to_csv(".../train_test_prep.csv")
print("Text pre-proessing is Done!")
count=0
removal,res = [],[]
for i in range(len(p_body_train)):
    res.append(p_body_train[i].count(" ")+1)
    if p_body_train[i] == '':
        removal.append(i)
        count+=1
print(count)

# getting text embeddings
"""print("Google Universal Sentence encoder start...")
module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
model = hub.load(module_url)
print("module %s loaded..." % module_url)

p_body_train_embeddings = model(p_body_train).numpy()"""

# saving embedding
print("saving start")
import pandas as pd
#pd.DataFrame(p_body_train_embeddings).to_csv(".../train_test_embeddings.csv")
print("Embedding Saving is Done!")

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('bert-base-nli-mean-tokens')
#Encoding:
p_body_train_embeddings = model.encode(p_body_train)

# import embedding
"""print("importing embeddings")
#pd.DataFrame(p_body_train_embeddings).to_csv(".../p_body_train_embeddings.csv")
a_ = pd.read_csv(".../p_body_train_embeddings2.csv")
a_.drop('Unnamed: 0', axis=1, inplace=True)
p_body_train_embeddings = a_.values.astype('float32')
print("Embedding Imported!")"""

# putting all info so far in a dataframe
import pandas as pd
post_df_train = pd.concat([pd.DataFrame(p_body_train_embeddings),
                           pd.DataFrame(p_category_train, columns=['category']),
                           pd.DataFrame(p_payout_train, columns=['payout']),
                           #pd.DataFrame(vote_count_train, columns=['num_vote']),
                           #pd.DataFrame(comment_count_train, columns=['num_comment'])
                           ], axis=1, ignore_index=True)

comment_df_train = pd.DataFrame(c_otype_train, columns=['otype'])

user_df_train = pd.DataFrame(u_otype_train, columns =['otype'])
dataset_train = pd.concat([post_df_train, comment_df_train], ignore_index=True, axis=0)

df_otype_train = pd.concat([pd.DataFrame(c_otype_train, columns=['otype']),
                     pd.DataFrame(u_otype_train, columns=['otype'])],ignore_index=True, axis=0)

from sklearn.preprocessing import OneHotEncoder
# creating instance of one-hot-encoder
enc_train = OneHotEncoder(handle_unknown='ignore')
# passing bridge-types-cat column (label encoded values of bridge_types)
enc_df_train = pd.DataFrame(enc_train.fit_transform(df_otype_train).toarray())
# merge with main df bridge_df on key values
df_otype_train = df_otype_train.join(enc_df_train)
# column names
df_otype_train.columns=['otype','user','comment']

# train
p_body_train = []
p_category_train = []
p_payout_train = []
#vote_count_train = []
#comment_count_train = []

u_otype_u_train = []
u_otype_c_train = []

c_otype_u_train = []
c_otype_c_train = []

for i in range(post_df_train.shape[0]):
    p_body_train.append(dataset_train.iloc[i, 0:768])
    p_category_train.append(dataset_train.iloc[i, 768])     # [] removed to have y=[] instead of y=[ ,1]
    p_payout_train.append(dataset_train.iloc[i, 769])
    #vote_count_train.append(dataset_train.iloc[i, 514])
    #comment_count_train.append(dataset_train.iloc[i, 515])

for i in range(len(p_body_train)):
    p_body_train[i] = p_body_train[i].tolist()

for i in range(len(dataset_train),len(df_otype_train)):
    u_otype_u_train.append([df_otype_train.iloc[i, 1]])
    u_otype_c_train.append([df_otype_train.iloc[i, 2]])
    #u_otype_c_train.append([0])

for i in range(post_df_train.shape[0], len(dataset_train)):
    c_otype_u_train.append([df_otype_train.iloc[i, 1]])
    c_otype_c_train.append([df_otype_train.iloc[i, 2]])
    #c_otype_c_train.append([0])

# classify the payout
"""for i in range(len(p_payout_train)):
    if p_payout_train[i] < 1:
        p_payout_train[i] = 0
    elif p_payout_train[i] >= 1:
        p_payout_train[i] = 1"""

"""avg = np.mean(p_payout_train)
std = np.std(p_payout_train)
#pd.DataFrame(p_payout_train).to_csv(".../round1.csv")

for i in range(len(p_payout_train)):
    if p_payout_train[i] < (avg-2*std):
        p_payout_train[i] = 0
    elif p_payout_train[i] >= (avg-2*std) and p_payout_train[i] < (avg-std):
        p_payout_train[i] = 1
    elif p_payout_train[i] >= (avg-std) and p_payout_train[i] < (avg+std):
        p_payout_train[i] = 2
    elif p_payout_train[i] >= (avg+std) and p_payout_train[i] < (avg+2*std):
        p_payout_train[i] = 3
    elif p_payout_train[i] >= (avg+2*std):
        p_payout_train[i] = 4"""

# number of post in each class
"""counting_payout = countFunction(p_payout_train)
print(counting_payout)
print("Avg: ", avg)
print("STD: ", std)"""

# convert to torch
p_body_train = torch.tensor(p_body_train).float()
p_category_train = torch.tensor(p_category_train).long()
p_payout_train = torch.tensor(p_payout_train).long()
#vote_count_train = torch.tensor(vote_count_train).float()
#comment_count_train = torch.tensor(comment_count_train).float()

u_otype_u_train = torch.tensor(u_otype_u_train).float()
u_otype_c_train = torch.tensor(u_otype_c_train).float()

c_otype_u_train = torch.tensor(c_otype_u_train).float()
c_otype_c_train = torch.tensor(c_otype_c_train).float()

# normalize
import torch.nn.functional as F
#vote_count_train = F.normalize(vote_count_train, dim=-1)
#comment_count_train = F.normalize(comment_count_train, dim=-1)

# creating edges
print("Start creating edges...")
vote_p_i_train = []
vote_p_j_train = []
vote_c_i_train = []
vote_c_j_train = []
authored_p_i_train = []
authored_p_j_train = []
authored_c_i_train = []
authored_c_j_train = []
reply_i_train = []
reply_j_train = []
v_p_otype_train = []
a_p_otype_train = []
v_c_otype_train = []
a_c_otype_train = []
r_otype_train = []
v_p_time_train = []
a_p_time_train = []
v_c_time_train = []
a_c_time_train = []
r_time_train = []

post_train, comment_train, user_train = {}, {}, {}
post_idx_train, comment_idx_train, user_idx_train = 0, 0, 0
for i in range(len(G_train.nodes())):
    if G_train.nodes()[i]['otype'] == 'user':
        user_train[i] = user_idx_train
        user_idx_train += 1
    elif G_train.nodes()[i]['otype'] == 'post':
        post_train[i] = post_idx_train
        post_idx_train += 1
    else:
        comment_train[i] = comment_idx_train
        comment_idx_train += 1

for i in range(len(G_train.nodes())):
    if G_train.nodes()[i]['otype'] == 'user':
        for j in range(len(G_train.nodes())):
            if (i,j) in G_train.edges():
                if G_train.edges[i,j]['otype'] == 'vote':

                    if G_train.nodes()[j]['otype'] == 'post':
                        #G.edges[i,j]['otype'] = 1  # 'otype' = vote
                        v_p_otype_train.append([torch.tensor(1)])
                        v_p_time_train.append([int(G_train.edges[i,j]['time'])])
                        vote_p_i_train.append(user_train[i])
                        vote_p_j_train.append(post_train[j])
                    elif G_train.nodes()[j]['otype'] == 'comment':
                        #G.edges[i,j]['otype'] = 1  # 'otype' = vote
                        v_c_otype_train.append([torch.tensor(1)])
                        v_c_time_train.append([int(G_train.edges[i,j]['time'])])
                        vote_c_i_train.append(user_train[i])
                        vote_c_j_train.append(comment_train[j])

                elif G_train.edges[i,j]['otype'] == 'authored':

                    if G_train.nodes()[j]['otype'] == 'post':
                        #G.edges[i,j]['otype'] = 0 # 'otype' = authored
                        a_p_otype_train.append([torch.tensor(0)])
                        a_p_time_train.append([int(G_train.edges[i,j]['time'])])
                        authored_p_i_train.append(user_train[i])
                        authored_p_j_train.append(post_train[j])
                    elif G_train.nodes()[j]['otype'] == 'comment':
                        #G.edges[i,j]['otype'] = 0 # 'otype' = authored
                        a_c_otype_train.append([torch.tensor(0)])
                        a_c_time_train.append([int(G_train.edges[i,j]['time'])])
                        authored_c_i_train.append(user_train[i])
                        authored_c_j_train.append(comment_train[j])

    elif G_train.nodes()[i]['otype'] == 'post':
        for j in range(len(G_train.nodes())):
            if (i,j) in G_train.edges():
                if G_train.edges[i,j]['otype'] == 'reply':
                    #G.edges[i,j]['otype'] = 2 # 'otype' = reply
                    reply_i_train.append(post_train[i])
                    reply_j_train.append(comment_train[j])
                    r_otype_train.append([torch.tensor(2)])
                    r_time_train.append([int(G_train.edges[i,j]['time'])])

print("Creating edges finished!")

# convert to torch
vote_p_i_train = torch.tensor(vote_p_i_train)
vote_p_j_train = torch.tensor(vote_p_j_train)
vote_c_i_train = torch.tensor(vote_c_i_train)
vote_c_j_train = torch.tensor(vote_c_j_train)
authored_p_i_train = torch.tensor(authored_p_i_train)
authored_p_j_train = torch.tensor(authored_p_j_train)
authored_c_i_train = torch.tensor(authored_c_i_train)
authored_c_j_train = torch.tensor(authored_c_j_train)
reply_i_train = torch.tensor(reply_i_train)
reply_j_train = torch.tensor(reply_j_train)
v_p_otype_train = torch.tensor(v_p_otype_train)
a_p_otype_train = torch.tensor(a_p_otype_train)
v_c_otype_train = torch.tensor(v_c_otype_train)
a_c_otype_train = torch.tensor(a_c_otype_train)
r_otype_train = torch.tensor(r_otype_train)
v_p_time_train = torch.tensor(v_p_time_train)
a_p_time_train = torch.tensor(a_p_time_train)
v_c_time_train = torch.tensor(v_c_time_train)
a_c_time_train = torch.tensor(a_c_time_train)
r_time_train = torch.tensor(r_time_train)

# reshape edges for building graph
vote_p_train = torch.cat((vote_p_i_train,vote_p_j_train)).reshape(-1,len(vote_p_i_train)).long()
vote_c_train = torch.cat((vote_c_i_train,vote_c_j_train)).reshape(-1,len(vote_c_i_train)).long()
authored_p_train = torch.cat((authored_p_i_train,authored_p_j_train)).reshape(-1,len(authored_p_i_train)).long()
authored_c_train = torch.cat((authored_c_i_train,authored_c_j_train)).reshape(-1,len(authored_c_i_train)).long()
reply_train = torch.cat((reply_i_train,reply_j_train)).reshape(-1,len(reply_i_train)).long()

# building heterogeneous graph
print("Start building heterogeneous graph...")
data_train = HeteroData()

# import propogated features
"""user_feat = pd.read_csv('.../user_feat_health_cat.csv')
user_feat.drop('Unnamed: 0', axis=1, inplace=True)
cm_feat = pd.read_csv('.../cm_feat_health_cat.csv')
cm_feat.drop('Unnamed: 0', axis=1, inplace=True)

user_feat_train, cm_feat_train = [], []

for i in range(user_feat.shape[0]):
    user_feat_train.append(user_feat.iloc[i, 0:512])
for i in range(cm_feat.shape[0]):
    cm_feat_train.append(cm_feat.iloc[i, 0:512])

user_feat_train = torch.tensor(user_feat_train).float()
cm_feat_train = torch.tensor(cm_feat_train).float()"""

# Post features
# group all your features of a single node type into one feature matrix
data_train['post'].x = torch.cat([p_body_train], dim=-1)
data_train['post'].y = p_category_train

# User features
data_train['user'].x = torch.cat([u_otype_u_train, u_otype_c_train], dim=-1)
#data_train['user'].x = torch.cat([user_feat_train], dim=-1)

# Comment features
# group all your features of a single node type into one feature matrix
data_train['comment'].x = torch.cat([c_otype_u_train, c_otype_c_train], dim=-1)
#data_train['comment'].x = torch.cat([cm_feat_train], dim=-1)

# edge indices
data_train['user', 'authored_post', 'post'].edge_index = authored_p_train
data_train['user', 'authored_comment', 'comment'].edge_index = authored_c_train
data_train['user', 'vote_post', 'post'].edge_index = vote_p_train
data_train['user', 'vote_comment', 'comment'].edge_index = vote_c_train
data_train['post', 'reply', 'comment'].edge_index = reply_train

# edge features
data_train['user', 'authored_post', 'post'].edge_attr = torch.cat([a_p_otype_train, a_p_time_train], dim=-1)
data_train['user', 'authored_comment', 'comment'].edge_attr = torch.cat([a_c_otype_train, a_c_time_train], dim=-1)
data_train['user', 'vote_post', 'post'].edge_attr = torch.cat([v_p_otype_train, v_p_time_train], dim=-1)
data_train['user', 'vote_comment', 'comment'].edge_attr = torch.cat([v_c_otype_train, v_c_time_train], dim=-1)
data_train['post', 'reply', 'comment'].edge_attr = torch.cat([r_otype_train, r_time_train], dim=-1)

# create edge reverse
import torch_geometric.transforms as T
data_train = T.ToUndirected()(data_train)
#data_train = T.RandomNodeSplit('train_rest', num_val=190, num_test=3)(data_train)

# saving the graphs
import dill
print("Saving Heterogeneous Graph!")
dill.dump(data_train, open('.../train+test.pk', 'wb'))
