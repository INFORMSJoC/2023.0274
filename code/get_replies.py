#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 19:39:16 2023

@author: negarmaleki
"""


# import libraries
import pandas as pd
from steem import Steem
import time
from datetime import datetime
s = Steem()

# Load Steemit Dataset
data_steemit = pd.read_csv(".../fivecat1000_data_same_num.csv")
#data_steemit = pd.read_csv(".../health_data_same_num.csv")
data_steemit.drop('Unnamed: 0', axis=1, inplace=True)

# Retriving posts with API
posts = []
for t in range(int(len(data_steemit)/2000)+1):
    print(t)
    if t == int(len(data_steemit)/2000):
        for j in range(t*2000, len(data_steemit)):
            if type(data_steemit.iloc[j,1]) != float and type(data_steemit.iloc[j,2]) != float:
                posts.append(s.get_content_replies(data_steemit.iloc[j,1],data_steemit.iloc[j,2]))
    else:
        for j in range(t*2000, (t+1)*2000):
            if type(data_steemit.iloc[j,1]) != float and type(data_steemit.iloc[j,2]) != float:
                posts.append(s.get_content_replies(data_steemit.iloc[j,1],data_steemit.iloc[j,2]))
    time.sleep(200)

replies_dataset = pd.DataFrame(posts)
replies_dataset.to_csv(".../fivecat300_data_replies_train+test.csv")
#replies_dataset.to_csv(".../health_data_replies_train+test.csv")
