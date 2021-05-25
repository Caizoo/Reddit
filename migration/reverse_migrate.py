import json 
import multiprocessing
import matplotlib.pyplot as plt 
from pandas.core import series
import pymongo 
from pymongo import MongoClient 
import argparse
import datetime as dt
import pandas as pd 
import itertools
from scipy import stats
import numpy as np 
from tqdm import tqdm
from functools import reduce
import pickle 

from migration.migrate_helper import fetch_rand_users_from_file

# TODO: redo this but temporaly, so by day/month and have a dict 'graph' instead of a single graph
def run_reverse_migration(args: dict):
    users = fetch_rand_users_from_file(args['sub'])  
    mongodb_address, username, password, database_str = args['mongodb_address'], args['username'], args['password'], args['database_str'] 
    client = MongoClient(f'mongodb://{username}:{password}@{mongodb_address}/', connect=False) if username!='' else MongoClient(f'mongodb://{mongodb_address}/', connect=False)
    db = client[database_str] 

    comments = pd.DataFrame(db['User_Comments'].find({'author': {'$in': users}})).sort_values(by='created_utc', ascending=True)
    comments['created'] = [dt.datetime.fromtimestamp(a, tz=dt.timezone.utc) for a in comments['created_utc']]

    all_graphs = {}
    all_author = {} 
    comments_d = comments.groupby(pd.Grouper(key='created', freq='1M'))
    comments_g_s = list(comments.groupby('subreddit').size().sort_values(ascending=False).index[:20].values)

    for d, comments_month in tqdm(comments_d):
        comments_g_a = comments_month.groupby('author') 

        graphs = []
        author_size = comments_g_a.ngroups
        for a, g in comments_g_a:
            edges = {k: {k: 0} for k in comments_g_s}
            edges = dict({'other': {k: 0} for k in comments_g_s}, **edges)
            subs = g['subreddit'].values 
            subs = [a if a in comments_g_s else 'other' for a in subs]
            for i in range(1, len(subs)):
                if subs[i-1] not in edges.keys(): edges[subs[i-1]] = {} 
                edges[subs[i-1]][subs[i]] = edges[subs[i-1]].get(subs[i], 0) + 1
            
            edges = pd.DataFrame(edges) 
            e_np = edges.to_numpy() 
            e_np = [[aa/len(g) for aa in a] for a in e_np]
            edges = pd.DataFrame(e_np, columns=edges.columns, index=edges.index)
            
            graphs.append(edges)

        graphs = pd.concat(graphs)
        by_row_index = graphs.groupby(graphs.index) 
        try:
            graphs = by_row_index.mean() 
        except pd.core.base.DataError as e:
            graphs = pd.DataFrame() 

        comments_g_s.sort() 
        graphs = graphs.reindex(comments_g_s+['other'])
        graphs = graphs[graphs.index]
        all_graphs[d] = graphs 
        all_author[d] = author_size

    pickle.dump(all_graphs, open('all_graphs.p', 'wb'))
    pickle.dump(all_author, open('all_author.p', 'wb'))
    # need to draw graph from DataFrame - print(graphs)
        