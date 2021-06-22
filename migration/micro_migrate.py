import json 
import multiprocessing
from os import remove
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
import seaborn as sns 

from migration.migrate_helper import fetch_top_users_from_file, fetch_rand_users_from_file

def migration_worker_sc(migrate_dict: dict):
    """ Multiprocessing worker for run_micro_sc

    Args:
        migrate_dict (dict): Same as args dict from run_micro_sc, with added 'user' key denoting the reddit user

    Returns:
        dict: Keys are subreddits, and values the probability of this user posting to that subreddit (no edges considered)
    """
    user = migrate_dict['user']
    unix_start = dt.datetime(migrate_dict['year'], 1, 1, tzinfo=dt.timezone.utc) 

    mongodb_address, username, password, database_str = migrate_dict['mongodb_address'], migrate_dict['username'], migrate_dict['password'], migrate_dict['database_str']
    client = MongoClient(f'mongodb://{username}:{password}@{mongodb_address}/', connect=False) if username!='' else MongoClient(f'mongodb://{mongodb_address}/', connect=False)
    db = client[database_str] 

    db_filter = {'author': 1, 'created_utc': 1, 'subreddit': 1} 

    # Fetch comments, submissions, or both made by this user across all subreddit (e.g. User_Comments rather than Comments)
    collection_str = 'User_Comments' if migrate_dict['d_type']=='comments' else 'User_Submissions'
    data = pd.DataFrame(db[collection_str].find({'author': user}, db_filter)) 
    if migrate_dict['d_type']=='both':
        other_data = pd.DataFrame(db['User_Comments'].find({'author': user}, db_filter)) 
        data = pd.concat([data, other_data])
    
    if not len(data): # If nothing on this user, return empty dict
        client.close() 
        return {'top': {}}

    # Group by subreddit and get number of posts for each subreddit, divide by total number of posts gives the probablility 
    data_dict = data.groupby('subreddit').size().sort_values(ascending=False).to_dict()
    data_dict = {k: data_dict[k]/len(data) for k in data_dict.keys()}

    return data_dict

def run_micro_sc(args: dict):
    """ Function to find an optimal set of subreddit for S_c by finding the top subreddits by post probability across a set of users.

    Args:
        args (dict):
            mongodb_address: mongodb address, 
            username: mongo username, 
            password: mongo password, 
            database_str: database name,
            d_type: comments or submissions,
            sub: subreddit name if using top users,
            file_str: file location of users of interest - if scalp/cache/top_users.json then use subreddit name to get top users of a subreddit,
    """
    users_oi = list() # users of interest
    if args['file_str']=='scalp/cache/top_users.json':
        users_oi = fetch_top_users_from_file(args['sub'])
    else:
        users_oi = list(json.load(open(args['file_str'], 'r'))['users']) 

    mongodb_address, username, password, database_str = args['mongodb_address'], args['username'], args['password'], args['database_str'] 
    client = MongoClient(f'mongodb://{username}:{password}@{mongodb_address}/', connect=False) if username!='' else MongoClient(f'mongodb://{mongodb_address}/', connect=False)
    db = client[database_str] 

    users_oi_dicts = [dict({'user': a}, **args) for a in users_oi]
    
    pool = multiprocessing.Pool(processes=args['cpus']) 
    l = list(tqdm.tqdm(pool.imap_unordered(migration_worker_sc, users_oi_dicts), total=len(users_oi_dicts)))
    pool.close() 

    # Mean over users subreddit probablities 
    tops = [pd.Series(ll, index=ll.keys()) for ll in l] 
    tops = pd.concat(tops, axis=1).fillna(0.0).mean(axis=1).sort_values(ascending=False)

    # Print the top 50 by probability 
    print(tops.head(50))  
    
    


def migration_worker(migrate_dict: dict):
    a, g, comments_g_s = migrate_dict['a'], migrate_dict['g'], migrate_dict['comments_g_s']
    edges = {} 
    subs = g['subreddit'].values 
    subs = [a if a in comments_g_s else 'other' for a in subs]
    for i in range(1, len(subs)):
        if subs[i-1] not in edges.keys(): edges[subs[i-1]] = {} 
        edges[subs[i-1]][subs[i]] = edges[subs[i-1]].get(subs[i], 0) + 1
        
    edges = pd.DataFrame(edges) 
    e_np = edges.to_numpy() 
    e_np = [[aa/len(g) for aa in a] for a in e_np]
    edges = pd.DataFrame(e_np, columns=edges.columns, index=edges.index)
    edges = edges.fillna(0.0)

    return edges 

def run_micro(args: dict):
    users = fetch_top_users_from_file(args['sub'])  
    mongodb_address, username, password, database_str = args['mongodb_address'], args['username'], args['password'], args['database_str'] 
    client = MongoClient(f'mongodb://{username}:{password}@{mongodb_address}/', connect=False) if username!='' else MongoClient(f'mongodb://{mongodb_address}/', connect=False)
    db = client[database_str] 

    unix_year = dt.datetime(args['year'], 1, 1, tzinfo=dt.timezone.utc).timestamp()
    comments = pd.DataFrame(db['User_Comments'].find({'author': {'$in': users}, 'created_utc': {'$gte': unix_year}})).sort_values(by='created_utc', ascending=True)

    comments_g_a = comments.groupby('author') 
    comments_g_s = list(comments.groupby('subreddit').size().sort_values(ascending=False).index[:args['top_x']].values)

    graphs = []
    migrate_list = [{'a': a, 'g': g, 'comments_g_s': comments_g_s} for a, g in comments_g_a]
    pool = multiprocessing.Pool()
    graphs = tqdm(pool.imap_unordered(migration_worker, migrate_list), total=len(migrate_list))
    pool.close() 

    graphs = pd.concat(graphs)
    by_row_index = graphs.groupby(graphs.index) 
    graphs = by_row_index.mean()

    df = graphs.copy() 
    df = df[df.index] 
    df = df.fillna(0.0)

    remove_loops = True 
    if remove_loops:
        for c in df.columns: 
            df.loc[c, c] = 0.0
    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(10, 6))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    g = sns.heatmap(df, cmap=cmap, center=0, linewidths=.5, cbar_kws={"shrink": .5}, square=True)
    g.set_xticklabels(g.get_xmajorticklabels(), fontsize = 8)
    g.set_yticklabels(g.get_ymajorticklabels(), fontsize = 8)
    #sns.heatmap(df)
    plt.tight_layout() 
    plt.show()