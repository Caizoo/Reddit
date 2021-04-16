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
import tqdm

from migration.migrate_helper import fetch_top_users_from_file

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
    pass 

def run_micro(args: dict):
    pass 