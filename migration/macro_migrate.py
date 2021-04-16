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


def run_macro(args: dict):
    """ Run macro migration and save the resulting graph

    Args:
        args (dict):
            mongodb_address: mongodb address, 
            username: mongo username, 
            password: mongo password, 
            database_str: database name,
            d_type: comments or submissions,
            sub: subreddit name if using top users,
            file_str: file location of users of interest - if scalp/cache/top_users.json then use subreddit name to get top users of a subreddit,
            plot_file_str: file location to save macro plot (_normalised added for normalised plot)
    """
    users_oi = list() # users of interest
    if args['file_str']=='scalp/cache/top_users.json':
        users_oi = fetch_top_users_from_file(args['sub'])
    else:
        users_oi = list(json.load(open(args['file_str'], 'r'))['users'])
    
    # connect to database
    mongodb_address, username, password, database_str = args['mongodb_address'], args['username'], args['password'], 'Reddit_Controversial'
    client = MongoClient(f'mongodb://{username}:{password}@{mongodb_address}/', connect=False) if username!='' else MongoClient(f'mongodb://{mongodb_address}/', connect=False)
    db = client[database_str] 

    # TODO: add both to this
    # Fetch specified data (comments or submissions) and add date data 
    d_type = 'Comments' if args['d_type']=='comments' else 'Submissions'
    data = pd.DataFrame(db['User_'+d_type].find({'author': {'$in': users_oi}}, {'_id': 1, 'subreddit': 1, 'author': 1, 'created_utc': 1}))
    data['date'] = [dt.datetime.fromtimestamp(a, tz=dt.timezone.utc) for a in data['created_utc']]

    # Group by author then subreddit
    data_author = data.groupby('author')
    dfs_norm = list()
    dfs = list() 
    for name, group in data_author:
        subs_series = list() 
        data_subreddit = group.groupby('subreddit') 
        for sub, sub_group in data_subreddit:
            # Create time series for user 'name' and subreddit 'sub' across time
            series_s = sub_group.groupby(pd.Grouper(key='date', freq='1D')).size() 
            series_s = series_s.rename(sub)
            subs_series.append(series_s)
        
        # Find the maximum value of all time series for user 'name'
        subs_series_max = max([max([max(a) for a in subs_series]), 1])
        df = pd.concat(subs_series, axis=1) 
        # Scale all time series by the value in subs_series_max
        df_np = [[aa/subs_series_max for aa in a] for a in df.to_numpy()]

        dfs_norm.append(pd.DataFrame(data=df_np, index=df.index, columns=df.columns)) # normalised time series
        dfs.append(df) # absolute time series 

    dfs = pd.concat(dfs)
    dfs_norm = pd.concat(dfs_norm)

    top_subs = dfs_norm.sum().sort_values(ascending=False).index[:20] # find top 20 subreddits based on absolute sum across all users

    # TODO: SAVE AS SVG using args['plot_file_str']

    # Average over users
    by_row_index = dfs.groupby(dfs.index)
    df_means = by_row_index.mean()
    df_means = df_means.loc['2019-11-01':]
    df_means = df_means[top_subs]
    
    legs = list() 
    for c in df_means.columns:
        df_c = df_means[c].ewm(7).mean() 
        df_c.plot() 
        legs.append(c) 

    plt.title('Summation of absolute user activity')
    plt.legend(legs) 
    plt.show() 

    by_row_index = dfs_norm.groupby(dfs_norm.index)
    df_means = by_row_index.mean()
    df_means = df_means.loc['2019-11-01':]
    df_means = df_means[top_subs]
    
    legs = list() 
    for c in df_means.columns:
        df_c = df_means[c].ewm(7).mean() 
        df_c.plot() 
        legs.append(c) 

    plt.title('Mean of normalised user activity')
    plt.legend(legs) 
    plt.show() 
        
        
        

