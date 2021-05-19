import numpy as np 
import pandas as pd 
import matplotlib as mpl 
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns 
import matplotlib.pyplot as plt 
import optuna
import json
import sklearn 
from scipy.sparse.linalg import eigsh 
import itertools 


from ipywidgets import interact, interactive, fixed, interact_manual, Layout
import ipywidgets as widgets
from matplotlib import animation, rc
from IPython.display import HTML, Image

import pymongo
from pymongo import MongoClient, errors
import pickle
import praw  
import prawcore 
from psaw import PushshiftAPI
import datetime as dt 
from dateutil.relativedelta import relativedelta
import json 
from multiprocessing.pool import Pool, ThreadPool
import argparse
import time
import os 
import shutil
import random
import multiprocessing
import tqdm
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA 

import tslearn 
from tslearn.clustering import TimeSeriesKMeans, KShape

# statsmodels 
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.stattools import adfuller, arma_order_select_ic, grangercausalitytests, coint
from statsmodels.tsa.seasonal import seasonal_decompose 
from statsmodels.tsa.api import acf, pacf, graphics
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.structural import UnobservedComponents
from statsmodels.tsa.vector_ar.vecm import coint_johansen as coint_johansen


# NLP 
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  
import gensim 
from gensim.utils import simple_preprocess 
from gensim.parsing.preprocessing import STOPWORDS 
from gensim.models import Word2Vec  
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import nltk 
from nltk.stem import WordNetLemmatizer, SnowballStemmer 
from nltk.stem.porter import * 
from nltk.corpus import stopwords

import re
import urllib

import notebook_library.windows
from notebook_library.windows import *# get_subreddit_top_data, get_user_windows
from notebook_library.windows import get_subreddit_top_data, get_user_windows, unix_range 
import notebook_library.word 
from notebook_library.word import *
from notebook_library.word import get_word_series
#import notebook_library.migrate
#from notebook_library.migrate import * 


# DEEP ML MODELS 
import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras import layers 
from tensorflow.keras import backend as K 
from tensorflow.python.ops import array_ops

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

#%load_ext rpy2.ipython

vader_analyzer = SentimentIntensityAnalyzer()

def intersection(lst1, lst2): 
    if type(lst1)!=list: return lst2
    if type(lst2)!=list: return lst1 
    return list(set(lst1) & set(lst2)) 

# SUB DICT DATA -----------------------------------------------------------------------------------------------------------------

def sentiment_worker(body: str):
    return vader_analyzer.polarity_scores(body)['compound']

def sub_worker(tup: tuple):
    counts = list()
    sub_dict_s = dict()

    for d_type in ['comments', 'submissions']:
        sub_dict_s[d_type] = tup[1][d_type][tup[1][d_type]['subreddit']==tup[0]]
        a = sub_dict_s[d_type].groupby(pd.Grouper(key='date', freq='1D')).size() 
        try: a.index = a.index.tz_convert(tz='utc')
        except:
            try: a.index = a.index.tz_localize(tz='utc')
            except: pass 
        counts.append(a)

    sub_dict_s['count'] = pd.concat(counts, axis=1)
    sub_dict_s['count'].columns = ['Comments', 'Submissions']
    for w in tup[2]:
        count_body = get_word_series(sub_dict_s['comments'], 'body', w)
        count_title = get_word_series(sub_dict_s['submissions'], 'title', w)
        count_selftext = get_word_series(sub_dict_s['submissions'], 'selftext', w)

        try:count_body.index = count_body.index.tz_convert(tz='utc')
        except:
            try: count_body.index = count_body.index.tz_localize(tz='utc')
            except Exception as e: pass
        try:count_title.index = count_title.index.tz_convert(tz='utc')
        except:
            try: count_title.index = count_title.index.tz_localize(tz='utc')
            except Exception as e: pass
        try:count_selftext.index = count_selftext.index.tz_convert(tz='utc')
        except:
            try: count_selftext.index = count_selftext.index.tz_localize(tz='utc')
            except Exception as e: pass
        
        sub_cols = list(filter(lambda a: len(a[0])>0, [(count_body, "Comments body"), (count_title, "Submissions title"), (count_selftext, "Submissions selftext")]))
        if len(sub_cols)>0:
            sub_dict_s[w] = pd.concat([a[0] for a in sub_cols], axis=1)
            sub_dict_s[w].columns = [a[1] for a in sub_cols]
        else:
            sub_dict_s[w] = pd.DataFrame()

    return (tup[0], sub_dict_s)

def get_sub_dict(sub_str: str, top_subs: int=10): 
    x = pickle.load(open(f"./notebook_library/cache/{sub_str}/sub_dict.p", "rb")) 
    top_submissions, top_comments = x['top_subs_submissions'], x['top_subs_comments']

    keep_subs = list(set(list(top_submissions.index)[:top_subs] + list(top_comments.index)[:top_subs]))

    new_dict = dict() 
    new_dict['users'] = x['users'] 
    new_dict['top_subs_comments'] = top_comments.index[:top_subs]
    new_dict['top_subs_submissions'] = top_submissions.index[:top_subs]
    new_dict['comments'] = x['comments']
    new_dict['submissions'] = x['submissions']


    for s in keep_subs:
        new_dict[s] = x[s]
    
    return new_dict

def refresh_sub_dict_cache(sub_str: str, db: pymongo.database.Database, top_x: int = 500, start_year: int = 2020):
    try: os.remove(f"./notebook_library/cache/{sub_str}/sub_dict.p")
    except: pass 
    try: os.mkdir(f"./notebook_library/cache/{sub_str}")
    except: pass 
    
    KEY_WORDS_LIST = json.load(open("./constants.json"))['KEY_WORDS_LIST']

    sub_dict = dict() 

    # GET SUBMISSIONS AND COMMENTS MADE ON SUBREDDIT
    subs, comms = (pd.DataFrame(db['Submissions'].find({'subreddit': sub_str})), 
                                  pd.DataFrame(db['Comments'].find({'subreddit': sub_str})))
    
    comms['date'] = [dt.datetime.fromtimestamp(a, tz=dt.timezone.utc) for a in comms['created_utc']]
    subs['date'] = [dt.datetime.fromtimestamp(a, tz=dt.timezone.utc) for a in subs['created_utc']]
    comms = comms.groupby(pd.Grouper(key='date', freq='1D')).size()
    subs = subs.groupby(pd.Grouper(key='date', freq='1D')).size()

    sub_dict['comments'] = comms 
    sub_dict['submissions'] = subs

    # TOP USER DATA 

    all_subs, all_comments, subs_user_windows, comment_user_windows = get_subreddit_top_data(sub_str, db, top_x=top_x, start_year=start_year)
    all_subs['date'] = [dt.datetime.fromtimestamp(a, tz=dt.timezone.utc) for a in all_subs['created_utc']]
    all_comments['date'] = [dt.datetime.fromtimestamp(a, tz=dt.timezone.utc) for a in all_comments['created_utc']]

    subs_user_windows = pd.Series(subs_user_windows).map(lambda a: [b[0] for b in a])
    comment_user_windows = pd.Series(comment_user_windows).map(lambda a: [b[0] for b in a])

    submissions_subs = all_subs.groupby(pd.Grouper(key='subreddit')).size().sort_values(ascending=False)
    comment_subs = all_comments.groupby(pd.Grouper(key='subreddit')).size().sort_values(ascending=False)

    subs_keep = list(set(submissions_subs.index[:10].append(comment_subs.index[:10]))) # keep top 20 subs to hold in cache 

    subs_keep = list(set(subs_keep + [sub_str])) # also keep the ego-centric sub in case it's not in top 20 (e.g. donaldtrump)
    sub_dict['top_subs_comments'] = comment_subs[:10] 
    sub_dict['top_subs_submissions'] = submissions_subs[:10]

    all_subs = all_subs[all_subs.subreddit.isin(subs_keep)]
    all_comments = all_comments[all_comments.subreddit.isin(subs_keep)]
    sub_dict['users'] = pd.concat([comment_user_windows, subs_user_windows], axis=1) 
    sub_dict['users'].columns = ["Comments", "Submissions"] 
    sub_dict['users']['Both'] = [intersection(row['Comments'], row['Submissions']) for index, row in sub_dict['users'].iterrows()]

    all_data = {'comments': all_comments, 'submissions': all_subs}     

    sub_worker_list = list() 
    for s in subs_keep:
        sub_worker_list.append((s, all_data.copy(), KEY_WORDS_LIST.copy()))
    
    pool = multiprocessing.Pool() 
    #res = list(pool.map(sub_worker, sub_worker_list)) 
    res = list(tqdm.tqdm(pool.imap_unordered(sub_worker, sub_worker_list), total=len(sub_worker_list)))
    pool.close()

    for s, d in res: sub_dict[s] = d

    pickle.dump(sub_dict, open(f"./notebook_library/cache/{sub_str}/sub_dict.p", "wb"))



# MIGRATION --------------------------------------------------------------------------------------------------------------------------

def get_migrate_dict(sub_str: str):
    return pickle.load(open(f"./notebook_library/cache/{sub_str}/migrate_dict.p", "rb")) 

def refresh_migrate_dict_cache(sub_str: str, db: pymongo.database.Database, top_x: int = 500, start_year: int = 2020):
    try: os.remove(f"./notebook_library/cache/{sub_str}/migrate_dict.p")
    except: pass 
    try: os.mkdir(f"./notebook_library/cache/{sub_str}")
    except: pass 

    KEY_WORDS_LIST = json.load(open("./constants.json"))['KEY_WORDS_LIST']

    migrate_dict = dict() 
    
    subs_user_windows, comment_user_windows = get_user_windows(sub_str, db, top_x=top_x, freq='1M', start_year=start_year)

    dates = list(subs_user_windows.keys()) + list(comment_user_windows.keys()) 
    dates = list(set(dates)) 
    dates.sort()  

    # TODO: multiprocessing of this 

    for d in dates:
        print(f"Collecting top user data for {d}")
        migrate_dict[d] = {'submissions': [], 'comments': [], 'top_subs_submissions_size': [], 'top_subs_comments_size': [], 'top_subs_comments': [], 'top_subs_submissions': []}
        users = list() 
        try: users = users + [a[0] for a in subs_user_windows[d]]
        except Exception as e: print(e) 
        try: users = users + [a[0] for a in comment_user_windows[d]]
        except Exception as e: print("Comment", e) 

        unix_low, unix_high = unix_range(d.year, d.month) 
        all_subs = db['User_Submissions'].find({'author': {'$in': users}})
        all_comms = db['User_Comments'].find({'author': {'$in': users}})

        all_subs = pd.DataFrame(all_subs)
        all_comms = pd.DataFrame(all_comms)

        if len(all_subs)==0 or len(all_comms)==0: continue

        all_subs_g = all_subs.groupby('subreddit') 
        all_comms_g = all_comms.groupby('subreddit')

        top_subs_submissions = list(all_subs_g.size().sort_values(ascending=False).iloc[:10].index)
        top_subs_comments = list(all_comms_g.size().sort_values(ascending=False).iloc[:10].index)

        subs_keep = list(set(top_subs_submissions + top_subs_comments)) # keep top 20 subs to hold in cache 
        subs_keep = list(set(subs_keep + [sub_str]))
        migrate_dict[d]['top_subs_comments'] = top_subs_comments
        migrate_dict[d]['top_subs_submissions'] = top_subs_submissions

        migrate_dict[d]['top_subs_submissions_size'] = all_subs_g.size().sort_values(ascending=False)
        migrate_dict[d]['top_subs_comments_size'] = all_comms_g.size().sort_values(ascending=False)

        all_subs['date'] = [dt.datetime.fromtimestamp(a, tz=dt.timezone.utc) for a in all_subs['created_utc']]
        all_comms['date'] = [dt.datetime.fromtimestamp(a, tz=dt.timezone.utc) for a in all_comms['created_utc']]
        all_subs = all_subs[all_subs['subreddit'].isin(subs_keep)]
        all_comms = all_comms[all_comms['subreddit'].isin(subs_keep)]

        all_subs_g = all_subs.groupby('subreddit') 
        all_comms_g = all_comms.groupby('subreddit')

        subs_count = dict()
        comms_count = dict() 

        for name, group in all_subs_g:
            g = group.groupby(pd.Grouper(key='date', freq='1D')).size() 
            subs_count[name] = g
        for name, group in all_comms_g:
            g = group.groupby(pd.Grouper(key='date', freq='1D')).size() 
            comms_count[name] = g

        migrate_dict[d]['submissions'] = subs_count
        migrate_dict[d]['comments'] = comms_count 

    pickle.dump(migrate_dict, open(f"./notebook_library/cache/{sub_str}/migrate_dict.p", "wb"))