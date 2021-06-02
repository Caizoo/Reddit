from pandas.core.frame import DataFrame
from pymongo import MongoClient
import pandas as pd 
import sklearn as sk 
import datetime as dt
import matplotlib.pyplot as plt 
import seaborn as sns 
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  
import multiprocessing
from tqdm import tqdm
import numpy as np 

def sent(comm):
    analzyer = SentimentIntensityAnalyzer() 
    try:
        comm['sentiment'] = analzyer.polarity_scores(comm['body'])['compound']
    except: 
        comm['sentiment'] = np.nan
    return comm 

def sent_sub(sub):
    analzyer = SentimentIntensityAnalyzer() 
    a, b = 0, 0
    try:
        a = analzyer.polarity_scores(sub['title'])['compound']
        b = analzyer.polarity_scores(sub['selftext'])['compound']
        sub['sentiment'] = np.mean([a, b])
    except: 
        sub['sentiment'] = np.nan
    
    return sub 

if __name__=="__main__":
    SUB = 'Coronavirus'
    client = MongoClient('mongodb://admin:password@192.168.1.123') 
    db = client['Reddit'] 

    comms, subs = pd.DataFrame(db['Comments'].find({'subreddit': SUB})), pd.DataFrame(db['Submissions'].find({'subreddit': SUB}))
    comms['date'], subs['date'] = [dt.datetime.fromtimestamp(a, tz=dt.timezone.utc) for a in comms['created_utc']], [dt.datetime.fromtimestamp(a, tz=dt.timezone.utc) for a in subs['created_utc']]

    comms = comms[comms['date']<dt.datetime(2021,4,1, tzinfo=dt.timezone.utc)]
    subs = subs[subs['date']<dt.datetime(2021,4,1,tzinfo=dt.timezone.utc)]

    comms_g = comms.groupby(pd.Grouper(key='date', freq='1D')) 
    subs_g = subs.groupby(pd.Grouper(key='date', freq='1D'))
    
    hours_dict_comms, hours_dict_subs = {}, {}
    for _, a in comms.iterrows():
        hours_dict_comms[a['date'].hour] = hours_dict_comms.get(a['date'].hour, 0) + 1
    for _, a in subs.iterrows():
        hours_dict_subs[a['date'].hour] = hours_dict_subs.get(a['date'].hour, 0) + 1

    hours_dict_comms = {k: hours_dict_comms[k]/len(comms) for k in hours_dict_comms.keys()}
    hours_dict_subs = {k: hours_dict_subs[k]/len(subs) for k in hours_dict_subs.keys()}

    hours_dict = pd.DataFrame([hours_dict_comms, hours_dict_subs], index=['Comments', 'Submissions']).T.sort_index()

    print(hours_dict)

    pool = multiprocessing.Pool()
    comms_items = [a for _, a in comms.iterrows()]
    comms = pd.DataFrame(tqdm(pool.imap_unordered(sent, comms_items), total=len(comms_items)))
    pool.close() 

    pool = multiprocessing.Pool() 
    subs_items = [a for _, a in subs.iterrows()]
    subs = pd.DataFrame(tqdm(pool.imap_unordered(sent_sub, subs_items), total=len(subs_items)))
    pool.close() 
    
    plt.figure()
    plt.plot(comms[['date', 'sentiment']].groupby(pd.Grouper(key='date', freq='1D')).mean()) 
    plt.plot(subs[['date', 'sentiment']].groupby(pd.Grouper(key='date', freq='1D')).mean())
    plt.xlabel('Date') 
    plt.ylabel('VADER sentiment')
    plt.legend(['Comments', 'Submissions'])
    
    ax = hours_dict.plot.bar()
    ax.set(title='Hour of posting to r/Coronavirus', xlabel='Hour', ylabel='Probability')

    plt.figure() 
    plt.xticks(rotation=90)
    plt.plot(comms_g.size())
    plt.title('Daily comments on r/Coronvirus')
    plt.xlabel('Date')
    plt.ylabel('Count')

    plt.show() 