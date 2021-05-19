import datetime as dt 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import multiprocessing
from pymongo import MongoClient

from sklearn.preprocessing import MinMaxScaler

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  
import gensim 
from gensim.utils import simple_preprocess 
from gensim.parsing.preprocessing import STOPWORDS 
import nltk 
from nltk.stem import WordNetLemmatizer, SnowballStemmer 
from nltk.stem.porter import * 
from nltk.corpus import stopwords

# NLP WORD FREQUENCY FUNCTIONS -----------------------------------------------------------------------------------------------------------------------------------------------------------------

def lemmatize_stemming(text):
    return WordNetLemmatizer().lemmatize(text, pos='v')

def preprocess(text):
    result=list()
    for token in gensim.utils.simple_preprocess(text) :
        if token not in gensim.parsing.preprocessing.STOPWORDS:
            result.append(lemmatize_stemming(token))
           
    return result

def get_word_count(docs: list, key_word: str):
    counter = 0
    for d in docs:
        text = preprocess(d) 
        counter += text.count(key_word) 
    return counter 

def get_word_series(df, c, key_word: str, name: str='', div_by_size=False):
    grouped = df.groupby(pd.Grouper(key='date', freq='1D'))
    counts = list() 
    ind = [dt.datetime(g.year,g.month,g.day, tzinfo=dt.timezone.utc) for g in grouped.groups]
    for g in grouped.groups:
        try:
            counts.append(get_word_count(grouped.get_group(g)[c], key_word))
        except:
            counts.append(0.0) 

    s = pd.Series(counts, index=ind, name=name)
    if div_by_size:
        return s.div(grouped.size())
    else:
        return s 

def get_key_word_counts(x, div_by_size=True):
    name, group, c, key_word = x
    return get_word_series(group, c, key_word, name=name, div_by_size=div_by_size) 
    
        
def content_subs_list(subs_list, mydb):
    comments = list() 
    submissions = list() 

    comments = list(mydb['Comments'].find({'subreddit': {"$in":subs_list}})) 
    submissions = list(mydb['Submissions'].find({'subreddit': {"$in":subs_list}})) 

    comment_bodies = [c['body'] for c in comments] 
    submission_titles = [s['title'] for s in submissions] 
    submission_bodies = [s['selftext'] for s in submissions]
    
    comments_lemma = list() 
    submissions_lemma = list() 
    for c in comment_bodies:
        cc = preprocess(c) 
        comments_lemma.append(cc)
    for s in submission_titles + submission_bodies:
        ss = preprocess(s) 
        submissions_lemma.append(ss) 
        

    dictionary = gensim.corpora.Dictionary(comments_lemma + submissions_lemma) 
    dictionary.filter_extremes(no_below=15, no_above=0.1, keep_n= 100000)

    bow_corpus = [dictionary.doc2bow(doc) for doc in comments_lemma] 

    freqs = dictionary.cfs 
    freqs = [(k,v) for k,v in freqs.items()] 
    freqs.sort(key=lambda a: -a[1]) 
    freqs = freqs[:100] 

    top_x = [dictionary[a[0]] for a in freqs] 

    # PRINT TOP WORDS FROM SUBS_LIST
    for x in top_x: print(x)  
    return top_x

def plot_subs_list(subs_list, key_words, mydb):
    comments = list() 
    submissions = list() 

    comments = list(mydb['Comments'].find({'subreddit': {"$in":subs_list}})) 
    submissions = list(mydb['Submissions'].find({'subreddit': {"$in":subs_list}})) 
    all_comments = pd.DataFrame(comments)
    all_submissions = pd.DataFrame(submissions)
    all_comments['date'], all_submissions['date'] = [dt.datetime.fromtimestamp(a, tz=dt.timezone.utc) for a in all_comments['created_utc']], [dt.datetime.fromtimestamp(a, tz=dt.timezone.utc) for a in all_submissions['created_utc']]

    df_grouped = all_comments.groupby(by='subreddit')  
    df_s = list() 
    for name, group in df_grouped: 
        print(f"Computing {name}")
        ss_s = [(name, group, 'body', k) for k in key_words]
        pool = multiprocessing.Pool() 
        ss_s = list(pool.map(get_key_word_counts, ss_s)) 
        pool.close() 

        df_s.append(pd.concat(ss_s, axis=1).fillna(0))


    #df_master = pd.concat(df_s)
    #df_master.groupby(df_master.index) 
    #df_means = df_master.mean() 
    return df_s

    #return get_key_word_counts(all_comments, 'body', key_words)