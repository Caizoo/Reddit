import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
import seaborn as sns 

from statsmodels.tsa.stattools import grangercausalitytests, coint



def plot_correlation(SUB, TO_SUB, VERSION, doc_type, sub_dict, to_sub_dict):
    data_sub = sub_dict[SUB][doc_type].groupby(pd.Grouper(key='date', freq='1d')).size()
    data_to_sub = to_sub_dict[TO_SUB][doc_type].groupby(pd.Grouper(key='date', freq='1d')).size()

    first_date = data_sub.index[0] if data_sub.index[0] > data_to_sub.index[0] else data_to_sub.index[0]
    last_date = data_sub.index[-1] if data_sub.index[-1] < data_to_sub.index[-1] else data_to_sub.index[-1]
        
    data_sub = data_sub.loc[first_date:last_date]
    data_to_sub = data_to_sub.loc[first_date:last_date]

    plt.scatter(data_sub, data_to_sub) 
    plt.xlabel(f"{SUB} daily {doc_type} count")
    plt.ylabel(f"{TO_SUB} daily {doc_type} count")
    plt.title(VERSION)
    plt.show()

def format_coint_func(v):
    return f"\tp-value:{v[1]}\n\tvalue:{v[0]}\n\tcritical-values:{v[2]}\n"

def plot_coint(SUB, TO_SUB, VERSION, doc_type, activity_type, sub_dict, to_sub_dict):
    import warnings
    warnings.filterwarnings('ignore')
    tt = ["date", "author"]
    if activity_type=='sentiment': tt = ["date","vader_body"] if doc_type=="comments" else ["date", "vader_title", "vader_selftext"]

    activity_sub = sub_dict[SUB][doc_type][tt].groupby(pd.Grouper(key='date', freq='1D'))
    activity_to_sub = to_sub_dict[TO_SUB][doc_type][tt].groupby(
                                                            pd.Grouper(key='date', freq='1D'))
    
    activity_sub = activity_sub.size() if activity_type=='Activity' else activity_sub.mean()
    activity_to_sub = activity_to_sub.size() if activity_type=='Activity' else activity_to_sub.mean()
    
    if doc_type=="submissions" and activity_type=="Sentiment":
        activity_sub = activity_sub.mean(axis=1)
        activity_to_sub = activity_to_sub.mean(axis=1)
        
    first_date = activity_sub.index[0] if activity_sub.index[0] > activity_to_sub.index[0] else activity_to_sub.index[0]
    last_date = activity_sub.index[-1] if activity_sub.index[-1] < activity_to_sub.index[-1] else activity_to_sub.index[-1]
    activity_sub = activity_sub.loc[first_date:last_date]
    activity_to_sub = activity_to_sub.loc[first_date:last_date]
    
    plt.xticks(rotation=90)
    plt.plot(activity_sub)
    plt.plot(activity_to_sub)
    plt.legend((SUB,TO_SUB))
    plt.show() 

    v1 = coint(activity_sub.to_numpy(), activity_to_sub.to_numpy(), trend='c', maxlag=20)
    v2 = coint(activity_sub.to_numpy(), activity_to_sub.to_numpy(), trend='ct', maxlag=20)
    v3 = coint(activity_sub.to_numpy(), activity_to_sub.to_numpy(), trend='ctt', maxlag=20)

    v4 = coint(activity_to_sub.to_numpy(), activity_sub.to_numpy(), trend='c', maxlag=20)
    v5 = coint(activity_to_sub.to_numpy(), activity_sub.to_numpy(), trend='ct', maxlag=20)
    v6 = coint(activity_to_sub.to_numpy(), activity_sub.to_numpy(), trend='ctt', maxlag=20)

    df_coint = pd.DataFrame(index=activity_sub.index)
    df_coint['sub'] = activity_sub
    df_coint['to_sub'] = activity_to_sub
    return df_coint,v1,v2,v3,v4,v5,v6

def run_granger_tests_activity(SUB, TO_SUB, sub_dict, to_sub_dict, d_type):
    df_sub = sub_dict[SUB][d_type].groupby(pd.Grouper(key='date', freq='1d')).size()
    df_to_sub = to_sub_dict[TO_SUB][d_type].groupby(pd.Grouper(key='date', freq='1d')).size()

    first_date = df_sub.index[0] if df_sub.index[0] > df_to_sub.index[0] else df_to_sub.index[0]
    last_date = df_sub.index[-1] if df_sub.index[-1] < df_to_sub.index[-1] else df_to_sub.index[-1]
    df_sub = df_sub[first_date:last_date]
    df_to_sub = df_to_sub[first_date:last_date]

    df_index = df_sub.index 
    df_sub = np.gradient(df_sub) 
    df_to_sub = np.gradient(df_to_sub)

    plt.figure() 
    plt.plot(df_sub)
    plt.plot(df_to_sub)
    plt.legend((SUB, TO_SUB))
    plt.show() 

    df_granger = pd.DataFrame(data={SUB: df_sub, TO_SUB: df_to_sub}, index=df_index) 
    df_granger = df_granger[[TO_SUB, SUB]] 

    results = grangercausalitytests(df_granger, maxlag=20, verbose=False) 

    p_vals = list() 
    f_vals = list() 

    for r in results:
        list_f, list_p = list(), [] 
        for k in results[r][0].keys():
            list_f.append(results[r][0][k][0])
            list_p.append(results[r][0][k][1])

        p_vals.append(np.mean(list_p)) 
        f_vals.append(np.mean(list_f)) 
    
    return p_vals, f_vals 


def run_granger_tests(SUB, TO_SUB, KEY_WORDS_LIST, sub_dict, to_sub_dict, doc_type):
    import warnings
    warnings.filterwarnings('ignore')
    sub_words_dicts = list() 
    to_sub_words_dicts = list() 
    for k in KEY_WORDS_LIST:
        x = sub_dict[SUB][k][doc_type]
        x = x.rename(k)
        x = x.fillna(0) 
        sub_words_dicts.append(x)

        xx = to_sub_dict[TO_SUB][k][doc_type]
        xx = xx.rename(k)
        xx = xx.fillna(0)
        to_sub_words_dicts.append(xx) 
    
    df_sub = pd.concat(sub_words_dicts, axis=1)
    df_to_sub = pd.concat(to_sub_words_dicts, axis=1)
    
    heatmap_data = dict() 
    heatmap_binary_data = dict() 
    heatmap_binary2_data = dict() 
    
    for KEY_WORD in KEY_WORDS_LIST:
        
        x_keyw_sub = df_sub[KEY_WORD]
        x_keyw_to_sub = df_to_sub[KEY_WORD]
        
        first_date = x_keyw_sub.index[0] if x_keyw_sub.index[0] > x_keyw_to_sub.index[0] else x_keyw_to_sub.index[0]
        last_date = x_keyw_sub.index[-1] if x_keyw_sub.index[-1] < x_keyw_to_sub.index[-1] else x_keyw_to_sub.index[-1]

        x_keyw_sub = x_keyw_sub.loc[first_date:last_date]
        x_keyw_to_sub = x_keyw_to_sub.loc[first_date:last_date]

        df_granger = pd.DataFrame(data={SUB: x_keyw_sub.values, TO_SUB: x_keyw_to_sub.values}, index=x_keyw_sub.index) 
        df_granger = df_granger[[TO_SUB, SUB]]

        results = grangercausalitytests(df_granger, maxlag=10, verbose=False)

        p_vals = list() 
        f_vals = list() 

        for r in results:
            list_f, list_p = list(), [] 
            for k in results[r][0].keys():
                list_f.append(results[r][0][k][0])
                list_p.append(results[r][0][k][1])

            p_vals.append(np.mean(list_p)) 
            f_vals.append(np.mean(list_f)) 


        heatmap_data[KEY_WORD] = p_vals

    return heatmap_data

def plot_grangers(SUB, TO_SUB, h, conf_level):

    if conf_level>0:
        for k in h.keys():
            p_vals = h[k] 
            h[k] = [1 if x>=conf_level else 0 for x in p_vals]

    import warnings
    warnings.filterwarnings('ignore')
    # HEATMAP NORMAL
    h = pd.DataFrame(h) 
    fig = plt.figure() 
    ax = sns.heatmap(h.T, yticklabels=True) 
    ax.set_title(f"{SUB} causes {TO_SUB} key words", fontdict={'fontsize': 8})
    ax.set_xlabel("Lags in days") 
    ax.set_ylabel("Key words")
    ax.tick_params(axis='both', which='major', labelsize=5)
    plt.tight_layout() 
    plt.show()