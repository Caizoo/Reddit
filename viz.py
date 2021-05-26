from pandas.core.frame import DataFrame
from pymongo import MongoClient
import pandas as pd 
import sklearn as sk 
import datetime as dt
import matplotlib.pyplot as plt 
import seaborn as sns 


if __name__=="__main__":
    SUB = 'NoNewNormal'
    client = MongoClient('mongodb://admin:password@192.168.1.123') 
    db = client['Reddit'] 

    comms, subs = pd.DataFrame(db['Comments'].find({'subreddit': SUB})), pd.DataFrame(db['Submissions'].find({'subreddit': SUB}))
    comms['date'], subs['date'] = [dt.datetime.fromtimestamp(a, tz=dt.timezone.utc) for a in comms['created_utc']], [dt.datetime.fromtimestamp(a, tz=dt.timezone.utc) for a in subs['created_utc']]

    comms_g = comms.groupby(pd.Grouper(key='date', freq='1D')) 
    
    hours_dict = {} 
    for _, a in comms.iterrows():
        hours_dict[a['date'].hour] = hours_dict.get(a['date'].hour, 0) + 1

    hours_dict = pd.Series(hours_dict) 
    plt.figure() 
    plt.bar(hours_dict.index, hours_dict.values)

    plt.figure() 
    plt.plot(comms_g.size())

    plt.show() 