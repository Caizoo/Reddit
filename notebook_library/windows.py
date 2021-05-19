import pandas as pd 
import matplotlib as mpl 
import matplotlib.pyplot as plt
import seaborn as sns  
import datetime as dt 
from dateutil.relativedelta import relativedelta
import pymongo 
from pymongo import MongoClient
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import multiprocessing
import time 
import json

# TOP USERS AND WINDOWS ------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def unix_range(year: int, month: int) -> (dt.datetime, dt.datetime):
    """[summary]

    Args:
        month (int): [description]
        dt ([type]): [description]

    Returns:
        [type]: [description]
    """
    d_1 = dt.datetime(year,month,1, tzinfo=dt.timezone.utc) 
    d_2 = d_1 + relativedelta(months=1) - relativedelta(seconds=1)
    return d_1.timestamp(), d_2.timestamp()  

def month_data(month: int, submission_window, comment_window, db) -> (list, list): 
    """[summary]

    Args:
        month (int, submission_window, comment_window, db): [description]
        list ([type]): [description]

    Returns:
        [type]: [description]
    """
    submissions = list() 
    comments = list() 
    unix_low, unix_high = unix_range(month) 
    for user, score in submission_window:
        subs_user = list(db['User_Submissions'].find({'author': user, "created_utc": {"$gte": unix_low,"$lte": unix_high}}))
        submissions += subs_user
    for user, score in comment_window:
        comm_user = list(db['User_Comments'].find({'author': user, "created_utc": {"$gte": unix_low,"$lte": unix_high}}))
        comments += comm_user
    return submissions, comments 

def int_to_day(a: int) -> str:
    """[summary]

    Args:
        a (int): [description]

    Returns:
        str: [description]
    """
    if a==0: return "Monday" 
    if a==1: return "Tuesday" 
    if a==2: return "Wednesday"
    if a==3: return "Thursday" 
    if a==4: return "Friday" 
    if a==5: return "Saturday" 
    if a==6: return "Sunday"

def find_top_users(df: pd.DataFrame, top_x: int):
    """[summary]

    Args:
        df (pd.DataFrame): [description]
        top_x (int): [description]

    Returns:
        [type]: [description]
    """
    user_dict = dict() 
    ignore_list = json.load(open("constants.json"))["IGNORE_USER_LIST"]

    for index, row in df.iterrows():
        user = row['author'] 
        if user=='None': continue
        if user in ignore_list: continue
        if user in user_dict.keys():
            user_dict[user] = user_dict[user] + 1 
        else:
            user_dict[user] = 1 

    user_tuples = [tuple(a) for a in user_dict.items()]
    user_tuples.sort(key=lambda a: -a[1]) 
    return user_tuples[:top_x]

def get_user_windows(sub_str: str, db: pymongo.database.Database, top_x: int, freq: str, start_year: int = 2020) -> (dict, dict):
    """[summary]

    Args:
        sub_str (str, db): [description]
        dict ([type]): [description]

    Returns:
        [type]: [description]
    """
    # if wanting to download subreddit data for an earlier date than the top users data
        # e.g. downloading subreddit data for 2016-, but only downloading user data from 2020- since there is a lot more
    unix_from = dt.datetime(start_year,1,1, tzinfo=dt.timezone.utc).timestamp() 
    submissions, comments = pd.DataFrame(), pd.DataFrame() 

    submissions = pd.DataFrame(db['Submissions'].find({'subreddit': sub_str, 'created_utc': {"$gte": unix_from}}).sort('created_utc',1))
    comments = pd.DataFrame(db['Comments'].find({'subreddit': sub_str, 'created_utc': {"$gte": unix_from}}).sort('created_utc',1))
    #comments = pd.DataFrame(db['Comments'].find({'subreddit': sub_str}).sort('created_utc',1))

    submissions['created'] = [dt.datetime.fromtimestamp(a['created_utc'], tz=dt.timezone.utc) for index, a in submissions.iterrows()]
    submissions.dropna() 

    submission_groups = submissions.groupby(pd.Grouper(key='created', freq='1M'))
    submissions_sizes = submission_groups.size() 
    top_users_submissions = dict()
    for index, month_subs in submission_groups:
        ind_stripped = dt.datetime(index.year, index.month, 1, tzinfo=dt.timezone.utc) 
        top_users_submissions[ind_stripped] = find_top_users(month_subs, top_x)

    comments['created'] = [dt.datetime.fromtimestamp(a['created_utc'], tz=dt.timezone.utc) for index, a in comments.iterrows()]
    comments.dropna() 
    
    comment_groups = comments.groupby(pd.Grouper(key='created', freq=freq))
    comment_sizes = comment_groups.size() 
    top_users_comments = dict()
    for index, month_subs in comment_groups:
        ind_stripped = dt.datetime(index.year, index.month, 1, tzinfo=dt.timezone.utc) #dt.datetime.strptime(index, "%Y-%m-%d %H:%M:%S")
        top_users_comments[ind_stripped] = find_top_users(month_subs, top_x)
    

    return top_users_submissions, top_users_comments

def get_subreddit_top_data(sub_str: str, db: pymongo.database.Database, top_x: int, freq: str='1M', start_year: int = 2020) -> (pd.DataFrame, pd.DataFrame):
    """[summary]

    Args:
        sub (str): [description]
        db (pymongo.database.Database): [description]
        top_x (int): [description]
        start_year (int, optional): [description]. Defaults to 2020.

    Returns:
        [type]: [description]
    """
    submission_windows, comment_windows = get_user_windows(sub_str, db, top_x, freq, start_year) 
    all_subs, all_comments = list(), [] 

    # TODO: make multiprocessing
    for d in submission_windows.keys():
        print(f"Fetching submissions for {d}") 
        
        unix_low, unix_high = unix_range(d.year, d.month) 
        user_list = [a[0] for a in submission_windows[d]]
        
        all_subs.append(pd.DataFrame(db['User_Submissions'].find({'$and':[
            {'author': {'$in': user_list}},
            {'created_utc': {"$gte": unix_low,"$lte": unix_high}}
        ]})))
    all_subs = pd.concat(all_subs)

    for d in comment_windows.keys():
        print(f"Fetching comments for {d}") 
        
        t = time.time() 
        unix_low, unix_high = unix_range(d.year, d.month) 
        user_list = [a[0] for a in comment_windows[d]]
        
        all_comments.append(pd.DataFrame(db['User_Comments'].find({'$and':[
            {'author': {'$in': user_list}},
            {'created_utc': {"$gte": unix_low,"$lte": unix_high}}
        ]})))
    all_comments = pd.concat(all_comments) 

    vader_analyzer = SentimentIntensityAnalyzer()

    all_subs, all_comments = pd.DataFrame(all_subs), pd.DataFrame(all_comments) 

    print("Adding extra submission data")
    all_subs['date'] = [dt.datetime.fromtimestamp(a, tz=dt.timezone.utc) for a in all_subs['created_utc']]
    all_subs['hour'] = [a.hour for a in all_subs['date']]
    all_subs['day'] = [a.weekday() for a in all_subs['date']]

    print("Adding extra comment data")
    all_comments['date'] = [dt.datetime.fromtimestamp(a, tz=dt.timezone.utc) for a in all_comments['created_utc']]
    all_comments['hour'] = [a.hour for a in all_comments['date']]
    all_comments['day'] = [a.weekday() for a in all_comments['date']]

    return all_subs, all_comments, submission_windows, comment_windows

def visualise_day_of_week(sub_dict, SUB, to_sub): 
    comments, submissions = sub_dict[to_sub]["comments"], sub_dict[to_sub]["submissions"]
    days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"] 

    c = comments.groupby(['day']).size() 
    s = submissions.groupby(['day']).size() 
    for d in range(7):
        if d not in c.index: c.loc[d] = 0
        if d not in s.index: s.loc[d] = 0
    
    c = [a/len(comments) for i, a in c.iteritems()]
    s = [a/len(submissions) for i, a in s.iteritems()]
    
    c_d = pd.DataFrame({'prob': c, 'day': days_of_week, 'type': ["comment" for _ in range(len(days_of_week))]}) 
    s_d = pd.DataFrame({'prob': s, 'day': days_of_week, 'type': ["submissions" for _ in range(len(days_of_week))]}) 
    d = c_d.append(s_d) 
    
    plt.xticks(rotation=90)
    sns.barplot(x='day', y='prob', hue='type', data=d).set(title=f"Probability of comment in hour of day {SUB}->{to_sub}")
    plt.tight_layout()
    plt.show() 

def visualise_hour_of_day(sub_dict, SUB, to_sub): 
    comments, submissions = sub_dict[to_sub]["comments"], sub_dict[to_sub]["submissions"]

    c = comments.groupby(['hour']).size()
    s = submissions.groupby(['hour']).size()
    for h in range(24):
        if h not in c.index: c.loc[h] = 0
        if h not in s.index: s.loc[h] = 0

    c = [a/len(comments) for i, a in c.iteritems()]
    s = [a/len(submissions) for i, a in s.iteritems()]

    c_d = pd.DataFrame({'prob': c, 'hour': list(range(24)), 'type': ["comment" for _ in range(24)]}) 
    s_d = pd.DataFrame({'prob': s, 'hour': list(range(24)), 'type': ["submissions" for _ in range(24)]}) 

    d = c_d.append(s_d) 
    
    sns.barplot(x='hour', y='prob', hue='type', data=d).set(title=f"Probability of comment in day of week {SUB}->{to_sub}")
    plt.show()    