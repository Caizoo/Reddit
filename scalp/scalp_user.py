# Data fetch and store 
from sys import api_version
import pymongo  
from pymongo import MongoClient, errors 
import praw 
import prawcore 

# Other imports 
import pandas as pd 
import datetime as dt 
from tqdm import tqdm
import multiprocessing 
import json 
import time 




# Local libraries 

from scalp.scalp_help import get_default_reddit_instance, insert_list

def download_user_data(user: str, data_type: str, db: pymongo.database.Database, verbose: bool=False) -> list:
    """ Download the submission or comment history of a user from the Reddit API

    Args:
        user (str): Name of user to download
        data_type (str): Download either "comments" or "submissions"
        db (pymongo.database.Database): Database instance to insert user data
        verbose (bool, optional): Whether to print download information. Defaults to True.

    Returns:
        list_data: [description]
    """
    list_data = list() 
    # whether to search comments or submissions of user 
    data_forest = user.comments if data_type=="comments" else user.submissions 

    # Download new, top, hot and controversial comments/submissions for user 
    try: 
        if verbose: print("\t[INFO]: Downloading new", flush=True)
        for sb in data_forest.new(limit=None): list_data.append(sb) 
        if verbose: print("\t[INFO]: Downloading top", flush=True)
        for sb in data_forest.top(limit=None): list_data.append(sb) 
        if verbose: print("\t[INFO]: Downloading hot", flush=True)
        for sb in data_forest.hot(limit=None): list_data.append(sb) 
        if verbose: print("\t[INFO]: Downloading controversial", flush=True)
        for sb in data_forest.controversial(limit=None): list_data.append(sb) 

    # If the user is forbidden or not found, place in Lost_Users -> shouldn't happen with fetchable checks when creating top user list
    except (prawcore.Forbidden, prawcore.NotFound) as e:
        print(f"[INFO]: {user} is forbidden", flush=True)
        try:
            db['Lost_Users'].insert_one({'_id': user, 'name': user})
        except:
            pass 
        return list_data
    
    # Any other exception, print and continue
    except Exception as e:
        print(e, flush=True)
        print(type(e), flush=True) 
        print("---------------------------------------------------", flush=True)
    
    return list_data

def scalp_user_comments(user_str: str, db: pymongo.database.Database, reddit: praw.reddit.Reddit):
    """ Scalp comment history of user

    Args:
        user_str (str): Username
        db (pymongo.database.Database): Database instance to insert data 
        reddit (praw.reddit.Reddit): Reddit API instance 
    """

    user = reddit.redditor(user_str)
    comments = download_user_data(user, "comments", db)

    insert_list(comments, db, "User_Comments")

def scalp_user_submissions(user_str: str, db: pymongo.database.Database, reddit: praw.reddit.Reddit):
    """ Scalp submission history of user

    Args:
        user_str (str): Username
        db (pymongo.database.Database): Database instance to insert data 
        reddit (praw.reddit.Reddit): Reddit API instance 
    """

    user = reddit.redditor(user_str)
    submissions = download_user_data(user, "submissions", db)

    insert_list(submissions, db, "User_Submissions")

def add_user_data(user: praw.reddit.Redditor, type: str, args: dict = dict()):
    """ Function to add user data on-the-fly while building top user lists and checking fetchable - separates normal, suspended and deleted accounts

    Args:
        user (praw.reddit.Redditor): Redditor instance from PRAW
        type (str): normal, suspended or deleted
        args (dict, optional): Arguments dict. Defaults to dict().
    """
    username, mongodb_address, password, database_str = args['username'], args['mongodb_address'], args['password'], args['database_str']
    client = MongoClient(f'mongodb://{username}:{password}@{mongodb_address}/', connect=False) if username!='' else MongoClient(f'mongodb://{mongodb_address}/', connect=False)
    db = client['Users']
    
    if type=='normal':
        fields = ['comment_karma', 'link_karma', 'created_utc', 'has_verified_email', 'is_employee', 'is_mod', 'is_gold']
        user_data = vars(user) 
        user_data = {k: user_data[k] for k in fields} 
        user_data['_id'] = str(user.name)
        insert_list([user_data], db, 'Users')
    elif type=='suspended':
        insert_list([{'_id': user.name, 'lost_type': 'deleted'}], db, 'Lost_Users', overwrite=True) 
    elif type=='deleted':
        insert_list([{'_id': user.name, 'lost_type': 'deleted'}], db, 'Lost_Users', overwrite=True)
    
    client.close() 

def is_user_fetchable(user_str: str, reddit: praw.reddit.Reddit, args: dict = dict()) -> bool:
    """ Check is user is fetchable using the Reddit API 

    Args:
        user_str (str): Username
        reddit (praw.reddit.Reddit): Reddit API instance
        args (dict, optional): Arguments dict. Defaults to dict().
    Returns:
        bool: If user is fetchable 
    """
    u = reddit.redditor(user_str) 
    try:
        created_utc = u.created_utc # If the user's creation date is available, user is fetchable
        add_user_data(u, 'normal', args=args)
        return True 
    except AttributeError as e:
        add_user_data(u, 'suspended', args=args) # user is suspended 
    except prawcore.exceptions.NotFound:
        add_user_data(u, 'deleted', args=args) # account deleted
    except Exception as e: pass
    
    return False

def find_top_users(df: pd.DataFrame, top_x: int, ensure_fetchable: bool = False, args: dict = dict()) -> tuple:
    """ Find top X users from a dataframe of comments or submissions (or both)

    Args:
        df (pd.DataFrame): DataFrame to search
        top_x (int): X for number of top X users
        ensure_fetchable (bool): Whether to check all top X users are fetchable (actual list or fetchable list of top users)
        args (dict, optional): Arguments dict. Defaults to dict().
    Returns:
        tuple: Tuple containing author/user and count of instances in df
    """

    reddit = get_default_reddit_instance() # returns Reddit API instance using key at index 0 from keys.json
    user_dict = dict() 
    for _, row in df.iterrows(): # iterate through df and keep tally of instances of authors
        user = row['author']
        if user=='None': continue # ignore None
        user_dict[user] = user_dict.get(user, 0) + 1 # author instance count
    user_tuples = [tuple(a) for a in user_dict.items()] 
    user_tuples.sort(key=lambda a: -a[1]) # sort by count

    new_tups = list() 
    if ensure_fetchable: # creates a new top X list with users that are fetchable only, rather than actual top users seen in user_tuples
        for i in tqdm(range(len(user_tuples))): # may not add up to 500 in size
            if is_user_fetchable(user_tuples[i][0], reddit, args=args): new_tups.append(user_tuples[i])
            if len(new_tups)>=top_x: break

    return user_tuples[:top_x], new_tups[:top_x] # return actual and fetchable lists, ensure_fetchable=False returns actual and empty

def get_user_windows(sub_str: str, db: pymongo.database.Database, top_x: int, start_year: int = 2020, args: dict = dict()) -> tuple:
    """ Get windowed lists of top users for each month

    Args:
        sub_str (str): Subreddit name
        db (pymongo.database.Database): Database instance to insert data 
        top_x (int): Top X users
        reddit (praw.reddit.Reddit): Reddit API instance 
        start_year (int, optional): Year to start looking for windows. Defaults to 2020.
        args (dict, optional): Arguments dict. Defaults to dict().
    Returns:
        tuple (dict): (Top X submittors each month, top X commentors each month) - each date has an 'actual' list and 'fetchable' list of top users
    """

    unix_start = dt.datetime(start_year, 1, 1, tzinfo=dt.timezone.utc).timestamp() # Loop through submissions starting from this time
    submissions, comments = pd.DataFrame(), pd.DataFrame() 
    top_users_submissions, top_users_comments = dict(), dict() 

    # Fetch submissions and comments for this subreddit from unix_start onwards
    submissions = pd.DataFrame(db['Submissions'].find({'subreddit': sub_str, 'created_utc': {'$gte': unix_start}}))
    comments = pd.DataFrame(db['Comments'].find({'subreddit': sub_str, 'created_utc': {'$gte': unix_start}}))

    # Add in created field for actual date from the UNIX epoch timestamp field created_utc
    submissions['created'] = [dt.datetime.fromtimestamp(a, tz=dt.timezone.utc) for a in submissions['created_utc']] 
    comments['created'] = [dt.datetime.fromtimestamp(a, tz=dt.timezone.utc) for a in comments['created_utc']] 
    submissions.dropna() 
    comments.dropna() 

    # Group into months
    subs_groups = submissions.groupby(pd.Grouper(key='created', freq='1M')) 
    comms_groups = comments.groupby(pd.Grouper(key='created', freq='1M')) 

    # Iterate over months and use find_top_users to find the top users for that group
    for index, month_subs in subs_groups:
        print(f"[INFO]: fetching submissions for {index}")
        actual_top, fetchable_top = find_top_users(month_subs, top_x, ensure_fetchable=True, args=args)
        top_users_submissions[str(index)] = {'actual': actual_top, 'fetchable': fetchable_top} # keep actual and fetchable lists

    for index, month_comms in comms_groups:
        print(f"[INFO]: fetching comments for {index}")
        actual_top, fetchable_top = find_top_users(month_comms, top_x, ensure_fetchable=True, args=args)
        top_users_comments[str(index)] = {'actual': actual_top, 'fetchable': fetchable_top}

    return top_users_submissions, top_users_comments

def user_scalp_worker(data_dict: dict):
    """ Worker to scalp users for comments and submissions

    Args:
        data_dict (dict): 
                            user: Username, 
                            mongodb_address: mongodb address, 
                            username: mongo username, 
                            password: mongo password, 
                            database_str: database name
    """
    user_str, mongodb_address, username, password, database_str = data_dict['user'], data_dict['mongodb_address'], data_dict['username'], data_dict['password'], data_dict['database_str']
    client = MongoClient(f'mongodb://{username}:{password}@{mongodb_address}/', connect=False) if username!='' else MongoClient(f'mongodb://{mongodb_address}/', connect=False)
    db = client[database_str]
    # arguments to pass to adding users to database in case of deleted or suspended account since top list was built
    args = {'mongodb_address': mongodb_address, 'username': username, 'password': password, 'database_str': database_str} 
    
    # Get Reddit API instance 
    api_keys = json.load(open("keys.json", "r"))["keys"]
    api_index = (int(multiprocessing.current_process().name.split("-")[1])-1)%16
    api_key = api_keys[api_index]

    REDDIT_ID = str(list(api_key.keys())[0])
    REDDIT_KEY = str(api_keys[api_index][list(api_key.keys())[0]])
    USER_AGENT = 'Reddit Research '+str(api_index+1)
    reddit = praw.Reddit(client_id=REDDIT_ID,
                        client_secret=REDDIT_KEY,
                        user_agent=USER_AGENT) 

    # Try for an hour in case no access to internet
    for _ in range(1200):
        try:
            scalp_user_submissions(user_str, db, reddit)
            scalp_user_comments(user_str, db, reddit)
            break
            
        # In case user becomes suspended or deleted between making the top user list and now 
        except prawcore.exceptions.Forbidden as e:
            try:
                u = reddit.redditor(user_str).name 
                add_user_data(u, 'suspended', args=args)
            except:
                pass 
            break
        except prawcore.exceptions.NotFound as e:
            try:
                u = reddit.redditor(user_str).name
                add_user_data(u, 'deleted', args=args)
            except:
                pass 
            break 

        # Just in case it's too large, don't keep trying
        except prawcore.exceptions.TooLarge as e:
            print(e, flush=True)
            break
        except Exception as e:
            print(e, flush=True)
            print("[INFO] trying again...", flush=True)
            time.sleep(2)
    
    client.close() 

