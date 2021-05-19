 # Data fetch anb store
from praw.models import user
import pymongo 
from pymongo import MongoClient, errors 
import praw 
import prawcore

# Other imports 
import argparse
import multiprocessing
from multiprocessing import Pool
import json
from tqdm import tqdm
import timeit

# Local libraries 
from scalp.scalp_subreddit import download_threaded_comments, push_search_submissions, push_search_comments, reddit_search_comments, reddit_search_submissions
from scalp.scalp_help import get_default_reddit_instance, insert_list
from scalp.pushshift import PushAPI
from scalp.scalp_user import get_user_random, get_user_windows, user_scalp_worker

# Scalp subreddits

def scalp_subreddit(args: dict):
    """ Scalp the submissions and comments of a subreddit using an API

    Args:
        args (dict): Dict of arguments:
                    sub: subreddit name
                    mongodb_address: address (IP) of mongo db without the mongodb://
                    username: mongo username
                    password: mongo password 
                    database_str: database name
    """
    assert args['sub']!='' # needs to be true
    sub_str, mongodb_address, username, password, database_str = args['sub'], args['mongodb_address'], args['username'], args['password'], 'Reddit'
    
    client = MongoClient(f'mongodb://{username}:{password}@{mongodb_address}/', connect=False) if username!='' else MongoClient(f'mongodb://{mongodb_address}/', connect=False)
    db = client[database_str]
    
    # if using Pushshift to download submissions
    if args['push_submissions']:
        push_search_submissions(sub_str, db, start_year=args['year'], incremental_comment_search=not args['push_comments'], args=args)

        # If using Pushshift to download comments, do below, else use incremental comment download method in the above to use 
        # Reddit API to download comments on-the-fly
        if args['push_comments']: push_search_comments(sub_str, db, start_year=args['year'], args=args)
    else:
        reddit_search_submissions(sub_str, db)
        sub_ids = [a['_id'] for a in list(db['Submissions'].find({'subreddit': sub_str}, {'_id': 1}))]
        download_threaded_comments(sub_ids, sub_str, args)
    
    client.close() 

def scalp_controversial_subreddit(args: dict): 
    """ Scalp controversial subreddit (banned or quarantined) for submissions and comments

    Args:
        args (dict): Dict of arguments:
                    sub: subreddit name
                    mongodb_address: address (IP) of mongo db without the mongodb://
                    username: mongo username
                    password: mongo password 
                    database_str: database name
    """
    assert args['sub']!=''
    sub_str, mongodb_address, username, password, database_str = args['sub'], args['mongodb_address'], args['username'], args['password'], 'Reddit_Controversial'

    client = MongoClient(f'mongodb://{username}:{password}@{mongodb_address}/', connect=False) if username!='' else MongoClient(f'mongodb://{mongodb_address}/', connect=False)
    db = client[database_str]

    # Just use pushshift to download everything since Reddit API gives no access
    push_search_submissions(sub_str, db, start_year=args['year'], args=args)
    push_search_comments(sub_str, db, start_year=args['year'], args=args)

    client.close() 

def scalp_random_subreddit(args: dict):
    """ Scalp a random subreddit for submissions and comments 

    Args:
        args (dict): Dict of arguments:
                    sub: subreddit name
                    mongodb_address: address (IP) of mongo db without the mongodb://
                    username: mongo username
                    password: mongo password 
                    database_str: database name
    """
    mongodb_address, username, password, database_str = args['mongodb_address'], args['username'], args['password'], 'Reddit_Random'
    
    client = MongoClient(f'mongodb://{username}:{password}@{mongodb_address}/', connect=False) if username!='' else MongoClient(f'mongodb://{mongodb_address}/', connect=False)
    db = client[database_str]

    # Fetch random subreddit
    reddit = get_default_reddit_instance()
    sub_str = str(reddit.subreddit('random').display_name) 

    # if using Pushshift to download submissions
    if args['push_submissions']:
        push_search_submissions(sub_str, db, start_year=args['year'], incremental_comment_search=not args['push_comments'], args=args)

        # If using Pushshift to download comments, do below, else use incremental comment download method in the above to use 
        # Reddit API to download comments on-the-fly
        if args['push_comments']: push_search_comments(sub_str, db, start_year=args['year'], args=args)
    else:
        reddit_search_submissions(sub_str, db)
        sub_ids = [a['_id'] for a in list(db['Submissions'].find({'subreddit': sub_str}, {'_id': 1}))]
        download_threaded_comments(sub_ids, sub_str, args)
    
    client.close() 

# Scalp users

def build_top_list(args: dict): # Loop subreddits in submissions, build list of top users for each month and save in JSON file scalp/cache/top_users.json with subreddit as key
    """ Build a JSON file of top users for en-masse downloading and reducing duplicate downloads of top users - as well as being able to quickly fetch the list without calling and
        aggregating from mongodb. Save to ./scalp/cache/top_users.json

    Args:
        args (dict): Dict of arguments:
                    sub: subreddit name
                    mongodb_address: address (IP) of mongo db without the mongodb://
                    username: mongo username
                    password: mongo password 
                    database_str: database name
    """
    mongodb_address, username, password, database_str = args['mongodb_address'], args['username'], args['password'], args['database_str']
    
    client = MongoClient(f'mongodb://{username}:{password}@{mongodb_address}/', connect=False) if username!='' else MongoClient(f'mongodb://{mongodb_address}/', connect=False)
    db = client[database_str]

    # Aggregate subreddits with at least one submission downloaded
    subreddits_list = [a['_id'] for a in list(db['Submissions'].aggregate([{'$group': {'_id': '$subreddit'}}]))]
    
    for sub in [args['sub']]:
        print(f'[INFO]: building top user list for {sub}') 
        top_submit, top_comm = get_user_windows(sub, db, 500, 2020, args=args) # Fetch windowed top users for sub

        # Store top users in json file 
        top_users_json = json.load(open('scalp/cache/top_users.json', 'r')) 
        top_users_json[sub] = {'submittors': top_submit, 'commentors': top_comm}
        json.dump(top_users_json, open('scalp/cache/top_users.json', 'w'))

    client.close() 

def build_random_list(args: dict):
    """ Build a JSON file of randomly sampled users with some threshold for en-masse downloading and reducing duplicate downloads of users - as well as 
        being able to quickly fetch the list without calling and aggregating from mongodb. Save to ./scalp/cache/rand_users.json

    Args:
        args (dict): Dict of arguments:
                    sub: subreddit name
                    mongodb_address: address (IP) of mongo db without the mongodb://
                    username: mongo username
                    password: mongo password 
                    database_str: database name
                    threshold: minimum number of posts (comments/submissions) to be considered for sampling
    """
    mongodb_address, username, password, database_str = args['mongodb_address'], args['username'], args['password'], args['database_str']
    
    client = MongoClient(f'mongodb://{username}:{password}@{mongodb_address}/', connect=False) if username!='' else MongoClient(f'mongodb://{mongodb_address}/', connect=False)
    db = client[database_str]

    # Aggregate subreddits with at least one submission downloaded
    subreddits_list = [a['_id'] for a in list(db['Submissions'].aggregate([{'$group': {'_id': '$subreddit'}}]))]
    
    for sub in [args['sub']]:
        print(f'[INFO]: building random user list for {sub}') 
        #top_submit, top_comm = get_user_windows(sub, db, 500, 2020, args=args) # Fetch windowed top users for sub

        rand_user_list = get_user_random(sub, db, num_users=args['top_x'], args=args) # use top_x to limit number of users sampled 
        print(len(rand_user_list))
        
        # Store top users in json file 
        rand_users_json = json.load(open('scalp/cache/rand_users.json', 'r')) 
        rand_users_json[sub] = rand_user_list
        json.dump(rand_users_json, open('scalp/cache/rand_users.json', 'w'))

    client.close() 

def scalp_users(args: dict):
    """ From build JSON list in ./scalp/cache/top_users.json, scalp user submissions and comments from all time (have to use hot, new, top, controversial 
        from Reddit API)

    Args:
        args (dict): Dict of arguments:
                    sub: subreddit name
                    mongodb_address: address (IP) of mongo db without the mongodb://
                    username: mongo username
                    password: mongo password 
                    database_str: database name
    """
    mongodb_address, username, password, database_str = args['mongodb_address'], args['username'], args['password'], 'Users' 

    # Fetch top user json file
    json_top_users = json.load(open(f'scalp/cache/{args["json_loc"]}.json'))
    set_top_users = set() 

    # Loop through json file and create a set of top users (no duplicates)
    sub_list = list(json_top_users.keys()) if args['sub']=='' else [args['sub']]
    for sub in sub_list:
        for d_type in json_top_users[sub].keys():
            for date in json_top_users[sub][d_type].keys():
                fetchable_top = [a[0] for a in json_top_users[sub][d_type][date]['fetchable']]
                set_top_users = set_top_users | set(fetchable_top)
    
    # Multiprocessing (max processes same as number of keys in keys.json)
    all_users = list(set_top_users)
    all_users = [{'user': a, 'mongodb_address': mongodb_address, 'username': username, 'password': password, 'database_str': database_str} for a in all_users]
    pool = Pool() 
    for _ in tqdm(pool.imap_unordered(user_scalp_worker, all_users), total=len(all_users)): pass # Check ThreadPool (bunch of threads) vs Pool
    pool.close()


if __name__=="__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument('--address', dest='mongodb_address', type=str, default='localhost:27017')
    parser.add_argument('--user', dest='username', type=str, default='')
    parser.add_argument('--pwd', dest='password', type=str, default='')
    parser.add_argument('--cpus', dest='cpus', type=int, default=multiprocessing.cpu_count())

    parser.add_argument('--sub', dest='sub', type=str, default='')
    parser.add_argument('--scalp', dest='scalp_method', type=str, default='subreddit') # Scalp a subreddit, scalp users or build top list
    parser.add_argument('--dstring', dest='database_str', type=str, default='Reddit') # Only used when building top list 
    parser.add_argument('--p_subs', dest='push_submissions', type=int, default=1) # Use PushAPI to download submissions
    parser.add_argument('--p_comms', dest='push_comments', type=int, default=0) # Use PushAPI to download comments

    parser.add_argument('--year', dest='year', type=int, default=2020)
    parser.add_argument('--topx', dest='top_x', type=int, default=500)
    parser.add_argument('--threshold', dest='threshold', type=int, default=10)
    parser.add_argument('--jloc', dest='json_loc', type=str, default='top_users')
    

    args = parser.parse_args() 

    assert args.scalp_method in ['subreddit', 'controversial', 'random', 'users', 'build_list', 'build_random']
    func_map = {'subreddit': scalp_subreddit, 'controversial': scalp_controversial_subreddit, 'random': scalp_random_subreddit, 
                'users': scalp_users, 'build_list': build_top_list, 'build_random': build_random_list}
            
    func_map[args.scalp_method](vars(args))