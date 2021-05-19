# Data fetch and store 
from re import L
import pymongo 
from pymongo import MongoClient, errors 
import praw 
import prawcore

# Other imports 
import datetime as dt
import json 
import multiprocessing 
from multiprocessing.pool import Pool, ThreadPool 
from tqdm import tqdm
import argparse 
import time 
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import BertTokenizer
import warnings

# Local libraries 
from scalp.pushshift import PushAPI 
from scalp.scalp_help import insert_list


# Helper functions
def find_subreddit_date_range(sub_str: str) -> tuple:
    """ Find the date range of a subreddit from Pushshift - note this may not actually be the real date range, rather what Pushshift offers

    Args:
        sub_str (str): Name of the subreddit

    Returns:
        tuple: (First submission, last submission, first comment, last comment)
    """
    api = PushAPI() 
    sub_s = dt.datetime.fromtimestamp(list(api.search_submissions(subreddit=sub_str,limit=1,sort="asc"))[0]['created_utc'], tz=dt.timezone.utc)
    sub_e = dt.datetime.fromtimestamp(list(api.search_submissions(subreddit=sub_str,limit=1,sort="desc"))[0]['created_utc'], tz=dt.timezone.utc)
    comm_s = dt.datetime.fromtimestamp(list(api.search_comments(subreddit=sub_str,limit=1,sort="asc"))[0]['created_utc'], tz=dt.timezone.utc)
    comm_e = dt.datetime.fromtimestamp(list(api.search_comments(subreddit=sub_str,limit=1,sort="desc"))[0]['created_utc'], tz=dt.timezone.utc)

    print(sub_str, flush=True)
    print(f"\t{sub_s} to {sub_e}", flush=True)
    print(f"\t{comm_s} to {comm_e}", flush=True)

    return (sub_s, sub_e, comm_s, comm_e)


def extra_info(item):
    warnings.filterwarnings("ignore")
    analyzer = SentimentIntensityAnalyzer() 
    #tokenizer = BertTokenizer('bert-base-uncased') 
    if 'body' in item.keys():
        score = analyzer.polarity_scores(item['body'])['compound']
        #tok_len=0
        #if item['body'] == '[deleted]': tok_len = 0
        #tok_len = len(tokenizer.encode(text=item['body'], verbose=False))
        item['vader_body'] = score 
        #item['bert_token_body'] = tok_len

    if 'title' in item.keys():
        score = analyzer.polarity_scores(item['title'])['compound']
        #tok_len=0
        #if item['title'] == '[deleted]': tok_len = 0
        #tok_len = len(tokenizer.encode(text=item['title'], verbose=False))
        item['vader_title'] = score 
        #item['bert_token_title'] = tok_len 
    
    if 'selftext' in item.keys():
        score = analyzer.polarity_scores(item['selftext'])['compound']
        #tok_len=0
        #if item['selftext'] == '[deleted]': tok_len = 0
        #tok_len = len(tokenizer.encode(text=item['selftext'], verbose=False))
        item['vader_selftext'] = score 
        #item['bert_token_selftext'] = tok_len

    return item


def vader_worker(dict_item):
    # TODO: implement this - check if comment or submission, then do vader on body or title&selftext
    dict_item = extra_info(dict_item)
    return dict_item

def make_praw_list_unique(items: list) -> list:
    """ Make a list of downloaded Reddit API items unique when using new, hot, top etc.

    Args:
        items (list): List of items

    Returns:
        new_items (list): Same list but unique
    """
    ids_done, new_items = list(), list() 
    for a in items:
        if a.id in ids_done: continue 
        new_items.append(a) 
        ids_done.append(a.id) 
    return new_items

# Reddit API search 
def reddit_search_submissions(sub_str: str, db: pymongo.database.Database):
    """ Use Reddit API to download submissions 

    Args:
        sub_str (str): Subreddit name
        db (pymongo.database.Database): Database instance to insert data
    """
    # Get API key based on multiprocessing worker number, or 0 if single process
    api_keys = json.load(open("keys.json", "r"))["keys"]
    api_index = 0
    api_key = ''
    try: 
        api_index = int(multiprocessing.current_process().name.split("-")[1])-1 
        api_key = api_keys[api_index]
    except Exception: 
        api_index = 0
        api_key = api_keys[0] 
    
    REDDIT_ID = str(list(api_key.keys())[0])
    REDDIT_KEY = str(api_keys[api_index][list(api_key.keys())[0]])
    USER_AGENT = 'Reddit Research '+str(api_index+1)
    reddit = praw.Reddit(client_id=REDDIT_ID,
                        client_secret=REDDIT_KEY,
                        user_agent=USER_AGENT)

    # Fetch subredddit, download new, top, hot and controversial with no limit
    sr = reddit.subreddit(sub_str) 
    submissions = list()
    unique_submissions = list()
    print(f"[INFO]:\tDownloading new for {sub_str}", flush=True)
    for sb in sr.new(limit=None): submissions.append(sb)
    print(f"[INFO]:\tDownloading top for {sub_str}", flush=True)
    for sb in sr.top(limit=None): submissions.append(sb)
    print(f"[INFO]:\tDownloading hot for {sub_str}", flush=True)
    for sb in sr.hot(limit=None): submissions.append(sb) 
    print(f"[INFO]:\tDownloading controversial for {sub_str}", flush=True)
    for sb in sr.controversial(limit=None): submissions.append(sb)

    submissions = make_praw_list_unique(submissions) # Make the list unique to remove duplicates

    # Multiprocessing for VADER sentiment computations on submissions
    print(f"[INFO]:\tComputing VADER on submissions")
    pool = Pool() 
    submissions = list(tqdm(pool.imap_unordered(vader_worker, submissions), total=len(submissions)))
    pool.close() 

    # Insert into database under collection 'Submissions'
    insert_list(submissions, db, "Submissions")

def reddit_search_comments(sub_str: str, submission_id: str, db, reddit: praw.reddit.Reddit, analyzer: SentimentIntensityAnalyzer):
    """ Scalp the comments from a given submission ID

    Args:
        sub_str (str): Subreddit name
        submission_id (str): Submissions ID 
        db ([type]): Database instance to insert comment data
        reddit (praw.reddit.Reddit): Reddit API instance 
    """
    sb = reddit.submission(submission_id) 
    all_comments = list() 

    # Loop comments
    for comment in sb.comments:
        if type(comment)==praw.models.MoreComments: # If it's a morecomments object, recursively use this function 
            all_comments = all_comments + reddit_search_morecomments(comment, sub_str, submission_id, analyzer)
        else:
            # Calculate VADER score, BERT token length and add comment into list
            c = {'_id': comment.id, 'subreddit': sub_str, 'author': str(comment.author), 'created_utc': comment.created_utc,
                'score': comment.score, 'total_awards_received': comment.total_awards_received, 'body': comment.body, 'submission': submission_id}
            c = extra_info(c)
            all_comments.append(c)

    insert_list(all_comments, db, "Comments")

def reddit_search_morecomments(more_comment: praw.models.MoreComments, sub_str: str, submission_id: str, analyzer: SentimentIntensityAnalyzer) -> list:
    """ Scalp a praw.models.MoreComments object for comments from a submissions. 
    Recursive function which calls itself if it encounters more MoreComments objects.

    Args:
        more_comment (praw.models.MoreComments): The MoreComments object
        sub_str (str): Subreddit name
        submission_id (str): ID of the submission

    Returns:
        list: List of dict objects for comment data
    """
    # Helper recursive function for reddit_search_comments
    all_comments = list()
    for comment in more_comment.comments():
        if type(comment)==praw.models.MoreComments:
            all_comments = all_comments + reddit_search_morecomments(comment, sub_str, submission_id, analyzer)
        else:
            # Calculate VADER score, BERT token length and add comment into list
            c = {'_id': comment.id, 'subreddit': sub_str, 'author': str(comment.author), 'created_utc': comment.created_utc,
                'score': comment.score, 'total_awards_received': comment.total_awards_received, 'body': comment.body, 'submission': submission_id}
            c = extra_info(c)
            all_comments.append(c)

    return all_comments

# Push API search 

def push_search_submissions(sub_str: str, db: pymongo.database.Database, start_year: int = 2020, incremental_comment_search: bool = False, args: dict = dict()):
    """ Use Pushshift to download all submissions of a subreddit from the given year

    Args:
        sub_str (str): Subreddit name
        db (pymongo.database.Database): Database instance to insert data
        start_year (int): Year to download data from. Defaults to 2020 
        incremental_comment_search (bool): Use Reddit API to search submissions for comments as they are downloaded from Pushshift
        args (dict, optional): Arguments dict. Defaults to dict()
    """
    # Create API instance 
    api = PushAPI()
    analyzer = SentimentIntensityAnalyzer()
    start_epoch = int(dt.datetime(start_year, 1, 1, tzinfo=dt.timezone.utc).timestamp()) # Search submissions from this time onwards
    ids = list()

    # Fields to return and keep in DB 
    fields = ['id', 'subreddit', 'author', 'created_utc', 'num_comments', 'score', 'upvote_ratio', 'total_awards_received', 'title', 'selftext']

    while True: # Keep looping until no more submissions are returned 
        submissions = list(api.search_submissions(after=start_epoch, subreddit=sub_str, filter=fields, limit=100, sort="asc")) # Fetch submissions

        if len(submissions) == 0: break # If nothing returned, break loop 

        temp_ids = list()

        for submission in submissions: 
            try:ids.append(submission['id']) # If submission has no id, continue
            except Exception: continue

            new_submission = {k: None for k in fields}
            for k in fields: 
                try: new_submission[k] = submission[k] # Try and adds as many fields as possible
                except Exception: continue

            # Calculate VADER score and BERT token length 
            new_submission = extra_info(new_submission)
            new_submission['_id'] = new_submission['id'] 
            new_submission.pop('id', None)

            temp_ids.append(new_submission)

        # Insert into DB
        insert_list(temp_ids, db, 'Submissions')

        submission_ids = [a['_id'] for a in temp_ids]

        # If using this method, multiprocess download the comments for these submissions on-the-fly
        if incremental_comment_search: download_threaded_comments(submission_ids, sub_str, args) # atm is sync multiprocessing, TODO: test with async function and threading

        # New start_epoch is the time of the last submission downloaded, so download the next lot of submissions from this time
        submission = submissions[-1]
        start_epoch = int(submission['created_utc'])
        print(f"{dt.datetime.fromtimestamp(start_epoch)} for {sub_str}", flush=True)

def push_search_comments(sub_str: str, db: pymongo.database.Database, start_year: int = 2020, args: dict = dict()):
    """ Use Pushshift to download all comments of a subreddit from the given year

    Args:
        sub_str (str): Subreddit name
        db (pymongo.database.Database): Database instance to insert data
        start_year (int): Year to download data from. Defaults to 2020 
        args (dict, optional): Arguments dict. Defaults to dict()
    """
        # Create API instance 
    api = PushAPI()
    analyzer = SentimentIntensityAnalyzer()
    start_epoch = int(dt.datetime(start_year, 1, 1, tzinfo=dt.timezone.utc).timestamp()) # Search submissions from this time onwards
    ids = list()

    # Fields to return and keep in DB 
    fields = ['id', 'subreddit', 'author', 'created_utc', 'num_comments', 'score', 'upvote_ratio', 'total_awards_received', 'body']

    while True:
        comments = list(api.search_comments(after=start_epoch, subreddit=sub_str, filter=fields, limit=100, sort="asc"))

        if len(comments) == 0:break

        temp_ids = list()
        
        for comment in comments: 
            try:ids.append(comment['id']) # If submission has no id, continue
            except Exception: continue

            new_comment = {k: None for k in fields}
            for k in fields: 
                try: new_comment[k] = comment[k] # Try and adds as many fields as possible
                except Exception: continue

            # Calculate VADER score and BERT token length 
            new_comment = extra_info(new_comment)
            new_comment['_id'] = new_comment['id'] 
            new_comment.pop('id', None)

            temp_ids.append(new_comment)

        # Insert into DB
        insert_list(temp_ids, db, 'Comments') 

        # New start_epoch is the time of the last submission downloaded, so download the next lot of submissions from this time
        comment = comments[-1]
        start_epoch = int(comment['created_utc'])
        print(f"{dt.datetime.fromtimestamp(start_epoch)} for {sub_str}", flush=True)

# Multiprocessing to download comments

def download_threaded_comments(sub_ids: list, sub_str: str, args: dict = dict()):
    """ Start a multiprocessing pool to download comments from list of submission IDs

    Args:
        sub_ids (list): List of submission IDs - single subreddit only
        sub_str (str): Subreddit of submission IDs
        args (dict): args (dict, optional): Arguments dict. Defaults to dict()
    """
    # TODO: put in asserts to make sure kwargs has right arguments

    # create tuple data for workers
    #new_ids = [(sub_str, 'mongodb_address': args['mongodb_address'], args['username'], args['password'], args['database_str'], a) for a in sub_ids]
    new_ids = [dict({'submission_id': a}, **args) for a in sub_ids]
    pool = Pool() 
    for _ in tqdm(pool.imap_unordered(master_comment_worker, new_ids), total=len(new_ids)):
        pass 
    pool.close() 

def master_comment_worker(data_dict: dict):
    """ Worker function for multiprocessing downloading of comments from a submission

    Args:
        data_dict (dict): 
                    user: Username, 
                    mongodb_address: mongodb address, 
                    username: mongo username, 
                    password: mongo password, 
                    database_str: database name
                    submission_id: submission id
    """
    # Fetch multiprocessing arguments 
    sub_str, mongodb_address, username, password = data_dict['sub'], data_dict['mongodb_address'], data_dict['username'], data_dict['password']
    database_str, submission_id = data_dict['database_str'], data_dict['submission_id']
    
    # Connect to DB and fetch Reddit API instance using worker ID to fetch key
    client = MongoClient(f'mongodb://{username}:{password}@{mongodb_address}/', connect=False) if username!='' else MongoClient(f'mongodb://{mongodb_address}/', connect=False)
    db = client[database_str] 
    analyzer = SentimentIntensityAnalyzer() 
    api_keys = json.load(open("keys.json", "r"))["keys"]
    api_index = (int(multiprocessing.current_process().name.split("-")[1])-1)%16
    api_key = api_keys[api_index]

    REDDIT_ID = str(list(api_key.keys())[0])
    REDDIT_KEY = str(api_keys[api_index][list(api_key.keys())[0]])
    USER_AGENT = 'Reddit Research '+str(api_index+1)
    reddit = praw.Reddit(client_id=REDDIT_ID,
                        client_secret=REDDIT_KEY,
                        user_agent=USER_AGENT)
    
    for _ in range(1800): # 1 hour
        try:
            reddit_search_comments(sub_str, submission_id, db, reddit, analyzer)
            break
        except prawcore.exceptions.Forbidden as e:
            print("Forbidden submission", flush=True)
            break 
        except prawcore.exceptions.NotFound as e:
            print("Submission not found", flush=True)
            break 
        except prawcore.exceptions.TooLarge as e:
            print(e, flush=True)
            break
        except Exception as e:
            print(e, flush=True)
            print(f"[INFO]: Trying {submission_id} again...", flush=True)
            time.sleep(2)
    
    client.close() 