# Data fetch and store 
import pymongo 
from pymongo import MongoClient, errors 
import praw 
import prawcore 
#from psaw import PushshiftAPI 

# Other imports 
import json 

# Local libraries 
import scalp.pushshift 
from scalp.pushshift import PushAPI

def hash_author(l_insert: list) -> list:
    return list(map(lambda a: a, l_insert))

def insert_list(l_insert: list, db: pymongo.database.Database, collection_str: str, overwrite: bool = True):
    """ Insert list of dict objects into a database

    Args:
        l_insert (list): List of dicts to insert
        db (pymongo.database.Database): PyMongo database instance to insert objects
        collection_str (str): String denoting the name of the collection to insert into 
    """



    try:
        # inserts new documents 
        db[collection_str].insert_many(l_insert, ordered=False, bypass_document_validation=True)
    except errors.BulkWriteError as e:
        # filter out document errors that weren't inserted
        panic_list = list(
            filter(lambda x: x['code'] != 11000, e.details['writeErrors'])) # If not 'document already exists' error then panic
        if len(panic_list) > 0:
            print(f"these are not duplicate errors {panic_list}", flush=True)

        # If to overwrite existing documents...
        if overwrite:
            to_overwrite_list = list(filter(lambda x: x['code']==11000, e.details['writeErrors']))
            for d in to_overwrite_list:
                doc = d['op']
                db[collection_str].replace_one({'_id': doc['_id']}, doc)
    except Exception as e:
        pass

def get_default_reddit_instance() -> praw.reddit.Reddit:
    """ Fetch a Reddit API instance using key at index 0

    Returns:
        praw.reddit.Reddit: [description]
    """
    api_keys = json.load(open("keys.json", "r"))["keys"]
    api_index = 0
    api_key = api_keys[api_index]
    REDDIT_ID = str(list(api_key.keys())[0])
    REDDIT_KEY = str(api_keys[api_index][list(api_key.keys())[0]])
    USER_AGENT = 'Reddit Research '+str(api_index+1)
    reddit = praw.Reddit(client_id=REDDIT_ID,
                        client_secret=REDDIT_KEY,
                        user_agent=USER_AGENT)

    return reddit