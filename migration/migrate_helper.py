import json 


def fetch_top_users_from_file(subreddit: str):
    json_top_users = json.load(open('scalp/cache/top_users.json', 'r'))
    set_top_users = set() 
    try:
        for d_type in json_top_users[subreddit].keys():
            for date in json_top_users[subreddit][d_type].keys():
                fetchable_top = [a[0] for a in json_top_users[subreddit][d_type][date]['fetchable']]
                set_top_users = set_top_users | set(fetchable_top)
    except Exception as e:
        pass 

    return list(set_top_users)