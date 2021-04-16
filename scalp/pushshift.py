import requests
from requests import adapters
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import time 

class PushAPI():

    def __init__(self):
        pass 

    def search_submissions(self, subreddit: str='', after: int=-1, before: int=-1, filter: list=list(), limit: int=100, sort: str='desc'):
        """ Custom API function to call Pushshift to search for submissions

        Args:
            subreddit (str, optional): Subreddit argument. Defaults to ''.
            after (int, optional): Submissions after this time in UNIX epoch. Defaults to -1.
            before (int, optional): Submissions before this time in UNIX epoch. Defaults to -1.
            filter (list, optional): List of fields to return. Defaults to [].
            limit (int, optional): Limit of submissions. Defaults to 100.
            sort (str, optional): Sort in asc or desc order. Defaults to 'desc'.
        """
        # Session to retry with about an hour of retry before moving on
        http = requests.session() 
        retry_strategy = Retry(total=1800)
        adapter = HTTPAdapter(timeout=2, max_retries=retry_strategy)
        http.mount("https://", adapter)
        http.mount("http://", adapter)

        while True:
            str_builder = 'https://api.pushshift.io/reddit/search/submission/?' # build API request string with arguments
            if subreddit!='': str_builder += f'subreddit={subreddit}'
            if after>=0: str_builder += f'&after={after}'
            if before>=0: str_builder += f'&before={before}'
            # Add in filter elements if there are any 
            if len(filter)>0: 
                str_builder += f'&fields='
                for f in filter[:-1]: str_builder += f'{f},'
                str_builder += f'{filter[-1]}'
            
            str_builder += f'&limit={limit}'
            str_builder += f'&sort={sort}'

            r = http.get(str_builder)
            
            # If good, return, else check with _non_200 to see what to do...atm just return 'keep going'
            if r.status_code==200:
                return r.json()['data']
            else:
                # TODO: make this loop and redo after cooldown given certain http codes
                todo = self._non_200(r.status_code)
                if todo: break
                time.sleep(2)
                


    def search_comments(self, subreddit: str='', after: int=-1, before: int=-1, filter: list=list(), limit: int=100, sort: str='desc'):
        """ Custom API function to call Pushshift to search for submissions

        Args:
            subreddit (str, optional): Subreddit argument. Defaults to ''.
            after (int, optional): Submissions after this time in UNIX epoch. Defaults to -1.
            before (int, optional): Submissions before this time in UNIX epoch. Defaults to -1.
            filter (list, optional): List of fields to return. Defaults to [].
            limit (int, optional): Limit of submissions. Defaults to 100.
            sort (str, optional): Sort in asc or desc order. Defaults to 'desc'.
        """
        # Session to retry with about an hour of retry before moving on
        http = requests.session() 
        retry_strategy = Retry(total=1800)
        adapter = HTTPAdapter(timeout=2, max_retries=retry_strategy)
        http.mount("https://", adapter)
        http.mount("http://", adapter)

        while True:
            str_builder = 'https://api.pushshift.io/reddit/search/comment/?' # build API request string with arguments
            if subreddit!='': str_builder += f'subreddit={subreddit}'
            if after>=0: str_builder += f'&after={after}'
            if before>=0: str_builder += f'&before={before}'
            # Add in filter elements if there are any 
            if len(filter)>0: 
                str_builder += f'&fields='
                for f in filter[:-1]: str_builder += f'{f},'
                str_builder += f'{filter[-1]}'
            
            str_builder += f'&limit={limit}'
            str_builder += f'&sort={sort}'

            r = http.get(str_builder)
            
            # If good, return, else check with _non_200 to see what to do...atm just return 'keep going'
            if r.status_code==200:
                return r.json()['data']
            else:
                # TODO: make this loop and redo after cooldown given certain http codes
                todo = self._non_200(r.status_code)
                if todo: break
                time.sleep(2) 


    def _non_200(self, code):
        # Print warning with code, retry connection
        print(f'[WARNING]: {code}', flush=True) 
        print("\tRetrying after backing off", flush=True)
        return False # whether to break loop or not  ATM just keep trying until manual intervention