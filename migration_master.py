import json
from migration.micro_migrate import run_micro_sc 
import multiprocessing
import matplotlib.pyplot as plt 
from pandas.core import series
import pymongo 
from pymongo import MongoClient 
import argparse
import datetime as dt
import pandas as pd 
import itertools
from scipy import stats
import numpy as np 
import tqdm


from migration.macro_migrate import run_macro


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--address', dest='mongodb_address', type=str, default='localhost:27017')
    parser.add_argument('--user', dest='username', type=str, default='')
    parser.add_argument('--pwd', dest='password', type=str, default='')
    parser.add_argument('--cpus', dest='cpus', type=int, default=multiprocessing.cpu_count())

    parser.add_argument('--mtype', dest='migrate_type', type=str, default='macro') # Type of migration to run 
    parser.add_argument('--fstr', dest='file_str', type=str, default='scalp/cache/top_users.json') # File to find top users 
    parser.add_argument('--pfstr', dest='plot_file_str', type=str, default='') 
    parser.add_argument('--dtype', dest='d_type', type=str, default='submissions') # comments, submissions, both
    parser.add_argument('--sub', dest='sub', type=str, default='') # Subreddit if using top users 
    parser.add_argument('--dstring', dest='database_str', type=str, default='Reddit')
    parser.add_argument('--year', dest='year', type=int, default=2020)
    parser.add_argument('--topx', dest='top_x', type=int, default=500) 
    

    args = parser.parse_args() 
    args_vars = vars(args)

    if args_vars['migrate_type']=='macro':
        run_macro(args_vars) 
    elif args_vars['migrate_type']=='micro_sc': # Find the best set S_c to minimise the 'other' node
        run_micro_sc(args_vars) 
    elif args_vars['migrate_type']=='micro':
        run_micro(args_vars)