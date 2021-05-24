#%%
import pickle
import matplotlib.pyplot as plt 
import matplotlib as mpl 
import seaborn as sns 
import datetime as dt 
from dateutil.relativedelta import relativedelta
import pandas as pd 

from IPython.display import clear_output
from ipywidgets import interact, interactive, fixed, interact_manual, Layout
import ipywidgets as widgets

from migration.migrate_helper import print_graph

mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['lines.linewidth'] = 0.5
sns.set_style("darkgrid")


all_graphs = pickle.load(open('all_graphs.p', 'rb')) 
all_author = pickle.load(open('all_author.p', 'rb'))
n_all_graphs = {str(k.to_pydatetime() + relativedelta(days=1) - relativedelta(months=1)): all_graphs[k] for k in all_graphs.keys()}
n_all_author = {str(k.to_pydatetime() + relativedelta(days=1) - relativedelta(months=1)): all_author[k] for k in all_author.keys()}

def interact_func(date):
    n_date = dt.datetime.strptime(date, '%Y-%m')
    df = n_all_graphs[str(n_date)+'+00:00']
    df = df.fillna(0.0)
    print_graph(df)

dates = pd.date_range(start=dt.datetime(2019,1,1, tzinfo=dt.timezone.utc), end=dt.datetime(2021,4,1, tzinfo=dt.timezone.utc), freq='1M').to_list()
dates = [d.strftime("%Y-%m") for d in dates]
#interact(interact_func,
#                date=widgets.SelectionSlider(options=dates, value='2019-11')
#)

for d in dates:
    n_date = dt.datetime.strptime(d, '%Y-%m')
    df = n_all_graphs[str(n_date)+'+00:00']
    df = df.fillna(0.0)
    print(f"{d}: {n_all_author[str(n_date)+'+00:00']}")
    #print_graph(df, title=d, save_loc=f'res/{d}.png')

# %%
