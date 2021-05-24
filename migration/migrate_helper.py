import pandas as pd 
import json 
import networkx as nx
import matplotlib.pyplot as plt 

def print_graph(df: pd.DataFrame, title: str='', save_loc: str=''):
    plt.figure() 
    cols = df.columns
    df.columns = list(range(len(df.columns))) 
    df.index = list(range(len(df.index)))

    edge_list = [] 
    for c in df.columns: 
        for i in df.index:
            edge_list.append({'from': c, 'to': i, 'weight': df.loc[i, c]}) 
    
    edge_list = pd.DataFrame(edge_list)
    
    G = nx.from_pandas_edgelist(edge_list, source='from', target='to', edge_attr='weight')
    pos = nx.circular_layout(G)  # positions for all nodes
    

    # nodes
    nx.draw_networkx_nodes(G, pos, node_size=100)

    # edges
    edges = G.edges()
    weights = [G[u][v]['weight'] for u,v in edges]
    nx.draw_networkx_edges(G, pos, width=weights)
    nx.draw_networkx_labels(G, pos, {k: k for k in df.columns}, font_size=16)

    for i, c in enumerate(cols):
        print(f'{c}: {i}')

    if save_loc!='':
        plt.title(title)
        plt.tight_layout()
        plt.savefig(save_loc, format="PNG")
    


def fetch_top_users_from_file(subreddit: str) -> list:
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

def fetch_rand_users_from_file(subreddit: str) -> list:
    json_users = json.load(open('scalp/cache/rand_users.json', 'r')) 
    return json_users[subreddit]