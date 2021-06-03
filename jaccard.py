from pymongo import MongoClient 
import json
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import numpy as np 




client = MongoClient('mongodb://admin:password@192.168.1.123') 
db = client['Reddit'] 

top_users = json.load(open('scalp/cache/top_users.json')) 

subs = {k: [] for k in top_users.keys()} 

for s in top_users.keys():
    curr_list = [] 
    for d in top_users[s].keys():
        for da in top_users[s][d].keys():
            u = [a[0] for a in top_users[s][d][da]['fetchable']]
            curr_list.append(u)

    subs[s] = list(set([item for sublist in curr_list for item in sublist])) 
    
subs_int = {k: 0 for k in subs.keys()} 
subs_int = {k: subs_int.copy() for k in subs.keys()} 

for s in subs_int.keys():
    for ss in subs_int[s].keys():
        intersect = len((set(subs[s]) & set(subs[ss])))
        uni = len((set(subs[s]) | set(subs[ss]))) 
        subs_int[s][ss] = intersect / uni if uni>0 else 0 

df = pd.DataFrame(subs_int) 

f, ax = plt.subplots(figsize=(10, 6))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)
mask = np.zeros_like(df)
mask[np.triu_indices_from(mask)] = True

# Draw the heatmap with the mask and correct aspect ratio
g = sns.heatmap(df, cmap=cmap, mask=mask, center=0, linewidths=.5, cbar_kws={"shrink": .5}, square=True)
g.set_xticklabels(g.get_xmajorticklabels(), fontsize = 8)
g.set_yticklabels(g.get_ymajorticklabels(), fontsize = 8)
#sns.heatmap(df)
plt.tight_layout() 
plt.show()
