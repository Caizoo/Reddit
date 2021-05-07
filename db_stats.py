from pymongo import MongoClient
import pandas as pd 
import numpy as np 
import seaborn as sns
from scipy.stats import skew
import matplotlib.pyplot as plt 




if __name__=="__main__":
    S = 'NoLockdownsNoMasks'
    client = MongoClient('mongodb://localhost') 
    db = client['Reddit'] 

    x = pd.DataFrame(db['Submissions'].find({'subreddit': S}))
    xx = pd.DataFrame(db['Comments'].find({'subreddit': S}))

    x_grouped = x.groupby('author').size() 
    xx_grouped = xx.groupby('author').size()
    
    

    print('#submissions:', len(x))
    print('#comments', len(xx))

    print('mean submissions:', x_grouped.mean())
    print('mean comments:', xx_grouped.mean()) 

    print('std.dev. submissions:', x_grouped.std()) 
    print('std.dev. comments:', xx_grouped.std())

    #print('skew submissions', skew(x_grouped))
    #print('skew comments', skew(xx_grouped))

    #x_n = np.sort(np.array(x_grouped))
    #plt.plot(x_n)

    #plt.show()