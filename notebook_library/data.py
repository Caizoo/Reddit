import numpy as np 
import pandas as pd 
import matplotlib as mpl 
import seaborn as sns 
import matplotlib.pyplot as plt 
import optuna 
import json
from sklearn.preprocessing import MinMaxScaler

from statsmodels.tsa.stattools import adfuller 


def one_diff(time_series):
    rts = [0] 
    for i in range(1, len(time_series)):
        rts.append(time_series[i]-time_series[i-1]) 
    return rts 






def find_stationize_method(x):
    # RETURN BEST STATIONIZING METHOD GIVEN THE DATA  
    
    # methods include:
        # diff, gradient, 2-nd order gradient, 3-rd order gradient
        # ma_3 subtraction, ma_7 subtraction 

    pass

def stationize(x, method):
    assert(type(x)==list or type(x)==np.ndarray or type(x)==pd.Series)
    time_series = x 
    if method=="None": time_series = x 
    if method=='Diff': time_series = one_diff(x)
    if method=='Grad': time_series = np.gradient(x) 

    return time_series

def getDelayedWindow(data, window, k_predict, i, output_features):

    X = data[i-window+1:i+1]
    Y = [data[i+k_predict][f] for f in output_features]
    
    return np.reshape(X, (X.shape[0], X.shape[1])), Y

def getDelayedDataset(data, window, k_predict, output_features):
    X = list()
    Y = list()
    for i in range(window, len(data)-k_predict):
        xx, yy = getDelayedWindow(data, window, k_predict, i, output_features=output_features)
        X.append(xx)
        Y.append(yy) 
    return np.asarray(X), np.asarray(Y)

def train_split(x, split):
    assert(type(x)==list or type(x)==np.ndarray or type(x)==pd.Series)
    train_split = int(len(x)*split) 
    return x[:train_split], x[train_split:] 

def normalize(x, split):
    assert(type(x)==list or type(x)==np.ndarray or type(x)==pd.Series)
    full_x = x.copy() 
    x = x[:int(len(x)*split)]
    x = np.asarray(x) 
    if len(x.shape)==1: x = x.reshape(-1,1)

    full_x = np.asarray(x)
    if len(full_x.shape)==1: full_x = full_x.reshape(-1,1)
    sc = MinMaxScaler() 
    sc.fit(x) 
    return sc.transform(full_x), sc  

def predict(x, model): 
    pass 

def denormalize(x, scaler):
    pass 

def destationize(x, method):
    pass 

def calculate_error(x, real):
    pass 

def plot_predictions(x, real): 
    pass 