import matplotlib.pyplot as plt  
import numpy as np 
import pandas as pd 
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
import multiprocessing
import optuna 

from statsmodels.tsa.ar_model import AutoReg, ar_select_order
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.stattools import adfuller, arma_order_select_ic, grangercausalitytests, coint, kpss
from statsmodels.tsa.seasonal import seasonal_decompose 
from statsmodels.tsa.api import acf, pacf, graphics
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.structural import UnobservedComponents

# STATIONARITY AND SEASONAL DECOMP -------------------------------------------------------------------------------------------------------------------------------------------------------------

def dickey_fuller_test(time_series, title, saveloc):
    fig, axes = plt.subplots(3,1)
    
    fig.suptitle(title)
    pd.plotting.autocorrelation_plot(time_series, ax=axes[0])
    dftest = adfuller(time_series, autolag='AIC') 
    print("ADF", dftest[0]) 
    print("P-Value", dftest[1]) 
    print("Num of lags", dftest[2]) 
    print("Num obvs for ADF reg", dftest[3]) 
    print("Critical vals") 
    for key, val in dftest[4].items(): print("\t", key, ":", val)

    print(kpss(time_series, nlags='auto', regression='c'))
    print(kpss(time_series, nlags='auto', regression='ct'))

    plot_acf(np.asarray(time_series), ax=axes[1], lags=np.arange(40))
    plot_pacf(np.asarray(time_series), ax=axes[2], lags=np.arange(40))
    plt.tight_layout() 
    
    #plt.savefig(saveloc) 
    plt.show()

def seasonal_decomp(time_series, title, saveloc):
    decomp = seasonal_decompose(time_series) 
    trend = decomp.trend 
    seasonal = decomp.seasonal 
    residual = decomp.resid 

    fig, axes = plt.subplots(4,1)
    
    axes[0].plot(time_series, label='Original') 
    axes[0].legend(loc='best')
    axes[1].plot(trend, label='Trend') 
    axes[1].legend(loc='best')
    axes[2].plot(seasonal, label='Seasonality') 
    axes[2].legend(loc='best')
    axes[3].plot(residual, label='Residuals') 
    axes[3].legend(loc='best')
    fig.suptitle(title)
    plt.tight_layout() 
    #plt.savefig(saveloc)
    plt.show() 


def ma_subtraction(time_series, window):
    ma = time_series.rolling(window).mean()
    time_series = time_series.iloc[window-1:] 
    time_series = time_series - ma 
    return time_series.dropna() 

def one_diff(time_series):
    rts = [0] 
    for i in range(1, len(time_series)):
        rts.append(time_series[i]-time_series[i-1]) 
    return rts 

def compute_decomps(sub_dict, SUB, to_sub, vis_type="both"): 
    comments, submissions = sub_dict[to_sub]["comments"], sub_dict[to_sub]["submissions"] 
    comments_count, submissions_count = list(comments[['date']].groupby(pd.Grouper(key='date', freq='1d')).size()), list(submissions[['date']].groupby(pd.Grouper(key='date', freq='1d')).size())
    comments_count, submissions_count = [float(a) for a in comments_count], [float(a) for a in submissions_count]

    comments, submissions = comments[['date']].groupby(pd.Grouper(key='date', freq='1d')).size(), submissions[['date']].groupby(pd.Grouper(key='date', freq='1d')).size()
    comments_diff_nograd = pd.Series(data=one_diff(comments) , index=comments.index)
    comments_diff = pd.Series(data=np.gradient(comments) , index=comments.index)
    comments_diff_2 = pd.Series(data=np.gradient(comments_diff.values), index=comments.index)
    comments_diff_3 = pd.Series(data=np.gradient(comments_diff_2.values), index=comments.index)
    comments_ma = ma_subtraction(comments, 3)
    comments_ma_7 = ma_subtraction(comments, 7)

    if vis_type=="ADF" or vis_type=="both":
        dickey_fuller_test(comments_count, 
                        f"Correlation for comment count {SUB}->{to_sub}", 
                        f"./res/{SUB}/corr/{SUB}_{to_sub}")
        dickey_fuller_test(comments_diff.values, 
                        f"Correlation for first-order diff comments {SUB}->{to_sub}", 
                        f"./res/{SUB}/corr/diff_{SUB}_{to_sub}")
        dickey_fuller_test(comments_diff_2.values, 
                        f"Correlation for second-order diff comments {SUB}->{to_sub}", 
                        f"./res/{SUB}/corr/diff2_{SUB}_{to_sub}")
        dickey_fuller_test(comments_diff_3.values, 
                        f"Correlation for third-order diff comments {SUB}->{to_sub}", 
                        f"./res/{SUB}/corr/diff3{SUB}_{to_sub}")
        dickey_fuller_test(comments_diff_nograd.values, 
                        f"Correlation for diff comments {SUB}->{to_sub}", 
                        f"./res/{SUB}/corr/diff_nograd_{SUB}_{to_sub}") 
        dickey_fuller_test(comments_ma.values, 
                        f"Correlation for MA_3 subtraction comments {SUB}->{to_sub}", 
                        f"./res/{SUB}/corr/ma3_{SUB}_{to_sub}")
        dickey_fuller_test(comments_ma_7.values, 
                        f"Correlation for MA_7 subtraction comments {SUB}->{to_sub}", 
                        f"./res/{SUB}/corr/ma7_{SUB}_{to_sub}") 

    if vis_type=="seasonal" or vis_type=="both":
        seasonal_decomp(comments, 
                        f"Seasonal for comment count {SUB}->{to_sub}", 
                        f"./res/{SUB}/season/{SUB}_{to_sub}")
        seasonal_decomp(comments_diff, 
                        f"Seasonal for first-order diff comments {SUB}->{to_sub}", 
                        f"./res/{SUB}/season/diff_{SUB}_{to_sub}")
        seasonal_decomp(comments_diff_2, 
                        f"Seasonal for second-order diff comments {SUB}->{to_sub}", 
                        f"./res/{SUB}/season/diff2_{SUB}_{to_sub}") 
        seasonal_decomp(comments_diff_3, 
                        f"Seasonal for third-order diff comments {SUB}->{to_sub}", 
                        f"./res/{SUB}/season/diff3_{SUB}_{to_sub}") 
        seasonal_decomp(comments_diff_nograd, 
                        f"Seasonal for diff comments {SUB}->{to_sub}", 
                        f"./res/{SUB}/season/diff_nograd_{SUB}_{to_sub}")
        seasonal_decomp(comments_ma, 
                        f"Seasonal for MA_3 subtraction comments {SUB}->{to_sub}", 
                        f"./res/{SUB}/season/ma3_{SUB}_{to_sub}")
        seasonal_decomp(comments_ma_7, 
                        f"Seasonal for MA_7 subtraction comments {SUB}->{to_sub}", 
                        f"./res/{SUB}/season/ma7_{SUB}_{to_sub}")  

# BASIC TIME SERIES MODELS ---------------------------------------------------------------------------------------------------------------------------------------------------------------------

def run_basic_models(sub_dict, SUB, TO_SUB, diff_type: str, LEVEL='ntrend', 
                    CYCLE=True, U_SEASON=7, ARIMA_ORDER=(0,0,0), SARIMA_ORDER=(0,0,0), SEASON_ORDER=(0,0,0,0)): 

    assert(diff_type in ["no", "diff", "grad", "grad_2", "ma_3", "ma_7"])
    comments, submissions = sub_dict[TO_SUB]["comments"], sub_dict[TO_SUB]["submissions"] 
    comments_count, submissions_count = (list(comments[['date']].groupby(pd.Grouper(key='date', freq='1d')).size()), 
                                         list(submissions[['date']].groupby(pd.Grouper(key='date', freq='1d')).size()))
    comments_count, submissions_count = [float(a) for a in comments_count], [float(a) for a in submissions_count]

    comments, submissions = comments[['date']].groupby(pd.Grouper(key='date', freq='1d')).size(), submissions[['date']].groupby(pd.Grouper(key='date', freq='1d')).size()
    
    
    comments_diff_nograd = pd.Series(data=one_diff(comments) , index=comments.index)
    comments_diff = pd.Series(data=np.gradient(comments) , index=comments.index)
    comments_diff_2 = pd.Series(data=np.gradient(comments_diff.values), index=comments.index)
    comments_ma = ma_subtraction(comments, 3)
    comments_ma_7 = ma_subtraction(comments, 7)
    
    
    df_use = comments.values 
    if diff_type=="diff": df_use = comments_diff_nograd.values
    elif diff_type=="grad": df_use = comments_diff.values
    elif diff_type=="grad_2": df_use = comments_diff_2.values
    elif diff_type=="ma_3": df_use = comments_ma.values
    elif diff_type=="ma_7": df_use = comments_ma_7.values

    train_size = int(len(df_use)*0.8)
    train, test = df_use[:train_size], df_use[train_size:]
    
    
    # UnobservedComponents 
    m = {"level": LEVEL, 'cycle': CYCLE, 'seasonal': U_SEASON}
    model = UnobservedComponents(train, **m) 
    u_res = model.fit() 
    preds_unobserved = u_res.get_forecast(steps=len(test), exog=[0 for _ in range(len(test))])._results
    conf_bands_unobserved = preds_unobserved.conf_int() 
    

    # ARIMA

    # find order value 
    model = ARIMA(endog=train, exog=[0 for _ in range(len(train))], order=ARIMA_ORDER) 
    arima_fit = model.fit()
    preds_arima = arima_fit.get_forecast(steps=len(test), exog=[0 for _ in range(len(test))])._results
    conf_bands_arima = preds_arima.conf_int() 
    
        
    # SARIMA

    model = SARIMAX(endog=train, exog=[0 for _ in range(len(train))], order=SARIMA_ORDER, seasonal_order=SEASON_ORDER)
    sarima_fit = model.fit()
    preds_sarima = sarima_fit.get_forecast(steps=len(test), exog=[0 for _ in range(len(test))])._results
    conf_bands_sarima = preds_sarima.conf_int() 
    

    # XGBOOST 


    return (train, test, u_res, arima_fit, sarima_fit, preds_unobserved, preds_arima, preds_sarima)

def plot_basic_models(train, test, u_res, arima_fit, preds_arima, conf_bands_arima, sarima_fit, preds_sarima, conf_bands_sarima):
    fig = u_res.plot_components()
    plt.tight_layout() 
    plt.show() 

    plt.figure()
    plt.plot(range(len(train), len(train)+len(test)), preds_arima.predicted_mean) 
    plt.plot(range(len(train), len(train)+len(test)), test)
    plt.fill_between(range(len(train), len(train)+len(test)), [c[0] for c in conf_bands_arima], [c[1] for c in conf_bands_arima], alpha=0.3)
    plt.plot(train) 
    plt.legend(("Predicted", "Real", "Train", "95% confidence"))
    plt.show() 
    plt.figure() 
    plt.title("Residuals") 
    plt.plot(arima_fit.fittedvalues)
    plt.show() 

    plt.figure() 
    plt.plot(range(len(train), len(train)+len(test)), preds_sarima.predicted_mean) 
    plt.plot(range(len(train), len(train)+len(test)), test)
    plt.fill_between(range(len(train), len(train)+len(test)), [c[0] for c in conf_bands_sarima], [c[1] for c in conf_bands_sarima], alpha=0.3)
    plt.plot(train) 
    plt.legend(("Predicted", "Real", "Train", "95% confidence"))
    plt.show() 
    plt.figure() 
    plt.title("Residuals") 
    plt.plot(sarima_fit.fittedvalues)
    plt.show() 

def init_best_vals(sub_dict, SUB, TO_SUB, diff_type: str):
    assert(diff_type in ["no", "diff", "grad", "grad_2", "ma_3", "ma_7"])
    comments, submissions = sub_dict[TO_SUB]["comments"], sub_dict[TO_SUB]["submissions"] 
    comments_count, submissions_count = (list(comments[['date']].groupby(pd.Grouper(key='date', freq='1d')).size()), 
                                         list(submissions[['date']].groupby(pd.Grouper(key='date', freq='1d')).size()))
    comments_count, submissions_count = [float(a) for a in comments_count], [float(a) for a in submissions_count]

    comments, submissions = comments[['date']].groupby(pd.Grouper(key='date', freq='1d')).size(), submissions[['date']].groupby(pd.Grouper(key='date', freq='1d')).size()
    
    
    comments_diff_nograd = pd.Series(data=one_diff(comments) , index=comments.index)
    comments_diff = pd.Series(data=np.gradient(comments) , index=comments.index)
    comments_diff_2 = pd.Series(data=np.gradient(comments_diff.values), index=comments.index)
    comments_ma = ma_subtraction(comments, 3)
    comments_ma_7 = ma_subtraction(comments, 7)
    
    
    df_use = comments.values 
    if diff_type=="diff": df_use = comments_diff_nograd.values
    elif diff_type=="grad": df_use = comments_diff.values
    elif diff_type=="grad_2": df_use = comments_diff_2.values
    elif diff_type=="ma_3": df_use = comments_ma.values
    elif diff_type=="ma_7": df_use = comments_ma_7.values

    train_size = int(len(df_use)*0.8)
    
    train, test = df_use[:train_size], df_use[train_size:]
    og_train, og_test = comments.values[:train_size], comments.values[train_size:]

    return train, test#, og_train, og_test 


def run_hyperparam(sub_dict, SUB, TO_SUB, n_trials):
    

    def objective_uc(trial):
        diff_type = trial.suggest_categorical('diff_type', ['no', 'diff', 'grad', 'grad_2', 'ma_3', 'ma_7'])
        level = trial.suggest_categorical('level', ["ntrend", "dconstant", "llevel", "rwalk", "dtrend", "lldtrend", "rwdrift", "lltrend", "strend", "rtrend"])
        cycle = trial.suggest_int('cycle', 0, 1)
        seasonal = trial.suggest_int('seasonal', 0, 14, step=2)
        train, test = init_best_vals(sub_dict, SUB, TO_SUB, diff_type)
        est = UCClass(level, cycle, seasonal) 
        est.fit(train, test)
        return est.score_norm() 

    def objective_arima(trial):
        diff_type = trial.suggest_categorical('diff_type', ['no', 'diff', 'grad', 'grad_2', 'ma_3', 'ma_7'])
        est = ARClass(trial.suggest_int('p', 0,5), trial.suggest_int('d', 0, 2), trial.suggest_int('q', 0, 5)) 
        train, test = init_best_vals(sub_dict, SUB, TO_SUB, diff_type)
        est.fit(train, test) 
        return est.score_norm()

    def objective_sarima(trial):
        diff_type = trial.suggest_categorical('diff_type', ['no', 'diff', 'grad', 'grad_2', 'ma_3', 'ma_7'])
        p = trial.suggest_int('p', 0,5)
        d = trial.suggest_int('d', 0, 2)
        q = trial.suggest_int('q', 0, 5)
        s = trial.suggest_int('s', 0, 14) if sum([p, d, q]) == 0 else trial.suggest_int('s', 2, 14)

        est = SRClass(p, d, q, p, d, q, s) 
        train, test = init_best_vals(sub_dict, SUB, TO_SUB, diff_type)

        # TODO: change this to prune trial 
        try:
            est.fit(train, test) 
            return est.score_norm()
        except ValueError as e:
            return 100000000000000
        except np.linalg.LinAlgError as e:
            return 100000000000000

    st_names = ['uc', 'arima', 'sarima']
    for st in st_names:
        st_n = st+f"_{SUB}_{TO_SUB}"
        study = optuna.create_study(study_name=st, storage="sqlite:///"+"notebook_library/optuna/"+st_n+".db", load_if_exists=True) 
        obj = list() 
        if st=='uc': obj = objective_uc 
        elif st=='arima': obj = objective_arima
        else: obj = objective_sarima
        study.optimize(obj, n_trials=n_trials) 
            
        print(f"Best params {study.study_name}:")
        print(study.best_params, "\n")

#def station_to_original(arr: np.array, diff_type: str): 


                

class UCClass():

    def __init__(self, level='no', cycle=False, seasonal=0):
        self.level = level 
        self.cycle = cycle
        self.seasonal = seasonal
        pass 

    def fit(self, train, test):
        self.train = train 
        self.test = test 
        self.model = UnobservedComponents(train, **{'level': self.level, 'cycle': self.cycle, 'seasonal': self.seasonal}) 
        self.u_res = self.model.fit() 
        self.preds_unobserved = self.u_res.get_forecast(steps=len(test), exog=[0 for _ in range(len(test))])._results

    def predict(self):
        return self.preds_unobserved.predicted_mean

    def score(self):
        return mean_squared_error(self.test, self.preds_unobserved.predicted_mean)

    def score_norm(self):
        sc = MinMaxScaler() 
        tt = sc.fit_transform(self.test.copy().reshape(-1,1)) 
        pt = sc.transform(self.preds_unobserved.predicted_mean.copy().reshape(-1,1))
        return mean_squared_error(tt, pt)

class ARClass():

    def __init__(self, p, d, q):
        self.p = p 
        self.d = d
        self.q = q
        pass 

    def fit(self, train, test):
        self.train = train 
        self.test = test 
        self.model = ARIMA(endog=train, exog=[0 for _ in range(len(train))], order=(self.p, self.d, self.q)) 
        self.arima_fit = self.model.fit()
        self.preds_arima = self.arima_fit.get_forecast(steps=len(test), exog=[0 for _ in range(len(test))])._results

    def predict(self):
        return self.preds_arima.predicted_mean

    def score(self):
        return mean_squared_error(self.test, self.preds_arima.predicted_mean)
    
    def score_norm(self):
        sc = MinMaxScaler() 
        tt = sc.fit_transform(self.test.copy().reshape(-1,1)) 
        pt = sc.transform(self.preds_arima.predicted_mean.copy().reshape(-1,1))
        return mean_squared_error(tt, pt)

class SRClass():

    def __init__(self, p, d, q, P, D, Q, s):
        self.p = p 
        self.d = d  
        self.q = q 
        self.P = P 
        self.D = D  
        self.Q = Q 
        self.s = s 
        pass 

    def fit(self, train, test):
        self.train = train 
        self.test = test 
        self.model = SARIMAX(endog=train, exog=[0 for _ in range(len(train))], order=(self.p, self.d, self.q), 
            seasonal_order=(self.P, self.D, self.Q, self.s))
        self.sarima_fit = self.model.fit()
        self.preds_sarima = self.sarima_fit.get_forecast(steps=len(test), exog=[0 for _ in range(len(test))])._results

    def predict(self):
        return self.preds_sarima.predicted_mean

    def score(self):
        return mean_squared_error(self.test, self.preds_sarima.predicted_mean)

    def score_norm(self):
        sc = MinMaxScaler() 
        tt = sc.fit_transform(self.test.copy().reshape(-1,1)) 
        pt = sc.transform(self.preds_sarima.predicted_mean.copy().reshape(-1,1))
        return mean_squared_error(tt, pt)
