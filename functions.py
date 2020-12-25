#%%
# import libraries
import matplotlib.pyplot as plt
import math
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.ar_model import AutoReg
import statsmodels.tsa.holtwinters as ets
from statsmodels.tsa.api import Holt
import itertools
import pandas as pd
import numpy as np
import seaborn as sns

# ACF calculation function
def fun_acf(x, lags=20):
    T = len(x)
    mean = sum(x)/len(x)
    k = [] 
    for i in range(len(x)):       
        if len(k)<=lags:
            k.append(i)
    tau = [1]

    # calculate denominator
    den=0
    for i in x:
        den+=((i-mean)**2)

    # calculate all numerators
    nums = []
    if T == 1:
        return tau # corner case
    else:
        for j in k[1:]:
            num = []
            for i in range(1, len(x)):
                t = x[i]
                t_lag = x[i-j]
                if (i-j) >= 0:
                    num.append((t - mean)*(t_lag - mean))
            nums.append(num)
    
    # append to tau
    for i in nums:
        tau.append(sum(i)/den)

    return tau

# ACF list
def list_acf(x):
    T_list = []
    for i in x:       
        if len(T_list)<=50: # max lag
            T_list.append(i)

    T_stem = []
    for i in reversed(T_list[1:]):
        T_stem.append(i)
    for i in T_list:
        T_stem.append(i)
    
    return(T_stem)

# ACF plot function
def plot_acf(x, lags=20):
    '''
    x: output of fun_acf()
    This function is the ACF plot
    '''
    k_list = []
    for i in x:       
        if len(k_list)<=lags: # default lag=20
            k_list.append(x.index(i))

    k_stem = []
    for i in reversed(k_list[1:]):
        k_stem.append(-1*i)
    for i in k_list:
        k_stem.append(i)
    #print(k_stem)

    T_list = []
    for i in x:       
        if len(T_list)<=50: # max lag 50
            T_list.append(i)

    T_stem = []
    for i in reversed(T_list[1:]):
        T_stem.append(i)
    for i in T_list:
        T_stem.append(i)
    #print(T_stem)

    plt.stem(k_stem, T_stem, use_line_collection = True) 
    plt.xlabel("Lags")
    plt.ylabel("Magnitude")
    plt.title("ACF Plot")
    plt.show()

# ADF test function - tests if input variable is stationary 
def ADF_Cal(x):
    result = adfuller(x)
    print("ADF Statistic: %f" %result[0])
    print("p-value: %f" %result[1])
    print("Critical Values:")
    for key, value in result[4].items():
        print('\t%s: %.3f' %(key, value))

# Correlation coefficient
def correlation_coefficient_cal(x, y):
    x_bar = sum(x)/len(x)
    y_bar = sum(y)/len(y)
    num = []
    den_1 = []
    den_2 = []
    for i in range(len(x)) :
        a = (x[i] - x_bar)*(y[i] - y_bar)
        num.append(a)
        b = (x[i] - x_bar)**2
        den_1.append(b)
        c = (y[i] - y_bar)**2
        den_2.append(c)
    r = sum(num)/math.sqrt((sum(den_1) * sum(den_2)))
    return r

# Stepwise function - forward selection - for choosing best variables in Multiple Linear Regression/LSE
def stepwise_fun(X,y):
    '''
    X: dataframe without a constant column added
    y: target variable
    '''
    all_combinations = []
    for r in range(len(X.columns.tolist()) + 1):    
        combinations_object = itertools.combinations(X.columns.tolist(), r)
        combinations_list = list(combinations_object)
        all_combinations += combinations_list
    
    stepwise_summary = pd.DataFrame({'Variables': [], 'AIC': [], 'BIC': [], 'Adj. R-squared': []})
    for i in range(1,len(all_combinations)):
        vars_i = np.asarray(all_combinations[i])
        X_i = X[vars_i]
        X_i = sm.add_constant(X_i)
        OLS_i = sm.OLS(y,X_i).fit()
        metrics_i = {'AIC': OLS_i.aic, 'Adj. R-squared': OLS_i.rsquared_adj, 'BIC': OLS_i.bic, 'Variables': [vars_i], 'R-squared': OLS_i.rsquared}
        stepwise_summary = stepwise_summary.append(metrics_i, ignore_index=True)
    return stepwise_summary

# no constant
def stepwise_fun_base(X,y):
    '''
    X: dataframe without a constant column added
    y: target variable
    '''
    all_combinations = []
    for r in range(len(X.columns.tolist()) + 1):    
        combinations_object = itertools.combinations(X.columns.tolist(), r)
        combinations_list = list(combinations_object)
        all_combinations += combinations_list
    
    stepwise_summary = pd.DataFrame({'Variables': [], 'AIC': [], 'BIC': [], 'Adj. R-squared': []})
    for i in range(1,len(all_combinations)):
        vars_i = np.asarray(all_combinations[i])
        X_i = X[vars_i]
        # X_i = sm.add_constant(X_i)
        OLS_i = sm.OLS(y,X_i).fit()
        metrics_i = {'AIC': OLS_i.aic, 'Adj. R-squared': OLS_i.rsquared_adj, 'BIC': OLS_i.bic, 'Variables': [vars_i], 'R-squared': OLS_i.rsquared}
        stepwise_summary = stepwise_summary.append(metrics_i, ignore_index=True)
    return stepwise_summary

# Moving average functions
# manual: enter window size and folding order when running function
def cal_MA_wFO_manual(x, window_size=0, folding_order=0):
    print('Enter window size:')
    window_size=int(input())
    print('Enter folding order (enter 0 if none):')
    folding_order=int(input())
    # corner cases
    if window_size <= 2:
        return("Window size must be greater than 2")
    if (window_size % 2 == 0):
        if (folding_order > 0):
            if (folding_order % 2 != 0):
                return("Window size and folding order must be either both even or both odd")
    if (window_size % 2 != 0):
        if (folding_order > 0):
            if (folding_order % 2 == 0):
                return("Window size and folding order must be either both even or both odd")

    # moving average: no folding order
    i = 0
    moving_averages = []
    while i < len(x) - window_size + 1:
        window = x[i : i + window_size]
        window_average = sum(window) / window_size
        moving_averages.append(window_average)
        i += 1

    if (folding_order < 1):
        return(moving_averages)
    # moving average: folding order
    else:
        ma_ma = []
        j = 0
        while j < len(moving_averages) - folding_order + 1:
            window_j = moving_averages[j : j + folding_order]
            window_average_j = sum(window_j) / folding_order
            ma_ma.append(window_average_j)
            j += 1
        return(ma_ma)
# not manual:  window size (and folding order) are inputs
def cal_MA_wFO(x, window_size, folding_order=0):
    # corner cases
    if window_size <= 2:
        return("Window size must be greater than 2")
    if (window_size % 2 == 0):
        if (folding_order > 0):
            if (folding_order % 2 != 0):
                return("Window size and folding order must be either both even or both odd")
    if (window_size % 2 != 0):
        if (folding_order > 0):
            if (folding_order % 2 == 0):
                return("Window size and folding order must be either both even or both odd")

    # moving average: no folding order
    i = 0
    moving_averages = []
    while i < len(x) - window_size + 1:
        window = x[i : i + window_size]
        window_average = sum(window) / window_size
        moving_averages.append(window_average)
        i += 1

    if (folding_order < 1):
        return(moving_averages)
    # moving average: folding order
    else:
        ma_ma = []
        j = 0
        while j < len(moving_averages) - folding_order + 1:
            window_j = moving_averages[j : j + folding_order]
            window_average_j = sum(window_j) / folding_order
            ma_ma.append(window_average_j)
            j += 1
        return(ma_ma)

# Estimate AR coefficients - enter # samples, AR order, list of coefficients - generates simulation based on inputs and uses AutoReg to estimate coeffs
def AR_est():
    print('Enter the number of samples:')
    T_AR = int(input())
    print('Enter the order number:')
    order = int(input())
    print('Enter the parameters (with a space in between each)')
    params = list(map(float, input("Enter a multiple value: ").split()))

    e_AR = np.random.normal(1, np.sqrt(2), size = T_AR) # WN mean and standard dev
    y_AR = np.zeros(len(e_AR))

    for i in range(len(e_AR)):
        if i == 0:
            y_AR[0] = e_AR[0]
        elif i == 1:
            y_AR[1] = params[0]*y_AR[0] + e_AR[1]
        else:
            if order > 1:
                order_len = order * 1
                order_list = [1]
                while order_len > 1:
                    order_list.append(order_list[-1]+1)
                    order_len -=1
            y_AR[i] = sum(params[j-1]*y_AR[i-j] for j in order_list) + e_AR[i]  
    
    model_AR = AutoReg(y_AR, lags=order)
    model_AR_fit = model_AR.fit()
    print('Estimated Coefficients: %s' % model_AR_fit.params)
    true_coeff = [1]
    for i in params:
        true_coeff.append(i)
    print('True Coefficients: %s' % true_coeff)

    return y_AR

# GPAC function - input ACF process from statsmodel library
def Cal_GPAC(acf, cols, rows):
    GPAC_table = np.zeros((cols, rows))
    mid = int(len(acf) / 2)
    for j in range(rows):
        for k in range(1, cols + 1):
            num = np.zeros((k, k))
            den = np.zeros((k, k))

            acf_counter = mid + j

            for c in range(k):
                k_counter = 0
                for r in range(k):
                    den[r, c] = acf[acf_counter + k_counter]
                    k_counter += 1
                acf_counter -= 1

            num[:, :-1] = den[:, :-1]

            acf_counter = mid + j
            for r in range(k):
                num[r, -1] = acf[acf_counter + 1]
                acf_counter += 1

            num_det = np.linalg.det(num)
            den_det = np.linalg.det(den)

            gpac_value = num_det / den_det
            GPAC_table[j, k-1] = gpac_value

    xticks = np.arange(1,k+1,1)

    plt.subplots(figsize=(15,10))
    ax = sns.heatmap(GPAC_table, vmin=-1, vmax=1, center=0, square=True, cmap='magma', annot=True, fmt='.3f')
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom + 0.5, top - 0.5)
    ax.set_xticklabels(xticks, horizontalalignment='center')
    ax.set(xlabel='k', ylabel='j')
    plt.title('Generalized Partial AutoCorrelation (GPAC) Table')
    plt.show()

# BASE MODELS - Average, Naive, Drift, Simple Exponential Smoothing (SES)
# Average Method
def average_method(train, test, times):
    avg_predict = []
    avg_forecast = []
    tot = 0
    count = 0

    for i in range(len(train)-1):
        tot += train[i]
        count += 1
        running_avg = tot / count
        avg_predict.append(running_avg)
    avg_predict = avg_predict[1:]

    avg_forecast_1 = np.mean(train)
    avg_forecast = np.ones(len(test)) * avg_forecast_1
    avg_forecast = avg_forecast.tolist()

    limit = len(train)
    times_train = times[:limit]
    times_test = times[limit:]

    plt.figure()
    plt.plot(times_train, train, label='Training')
    plt.plot(times_test, test, label='Testing')
    plt.plot(times_test, avg_forecast, label='Forecast')
    plt.xlabel('Time')
    plt.xticks(np.arange(0,len(times),round(len(times)/10)), rotation=60)
    plt.ylabel('Amount')
    plt.title('Average Method')
    plt.legend()
    plt.show()

    train_avg_error = train[2:]
    avg_predict_error = train_avg_error - avg_predict
    avg_predict_error_sq = avg_predict_error ** 2
    avg_predict_MSE = sum(avg_predict_error_sq) / len(avg_predict_error_sq)
    print('Average method 1-step prediction MSE:',avg_predict_MSE)

    avg_forecast_error = test - avg_forecast
    avg_forecast_error_sq = avg_forecast_error ** 2
    avg_forecast_MSE = sum(avg_forecast_error_sq) / len(avg_forecast_error_sq)
    print('Average method h-step forecast MSE:',avg_forecast_MSE)

    avg_predict_error_mean = sum(avg_predict_error) / len(avg_predict_error)
    avg_predict_var1 = []
    for i in range(len(avg_predict_error)):
        avg_predict_var1.append((avg_predict_error[i] - avg_predict_error_mean) ** 2)
    avg_predict_var = sum(avg_predict_var1) / len(avg_predict_var1)
    print('Average method 1-step prediction error variance:',avg_predict_var)

    avg_forecast_error_mean = sum(avg_forecast_error) / len(avg_forecast_error)
    avg_forecast_var1 = []
    for i in range(len(avg_forecast_error)):
        avg_forecast_var1.append((avg_forecast_error[i] - avg_forecast_error_mean) ** 2)
    avg_forecast_var = sum(avg_forecast_var1) / len(avg_forecast_var1)
    print('Average method h-step forecast error variance:',avg_forecast_var)

    q_avg = fun_acf(avg_forecast_error, 20)
    plot_acf(q_avg)
    q_avg = q_avg[1:]

    q_avg_total = 0
    for i in q_avg:
        q_avg_total += (i ** 2)

    q_avg_total = q_avg_total * len(avg_forecast_error)
    print('Average method Q value:', q_avg_total)


# Naive Method
def naive_method(train, test, times):
    naive_predict = []
    for i in range(1, len(train)):
        naive_predict.append(train[(i - 1)])

    naive_forecast_1 = train[-1]
    naive_forecast = np.ones(len(test)) * naive_forecast_1
    naive_forecast = naive_forecast.tolist()

    limit = len(train)
    times_train = times[:limit]
    times_test = times[limit:]

    plt.figure()
    plt.plot(times_train, train, label='Training')
    plt.plot(times_test, test, label='Testing')
    plt.plot(times_test, naive_forecast, label='Forecast')
    plt.xlabel('Time')
    plt.xticks(np.arange(0,len(times),round(len(times)/10)), rotation=60)
    plt.ylabel('Amount')
    plt.title('Naive Method')
    plt.legend()
    plt.show()

    train_naive_error = train[1:]
    naive_predict_error = train_naive_error - naive_predict
    naive_predict_error_sq = naive_predict_error ** 2
    naive_predict_MSE = sum(naive_predict_error_sq) / len(naive_predict_error_sq)
    print('Naive method 1-step forecast MSE:', naive_predict_MSE)

    naive_forecast_error = test - naive_forecast
    naive_forecast_error_sq = naive_forecast_error ** 2
    naive_forecast_MSE = sum(naive_forecast_error_sq) / len(naive_forecast_error_sq)
    print('Naive method h-step forecast MSE:', naive_forecast_MSE)

    naive_predict_error_mean = sum(naive_predict_error) / len(naive_predict_error)
    naive_predict_var1 = []
    for i in range(len(naive_predict_error)):
        naive_predict_var1.append((naive_predict_error[i] - naive_predict_error_mean) ** 2)
    naive_predict_var = sum(naive_predict_var1) / len(naive_predict_var1)
    print('Naive method 1-step prediction error variance:', naive_predict_var)

    naive_forecast_error_mean = sum(naive_forecast_error) / len(naive_forecast_error)
    naive_forecast_var1 = []
    for i in range(len(naive_forecast_error)):
        naive_forecast_var1.append((naive_forecast_error[i] - naive_forecast_error_mean) ** 2)
    naive_forecast_var = sum(naive_forecast_var1) / len(naive_forecast_var1)
    print('Naive method h-step forecast error variance:',naive_forecast_var)

    q_naive = fun_acf(naive_forecast_error, 20)
    plot_acf(q_naive)
    q_naive = q_naive[1:]

    q_naive_total = 0
    for i in q_naive:
        q_naive_total += (i ** 2)

    q_naive_total = q_naive_total * len(naive_forecast_error)
    print('Naive Method Q value:', q_naive_total)


# Drift Method
def drift_method(train, test, times):
    drift_predict = []
    drift_forecast = []

    for i in range(1, len(train)-1):
        slope = ((train[i]-train[0])/(i))
        intercept = train[0] - slope
        drift_predict.append(intercept + ((i+2) * slope))

    m = (train[-1] - train[0]) / (len(train)-1)
    for i in range(len(test)):
        drift_forecast.append(train[-1] + ((i+1) * m))

    limit = len(train)
    times_train = times[:limit]
    times_test = times[limit:]

    plt.figure()
    plt.plot(times_train, train, label='Training')
    plt.plot(times_test, test, label='Testing')
    plt.plot(times_test, drift_forecast, label='Forecast')
    plt.xlabel('Time')
    plt.xticks(np.arange(0,len(times),round(len(times)/10)), rotation=60)
    plt.ylabel('Amount')
    plt.title('Drift Method')
    plt.legend()
    plt.show()

    train_drift_error = train[2:]
    drift_predict_error = train_drift_error - drift_predict
    drift_predict_error_sq = drift_predict_error ** 2
    drift_predict_MSE = sum(drift_predict_error_sq) / len(drift_predict_error_sq)
    print('Drift method 1-step predict MSE:', drift_predict_MSE)

    drift_forecast_error = test - drift_forecast
    drift_forecast_error_sq = drift_forecast_error ** 2
    drift_forecast_MSE = sum(drift_forecast_error_sq) / len(drift_forecast_error_sq)
    print('Drift method h-step forecast MSE:', drift_forecast_MSE)

    drift_predict_error_mean = sum(drift_predict_error) / len(drift_predict_error)
    drift_predict_var1 = []

    for i in range(len(drift_predict_error)):
        drift_predict_var1.append((drift_predict_error[i] - drift_predict_error_mean) ** 2)
    drift_predict_var = sum(drift_predict_var1) / len(drift_predict_var1)
    print('Drift method 1-step prediction error variance:', drift_predict_var)

    drift_forecast_error_mean = sum(drift_forecast_error) / len(drift_forecast_error)
    drift_forecast_var1 = []
    for i in range(len(drift_forecast_error)):
        drift_forecast_var1.append((drift_forecast_error[i] - drift_forecast_error_mean) ** 2)
    drift_forecast_var = sum(drift_forecast_var1) / len(drift_forecast_var1)
    print('Drift method h-step forecast error variance:', drift_forecast_var)

    q_drift = fun_acf(drift_forecast_error, 20)
    plot_acf(q_drift)
    q_drift = q_drift[1:]

    q_drift_total = 0
    for i in q_drift:
        q_drift_total += (i ** 2)

    q_drift_total = q_drift_total * len(drift_forecast_error)
    print('Drift method Q Value:', q_drift_total)


# Simple Exponential Smoothing (SES)
def ses_method(train, test, times, alpha):
    ses_predict = []
    ses_forecast = []
    ses_one = train[0]

    for i in range(1, len(train)):
        ses_value = (alpha * train[i - 1]) + ((1 - alpha) * ses_one)
        ses_one = ses_value
        ses_predict.append(ses_value)

    for i in range(len(test)):
        ses_forecast.append(ses_predict[-1])

    limit = len(train)
    times_train = times[:limit]
    times_test = times[limit:]

    plt.figure()
    plt.plot(times_train, train, label='Training Data')
    plt.plot(times_test, test, label='Testing Data')
    plt.plot(times_test, ses_forecast, label='Forecast')
    plt.xlabel('Time')
    plt.xticks(np.arange(0,len(times),round(len(times)/10)), rotation=60)
    plt.ylabel('Amount')
    plt.title('Simple Exponential Smoothing (SES)')
    plt.legend()
    plt.show()

    train_ses_error = train[1:]
    ses_predict_error = train_ses_error - ses_predict
    ses_predict_error_sq = ses_predict_error ** 2
    ses_predict_MSE = sum(ses_predict_error_sq) / len(ses_predict_error_sq)
    print('SES 1-step prediction MSE:', ses_predict_MSE)

    ses_forecast_error = test - ses_forecast
    ses_forecast_error_sq = ses_forecast_error ** 2
    ses_forecast_MSE = sum(ses_forecast_error_sq) / len(ses_forecast_error_sq)
    print('SES h-step forecast MSE:', ses_forecast_MSE)

    ses_predict_error_mean = sum(ses_predict_error) / len(ses_predict_error)
    ses_predict_var1 = []
    for i in range(len(ses_predict_error)):
        ses_predict_var1.append((ses_predict_error[i] - ses_predict_error_mean) ** 2)
    ses_predict_var = sum(ses_predict_var1) / len(ses_predict_var1)
    print('SES 1-step prediction error variance:', ses_predict_var)

    ses_forecast_error_mean = sum(ses_forecast_error) / len(ses_forecast_error)
    ses_forecast_var1 = []
    for i in range(len(ses_forecast_error)):
        ses_forecast_var1.append((ses_forecast_error[i] - ses_forecast_error_mean) ** 2)
    ses_forecast_var = sum(ses_forecast_var1) / len(ses_forecast_var1)
    print('SES h-step forecast error variance:', ses_forecast_var)

    q_ses = fun_acf(ses_forecast_error, 20)
    plot_acf(q_ses)
    q_ses = q_ses[1:]

    q_ses_total = 0
    for i in q_ses:
        q_ses_total += (i ** 2)

    q_ses_total = q_ses_total * len(ses_forecast_error)
    print('Q Value for SES is:', q_ses_total)


# HOLT-WINTER'S
def holt_winters_method(train, test, times, season_num=2, trend_hw='add', seasonal_hw='add'):
    model = ets.ExponentialSmoothing(train, trend=trend_hw, seasonal =seasonal_hw, seasonal_periods=season_num, damped_trend=True)

    fit_forecast = model.fit()
    holtwin_forecast = fit_forecast.forecast(steps=len(test))

    # cross validation
    holtwin_predict = []
    for i in range(season_num,len(train)):
        window = train[:i+season_num]
        model = ets.ExponentialSmoothing(window, trend=trend_hw, seasonal=seasonal_hw,seasonal_periods=season_num)
        fit_i = model.fit()
        holtwin_predict_i = fit_i.forecast(steps=len(train[:i+season_num]))
        holtwin_predict_list = holtwin_predict_i.tolist()
        holtwin_predict.append(holtwin_predict_list[-1])

    limit = len(train)
    times_train = times[:limit]
    times_test = times[limit:]

    plt.figure()
    plt.plot(times_train, train, label = 'Training')
    plt.plot(times_test, test, label = 'Test')
    plt.plot(times_test, holtwin_forecast, label = 'Forecast')
    plt.legend(loc = 0)
    plt. title("Holt-Winters Method")
    plt.ylabel('Amount')
    plt.xlabel('Time')
    plt.xticks(np.arange(0,len(times),round(len(times)/10)), rotation=60)
    plt.show()

    # calc holt win predict error
    train_holtwin_error = train[(len(train)-len(holtwin_predict)):]
    holtwin_predict_error = []
    for i in range(len(train_holtwin_error)):
        predict_error = train_holtwin_error[i] - holtwin_predict[i]
        holtwin_predict_error.append(predict_error)

    holtwin_predict_error_sq = []
    for i in range(len(holtwin_predict_error)):
        holtwin_predict_error_sq.append(holtwin_predict_error[i] ** 2)
    holtwin_predict_MSE = np.sum(holtwin_predict_error_sq) / len(holtwin_predict_error_sq)
    print('Holt-Winters 1-step prediction MSE:', holtwin_predict_MSE)

    holtwin_forecast_error = test - holtwin_forecast

    holtwin_forecast_error_sq = holtwin_forecast_error ** 2

    holtwin_forecast_MSE = sum(holtwin_forecast_error_sq) / len(holtwin_forecast_error_sq)
    print('Holt-Winters h-step forecast MSE:', holtwin_forecast_MSE)

    holtwin_predict_error_mean = sum(holtwin_predict_error) / len(holtwin_predict_error)
    holtwin_predict_variance = []
    for i in range(len(holtwin_predict_error)):
        holtwin_predict_vari = (holtwin_predict_error[i] - holtwin_predict_error_mean) ** 2
        holtwin_predict_variance.append(holtwin_predict_vari)
    holtwin_predict_var = np.sum(holtwin_predict_variance) / len(holtwin_predict_variance)
    print('Holt-Winters 1-step prediction error variance:', holtwin_predict_var)

    holtwin_forecast_error_mean = sum(holtwin_forecast_error) / len(holtwin_forecast_error)
    holtwin_forecast_variance = []
    for i in range(len(holtwin_forecast_error)):
        holtwin_forecast_vari = (holtwin_forecast_error[i] - holtwin_forecast_error_mean) ** 2
        holtwin_forecast_variance.append(holtwin_forecast_vari)
    holtwin_forecast_var = sum(holtwin_forecast_variance) / len(holtwin_forecast_variance)
    print('Holt-Winters h-step forecast error variance:', holtwin_forecast_var)

    q_holtwin = fun_acf(holtwin_forecast_error)
    plot_acf(q_holtwin)
    q_holtwin = q_holtwin[1:]

    q_holtwin_total = 0
    for i in q_holtwin:
        q_holtwin_total += (i ** 2)

    q_holtwin_total = q_holtwin_total * len(holtwin_forecast_error)
    print('Q Value for Holt-Winters is:', q_holtwin_total)

    holtwin_cor = correlation_coefficient_cal(holtwin_forecast_error, test)
    print('Holt-Winters method correlation coefficient between forecast errors and test set:', holtwin_cor)
