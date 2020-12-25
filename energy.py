# Appliance Energy Usage
#%%
# import libraries
import statistics
import random
import math
import datetime as dt
import statsmodels.api as sm
import numpy as np
import pandas as pd
from pandas.plotting import lag_plot
from pandas import concat
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import itertools
import seaborn as sns
from pandas.plotting import register_matplotlib_converters
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import STL
import statsmodels.tsa.holtwinters as ets
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima_process import ArmaProcess
from scipy import signal
from scipy import stats, linalg
from lifelines import KaplanMeierFitter
from functions import *
import warnings
warnings.filterwarnings('ignore')

#%%
# 1- Load the energy
df = pd.read_csv('C:/Users/sjg27/OneDrive/Documents/GWU Data Science/Fall 20/Time Series/project/data/energydata_complete_halfhourly.csv', header=0)
df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['half_hour'])
df.info()
df.head()

# plot dependent variable 'Sum of Appliances'
plt.figure(figsize=(10, 6))
plt.plot(df['datetime'], df['Sum of Appliances'], label = 'Appliances')
plt.legend(loc = 0)
plt.xlabel('Date')
plt.ylabel('Appliances (Wh)')
plt.title('Energy Usage of Appliances in Wh')
plt.xticks(rotation=75)
# plt.xticks(np.arange(0, len(df['datetime']), 10), rotation=75)
plt.show()

# ADF - stationary data check
appliances = df['Sum of Appliances']
appliances_list = appliances.tolist()
ADF_Cal(appliances_list)
# appliances is a stationary variable

# ACF analysis
appliances_acf = fun_acf(appliances_list,50)
plot_acf(appliances_acf,50)

#%%
# correlation matrix
sns.heatmap(df.corr());

#%%
# decomposition

# 3-MA plot (moving average window size 3)
window = 3 
ma_3 = cal_MA_wFO(appliances, window)
k_3=int((window-1)/2) 
detrended_ma_3 = np.array(ma_3) - np.array(appliances[k_3:len(appliances)-k_3])
datetime = df['datetime']

plt.figure(figsize=(10, 6))
plt.plot(datetime, appliances, label='Original', alpha=.7)
plt.plot(datetime[k_3:len(appliances)-k_3], ma_3, label='MA Trend')
plt.plot(datetime[k_3:len(appliances)-k_3], ma_3, label='Detrended', alpha=.5)
plt.xticks(rotation=60)
plt.xlabel('Time Period')
plt.ylabel('Appliance Usage Wh')
plt.title('Appliance Usage: Original & MA=3')
plt.legend()
plt.show()

# 2x24-MA plot (moving average window size 24 and folding order 2)
window = 24 
foldingorder = 2
ma_2x24 = cal_MA_wFO(appliances, window, foldingorder)
k_2x24=int(((window-2)/2)+(foldingorder/2)) 
detrended_ma_2x24 = np.array(ma_2x24) - np.array(appliances[k_2x24:len(appliances)-k_2x24])
datetime = df['datetime']

plt.figure(figsize=(10, 6))
plt.plot(datetime, appliances, label='Original', alpha=.7)
plt.plot(datetime[k_2x24:len(appliances)-k_2x24], ma_2x24, label='MA Trend')
plt.plot(datetime[k_2x24:len(appliances)-k_2x24], ma_2x24, label='Detrended', alpha=.5)
plt.xticks(rotation=60)
plt.xlabel('Time Period')
plt.ylabel('Appliance Usage Wh')
plt.title('Appliance Usage: Original & MA=24 FO=2')
plt.legend()
plt.show()
#%%
# STL decomposition
df_STL = pd.Series(appliances, name = 'Appliances Usage (Wh)')

res = STL(df_STL, period=48).fit()
res.plot()
plt.xlabel('Time Interval')
plt.show()

# seasonally adjusted data vs original data
plt.figure(figsize=(10, 6))
plt.plot(datetime, appliances, label='Original')
plt.plot(datetime, res.seasonal, label='Seasonally Adjusted')
plt.xticks(rotation=60)
plt.xlabel('Time Period')
plt.ylabel('Amount')
plt.title('Original & Seasonaly Adjusted Data')
plt.legend()
plt.show()

# trend strength calc (0 to 1 scale - 1 highly trended)
var_r = res.resid.var()
var_tr = (res.trend + res.resid).var()
strength_trend = max(0, 1-(var_r / var_tr))
print('The strength of trend is: ', strength_trend)

# seasonal strength calc (0 to 1 scale - 1 highly seasonal)
var_sr = (res.seasonal + res.resid).var()
strength_seasonality = max(0, 1-(var_r / var_sr))
print('The strength of seasonality is: ', strength_seasonality)

#%%
# Multiple Linear Regression
# feature Selection - define variables from dataframe
df_var_name = np.asarray(df.columns)
df_var_name = df_var_name[2:-1]
df_vars = df[df_var_name]

# train/test split
train, test = train_test_split(df_vars, train_size=0.8, shuffle=False)

df_preds = df_var_name[1:]
y_train = train[['Sum of Appliances']] 
y_test = test[['Sum of Appliances']]
X_train = train[df_preds]
X_test = test[df_preds]

# too many variables to run the stepwise function - must eliminate some variables
X_train_all = train[df_preds]
X_train_all = sm.add_constant(X_train)
OLS_all = sm.OLS(y_train, X_train_all).fit()
print(OLS_all.summary())

# after reviewing summary - removing variables with a higher p value than .2
df_preds_new = ['Average of RH_1', 'Average of T2', 'Average of RH_2', 'Average of T3', 'Average of RH_3', 'Average of T4', 'Average of T6', 'Average of RH_6', 'Average of T8', 'Average of RH_8', 'Average of T9', 'Average of RH_9', 'Average of T_out', 'Average of Press_mm_hg', 'Average of Windspeed', 'Average of Visibility']

X_train = train[df_preds_new]
X_test = test[df_preds_new]

#%%
# DON'T RERUN IN TESTING (COMMENT OUT)
# stepwise regression - forward step - takes a long time to run with many variables! - about 20 minutes with 16 variables
stepwise_reg = stepwise_fun_base(X_train, y_train)
stepwise_reg_top = stepwise_reg.sort_values('Adj. R-squared',ascending = False)
stepwise_reg_top.head()
stepwise_reg_top_vars = stepwise_reg_top['Variables'].iloc[0][0].tolist()
print('The best model includes these variables:', stepwise_reg_top_vars)
#%%
# t-test and F-test on the best final model
# t-test: stat for each coefficient - look at p value - want small for each (below .05 confidence threshold)
# F-test for entire model - look at p value (Prob (F-statistic)) - want small (below .05 confidence threshold)

# stepwise_reg_top_vars = ['Average of RH_1', 'Average of T2', 'Average of RH_2', 'Average of T3', 'Average of RH_3', 'Average of T4', 'Average of T6', 'Average of RH_6', 'Average of T8', 'Average of RH_8', 'Average of T9', 'Average of RH_9', 'Average of T_out', 'Average of Windspeed', 'Average of Visibility'] # FOR TESTING
X_train_best = train[stepwise_reg_top_vars]
X_test_best = test[stepwise_reg_top_vars]
OLS_model_best = sm.OLS(y_train, X_train_best).fit()
OLS_model_predict_best = OLS_model_best.predict(X_train_best)
OLS_model_details_best = OLS_model_best.summary()
print(OLS_model_details_best)
print('\n')
print('The F-statistic for the best model is:', OLS_model_best.fvalue.round(4))
print('The p-value of the F-statistic for the best model is:', OLS_model_best.f_pvalue)

# %%
# plot of one-step predictions and h-step forecasts
OLS_model_forecast_best = OLS_model_best.predict(X_test_best)

length = []
for i in range(len(df_vars)):
    length.append(i+1)

limit = len(X_train_best)
length_train = length[:limit]
length_test = length[limit:]

# plot of the train, test and predicted values in one graph
plt.figure(figsize=(10, 6))
plt.plot(length_train, y_train, label='training data')
plt.plot(length_train, OLS_model_predict_best, label='prediction')
plt.plot(length_test, y_test, label='testing data')
plt.plot(length_test, OLS_model_forecast_best, label='forecast')
plt.axvline(limit, linestyle='dashed', color='black', alpha=.3, label='train/test split')
plt.xlabel('sample')
plt.xticks(np.arange(0,len(length),round(len(length)/10)), rotation=60)
plt.ylabel('Wh')
plt.title('Appliance Usage (Wh) Predictions & Forecasts using OLS')
plt.legend()
plt.show()

#%%
# MLE residuals
app_train, app_test = train_test_split(appliances, train_size=0.8, shuffle=False)
mle_predict_error = []
for i in range(0, len(app_train)):
    mle_predict_error.append(app_train[i] - OLS_model_predict_best[i])

index_train = len(app_train)
mle_forecast_error = []
for i in range(0, len(app_test)):
    mle_forecast_error.append(app_test[i+index_train] - OLS_model_forecast_best[i+index_train])

# plot of residuals
plt.figure(figsize=(10, 6))
plt.plot(length_train, mle_predict_error, label='predict error')
plt.plot(length_test, mle_forecast_error, label='forecast error')
plt.axvline(limit, linestyle='dashed', color='black', alpha=.3, label='train/test split')
plt.xlabel('sample')
plt.xticks(np.arange(0,len(length),round(len(length)/10)), rotation=60)
plt.ylabel('residual')
plt.legend()
plt.title('MLE Best Model Prediction and Forecast Residuals')
plt.show()

# ACF of predict error
#mle_pred_acf = fun_acf(mle_predict_error,50)
#plot_acf(mle_pred_acf,50)

# ACF of forecast errir
mle_fcst_acf = fun_acf(mle_forecast_error,20)
plot_acf(mle_fcst_acf,20)

# var, std dev of predict error & forecast error
mle_predict_error_mean = sum(mle_predict_error) / len(mle_predict_error)
mle_predict_var1 = []
for i in range(len(mle_predict_error)):
    mle_predict_var1.append((mle_predict_error[i] - mle_predict_error_mean) ** 2)
mle_predict_var = sum(mle_predict_var1) / len(mle_predict_var1)
print('Best MLE 1-step prediction error variance:',mle_predict_var)
mle_predict_std = math.sqrt(mle_predict_var)
print('Best MLE 1-step prediction error standard deviation:', mle_predict_std)

mle_forecast_error_mean = sum(mle_forecast_error) / len(mle_forecast_error)
mle_forecast_var1 = []
for i in range(len(mle_forecast_error)):
    mle_forecast_var1.append((mle_forecast_error[i] - mle_forecast_error_mean) ** 2)
mle_forecast_var = sum(mle_forecast_var1) / len(mle_forecast_var1)
print('Best MLE h-step forecast error variance:',mle_forecast_var)
mle_forecast_std = math.sqrt(mle_forecast_var)
print('Best MLE h-step forecast error standard deviation:', mle_forecast_std)
#%%
q_mle = fun_acf(mle_forecast_error, 20)
q_mle = q_mle[1:]

q_mle_total = 0
for i in q_mle:
    q_mle_total += (i ** 2)

q_mle_total = q_mle_total * len(mle_forecast_error)
print('MLE Q value:', q_mle_total)
print('MLE h-step MSE: ', mean_squared_error(y_test,OLS_model_forecast_best))

# %%
# GPAC
target = appliances
acf_list = list_acf(target)
Cal_GPAC(acf_list, 8, 8)

# %%
# ARMA - statsmodels
from statsmodels.tsa.arima_model import ARIMA as ARMA
from statsmodels.graphics.api import qqplot

# ARMA (1,3)
app_train, app_test = train_test_split(appliances, train_size=0.8, shuffle=False)
arma_mod = ARMA(app_train, order=(1, 0, 3)).fit()
print(arma_mod.params)

resid = arma_mod.resid
stats.normaltest(resid)

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111)
fig = qqplot(resid, line='q', ax=ax, fit=True)

print(arma_mod.summary())

# plot residual errors
residuals = pd.DataFrame(arma_mod.resid)
residuals.plot(title='Residuals', legend=False)
plt.legend()
plt.show()
residuals.plot(kind='kde', title='Density Plot of Residuals')
plt.show()
print(residuals.describe())

print('AR roots: ', arma_mod.arroots)
print('MA roots: ', arma_mod.maroots)

# generate predicted and forecasted values for arma_mod
date_periods = []
for i in range(0,len(datetime)):
    date_periods.append(i+1)

limit = len(app_train)
length_train = date_periods[:limit]
length_test = date_periods[limit:]

arma_mod_predict = arma_mod.fittedvalues
arma_mod_forecast = arma_mod.predict(limit, 6579, dynamic=True)

# plot of the train, test and predicted values in one graph
plt.figure(figsize=(10, 6))
plt.plot(length_train, app_train, label='training data')
plt.plot(length_train, arma_mod_predict, label='prediction')
plt.plot(length_test, app_test, label='testing data')
plt.plot(length_test, arma_mod_forecast[:-1], label='forecast') ##
plt.axvline(limit, linestyle='dashed', color='black', alpha=.3, label='train/test split')
plt.xlabel('sample')
# plt.xticks(np.arange(0,len(length),round(len(length)/10)), rotation=60)
plt.ylabel('Appliance Usage (Wh)')
plt.title('ARMA(1,3) Appliance Usage (Wh) Predictions & Forecasts')
plt.legend()
plt.show()

#%%
# residual acf

# one-step
one_step_acf = fun_acf(residuals[0].tolist())
plot_acf(one_step_acf)

q_one_step_total = 0
for i in one_step_acf:
    q_one_step_total += (i ** 2)

q_one_step_total = q_one_step_total * len(one_step_acf)
print('One-step Q value:', q_one_step_total)

# h-step
h_step_error = []
for i in range(0,len(app_test)):
    ind = i + 5263
    diff = app_test[ind] - arma_mod_forecast[ind]
    h_step_error.append(diff)

h_step_acf = fun_acf(h_step_error)
plot_acf(h_step_acf)

q_h_step_total = 0
for i in h_step_acf:
    q_h_step_total += (i ** 2)

q_h_step_total = q_h_step_total * len(h_step_acf)
print('H-step Q value:', q_h_step_total)

#%%
print('Variance of prediction error: ', statistics.variance(residuals[0].tolist()))
print('Variance of forecast error: ', statistics.variance(h_step_error))
print('ARMA(1,3) h-step MSE: ', mean_squared_error(y_test,arma_mod_forecast[:-1]))
#%%
# covariance
covar = pd.DataFrame(arma_mod.cov_params())
covar

#%%
# ARMA (0,3)
arma_mod2 = ARMA(app_train, order=(0, 0, 3)).fit()
print(arma_mod2.params)

resid2 = arma_mod2.resid
stats.normaltest(resid2)

fig = plt.figure(figsize=(10,8))
ax = fig.add_subplot(111)
fig = qqplot(resid2, line='q', ax=ax, fit=True)

print(arma_mod2.summary())
# plot residual errors
residuals = pd.DataFrame(arma_mod2.resid)
residuals.plot(title='Residuals')
plt.show()
residuals.plot(kind='kde', title='Density Plot of Residuals')
plt.show()
print(residuals.describe())

print('AR roots: ', arma_mod2.arroots)
print('MA roots: ', arma_mod2.maroots)

arma_mod2_predict = arma_mod2.fittedvalues
arma_mod2_forecast = arma_mod2.predict(limit, 6579, dynamic=True)

# plot of the train, test and predicted values in one graph
plt.figure(figsize=(10, 6))
plt.plot(length_train, app_train, label='training data')
plt.plot(length_train, arma_mod2_predict, label='prediction')
plt.plot(length_test, app_test, label='testing data')
plt.plot(length_test, arma_mod2_forecast[:-1], label='forecast') ##
plt.axvline(limit, linestyle='dashed', color='black', alpha=.3, label='train/test split')
plt.xlabel('sample')
# plt.xticks(np.arange(0,len(length),round(len(length)/10)), rotation=60)
plt.ylabel('Appliance Usage (Wh)')
plt.title('ARMA(0,3) Appliance Usage (Wh) Predictions & Forecasts')
plt.legend()
plt.show()

# residual acf

# one-step
one_step_acf = fun_acf(residuals[0].tolist())
plot_acf(one_step_acf)

q_one_step_total = 0
for i in one_step_acf:
    q_one_step_total += (i ** 2)

q_one_step_total = q_one_step_total * len(one_step_acf)
print('One-step Q value:', q_one_step_total)

# h-step
h_step_error = []
for i in range(0,len(app_test)):
    ind = i + 5263
    diff = app_test[ind] - arma_mod_forecast[ind]
    h_step_error.append(diff)

h_step_acf = fun_acf(h_step_error)
plot_acf(h_step_acf)

q_h_step_total = 0
for i in h_step_acf:
    q_h_step_total += (i ** 2)

q_h_step_total = q_h_step_total * len(h_step_acf)
print('H-step Q value:', q_h_step_total)

#%%
print('Variance of prediction error: ', statistics.variance(residuals[0].tolist()))
print('Variance of forecast error: ', statistics.variance(h_step_error))
print('ARMA(0,3) h-step MSE: ', mean_squared_error(y_test,arma_mod2_forecast[:-1]))
#%%
# covariance
covar2 = pd.DataFrame(arma_mod2.cov_params())
covar2

#%%
# Base models
app_train_arr = np.asarray(app_train)
app_test_arr = np.asarray(app_test)
date_periods_arr = np.asarray(date_periods)

#%%
# Average Method
print("Average Method:")
average_method(app_train_arr, app_test_arr, date_periods_arr)

# %%
# Naive Method
print("Naive Method:")
naive_method(app_train_arr, app_test_arr, date_periods_arr)

# %%
# Drift Method
print("Drift Method:")
drift_method(app_train_arr, app_test_arr, date_periods_arr)

# %%
# Simple Exponential Smoothing (SES) Method
print("Simple Exponential Smoothing (SES) with alpha=0:")
ses_method(app_train_arr, app_test_arr, date_periods_arr, 0)
# %%
print("Simple Exponential Smoothing (SES) with alpha=0.5:")
ses_method(app_train_arr, app_test_arr, date_periods_arr, 0.5)
# %%
print("Simple Exponential Smoothing (SES) with alpha=0.99:")
ses_method(app_train_arr, app_test_arr, date_periods_arr, 0.99)

# %%
# Holt-Winters
# use additive method because seasonal variations are roughly constant throughout the data series
print("Holt-Winters:")
holt_winters_method(app_train_arr, app_test_arr, date_periods_arr, season_num=48)
# %%
print("Holt-Winters:") 
holt_winters_method(app_train_arr, app_test_arr, date_periods_arr, season_num=12)
# %%
print("Holt-Winters:")
holt_winters_method(app_train_arr, app_test_arr, date_periods_arr, season_num=24)

# %%
# best model plot - MLE forecast vs test
plt.figure(figsize=(10, 6))
plt.plot(length_test, y_test, label='testing data')
plt.plot(length_test, OLS_model_forecast_best, label='forecast')
plt.xlabel('sample')
plt.ylabel('Wh')
plt.title('Appliance Usage (Wh) Forecasts: Best OLS Model')
plt.legend()
plt.show()
# %%
