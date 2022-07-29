import pandas as pd  
import pandas_datareader as web
from pandas.plotting import autocorrelation_plot
import numpy as np
from numpy import array
import matplotlib.pyplot as plt
import scipy.optimize as sco
from scipy.stats.mstats import winsorize
plt.style.use('fivethirtyeight')
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Normalization 
from keras import regularizers 
from keras.callbacks import ModelCheckpoint
from sklearn.metrics import mean_squared_error
import math
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
np.random.seed(777) #It is used to obtain always the same casual sequence

#Function to report outliers (beyond the percentile) to the border value of the range
def correct_outline(dataset,perc,col_name):
    mask=winsorize(np.array(dataset), limits=perc) 
    dataset=pd.DataFrame(mask,index=dataset.index,columns=[col_name])
    return dataset

#Function to deal with missing data
def correct_nan(dataset):
    for x in dataset:
        if dataset[x].isnull().values.any():
            for i in range(1,len(dataset)):
                if dataset[x].isnull().iloc[i]:
                    dataset[x].iloc[i]=dataset[x].iloc[i-1] 
    dataset.dropna()           
    return dataset

#Here I set the number of stocks which form my portfolio
n_tick=10

#I create a list with the names of the 10 best companies
ticker=['TIT','TRN','HER','DIA','PIA','BRE','IGD','AMP','ZV','FNM']

#I create the matrix with the prices of the actions by reading the values from YAHOO Finance
#I set starting and ending date
data_start='2012-01-01'
data_end='2021-12-31'

for x in ticker:
    if x==ticker[0]:
        tick_val = pd.DataFrame(web.DataReader(x+'.MI', data_source='yahoo', start=data_start, end=data_end), columns=['Close'])
    else:
        tick_val = pd.concat([tick_val, pd.DataFrame(web.DataReader(x+'.MI', data_source='yahoo', start=data_start, end=data_end), columns=['Close'])],axis=1)
        
    tick_val.rename(columns={'Close': x}, inplace=True)

#Correct values which contain non-valid values
tick_val=correct_nan(tick_val)

#Print the trend of the stocks
plt.figure(figsize=(14, 7))
for c in tick_val.columns.values:
    plt.plot(tick_val.index, tick_val[c], lw=3, alpha=0.8,label=c)
plt.legend(loc='upper left', fontsize=12)
plt.ylabel('price')

#I calculate daily returns
tick_ret = tick_val.pct_change()
tick_ret=tick_ret.dropna()

#I print the trend of the returns
plt.figure(figsize=(14, 7))
for c in tick_ret.columns.values:
    plt.plot(tick_ret.index, tick_ret[c], lw=3, alpha=0.8,label=c)
plt.legend(loc='upper right', fontsize=12)
plt.ylabel('daily returns')

### markowitz ###
def portfolio_annualised_performance(weights, mean_returns, cov_matrix):
    returns = np.sum(mean_returns*weights ) *252
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
    return std, returns
  
def random_portfolios(num_portfolios, mean_returns, cov_matrix, risk_free_rate):
    results = np.zeros((3,num_portfolios))
    weights_record = []
    for i in range(num_portfolios):
        weights = np.random.random(n_tick)
        weights /= np.sum(weights)
        weights_record.append(weights)
        portfolio_std_dev, portfolio_return = portfolio_annualised_performance(weights, mean_returns, cov_matrix)
        results[0,i] = portfolio_std_dev
        results[1,i] = portfolio_return
        results[2,i] = (portfolio_return - risk_free_rate) / portfolio_std_dev
    return results, weights_record

def min_variance(mean_returns, cov_matrix):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0,1.0)
    bounds = tuple(bound for asset in range(num_assets))
    result = sco.minimize(portfolio_volatility, num_assets*[1./num_assets,], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)
    return result

def portfolio_volatility(weights, mean_returns, cov_matrix):
    return portfolio_annualised_performance(weights, mean_returns, cov_matrix)[0]

def neg_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
    p_var, p_ret = portfolio_annualised_performance(weights, mean_returns, cov_matrix)
    return -(p_ret - risk_free_rate) / p_var

def max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = (0.0,1.0)
    bounds = tuple(bound for asset in range(num_assets))
    result = sco.minimize(neg_sharpe_ratio, num_assets*[1./num_assets,], args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)
    return result

def efficient_return(mean_returns, cov_matrix, target):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)

    def portfolio_return(weights):
        return portfolio_annualised_performance(weights, mean_returns, cov_matrix)[1]

    constraints = ({'type': 'eq', 'fun': lambda x: portfolio_return(x) - target},
                   {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0,1) for asset in range(num_assets))
    result = sco.minimize(portfolio_volatility, num_assets*[1./num_assets,], args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return result

def efficient_frontier(mean_returns, cov_matrix, returns_range):
    efficients = []
    for ret in returns_range:
        efficients.append(efficient_return(mean_returns, cov_matrix, ret))
    return efficients

def display_simulated_ef_with_random(mean_returns, cov_matrix, num_portfolios, risk_free_rate):
    results, weights = random_portfolios(num_portfolios,mean_returns, cov_matrix, risk_free_rate)
    
    max_sharpe_idx = np.argmax(results[2])
    sdp, rp = results[0,max_sharpe_idx], results[1,max_sharpe_idx]
    max_sharpe_allocation = pd.DataFrame(weights[max_sharpe_idx],index=tick_val.columns,columns=['allocation'])
    max_sharpe_allocation.allocation = [round(i*100,2)for i in max_sharpe_allocation.allocation]
    max_sharpe_allocation = max_sharpe_allocation.T
    
    min_vol_idx = np.argmin(results[0])
    sdp_min, rp_min = results[0,min_vol_idx], results[1,min_vol_idx]
    min_vol_allocation = pd.DataFrame(weights[min_vol_idx],index=tick_val.columns,columns=['allocation'])
    min_vol_allocation.allocation = [round(i*100,2)for i in min_vol_allocation.allocation]
    min_vol_allocation = min_vol_allocation.T
    
    print("-"*80)
    print("Maximum Sharpe Ratio Portfolio Allocation\n")
    print("Annualised Return:", round(rp,2))
    print("Annualised Volatility:", round(sdp,2))
    print("\n")
    print(max_sharpe_allocation)
    print("-"*80)
    print("Minimum Volatility Portfolio Allocation\n")
    print("Annualised Return:", round(rp_min,2))
    print("Annualised Volatility:", round(sdp_min,2))
    print("\n")
    print(min_vol_allocation)
    
    plt.figure(figsize=(10, 7))
    plt.scatter(results[0,:],results[1,:],c=results[2,:],cmap='YlGnBu', marker='o', s=10, alpha=0.3)
    plt.colorbar()
    plt.scatter(sdp,rp,marker='*',color='r',s=500, label='Maximum Sharpe ratio')
    plt.scatter(sdp_min,rp_min,marker='*',color='g',s=500, label='Minimum volatility')
    plt.title('Simulated Portfolio Optimization based on Efficient Frontier')
    plt.xlabel('annualised volatility')
    plt.ylabel('annualised returns')
    plt.legend(labelspacing=0.8)

#I calculate the expected value of the return of the actions
mean_returns = tick_ret.mean()
#I compute the variance and covariance matrix
cov_matrix = tick_ret.cov()
#I set the number of portfolios to be generated in a random way
num_portfolios = 25000
#I set the value of the return of the investment considered as safe (BTP 10Y)
risk_free_rate = 0.01
#I recall the function for the generation of the frontier of the portfolios
display_simulated_ef_with_random(mean_returns, cov_matrix, num_portfolios, risk_free_rate)

#I obtain the value of the portfolio for the past years
#I set the relative weights to Maximum Sharpe Ratio Portfolio Allocation
wgh=[[0.62],[2.74],[5.18],[18.84],[0.6],[13.15],[0.57],[19.53],[20.48],[18.29]]
#I decide the value of my portfolio at the beginning
val_port_ini=1000000
#I evaluate the weights of the single stocks
bdg_x_tit=np.array((val_port_ini/100)*np.array(wgh))

#I consider the number of actions, for each stock in my starting portfolio
n_tit=[[0],[0],[0],[0],[0],[0],[0],[0],[0],[0]]
n_tit=np.array(n_tit)
i=0
for t in ticker:
    n_tit[i,0]=bdg_x_tit[i,0]/np.array(tick_val[t].iloc[:1].values)
    i+=1

#I create the historical series for each value in my portfolio
port_val=np.dot(np.array(tick_val),n_tit)
port_val=pd.DataFrame(port_val,index=tick_val.index)
port_val.rename(columns={0:'Portfolio'}, inplace=True)




### Econometric model ###

#I compute the first order of differencing
port_diff_1 = port_val.diff()
port_diff_1=port_diff_1.dropna()

#I calculate the second order of differencing
port_diff_2 = port_diff_1.diff()
port_diff_2=port_diff_2.dropna()

#I carry out Fuller test on my original series
ad_test = adfuller(port_val.values)
print('Portfolio senza differenze\n')
print('ADF Statistic: %f' % ad_test[0])
print('p-value: %f' % ad_test[1])
if ad_test[1] <= 0.05:
    print("Strong evidence against the null hypothesis")
    print("Reject the null hypothesis")
    print("Data has no unit root and is stationary")
else:
    print("Weak evidence against the null hypothesis")
    print("Fail to reject the null hypothesis")
    print("Data has a unit root and is non-stationary\n")
print('\n')

#I carry out Fuller test on the first differencing
ad_test = adfuller(port_val.diff().dropna().values)
print('Portfolio 1st Order Differencing\n')
print('ADF Statistic: %f' % ad_test[0])
print('p-value: %f' % ad_test[1])
if ad_test[1] <= 0.05:
    print("Strong evidence against the null hypothesis")
    print("Reject the null hypothesis")
    print("Data has no unit root and is stationary")
else:
    print("Weak evidence against the null hypothesis")
    print("Fail to reject the null hypothesis")
    print("Data has a unit root and is non-stationary\n")
print('\n')

#Autocorrelation graph original series
plot_acf(port_val.values) 

#Autocorrelation graph 1st order Differencing
plot_acf(port_val.diff().dropna().values, title='Autorrelation 1st Order Differencing') 

#Autocorrelation graph 2nd order Differencing
plot_acf(port_val.diff().diff().dropna().values, title='Autorrelation 2nd Order Differencing')

#Partial autocorrelation graph original series
plot_pacf(port_val.values) 

#Partial autocorrelation graph 1st order Differencing
plot_pacf(port_val.diff().dropna().values, title='Partial autorrelation 1st Order Differencing') 

#Partial autocorrelation graph 2nd order Differencing
plot_pacf(port_val.diff().diff().dropna().values, title='Partial autorrelation 2nd Order Differencing')

plt.show()

#I set the hyperparameters of the econometric model as suggested from the analysis of the graphs
p=5
d=1
q=1

#Graph of the autocorrelation of the series related to the first order of differencing
autocorrelation_plot(port_val.diff().dropna())
plt.show()

#I call ARIMA model and I pass to it the entire historical series and the parameters previously calculated
model = ARIMA(port_val, order=(p,d,q))
model_fit = model.fit()

#Fit model
print(model_fit.summary())

#Line plot of residuals
residuals = pd.DataFrame(model_fit.resid)
residuals.plot()
plt.show()

#Density plot of residuals
residuals.plot(kind='kde')
plt.show()

#Summary stats of residuals
print(residuals.describe())

#Test predictions

#Split into train and test sets
X = port_val.values
size = int(len(X) * 0.80)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()

#Walk-forward validation
for t in range(len(test)):
	model = ARIMA(history, order=(p,d,q))
	model_fit = model.fit()
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test[t]
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))

#Evaluate forecasts
rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)

#Plot forecasts against actual outcomes
plt.plot(test)
plt.plot(predictions, color='red')
plt.show()

#Prevision (same procedure as before)
d_prev=30 #number of days to be predicted
X = port_val.values
train = X[:]
history = [x for x in train]
predictions = list()
for t in range(d_prev):
    model = ARIMA(history, order=(p,d,q))
    model_fit = model.fit()
    output = model_fit.forecast()
    yhat = output[0]
    supp=[]
    supp.insert(0,output[0])
    predictions.append(yhat)
    history.append(supp) #This time I set y predicted instead of test value

#I print the predicted values which follow the last one hundred values detected
print('predicted=%f' % yhat)
plt.plot(train[-100:])
plt.plot(pd.DataFrame(predictions,index=range(100,100+d_prev)), color='red')
plt.show()



### Neural network ###

#Function to transform features by scaling each one of them to a given range (0,1 in this case)
scaler=MinMaxScaler(feature_range=(0,1))

#Fit to data, then transform it
df=scaler.fit_transform(np.array(port_val).reshape(-1,1))
dates = df[:, 0]

##Splitting dataset into train and test sets
training_size=int(len(df)*0.80)
test_size=len(df)-training_size
train_data,test_data=df[0:training_size,:],df[training_size:len(df),:1]

#Function to convert an array of values into a dataset matrix
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

#Reshape dataset into X=t,t+1,t+2,t+3 and Y=t+4
time_step = 4
X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)

#Reshape input to be [samples, time steps, features] which is required for LSTM keras
X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1) 
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)  

#Create the Stacked LSTM model
normalization_layer = Normalization()
normalization_layer.adapt(X_train)

model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(time_step,1)))
#model.add(normalization_layer)
model.add(Dropout(0.5))
model.add(LSTM(50,return_sequences=True))
#model.add(normalization_layer)
model.add(Dropout(0.2))
model.add(LSTM(50))
#model.add(normalization_layer)
model.add(Dropout(0.2))
model.add(Dense(100,activation="relu", activity_regularizer=regularizers.L2(1e-5)))
#model.add(Dropout(0.2)) usually Dropout in most of the literatures is not added before the last hidden layer before the final output
model.add(Dense(1))

#Save the weights of the best model
checkpoint=ModelCheckpoint("next_words.h5", monitor='loss',verbose=1,save_best_only=True)

#Compile the model
model.compile(loss='mean_squared_error',optimizer='adam')

#Fit the model
model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=10,batch_size=16,verbose=1,callbacks=[checkpoint])

#Load the best model previously saved
filename="next_words.h5"
model.load_weights(filename)

#Train predictions
train_predict=model.predict(X_train)

#Test predictions
test_predict=model.predict(X_test)

#Transformback to original form
train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)

#Calculate RMSE performance metrics
math.sqrt(mean_squared_error(y_train,train_predict))

#Test Data RMSE
math.sqrt(mean_squared_error(ytest,test_predict))



### Plotting 
# shift train predictions for plotting
look_back=4
trainPredictPlot = np.empty_like(df)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
# shift test predictions for plotting
testPredictPlot = np.empty_like(df)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(df)-1, :] = test_predict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(df))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()


#Prepare data for forecasting
x_input=test_data[(len(test_data)-time_step):].reshape(1,-1)
temp_input=list(x_input)
temp_input=temp_input[0].tolist()

#Forecasting for next 30 days
lst_output=[]
n_steps=time_step
i=0
while(i<30):    
  if(len(temp_input)>time_step):
       #print(temp_input)
       x_input=np.array(temp_input[1:])
       print("{} day input {}".format(i,x_input))
       x_input=x_input.reshape(1,-1)
       x_input = x_input.reshape((1, n_steps, 1))
       #print(x_input)
       yhat = model.predict(x_input, verbose=0)
       print("{} day output {}".format(i,yhat))
       temp_input.extend(yhat[0].tolist())
       temp_input=temp_input[1:]
       #print(temp_input)
       lst_output.extend(yhat.tolist())
       i=i+1
  else:
       x_input = x_input.reshape((1, n_steps,1))
       yhat = model.predict(x_input, verbose=0)
       print(yhat[0])
       temp_input.extend(yhat[0].tolist())
       print(len(temp_input))
       lst_output.extend(yhat.tolist())
       i=i+1
    
#Plot of forecasting
m_fc=scaler.inverse_transform(lst_output)
p_df=scaler.inverse_transform(df[2437:])
fc=pd.DataFrame(m_fc,index=pd.date_range("2022-01-01", periods=30,freq="D"),columns=['Forecast'])
pf=pd.DataFrame(p_df,index=port_val[2437:].index,columns=['Original'])
fig, ax = plt.subplots(figsize=(12, 8))
plt.title('Forecasting')
plt.plot(fc['Forecast'])
plt.plot(pf['Original'])
plt.show()