#!/usr/bin/env python
# coding: utf-8

# In[18]:


# Importing all the required libraries

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import style
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
from pandas_datareader import data as pdr
import yfinance as yf
import seaborn as sns
from pandas_datareader import data
yf.pdr_override()
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# List of the tech stocks 
#AAPL: Apple
#GOOG: Google
#MSFT: Microsoft
#TSLA: Tesla

stock_list= ['AAPL','GOOG','MSFT','TSLA']


# In[3]:


# Time Period
end= datetime.now()
start= datetime(2012,5,19)


# In[4]:


#for loop for grabbing finance data and setting it as a dataframe. 
#globals()- taking the string 'stock' and makes it a global variable(Setting the string name as data variable)
for stock in stock_list:
    globals()[stock] = pdr.get_data_yahoo(stock,start,end)


# In[5]:


TSLA.describe()


# In[6]:


TSLA.info()


# In[7]:


TSLA['Close'].plot(legend=True,figsize=(10,4))


# In[8]:


# Visualizing the stock prices
AAPL['Adj Close'].plot(legend=True,label='AAPL', figsize=(15, 9), 
    title='Adjusted Closing Price', color='red', linewidth=1.0, grid=True)


# In[9]:


#calculating moving average for stock
ma_day = [10,20,60]

for ma in ma_day:
    column_name= 'MA for %s days'%(str(ma))
    TSLA[column_name]= TSLA['Adj Close'].rolling(ma).mean()
    AAPL[column_name]= AAPL['Adj Close'].rolling(ma).mean()
    GOOG[column_name]= GOOG['Adj Close'].rolling(ma).mean()
    MSFT[column_name]= MSFT['Adj Close'].rolling(ma).mean()


# In[10]:


#plotting all additional moving averages
TSLA[['Adj Close','MA for 10 days','MA for 20 days','MA for 60 days']].plot(subplots=False,figsize=(20,10))


# In[11]:



AAPL[['Adj Close','MA for 10 days','MA for 20 days','MA for 60 days']].plot(subplots=False,figsize=(20,10))


# In[12]:


GOOG[['Adj Close','MA for 10 days','MA for 20 days','MA for 60 days']].plot(subplots=False,figsize=(20,10))


# In[13]:


MSFT[['Adj Close','MA for 10 days','MA for 20 days','MA for 60 days']].plot(subplots=False,figsize=(20,10))


# In[14]:


#daily returns and risk of stocks. Daily return column= percent change in the adjusted price column
TSLA['Daily Return'] = TSLA['Adj Close'].pct_change()
TSLA['Daily Return'].plot(figsize=(20,10),legend=True,linestyle='dotted',marker='o')


# In[15]:


#plotting a histogram for daily returns of apple for past year
sns.distplot(TSLA['Close'].dropna(),bins=100,color='red')


# In[16]:


TSLA['Close'].hist(bins=100,figsize=(10,4),color='red')


# In[19]:


#analyzing the returns of all the stocks on our list, specifying only adj close price
closing_df= data.DataReader(stock_list,start,end)['Adj Close']
closing_df.head()


# In[20]:


corr = closing_df.corr()
sns.heatmap(corr,annot=True)


# In[21]:


#Clustermap the correlations
#Use seaborn's clustermap to cluster the correlations together:
sns.clustermap(corr, cmap="coolwarm", annot=True, figsize= (8,5))


# In[22]:


#Tsla Stock prdicting.  
TSLA = TSLA.dropna(axis = 0, how ='any', thresh = None, subset = None, inplace=False)


# In[23]:


# Number of days for which to predict the stock prices
predict_days = 30


# In[24]:


# Shifting by the Number of Predict days for Prediction array

TSLA['Prediction'] = TSLA['Adj Close'].shift(-predict_days)
# print(df['Prediction'])
# print(df['Adj Close'])


# In[26]:


# Dropping the Prediction Row

X = np.array(TSLA.drop(['Prediction'], axis = 1))
X = X[:-predict_days]      # Size upto predict days
# print(X)
print(X.shape)


# In[27]:


# Creating the Prediction Row

y = np.array(TSLA['Prediction'])
y = y[:-predict_days]      # Size upto predict_days
# print(y)
print(y.shape)


# In[28]:


# Splitting the data into Training data & Testing data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)      #Splitting the data into 70% for training & 20% for testing
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[29]:


# Defining the Linear Regression Model

linear_model = LinearRegression()
linear_model.fit(X_train, y_train)      # Training the algorithm


# In[30]:


# Score of the Linear Regression Model (Using the Test Data)

linear_model_score = linear_model.score(X_test, y_test)
print('Linear Model score:', linear_model_score)


# In[31]:


# Define the Real & Prediction Values

X_predict = np.array(TSLA.drop(['Prediction'], 1))[-predict_days:]

linear_model_predict_prediction = linear_model.predict(X_predict)
linear_model_real_prediction = linear_model.predict(np.array(TSLA.drop(['Prediction'], 1)))


# In[32]:


# Defining some Parameters

predicted_dates = []
recent_date = TSLA.index.max()
display_at = 1000
alpha = 0.5

for i in range(predict_days):
    recent_date += (timedelta(days=1))
    predicted_dates.append(recent_date)


# In[34]:


# Plotting the Actual and Prediction Prices

plt.figure(figsize=(15, 9))
plt.plot(TSLA.index[display_at:], linear_model_real_prediction[display_at:], 
         label='Linear Prediction', color='blue', alpha=alpha)
plt.plot(predicted_dates, linear_model_predict_prediction, 
         label='Forecast', color='green', alpha=alpha)
plt.plot(TSLA.index[display_at:], TSLA['Close'][display_at:], 
         label='Actual', color='red')
plt.legend()


# In[35]:


# Defining the Ridge Regression Model

ridge_model = Ridge()
ridge_model.fit(X_train, y_train)     # Training the algorithm


# In[36]:


# Score of the Ridge Regression Model (Using the Test Data)

ridge_model_score = ridge_model.score(X_test, y_test)
print('Ridge Model score:', ridge_model_score)


# In[37]:


# Define the Real & Prediction Values

ridge_model_predict_prediction = ridge_model.predict(X_predict)
ridge_model_real_prediction = ridge_model.predict(np.array(TSLA.drop(['Prediction'], 1)))


# In[38]:


# Plotting the Actual and Prediction Prices

plt.figure(figsize=(15, 9))
plt.plot(TSLA.index[display_at:], ridge_model_real_prediction[display_at:], 
         label='Ridge Prediction', color='blue', alpha=alpha)
plt.plot(predicted_dates, ridge_model_predict_prediction, 
         label='Forecast', color='green', alpha=alpha)
plt.plot(TSLA.index[display_at:], TSLA['Close'][display_at:], 
         label='Actual', color='red')
plt.legend()


# In[ ]:




