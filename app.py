import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
from datetime import datetime
import streamlit as st
st.title("Stock Trend Prediction")
user_input=st.text_input('Enter the stock ','TSLA')
df=yf.Ticker(user_input)
df=df.history(period="max")


#Describing the data
st.subheader("Data from inception - till date")
st.write(df.describe())
#Visualisations
st.subheader('Closing Price vs Time Chart')
fig=plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)
st.subheader('Closing Price vs Time Chart(100 MA)')

ma100=df.Close.rolling(100).mean()
ma200=df.Close.rolling(200).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma100,label="MA100")
plt.plot(df.Close,label="Closing Price")
plt.legend()
st.pyplot(fig)
st.subheader('Moving Average 100 vs Moving Average 200')
st.write("Trend analysis of stocks by comparing the 100-day moving average (MA100) and the 200-day moving average (MA200) is a common method used by traders and investors to assess the long-term and intermediate-term trends of a stock. This analysis helps to identify potential buy or sell signals based on the crossovers and relationships between these moving averages.")
st.write("Pay attention to crossovers between MA100 and MA200. When the MA100 crosses above the MA200, it's often considered a bullish signal, indicating that the stock's short-term trend is becoming stronger than its long-term trend. This could be seen as a potential buy signal. Conversely, when the MA100 crosses below the MA200, it could be considered a bearish signal, suggesting a potential downtrend and a potential sell signal.")
ma100=df.Close.rolling(100).mean()
ma200=df.Close.rolling(200).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(ma100,'r',label="MA100",)
plt.plot(ma200,'g',label="MA200",)
# plt.plot(df.Close,label="Closing Price")
plt.legend()
st.pyplot(fig)
#Splitting the data into training and testing
data_training=pd.DataFrame(df['Close'][0:int(len(df)*0.7)])
data_testing=pd.DataFrame(df["Close"][int(len(df)*0.7):int(len(df))])
print(data_training.shape)
print(data_testing.shape)
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
data_training_array=scaler.fit_transform(data_training)
#scaler.scale_
#data_training_array.shape

#Load the model
model=load_model('keras_model.h5')
#Testing part
past_100_days=data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
final_df.head()
input_data=scaler.fit_transform(final_df)
#input_data
#input_data.shape
X_test=[]
y_test=[]
for i in range(100,input_data.shape[0]):
    X_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])   
    
X_test, y_test=np.array(X_test),np.array(y_test)
#print(X_test.shape)
#print(y_test.shape)

y_predicted=model.predict(X_test)
#y_predicted.shape
#y_predicted
#scaler.scale_
scale_factor=1/scaler.scale_
#scale_factor
y_predicted=y_predicted*scale_factor
y_test=y_test*scale_factor

st.subheader("Predictions vs Original")
fig2=plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label="Original Price")
plt.plot(y_predicted, 'r',label="Predicted Price")
plt.legend()
st.pyplot(fig2)
#Stock Sentiment Analysis
from urllib.request import urlopen,Request
from bs4 import BeautifulSoup
news_tables={}
finviz_url="https://finviz.com/quote.ashx?t="
ticker=user_input
url=finviz_url+ticker
req=Request(url=url,headers={'user-agent':'Stock Prediction App'})
response=urlopen(req)
html=BeautifulSoup(response,'html')
news_table=html.find(id='news-table')
news_tables[ticker]=news_table
print(news_tables)


