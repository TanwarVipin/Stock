import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objects as go
import plotly.express as px

import streamlit as st

import yfinance as yf 

from datetime import datetime

st.title('Stock Forecasting App')



start_date=st.date_input('Start Date')

end_date=st.date_input('End Date')

ticker=st.text_input('Ticker')

df=yf.download(ticker,start=start_date,end=end_date)


df.reset_index(inplace=True)

st.table(df)

def plot_raw_data(df):
  fig=go.Figure()
  fig.add_trace(go.Scatter(x=df['Date'],y=df['Close'],name='Stock_Final_Price'),line=dict(color='red'))
  fig.add_trace(go.Scatter(x=df['Date'],y=df['Open'],name='Stock_Open_Price'),line=dict(color='blue'))
  fig.add_trace(go.Scatter(x=df['Date'],y=df['Close'].rolling(100).mean(),name='Stock_Moving_Average'),line=dict(color='green'))
  fig.layout.update(title_text='Time Series Data',xaxis_rangeslider_visible=True)
  st.plotly_chart(fig)


plot_raw_data(df)


train_df=df[['Date','Close']].copy()

train_df=train_df.rename(columns={'Date':'ds','Close':'y'})

year=st.select_slider('No_of_Year',options=[1,2,3,4,5,6,7,8,9,10])

m=Prophet()
m.fit(train_df)

future=m.make_future_dataframe(periods=year*365)


forecast=m.predict(future)

st.subheader('Forecast Data')
st.table(forecast.tail(10))


fig1=plot_plotly(m,forecast)
st.write(fig1)


st.subheader('Forecast Components')

fig2=m.plot_components(forecast)
st.write(fig2)


