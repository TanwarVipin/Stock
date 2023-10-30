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

default_date = "2020-01-01"

start_date=st.date_input('Start Date',value=None)

end_date=st.date_input('End Date')

ticker=st.text_input('Ticker','AAPL')
if start_date:
  df=yf.download(ticker,start=start_date,end=end_date)
  df.reset_index(inplace=True)
  st.table(df.head(10))


  fig=go.Figure()
  fig.add_trace(go.Scatter(x=df['Date'],y=df['Close'],name='Stock_Final_Price',line=dict(color='red')))
  fig.add_trace(go.Scatter(x=df['Date'],y=df['Open'],name='Stock_Open_Price',line=dict(color='blue')))
  fig.add_trace(go.Scatter(x=df['Date'],y=df['Close'].rolling(100).mean(),name='Stock_Moving_Average',line=dict(color='green')))
  fig.layout.update(title_text='Time Series Data',xaxis_rangeslider_visible=True)
  st.plotly_chart(fig)

  data=df.copy()
  trace = go.Candlestick(x=data['Date'],
                       open=data['Open'],
                       high=data['High'],
                       low=data['Low'],
                       close=data['Close'])

# Create a layout for the chart
  layout = go.Layout(title='Candlestick Chart',
                   xaxis=dict(title='Date'),
                   yaxis=dict(title='Price'))
                  

# Create a figure and add the candlestick trace to it
  fig1 = go.Figure(data=[trace], layout=layout)

  st.plotly_chart(fig1)

  
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

else:
  st.write("Enter Start Date and waits for Data Loading")



