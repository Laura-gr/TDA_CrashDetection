#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 16:36:12 2023

@authors: slepot, laura-gr
"""

import streamlit as st
import yfinance as yf
import datetime
import plotly.graph_objs as go
import requests
import json

#### Title

st.title('_Topological data analysis for crash detection_ \n based on insert ref of paper')

st.markdown('Here we showcase different plots related to closing stock market values. These plots are based on the paper [Topological data analysis of financial time series: Landscapes of crashes](https://www.sciencedirect.com/science/article/abs/pii/S0378437117309202) by _Marian Gidea_ and _Yuri Katz_ and are computed using some _Topological Data Analysis_ tools. \n Their idea is the following.')
st.markdown('>We use persistence homology to detect and quantify topological patterns that appear in multidimensional time series. Using a sliding window, we extract time-dependent point cloud data sets, to which we associate a topological space. We detect transient loops that appear in this space, and we measure their persistence. This is encoded in real-valued functions referred to as a ’persistence landscapes’. We quantify the temporal changes in persistence landscapes via their $L^p$-norms.')
##### General input parameters

st.header('Choosing the parameters')

with st.expander('Parameters for the analysis. :small[Here you can choose the stock indexes you want to look at and some other technical parameters for the computations]'):

    ### Date
    min_date = datetime.date(2020, 1, 1)
    max_date = datetime.date(2022, 12, 31)


    start_date = st.date_input("Start date", min_value=min_date, max_value=max_date, value=min_date)
    end_date = st.date_input("End date", min_value=min_date, max_value=max_date, value=max_date)

    

### TDA params to select


    st.subheader('TDA parameters for the analysis')


    size_persistence_window=st.selectbox('Please choose the size (in days) of the rolling window',(50,100,150,200,250))


    cut_freq=st.selectbox('Please choose the scale for the cut frequency', (3,2,1,0,-1,'None'))
    st.caption('_If you choose n as the parameter, then the frequency is $10^{-n}$. In particular, if you choose -1, then the frequency is 10_. ')


    norm=st.selectbox('Please select the norm',(1,2,3))
    st.caption('_Here you are choosing $L^p$ norm will be used._')


    type_of_filter=st.selectbox('Please select the filter',('None','low','high'))
    st.caption('_Filters can be applied during the computations._')


    type_of_plot=st.radio('Pick the type of plot', ['Average PSD','Average STD'])


    size_computation_window=st.radio('Select the size of the computation window', (100,250,500))
### Stocks  to select

    stocks=st.multiselect('Choose a stock index', ['AAPL','TSLA','AMZN','MSFT'])
    
if len(stocks)<3:
           st.error('You need to chose at least three stock indices')

    
if start_date <= end_date:
        st.success(" Chosen start date: `{}`\n\n Chosen end date:`{}`.".format(start_date, end_date))
else:
        st.error("Error: End date must be after start date.")

if type_of_filter=='None':
       st.write('You chose to apply no filter to your data.')
else :
       st.write('You are considering a {} filter'.format(type_of_filter))

st.write('Moreover, the parameters you chose are the following. You consider a rolling window of {size_window} days, an $L^{p}$ norm. The plot you are looking at is an {plot_chosen} with frequency cut ${freq_cut}$ \n You are looking at the {stocks_chosen}'.format(size_window=size_persistence_window, p=norm, plot_chosen=type_of_plot, freq_cut= 10**-cut_freq, stocks_chosen=stocks))

for stock_name in stocks :
    stock_data = yf.download(stock_name, start=start_date, end=end_date)
    stock_data.reset_index(inplace=True)


    fig = go.Figure()
    fig.add_trace(go.Scatter(x=stock_data.Date, y=stock_data['Close'], name='Close'))
    fig.update_layout(title=f"{stock_name} Stock Price")
    fig.update_xaxes(title_text='Date')
    fig.update_yaxes(title_text='Close value')
#fig.update_layout(y_label='Close value', x_label='Number of days')
    st.plotly_chart(fig)