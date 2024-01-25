#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 16:36:12 2023

@author: slepot
"""

import streamlit as st
import yfinance as yf
import datetime
import plotly.graph_objs as go
import requests
import json

#### Title
st.title('_Topological data analysis_ for crash detection \n based on insert ref of paper')

st.markdown('Here we showcase different plots related to closing stock market values. These plots are based on INSERT REF and are computed using some _Topological Data Analysis_ tools.')
##### General input parameters

st.header('Choosing the parameters')

st.subheader('Time scope for the analysis')


tab1, tab2, tab3=st.tabs(['Date scope','TDA parametes', 'Stock parameters'])


with tab1 :
    ### Date
    min_date = datetime.date(2020, 1, 1)
    max_date = datetime.date(2022, 12, 31)


    start_date = st.date_input("Start date", min_value=min_date, max_value=max_date, value=min_date)
    end_date = st.date_input("End date", min_value=min_date, max_value=max_date, value=max_date)

    if start_date <= end_date:
        st.success(" Chosen start date: `{}`\n\n Chosen end date:`{}`".format(start_date, end_date))
    else:
        st.error("Error: End date must be after start date.")


### TDA params to select

with tab2 :

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

with tab3 :
     stock_name = st.selectbox('Please choose at least three stocks to analyze', ('AAPL','TSLA','AMZN','MSFT'))


stock_data = yf.download(stock_name, start=start_date, end=end_date)
stock_data.reset_index(inplace=True)

fig = go.Figure()
fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], name='Close'))
fig.update_layout(title=f"{stock_name} Stock Price")
fig.update_xaxes(title_text='Number of days')
fig.update_yaxes(title_text='Close value')
#fig.update_layout(y_label='Close value', x_label='Number of days')
st.plotly_chart(fig)