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
import pandas as pd
import seaborn as sns
import numpy as np

import sys


import requests
#import json


from TDA_tools import tda_class



#### Title

st.title('_Topological data analysis for crash detection in stock markets_')

#st.write(tickers[:5])

st.markdown('Here we showcase different plots related to closing stock market values. These plots are based on the paper [Topological data analysis of financial time series: Landscapes of crashes](https://www.sciencedirect.com/science/article/abs/pii/S0378437117309202) by _Marian Gidea_ and _Yuri Katz_ and are computed using some _Topological Data Analysis_ tools. \n Their idea is the following.')
st.markdown('>We use persistence homology to detect and quantify topological patterns that appear in multidimensional time series. [...] Our study suggests that TDA provides a new type of econometric analysis, which complements the standard statistical measures. \n The method can be used to detect early warning signals of imminent market crashes. We believe that this approach can be used beyond the analysis of financial time series presented here.')
##### General input parameters


### Date
min_date = datetime.date(2000, 1, 1)
max_date = datetime.date(2023, 12, 31)


start_date = st.date_input("Start date", min_value=min_date, max_value=max_date, value=min_date, help='For reasons of computation costs, we limit the starting date to January 2000.')

end_date = st.date_input("End date", min_value=min_date, max_value=max_date, value=max_date)
    
### Stocks  to select

indices_dict={'Russell 2000':'^RUT','Dow Jones Industrial Average':'^DJI','NASDAQ Composite':'^IXIC','NYSE Composite':'^NYA', 'CAC40':'^FCHI', 'DAX Germany':'DAX','AEX Amsterdam':'^AEX', 'FTSE 100':'^FTSE','IBEX 35':'^IBEX', 'Euronext 100':'^N100','Nikkei 225':'^N225','BEL 20':'^BFX','MOEX Russia':'IMOEX.ME','Hang Seng':'^HSI','All Ordinaries':'^AORD','IPC Mexico':'^MXX','MERVAL':'^MERV' }

indices_names=list(x for x in indices_dict.keys())

stocks_input=st.multiselect('Choose a stock index', indices_names)

stocks=[indices_dict[x] for x in stocks_input]
    
if len(stocks)<3:
    st.error('You need to chose at least three stock indices')
    sys.exit()
           

    
if start_date <= end_date:
    st.success(" Chosen start date: `{}`\n\n  Chosen end date: `{}`.".format(start_date, end_date))
else:
    st.error("Error: End date must be after start date.")


### Creating a DF with all stocks in it and index=date in time window specified by user


dict_of_crashes={'Dot Com':datetime.date(2000,3,10), 'September 11 Attacks':datetime.date(2001,9,11),'Lehman bankruptcy':datetime.date(2008,9,15)}


stocks_all=pd.DataFrame()
fig = go.Figure()

for stock_name in stocks_input :
    stock_data = yf.download(indices_dict[stock_name], start=start_date, end=end_date)
    stocks_all[stock_name]=stock_data['Close']
    stocks_all.index=pd.to_datetime(stocks_all.index.date)
    ## Creating plots (but hidden in expanding boxes) for each of the chosen stocks
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], name='{}'.format(stock_name)))
    
for crisis in dict_of_crashes :
    if start_date <= dict_of_crashes[crisis] <= end_date :
        st.write('The {} crisis happened during the period you are looking at'.format(crisis))
        #fig.add_vline(x=dict_of_crashes[crisis], annotation_text='hey',line_dash='dash')
    
fig.update_xaxes(title_text='Date')
fig.update_yaxes(title_text='Close value')
fig.update_layout(title=f'{stocks_input} closing values')
st.plotly_chart(fig)

    

st.subheader('Measures of volatility')

st.write("For predecting crashes, looking at the closing values isn't enough. \n The authors suggest new measures of volatility for markets that could provide early warning for imminent crashes.")

with st.expander('Parameters for the analysis'):

    ### TDA params to select

    size_persistence_window=st.selectbox('Choose the size (in days) of the rolling window',(50,100,150,200,250))

    norm_input=st.selectbox('Select the norm',(1,2,3))
    norm=[int(norm_input)]
    st.caption('_Here you are choosing $L^p$ norm will be used._')

type_of_plot=st.radio('Choose which measure you want to display.', ['Persistence norm','Average Power Spectral density','Average variance'], captions=["Using a sliding window, we extract time-dependent point \
                  cloud data sets, to which we associate a topological space. We detect transient loops that appear in this space, and we measure their persistence. \
                 This is encoded in real-valued functions referred to as a persistence landscapes’. We quantify the temporal changes in persistence landscapes via their Lp-norms.","Power Spectral density describes how the power of a signal or time series is distributed over frequency. \n Here we compute its average over the user-selected frequencies.","We compute the variance of the signal over the user-selected frequencies."])



if type_of_plot!='Persistence norm' :

    with st.expander('For the plots you chose, you need to specify the following.') :
        cut_freq=st.selectbox('Choose the scale for the cut frequency', (3,2,1,0,-1,'None'))
        st.caption('_If you choose n as the parameter, then the frequency is $10^{-n}$. In particular, if you choose -1, then the frequency is 10_. ')

        type_of_filter=st.selectbox('Select the filter',(None,'low','high'))
        st.caption('_Filters can be applied during the computations._')

        size_computation_window=st.radio('Select the size of the computation window', (100,250,500))
    
    st.write('Moreover, the parameters you chose are the following. You consider a rolling window of {size_window} days, an $L^p$ norm. The plot you are looking at is an {plot_chosen} with frequency cut ${freq_cut}$ \n You are looking at the {stocks_chosen} stocks.'.format(size_window=size_persistence_window, p=norm, plot_chosen=type_of_plot, freq_cut= 10**-cut_freq, stocks_chosen=stocks_input))



    if type_of_filter=='None':
       st.write('You chose to apply no filter to your data.')
    else :
       if type_of_filter==None:
              st.write('You chose to apply no filter.')
       else :
            st.write('You are considering a {} filter'.format(type_of_filter))

else :
    cut_freq=0
    type_of_filter=None
    size_computation_window=1
    st.write('Moreover, the parameters you chose are the following. You consider a rolling window of {size_window} days, an $L^p$ norm. The plot you are looking at is an {plot_chosen} \n You are looking at the {stocks_chosen} stocks.'.format(size_window=size_persistence_window, p=norm, plot_chosen=type_of_plot, stocks_chosen=stocks_input))







stocks_tda=tda_class.computation_tda(data=stocks_all, window_tda=size_persistence_window, scaling=None, p_norms=norm, window_freq=size_computation_window, freq_cut=cut_freq, filter_keep=type_of_filter)


if type_of_plot=='Persistence norm' :
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=stocks_tda.avg_PSD.index,y=stocks_tda.persistence_norms.iloc[:,0]))
    fig.update_layout(title='{}'.format(type_of_plot))
    fig.update_traces(mode="markers+lines", hovertemplate=None)
    fig.update_xaxes(title_text='Date')
    fig.update_yaxes(title_text='{}'.format(type_of_plot))
    st.plotly_chart(fig)

if type_of_plot=='Average Power Spectral density':
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=stocks_tda.avg_PSD.index,y=stocks_tda.avg_PSD.iloc[:,0]))
    fig.update_layout(title='{}'.format(type_of_plot))
    fig.update_traces(mode="markers+lines", hovertemplate=None)
    fig.update_xaxes(title_text='Date')
    fig.update_yaxes(title_text='{}'.format(type_of_plot))
    st.plotly_chart(fig)

if type_of_plot == 'Average variance' :
    fig=go.Figure()
    fig.add_trace(go.Scatter(x=stocks_tda.avg_PSD.index,y=stocks_tda.norms_var.iloc[:,0]))
    fig.update_layout(title='{}'.format(type_of_plot))
    fig.update_traces(mode="markers+lines", hovertemplate=None)
    fig.update_xaxes(title_text='Date')
    fig.update_yaxes(title_text='{}'.format(type_of_plot))
    st.plotly_chart(fig)
