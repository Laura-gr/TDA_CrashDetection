#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 16:36:12 2023

@authors: slepot, laura-gr
"""

import streamlit as st

import yfinance as yf
import datetime
import time

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
min_date = datetime.date(1990, 1, 1)
max_date = datetime.date(2023, 12, 31)


start_date = st.date_input("Start date", min_value=min_date, max_value=max_date, value=min_date, help='For reasons of computation costs, we limit the starting date to January 2000.')

end_date = st.date_input("End date", min_value=min_date, max_value=max_date, value=max_date)

start_date = datetime.datetime(start_date.year, start_date.month, start_date.day)
end_date = datetime.datetime(end_date.year, end_date.month, end_date.day)

### Stocks  to select

indices_dict={'S&P 500':'^SPX','Russell 2000':'^RUT','Dow Jones Industrial Average':'^DJI','NASDAQ Composite':'^IXIC','NYSE Composite':'^NYA', 'CAC40':'^FCHI','AEX Amsterdam':'^AEX', 'FTSE 100':'^FTSE','IBEX 35':'^IBEX', 'Euronext 100':'^N100','Nikkei 225':'^N225','BEL 20':'^BFX','MOEX Russia':'IMOEX.ME','Hang Seng':'^HSI','All Ordinaries':'^AORD','IPC Mexico':'^MXX','MERVAL':'^MERV' }

indices_names=list(x for x in indices_dict.keys())

stocks_input=st.multiselect('Choose a stock index', indices_names)

stocks=[indices_dict[x] for x in stocks_input]
    
if len(stocks)<3:
    st.error('You need to chose at least three stock indices')
    sys.exit()

stock_str=' '
for x in stocks_input[:-1]:
    stock_str += x+', '
stock_str += 'and '+stocks_input[-1]

           

    
if start_date <= end_date:
    st.success(" Chosen start date: `{}`\n\n  Chosen end date: `{}`.".format(start_date, end_date))
else:
    st.error("Error: End date must be after start date.")


### Creating a DF with all stocks in it and index=date in time window specified by user


dict_of_crashes={'Dot Com':datetime.datetime(2000,3,10), 'September 11 Attacks':datetime.datetime(2001,9,11),'Lehman bankruptcy':datetime.datetime(2008,9,15)}


stocks_all=pd.DataFrame()
fig = go.Figure()

for stock_name in stocks_input :
    stock_data = yf.download(indices_dict[stock_name], start=start_date, end=end_date)
    stocks_all[stock_name]=stock_data['Close']
    fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], name='{}'.format(stock_name)))


for crisis in dict_of_crashes :
    if start_date <= dict_of_crashes[crisis] and dict_of_crashes[crisis] <= end_date :
        #st.write('The {} crisis happened during the period you are looking at'.format(crisis))
        fig.add_vline(x=dict_of_crashes[crisis].timestamp()*1000 ,line_dash='dash', line_color = 'green')
        fig.add_annotation(x=dict_of_crashes[crisis].timestamp()*1000, y=0.85, text = crisis, textangle = -30, yref = 'y domain', xanchor = 'left', yanchor = 'bottom', showarrow = False)


fig.update_xaxes(title_text='Date')
fig.update_yaxes(title_text='Close value')
fig.update_layout(title=f'{stocks_input} closing values')
st.plotly_chart(fig)


st.subheader('Measures of volatility')

st.write("For predecting crashes, looking at the closing values isn't enough. \n The authors suggest new measures of volatility for markets that could provide early warning for imminent crashes.")


type_of_plot=st.radio('Choose which measure you want to display.', ['Persistence norm','Average Power Spectral density','Average variance'], captions=["Using a sliding window, we extract time-dependent point \
                  cloud data sets, to which we associate a topological space. We detect transient loops that appear in this space, and we measure their persistence. \
                 This is encoded in real-valued functions referred to as a persistence landscapes’. We quantify the temporal changes in persistence landscapes via their Lp-norms.","Power Spectral density describes how the power of a signal or time series is distributed over frequency. \n Here we compute its average over the user-selected frequencies.","We compute the variance of the signal over the user-selected frequencies."])

with st.expander('Parameters for the analysis'):

    ### TDA params to select

    size_persistence_window=st.selectbox('Choose the size (in days) of the rolling window',(50,100,150,200,250))

    norm_input=st.selectbox('Select the norm',(1,2,3))
    norm=[int(norm_input)]
    st.caption('_Here you are choosing $L^p$ norm will be used._')



if type_of_plot!='Persistence norm' :

    with st.expander('For the plot you chose, you need to specify the following.') :
        st.write('Here we decompose our signal in the Fourier domain. As such, we can choose to focus on different types of frequencies and to set the thresholds between those frequencies at different levels.')
        filter=st.selectbox('Select the type of filter for the frequencies.',('No filter','low-pass','high-pass'), help='Here you choose if you want to look higher or lower than your threshold or to look at all frequencies. If you look at all frequencies and then there is no need to choose a threshold.')
        
        
        if filter=='No filter' :
            type_of_filter=None
            cut_freq_type='all'
            cut_freq=None
            
        else :
            type_of_filter=filter.split('-')[0]
            
            cut_freq_type = st.radio('Pick the threshold', ['low','medium low', 'medium high', 'high', 'custom'],help='You can cut the frequencies at different scales. We provide some values or you can enter a custom frequency cut value.')
        
            cut_freq_dict={'low':4.27,'medium low':19.5,'medium high':60.2,'high':90.7}

            if cut_freq_type=='custom':
                cut_freq_100=st.number_input('Insert a number between 0 and 100')

                if cut_freq_100 >100 or cut_freq_100 < 0 :
                    st.error('The value should be between 0 and 100')
                    sys.exit()
            else :
                cut_freq_100 = cut_freq_dict[cut_freq_type]
        
            cut_freq=0.00492 * cut_freq_100 + 0.004
            st.write('You are considering a {}-pass filter with a {} cutoff frequency.'.format(type_of_filter,cut_freq_type))

        size_computation_window=st.radio('Select the size of the computation window', (100,250,500))
        
    st.write('The parameters you chose are the following. \n \n You consider a rolling window of {size_window} days, an $L^{p}$ norm. The plot it will display at is a {plot_chosen} plot '.format(size_window=size_persistence_window, p=norm[0], plot_chosen=type_of_plot)+ 'for the '+stock_str+ ' indices.')

    if type_of_filter==None:
       st.write('You chose to apply no filter to your data and therefore considering all frequencies.')
    else :
        st.write('You are considering a {}-pass filter with a {} cutoff frequency.'.format(type_of_filter,cut_freq_type))

else :
    cut_freq=0
    type_of_filter=None
    size_computation_window=1
    st.write('The parameters you chose are the following. \n \n You consider a rolling window of {size_window} days, an $L^{p}$ norm. The plot it will display at is a {plot_chosen} plot '.format(size_window=size_persistence_window, p=norm[0], plot_chosen=type_of_plot)+ 'for the '+stock_str+ ' indices.')





if st.button("Compute and plot."):
    with st.spinner('Wait for it... Maybe go fetch a coffee in the meantime.'):
        stocks_tda=tda_class.computation_tda(data=stocks_all, window_tda=size_persistence_window, scaling=None, p_norms=norm, window_freq=size_computation_window, freq_cut=cut_freq, filter_keep=type_of_filter)
        
        
        if type_of_plot=='Persistence norm' :
            fig=go.Figure()
            fig.add_trace(go.Scatter(x=stocks_tda.avg_PSD.index,y=stocks_tda.persistence_norms.iloc[:,0]))
            fig.update_layout(title='{}'.format(type_of_plot))
            fig.update_traces(mode="markers+lines", hovertemplate=None)
            fig.update_xaxes(title_text='Date')
            fig.update_yaxes(title_text='{}'.format(type_of_plot))
            for crisis in dict_of_crashes :
                if start_date <= dict_of_crashes[crisis] and dict_of_crashes[crisis] <= end_date :
                    #st.write('The {} crisis happened during the period you are looking at'.format(crisis))
                    fig.add_vline(x=dict_of_crashes[crisis].timestamp()*1000 ,line_dash='dash', line_color = 'green')
                    fig.add_annotation(x=dict_of_crashes[crisis].timestamp()*1000, y=0.85, text = crisis, textangle = -30, yref = 'y domain', xanchor = 'left', yanchor = 'bottom', showarrow = False)

            st.plotly_chart(fig)

        if type_of_plot=='Average Power Spectral density':
            fig=go.Figure()
            fig.add_trace(go.Scatter(x=stocks_tda.avg_PSD.index,y=stocks_tda.avg_PSD.iloc[:,0]))
            fig.update_layout(title='{}'.format(type_of_plot))
            fig.update_traces(mode="markers+lines", hovertemplate=None)
            fig.update_xaxes(title_text='Date')
            fig.update_yaxes(title_text='{}'.format(type_of_plot))
            for crisis in dict_of_crashes :
                if start_date <= dict_of_crashes[crisis] and dict_of_crashes[crisis] <= end_date :
                    #st.write('The {} crisis happened during the period you are looking at'.format(crisis))
                    fig.add_vline(x=dict_of_crashes[crisis].timestamp()*1000 ,line_dash='dash', line_color = 'green')
                    fig.add_annotation(x=dict_of_crashes[crisis].timestamp()*1000, y=0.85, text = crisis, textangle = -30, yref = 'y domain', xanchor = 'left', yanchor = 'bottom', showarrow = False)

            st.plotly_chart(fig)

        if type_of_plot == 'Average variance' :
            fig=go.Figure()
            fig.add_trace(go.Scatter(x=stocks_tda.avg_PSD.index,y=stocks_tda.norms_var.iloc[:,0]))
            fig.update_layout(title='{}'.format(type_of_plot))
            fig.update_traces(mode="markers+lines", hovertemplate=None)
            fig.update_xaxes(title_text='Date')
            fig.update_yaxes(title_text='{}'.format(type_of_plot))
            for crisis in dict_of_crashes :
                if start_date <= dict_of_crashes[crisis] and dict_of_crashes[crisis] <= end_date :
                    #st.write('The {} crisis happened during the period you are looking at'.format(crisis))
                    fig.add_vline(x=dict_of_crashes[crisis].timestamp()*1000 ,line_dash='dash', line_color = 'green')
                    fig.add_annotation(x=dict_of_crashes[crisis].timestamp()*1000, y=0.85, text = crisis, textangle = -30, yref = 'y domain', xanchor = 'left', yanchor = 'bottom', showarrow = False)

            st.plotly_chart(fig)
