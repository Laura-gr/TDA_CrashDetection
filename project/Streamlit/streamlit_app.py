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

min_date = datetime.date(2020, 1, 1)
max_date = datetime.date(2022, 12, 31)

stock_name = st.selectbox('Please choose stock name', ('AAPL','TSLA','AMZN','MSFT'))

start_date = st.date_input("Start date", min_value=min_date, max_value=max_date, value=min_date)
end_date = st.date_input("End date", min_value=min_date, max_value=max_date, value=max_date)

if start_date <= end_date:
    st.success("Start date: `{}`\n\nEnd date:`{}`".format(start_date, end_date))
else:
    st.error("Error: End date must be after start date.")

stock_data = yf.download(stock_name, start=start_date, end=end_date)
stock_data.reset_index(inplace=True)

fig = go.Figure()
fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['Close'], name='Close'))
fig.update_layout(title=f"{stock_name} Stock Price")
fig.update_xaxes(title_text='Number of days')
fig.update_yaxes(title_text='Close value')
#fig.update_layout(y_label='Close value', x_label='Number of days')
st.plotly_chart(fig)