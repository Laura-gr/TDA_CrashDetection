#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 17:05:34 2023

@author: slepot
"""

import yfinance as yf
import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import plotly.io as pio
import plotly.express as px
pio.renderers.default='browser'

stock_names = ['AAPL','TSLA','AMZN']

start_date = datetime.date(2020, 1, 1)
end_date = datetime.date(2022, 12, 31)

# stock_data = {stock: yf.download(stock, start=start_date, end=end_date) 
#               for stock in stock_names}

stock_data = yf.download(['AAPL','TSLA','AMZN'], 
                         start=start_date, end=end_date)['Close'] 
scaler = StandardScaler(with_mean = False)
stock_data = pd.DataFrame(scaler.fit_transform(stock_data),
                          columns = stock_data.columns,
                          index=stock_data.index)
fig,ax = plt.subplots(figsize = (12,8))
ax.scatter(stock_data[stock_data.columns[0]],
            stock_data[stock_data.columns[2]])
ax.set_xlabel(stock_data.columns[0])
ax.set_ylabel(stock_data.columns[2])
plt.show()

fig = px.scatter_3d(stock_data, x='AAPL', y='TSLA', z='AMZN')
fig.show()
