
# some_file.py
import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '../TDA_tools/tda_class.py')

import tda_class

import yfinance as yf
import datetime
import plotly.graph_objs as go
import requests
import json
import pandas as pd

start_date = datetime.date(2020, 1, 1)
end_date = datetime.date(2022, 12, 31)


stocks= ['AAPL','TSLA','AMZN','MSFT']

stocks_tda=pd.DataFrame()

for stock_name in stocks :
    stock_data = yf.download(stock_name, start=start_date, end=end_date)
    #stock_data.reset_index(inplace=True)
    stocks_tda[stock_name]=stock_data['Close']

print(stocks_tda.head())


