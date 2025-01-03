import pandas as pd
import numpy as np
import yfinance as yf
from yahooquery import Ticker
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

def get_price(simbol):
    df = yf.download(simbol)

    df.columns = df.columns.droplevel(1)
    df.columns.names = [None]
    df = df.reset_index()

    return df

def get_statement(simbol):
    data = Ticker('aapl')
    print(data.summary_detail)

    return data

def data_adjust(df):
    df_adjusted = df.drop(['Adj Close'], axis = 1)
    df_adjusted = df_adjusted.loc[df_adjusted['Date'] >= f'{datetime.today().year - 10}-01-01',:].reset_index(drop = True)
    
    dates = df_adjusted['Date']
    df_adjusted = df_adjusted.drop('Date', axis = 1)
    return df_adjusted, dates

def train_test_split(df, train_size = int, seq_length = int):
    train_set = df[0 : train_size]
    test_set  = df[train_size - seq_length:].reset_index(drop = True)

    return train_set, test_set

def scailing(train_set, test_set):
    # Imput scale
    scaler_x = MinMaxScaler()
    scaler_x.fit(train_set.iloc[:, 1:])

    trainX_scailed = scaler_x.transform(train_set.iloc[:, 1:])
    testX_scailed  = scaler_x.transform(test_set.iloc[:, 1:])

    # Output scale
    scaler_y = MinMaxScaler()
    scaler_y.fit(train_set.iloc[:, [0]])

    trainY_scailed = scaler_y.transform(train_set.iloc[:, [0]])
    testY_scailed  = scaler_y.transform(test_set.iloc[:, [0]])

    train_scailed = np.concatenate(trainX_scailed, trainY_scailed, axis = 1)
    test_scailed  = np.concatenate(testX_scailed, testY_scailed, axis = 1)

    return scaler_x, scaler_y, train_scailed, test_scailed

def build_dataset(time_series, seq_length):
    dataX = []
    dataY = []
    for i in range(0, len(time_series)-seq_length):
        _x = time_series[i:i+seq_length, :]
        _y = time_series[i+seq_length, [0]]
        # print(_x, "-->",_y)
        dataX.append(_x)
        dataY.append(_y)

    return np.array(dataX), np.array(dataY)

