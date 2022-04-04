import pandas as pd

import pandas as pd
import datetime as dt
import urllib.request, json
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import sys
from tensorflow import keras
import matplotlib.pyplot as plt


stonks = ['INTC']
# , 'MMM', 'UAL', 'NFLX']

#https://towardsdatascience.com/predicting-stock-prices-using-a-keras-lstm-model-4225457f0233

for s in stonks:
    stock = pd.read_csv("../DATA/%s_daily.csv" % (s))
    prices = stock['close'].values[::-1].reshape(len(stock), 1)
    Ms = MinMaxScaler(feature_range=(0,1))

    train_test_split = (int)(0.8 * len(prices))
    prices_train = prices[0:train_test_split]
    prices_test = prices[train_test_split:]

    prices_train_scaled = Ms.fit_transform(prices_train)

    X_train = []
    y_train = []
    for i in range(60, len(prices_train_scaled)):
        X_train.append(prices_train_scaled[i-60:i])
        y_train.append(prices_train_scaled[i])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    
    model = keras.models.Sequential()
    model.add(keras.layers.LSTM(units=50,return_sequences=True,input_shape=(X_train.shape[1], 1)))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.LSTM(units=50,return_sequences=True))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.LSTM(units=50,return_sequences=True))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.LSTM(units=50))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(units=1))
    model.compile(optimizer='adam',loss='mean_squared_error')
    model.fit(X_train,y_train,epochs=20,batch_size=32)

    inputs = prices[len(prices) - len(prices_test) - 60:]
    inputs = inputs.reshape(-1,1)
    inputs = Ms.transform(inputs)
    X_test = []
    for i in range(60, len(prices_test)):
        X_test.append(inputs[i-60:i, 0])
    # print(X_test.shape)
    X_test = np.array(X_test)   
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    predicted_stock_price = model.predict(X_test)
    predicted_stock_price = Ms.inverse_transform(predicted_stock_price)

    print(predicted_stock_price)
    plt.plot(prices_test[:50], color = 'black', label = '%s Stock Price' % (s))
    plt.plot(predicted_stock_price[:50], color = 'green', label = 'Predicted %s Stock Price' % (s))
    plt.title('%s Stock Price Prediction' % (s))
    plt.xlabel('Time')
    plt.ylabel('%s Stock Price' % (s))
    plt.legend()
    plt.show()