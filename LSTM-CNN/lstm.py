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

# callback = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=7)

sliding_window = 30
stonks = [('INTC', True), ('MMM', False), ('UAL', True), ('NFLX', True)]

#https://towardsdatascience.com/predicting-stock-prices-using-a-keras-lstm-model-4225457f0233

for (s, vol) in stonks:
    # stock = pd.read_csv("../DATA/%s_daily.csv" % (s))
    #For final test, does the past 30 days (as of May 7)
    stock = pd.read_csv("../DATA_APRIL/%s_daily.csv" % (s))
    prices = stock['close'].values[::-1].reshape(len(stock), 1)
    Ms = MinMaxScaler(feature_range=(0,1))

    train_test_split = (int)(len(prices) - 30)
    prices_train = prices[:train_test_split]
    prices_test = prices[train_test_split:]

    prices_train_scaled = Ms.fit_transform(prices_train)

    X_train = []
    y_train = []
    for i in range(sliding_window, len(prices_train_scaled)):
        X_train.append(prices_train_scaled[i-sliding_window:i])
        y_train.append(prices_train_scaled[i])
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    print(X_train.shape)
    
    model = keras.models.Sequential()
    model.add(keras.layers.Conv1D(32, 3))
    model.add(keras.layers.AvgPool1D(3))
    model.add(keras.layers.LSTM(units=50,return_sequences=True))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.LSTM(units=50))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(units=1))

    opt = keras.optimizers.Adam(learning_rate=0.0005)
    # opt = keras.optimizers.RMSprop()
    model.compile(optimizer=opt,loss='mse', metrics=[keras.metrics.RootMeanSquaredError(name='rmse'), 'mean_absolute_error'])
    model.fit(X_train,y_train,epochs=50,batch_size=32,validation_split=0.1,verbose=0) #callback not included

    inputs = prices[len(prices) - len(prices_test) - sliding_window:]
    inputs = inputs.reshape(-1,1)
    inputs = Ms.transform(inputs)
    X_test = []
    y_test = []
    for i in range(0, len(prices_test)):
        X_test.append(inputs[i:i + sliding_window, 0])
        y_test.append(inputs[i + sliding_window][0])
    # print(X_test.shape)
    X_test = np.array(X_test)   
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    predicted_stock_price = model.predict(X_test)
    predicted_stock_price = Ms.inverse_transform(predicted_stock_price)

    model.evaluate(X_test, np.array(y_test))

    y_test = Ms.inverse_transform(np.array(y_test).reshape(-1, 1))

    P = [1000, 0]

    trade_length = min(len(predicted_stock_price), len(y_test))

    initial_trade = False
    if vol:
        for i in range(1, trade_length):
            if predicted_stock_price[i] > predicted_stock_price[i-1] and P[0] > 0:
                P[1] = P[0] / y_test[i]
                P[0] = 0
                print("BUY at day: ", i, ", For price: ", y_test[i])
            elif predicted_stock_price[i] < predicted_stock_price[i-1] and P[1] > 0:
                P[0] = P[1] * y_test[i]
                P[1] = 0
                print("SELL at day: ", i, ", For price: ", y_test[i])

    # If stock is not volatile (MMM)
    else:
        for i in range(1, trade_length):
            if predicted_stock_price[i] > y_test[i-1] and P[0] > 0:
                P[1] = P[0] / y_test[i]
                P[0] = 0
                print("BUY at day: ", i, ", For price: ", y_test[i])
            elif predicted_stock_price[i] < y_test[i-1] and P[1] > 0:
                P[0] = P[1] * y_test[i]
                P[1] = 0
                print("SELL at day: ", i, ", For price: ", y_test[i])

    print("Hold amount: ", 1000 * y_test[len(y_test) - 1] / y_test[0])
    print("Final amount: ", P[0] + P[1] * y_test[trade_length - 1])

    plt.plot(prices_test, color = 'black', label = '%s Stock Price' % (s))
    plt.plot(predicted_stock_price, color = 'green', label = 'Predicted %s Stock Price' % (s))
    plt.title('%s Stock Price Prediction' % (s))
    plt.xlabel('Time')
    plt.ylabel('%s Stock Price' % (s))
    plt.legend()
    plt.show()