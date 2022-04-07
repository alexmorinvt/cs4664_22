import pandas as pd
from sklearn.model_selection import train_test_split
import pandas as pd
import datetime as dt
import urllib.request, json
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import sys
from tensorflow import keras
import matplotlib.pyplot as plt


stonks = ['NFLX']
# , 'MMM', 'UAL', 'NFLX']

#https://towardsdatascience.com/predicting-stock-prices-using-a-keras-lstm-model-4225457f0233

for s in stonks:
    stock = pd.read_csv("./combine.csv")
    #print(stock.iloc[0])
    stock.dropna()

    X = stock[['neutral', 'positive', 'negative','open','high','low','close','volume']]
    #print(X[0:5])
    Y = stock['close'].values[::-1].reshape(len(stock), 1)
    #print(Y[0:5])
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    #prices = stock['close'].values[::-1].reshape(len(stock), 1)
    Ms = MinMaxScaler(feature_range=(0,1))

    #train_test_split = (int)(0.8 * len(prices))
    #prices_train = prices[0:train_test_split]
    #prices_test = prices[train_test_split:]

    prices_train_scaled = Ms.fit_transform(X_train)
    Y_train = Y_train.reshape(-1,1)
    prices_test_scaled = Ms.fit_transform(Y_train)
    #print(prices_train_scaled.shape)
    #print(prices_test_scaled.shape)

    train = []
    #val = []
    for i in range(60, len(prices_train_scaled)):
        train.append(prices_train_scaled[i-30:i])
        #val.append(prices_test_scaled[i+60:])
    #print(train.shape)
    train = np.asarray(train,dtype=object).astype(np.float32)

    val=prices_test_scaled[60:len(prices_test_scaled)]
    val = np.asarray(val,dtype=object).astype(np.float32)

    train = np.reshape(train, (train.shape[0], train.shape[1], 8))

    val = val.reshape(-1,1)
  
    
    model = keras.models.Sequential()
    model.add(keras.layers.LSTM(units=50,return_sequences=True,input_shape=(X_train.shape[1], 8)))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.LSTM(units=50,return_sequences=True))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.LSTM(units=50,return_sequences=True))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.LSTM(units=50))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(units=1))
    model.compile(optimizer='adam',loss='mean_squared_error')
    model.fit(train,val,epochs=20,batch_size=32)

    #inputs = prices[len(prices) - len(prices_test) - 60:]
    #inputs = inputs.reshape(-1,1)
    #inputs = Ms.transform(inputs)
    #X_test = []
    #for i in range(60, len(prices_test)):
    #    X_test.append(inputs[i-60:i, 0])
    # print(X_test.shape)
    #X_test = np.array(X_test)   
    #X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))




    pre_train_scaled = Ms.fit_transform(X_test)
    Y_test = Y_test.reshape(-1,1)
    pre_test_scaled = Ms.fit_transform(Y_test)

    input = []
    for i in range(60, len(pre_train_scaled)):
        input.append(pre_train_scaled[i-60:i])

    test = pre_test_scaled[60:len(pre_test_scaled)]
    input, test = np.asarray(input,dtype=object).astype(np.float32), np.asarray(test,dtype=object).astype(np.float32)
    input = np.reshape(input, (input.shape[0], input.shape[1], 8))

    predicted_stock_price = model.predict(input)
    predicted_stock_price = Ms.inverse_transform(predicted_stock_price)

    #val = val.reshape(-1,1)

    #print(predicted_stock_price)
    plt.plot(Y_test[60:len(Y_test)], color = 'black', label = '%s Stock Price' % (s))
    plt.plot(predicted_stock_price[:len(predicted_stock_price)], color = 'green', label = 'Predicted %s Stock Price' % (s))
    plt.title('%s Stock Price Prediction' % (s))
    plt.xlabel('Time')
    plt.ylabel('%s Stock Price' % (s))
    plt.legend()
    plt.show()
    plt.savefig('nlp.png')