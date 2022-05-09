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
from tqdm import tqdm
from finBERT import findPercentageBySentences
from util import data_preprocessing
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")

model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert",output_hidden_states=True)




stonks = ['NFLX']
# , 'MMM', 'UAL', 'NFLX']

#https://towardsdatascience.com/predicting-stock-prices-using-a-keras-lstm-model-4225457f0233

for s in stonks:
    stock = pd.read_csv("./new_combine.csv")
    data=[]
    for i, line in tqdm(enumerate(stock['title'])):
        
        line=data_preprocessing(line)
        encoding = tokenizer(line, return_tensors="pt")

        # forward pass
        outputs = model(**encoding)
        # get feature vector 
        feature_vector = outputs.hidden_states[0][:,:,-1]
        print(outputs.hidden_states[1].shape)
        #list = {'pos': 0.8, 'neg': 0.1, 'neu': 0.1}
        #senData[i]=list
        #data.append([stock.iloc[i]['date_time'], stock.iloc[i]['open'],stock.iloc[i]['high'],stock.iloc[i]['low'],stock.iloc[i]['close'],line,list['pos'],list['neg'],list['neu']])
        #data.append(['date': netflix.iloc[i]['date'], 'time': netflix.iloc[i]['time'], 'headline':line,'pos':list['pos'],'neg':list['neg'],'neu':list['neu']},ignore_index=True)

    df = pd.DataFrame(columns =['date','open','high','low','close','headline', 'positive','negative','neutral'], data=data)
    #print(stock.iloc[0])
    df.to_csv('new_sen.csv',index=False)
    stock=df
    stock.dropna()

    X = stock[['close','open','high','low']]#'neutral', 'positive', 'negative','open','high','low',
    #print(X[0:5])
    Y = stock['close'].values[::-1].reshape(len(stock), 1)
    #print(Y[0:5])
    train_test_split = (int)(0.8 * len(X))
    X_train = X[0:(int)(0.8 * len(X))]
    #Y_train = Y[0:(int)(0.8 * len(Y))]
    X_test = X[(int)(0.8 * len(X)):]
    #Y_test = Y[(int)(0.8 * len(Y)):]
    #X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    #prices = stock['close'].values[::-1].reshape(len(stock), 1)
    Ms = MinMaxScaler(feature_range=(0,1))

    #train_test_split = (int)(0.8 * len(prices))
    #prices_train = prices[0:train_test_split]
    #prices_test = prices[train_test_split:]

    prices_train_scaled = Ms.fit_transform(X_train)
    prices_test_scaled=prices_train_scaled#[:,6]
    #Y_train = Y_train.reshape(-1,1)
    #prices_test_scaled = Ms.fit_transform(Y_train)
    #print(prices_train_scaled.shape)
    #print(prices_test_scaled.shape)

    train = []
    #val = []
    for i in range(60, len(prices_train_scaled)):
        train.append(prices_train_scaled[i-60:i])
        #val.append(prices_test_scaled[i+60:])
    #print(train.shape)
    train = np.asarray(train,dtype=object).astype(np.float32)

    val = prices_test_scaled[60:]
    val = np.asarray(val,dtype=object).astype(np.float32)
    val = val[:, 0] # only predict stock price
    
    print(train[2:5, -1])
    print('fdg')
    print(val[2:5])
    
    model = keras.models.Sequential()
    model.add(keras.layers.LSTM(units=50,return_sequences=True,input_shape=(60, 4)))
    model.add(keras.layers.Dropout(0.2))
    #model.add(keras.layers.LSTM(units=50,return_sequences=True))
    #model.add(keras.layers.Dropout(0.2))
    #model.add(keras.layers.LSTM(units=50,return_sequences=True))
    #model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.LSTM(units=50))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(units=1))
    model.compile(optimizer='adam',loss='mean_squared_error',metrics=[keras.metrics.RootMeanSquaredError(name='rmse'), 'mean_absolute_error'])
    model.fit(train,val,epochs=100,batch_size=32)

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
    pre_test_scaled=pre_train_scaled
    #Y_test = Y_test.reshape(-1,1)
    #pre_test_scaled = Ms.fit_transform(Y_test)

    input = []
    for i in range(60, len(pre_train_scaled)):
        input.append(pre_train_scaled[i-60:i])

    test = pre_test_scaled[60:]
    input, test = np.asarray(input,dtype=object).astype(np.float32), np.asarray(test,dtype=object).astype(np.float32)

    predicted_stock_price = np.zeros((input.shape[0], input.shape[2]))
    predicted_stock_price[:, 0] = model.predict(input)[:, 0]
    predicted_stock_price = Ms.inverse_transform(predicted_stock_price)
    predicted_stock_price = pd.DataFrame({'close': predicted_stock_price[:, 0]}, index=X_test.index[60:])

    #print(input[2:5])
    print(predicted_stock_price[2:5])
    print('fdg')
    print(X_test[['close']][2:5])

    model.evaluate(input, np.array(test)[:, 0])

    #print(predicted_stock_price)
    plt.plot(X_test[['close']][60:], color = 'black', label = '%s Stock Price' % (s))
    plt.plot(predicted_stock_price, color = 'green', label = 'Predicted %s Stock Price' % (s))
    plt.title('%s Stock Price Prediction' % (s))
    plt.xlabel('Time')
    plt.ylabel('%s Stock Price' % (s))
    plt.legend()
    plt.show()
    plt.savefig('nlp.png')