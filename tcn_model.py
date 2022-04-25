from model import Model

import pandas as pd

from tcn import TCN, tcn_full_summary # file is named tcn_ to avoid conflict with this package
import datetime as dt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
import matplotlib.pyplot as plt


class TCN_(Model):
    """Stock predict using TCN, use on algo-trading."""
    config = {
        "filters": {"min": 8, "max": 128, "by": 2, "log": True},
        "ker_size": {"min": 1, "max": 16, "by": 2, "log": True},
    }


    def train(self, stocks, texts):
        """Train a TCN to predict stock prices."""
        stock = stocks[0]
        prices = stock['close'].values[::].reshape(len(stock), 1)
        self.Ms = MinMaxScaler(feature_range=(0,1))

        prices_scaled = self.Ms.fit_transform(prices)

        X_train = []
        y_train = []
        for i in range(60, len(prices_scaled)):
            X_train.append(prices_scaled[i-60:i])
            y_train.append(prices_scaled[i])
        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        
        self.model = keras.models.Sequential([
            TCN(input_shape=(X_train.shape[1], 1), nb_filters=self.filters,
                kernel_size=self.ker_size,
                use_skip_connections=False,
                use_batch_norm=False,
                use_weight_norm=False,
                use_layer_norm=False
            ),
            keras.layers.Dense(1, activation='linear')
        ])
        self.model.compile(optimizer='adam',loss='mean_squared_error')
        self.model.fit(X_train,y_train,epochs=20,batch_size=32)


    def test(self, stocks, texts, portfolio):
        """Predict next day's price and trade."""
        stock = stocks[0]
        prices = stock['close'].values[::].reshape(len(stock), 1)
        inputs = prices[-60:]
        inputs = inputs.reshape(-1,1)
        inputs = self.Ms.transform(inputs)
        X_test = []
        X_test.append(inputs)
        X_test = np.array(X_test)   
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        predicted_stock_price = self.model.predict(X_test)
        predicted_stock_price = self.Ms.inverse_transform(predicted_stock_price)

        # Convert price prediction trading decision
        last = prices[-1][0]
        diff = predicted_stock_price[0][0] / last if last else 1
        THRESH, MAG = 0.01, 0.5
        if abs(diff - 1) < THRESH:
            return [0] * len(stock)
        elif diff < 1:
            return [-portfolio[0] * MAG] * len(stock)
        else:
            return [portfolio[1] / last * MAG] * len(stock)
