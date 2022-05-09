from model import Model

from tcn import TCN     # file is named tcn_ to avoid conflict with this package
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow import keras


class TCN_(Model):
    """Stock predict using TCN, use on algo-trading."""
    config = {
        'filters': {'min': 16, 'max': 64, 'by': 2, 'log': True, 'train': True},
        'ker_size': {'min': 2, 'max': 8, 'by': 2, 'log': True, 'train': True},
        'window': {'min': 25, 'max': 100, 'by': 25, 'log': False, 'train': True},
        'all_in': {'min': 0, 'max': 1, 'by': 1, 'log': False, 'train': False},
        'alpha': {'min': 1e2, 'max': 128e2, 'by': 2, 'log': True, 'train': False},
    }


    def train(self, stocks, texts):
        """Train a TCN to predict stock prices."""
        self.convert = True

        if not stocks: return
        stock = stocks[0]
        prices = stock['close'].values[::].reshape(len(stock), 1)
        self.Ms = StandardScaler(with_mean=False)
        
        # Log transform + 1st order differencing
        prices_scaled = np.diff(np.log(prices), axis=0)
        prices_scaled = self.Ms.fit_transform(prices_scaled)

        X_train = []
        y_train = []
        for i in range(self.window, len(prices_scaled)):
            X_train.append(prices_scaled[i-self.window:i])
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


    def test(self, stock, text, portfolio, alpha=1e3, all_in=False):
        """Predict next day's price and trade."""
        predicted_diff = self.pred[self.test_idx]
        self.test_idx += 1
        fun = round if all_in else lambda x: x
        return [fun(np.tanh(alpha*predicted_diff).item())] * len(stock)


    def test_all(self, stock, text, index, **hyper):
        """Predict in batches using the model."""
        self.test_idx = 0
        if hasattr(self, 'pred'): return
        if not stock: return
        stock = stock[0]
        self.pred = np.zeros((stock.shape[0]-index, 1))
        prices = stock['close'].values[::].reshape(len(stock), 1)
        inputs = prices[-(self.window+self.pred.shape[0]):]
        inputs = np.diff(np.log(inputs), axis=0)
        inputs = self.Ms.transform(inputs)
        X_test = [inputs[i-self.window:i] for i in range(self.window, len(inputs))]
        predicted_diff = self.model(np.array(X_test))
        predicted_diff = self.Ms.inverse_transform(predicted_diff)
        self.pred[-predicted_diff.shape[0]:] = predicted_diff
