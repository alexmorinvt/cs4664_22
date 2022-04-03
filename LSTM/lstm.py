import pandas as pd

import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import urllib.request, json
import os
import numpy as np
import tensorflow as tf # This code has been tested with TensorFlow 1.6
from sklearn.preprocessing import MinMaxScaler

stonks = ['INTC']
# , 'MMM', 'UAL', 'NFLX']

for s in stonks:
    stock = pd.read_csv("../DATA/%s_daily.csv" % (s))
    print(stock.head)
    prices = stock[['open', 'close']]
    Ms = MinMaxScaler()
    prices[prices.columns] = Ms.fit_transform(prices)
