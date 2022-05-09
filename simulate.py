#!/usr/bin/env python3
"""CS4664: Data-Centric Computing - Final project.
Partly based on our work for MCM 2022 problem C.
Refer to README.md for installation instructions.
"""

import pandas as pd
import numpy as np
from tqdm import tqdm

# TODO: put this into a function
# TODO: command-line arguments
date_start, date_end = '2016-08-23', '2023-12-10'

# Load stock data
# NOTE: only NFLX (for now)
# TODO: download data
stock = [pd.read_csv("./DATA/NFLX_daily.csv")[::-1]]
# NOTE: only specific section (for now)
for s in stock:
    s['date_time'] = pd.to_datetime(s.timestamp)
stock = [s.drop(columns=['Unnamed: 0', 'timestamp']) for s in stock]
between = lambda d, s, e: d[(s <= d.date_time) & (d.date_time <= e)]
stock = [between(s, date_start, date_end) for s in stock]

# Load text data
#text = [pd.read_csv("./DATA/TEXT/netflix_bert_sen.csv", parse_dates=[['date', 'time']])[::-1]]
text = [pd.read_csv("./DATA/netflix.csv")[::-1]]
for t in text:
    t['date_time'] = pd.to_datetime(s.date_time)
text = [t.drop(columns=['Unnamed: 0']) for t in text]
text = [between(t, date_start, date_end) for t in text]

# Split into train and test
# TODO: rolling cross-validation
split = round(len(stock[0]) * 0.8)
stock_train = [s[:split] for s in stock]
text_match = lambda stock_data: [between(t, s.iloc[0].date_time, s.iloc[-1].date_time) for s, t in zip(stock_data, text)]
text_train = text_match(stock_train)

# Create new model
# TODO: choose which model to load
# from example_models import Null, Hold
# from tcn_model import TCN_
from combine_example import Combine
model = Combine([0.0])

# Train model
table=model.train(stock_train, text_train)
#df = pd.DataFrame(np.asarray(table).reshape(99, 10), columns=['open','high','low','close','volume','date_time','headline','positive','negative','neutral'])
df = pd.DataFrame(np.asarray(table).reshape(1130, 7), columns=['open','high','low','close','volume','date_time','title'])
df.to_csv('./nlpZoe/new_combine.csv')

# Test model
# TODO: conversion
# TODO: better simulation
portfolio = [0, 1000]
totals = [1000]
for i in range(len(stock_train[0]), len(stock[0])):
    stock_test = [s[:i+1] for s in stock]
    action = model.test(stock_test, None, portfolio)
    xchg = stock_test[0].iloc[-1]['close']
    amt = action[0]
    if model.convert:
        if amt > 0:
            amt *= portfolio[-1] / xchg
        elif amt < 0:
            amt *= portfolio[0]
    portfolio[0] += amt
    portfolio[1] -= amt * xchg
    assert(p >= 0 for p in portfolio)
    totals.append(portfolio[0] * xchg + portfolio[1])

# Liquidate all assets
xchg = stock[0].iloc[-1]['close']
total = portfolio[0] * xchg + portfolio[1]
print(f"[ {portfolio[0]:.3f} NFLX,\t ${portfolio[1]:.2f} ]\tTotal: ${total:.2f}")

# Plot assets over time
import matplotlib.pyplot as plt
s = 'INTL'
plt.plot(totals, color = 'black', label = 'TCN')
plt.title('%s Portfolio value using TCN' % (s))
plt.xlabel('Time')
plt.ylabel('%s Portfolio value' % (s))
plt.legend()
plt.show()