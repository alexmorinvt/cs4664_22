#!/usr/bin/env python3
"""CS4664: Data-Centric Computing - Final project.
Partly based on our work for MCM 2022 problem C.
Refer to README.md for installation instructions.
"""

import pandas as pd
from tqdm import tqdm

# TODO: put this into a function
# TODO: command-line arguments
from utils.data import between, text_match, train, valid
date_start, date_end = '2021-09-07', '2021-09-23'

# Load stock data
# NOTE: only NFLX (for now)
# TODO: download data
stock = [pd.read_csv("./DATA/NFLX_1min_2years.csv")[::-1]]
# NOTE: only specific section (for now)
for s in stock:
    s['date_time'] = pd.to_datetime(s.time)
stock = [s.drop(columns=['Unnamed: 0', 'time']) for s in stock]
stock = [between(s, date_start, date_end) for s in stock]

# Load text data
text = [pd.read_csv("./DATA/TEXT/netflix_bert_sen.csv", parse_dates=[['date', 'time']])[::-1]]
text = [t.drop(columns=['Unnamed: 0']) for t in text]
text = [between(t, date_start, date_end) for t in text]

# Split into train and test
# TODO: rolling cross-validation
split = 0.8
stock_train, text_train = train(stock, text, split)

# Create new model
# TODO: choose which model to load
from example_models import Null, Hold
from tcn_model import TCN_
from combine_example import Combine
model = Hold([0.0])

# Train model
model.train(stock_train, text_train)

# Test model
# TODO: conversion
# TODO: better simulation
portfolio = [0, 1000]
totals = [1000]
for stock_test, text_test in valid(stock, text, split):
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
plt.savefig('portfolio.pdf')
plt.show()