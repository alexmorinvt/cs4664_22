#!/usr/bin/env python3
"""CS4664: Data-Centric Computing - Final project.
Partly based on our work for MCM 2022 problem C.
Refer to README.md for installation instructions.
"""

import pandas as pd
from tqdm import tqdm

# TODO: put this into a function
# TODO: command-line arguments

# Load stock data
# NOTE: only NFLX (for now)
# TODO: download data
stock = [pd.read_csv("./DATA/NFLX_1min_2years.csv")[::-1]]
# NOTE: only specific section (for now)
stock = [s[('2021-09-05' <= s.time) & (s.time <= '2021-09-22')] for s in stock]

# Load text data
# TODO: load text data

# Split into train and test
# TODO: rolling cross-validation
split = round(len(stock[0]) * 0.8)
stock_train = [s[:split] for s in stock]
text_train = [None]

# Create new model
# TODO: choose which model to load
from example_models import Null, Hold
from tcn_model import TCN_
model = TCN_([0.0])

# Train model
model.train(stock_train, text_train)

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