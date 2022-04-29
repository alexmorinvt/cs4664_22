#!/usr/bin/env python3
"""CS4664: Data-Centric Computing - Final project.
Partly based on our work for MCM 2022 problem C.
Refer to README.md for installation instructions.
"""

import pandas as pd

# TODO: put this into a function
# TODO: command-line arguments
from utils.data import between

# NOTE: only NFLX (for now)
# NOTE: only specific section (for now)
names = ['NFLX']
date_start, date_end = '2021-09-07', '2021-09-23'

# TODO: download data

stock, text = [], []
for name in names:

    # Load stock data
    s = pd.read_csv(f"./DATA/{name}_1min_2years.csv")[::-1]
    s.columns.name = name
    s['date_time'] = pd.to_datetime(s.time)
    s.drop(columns=['Unnamed: 0', 'time'], inplace=True)
    s = between(s, date_start, date_end)
    stock.append(s)

    # Load text data
    t = pd.read_csv(f"./DATA/TEXT/{name}_bert_sen.csv", parse_dates=[['date', 'time']])[::-1]
    t.columns.name = name
    t.drop(columns=['Unnamed: 0'], inplace=True)
    t = between(t, date_start, date_end)
    text.append(t)

# Create new model
# TODO: choose which model to load
from example_models import Null, Hold
from tcn_model import TCN_
from combine_example import Combine

# Evaluate the model
from utils.simulate import Simulation, evaluate
from utils.crossval import none
sim = Simulation([0.0], 1000)
model, args = TCN_, {'filters': 32, 'ker_size': 8}
for fold, index in none((stock, text), 0.8):
    totals = evaluate(model(sim.fees, **args), sim, fold, index)

# Plot assets over time
import matplotlib.pyplot as plt
s, m = names[0], model.__name__
plt.plot(totals, color = 'black', label = m)
plt.title('%s Portfolio value using %s' % (s, m))
plt.xlabel('Time')
plt.ylabel('%s Portfolio value' % (s))
plt.legend()
plt.savefig('portfolio.pdf')
plt.show()

# Try hyperparameter tuning with TCN model
from utils.hyper import sweep
from utils.crossval import sliding
sweep(model, sim=sim, train_val=(stock, text), partition=sliding, split=0.8, folds=5)
