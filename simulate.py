#!/usr/bin/env python3
"""CS4664: Data-Centric Computing - Final project.
Partly based on our work for MCM 2022 problem C.
Refer to README.md for installation instructions.
"""

import pandas as pd
from tqdm import tqdm

# TODO: put this into a function
# TODO: command-line arguments
from utils.data import between
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

# Create new model
# TODO: choose which model to load
from example_models import Null, Hold
from tcn_model import TCN_
from combine_example import Combine
rates = [0.0]
principal = 1000
split = 0.8
model = Hold(rates)

# Evaluate the model
from utils.simulate import Simulation, evaluate
sim = Simulation(rates, principal)
totals = evaluate(sim, model)

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