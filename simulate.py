#!/usr/bin/env python3
"""CS4664: Data-Centric Computing - Final project.
Partly based on our work for MCM 2022 problem C.
Refer to README.md for installation instructions.
"""

# TODO: command-line arguments
# NOTE: only NFLX (for now)
# NOTE: only specific section (for now)
from utils.data import load_data
names = ['NFLX']
stock, text = load_data(names, '2021-09-07', '2021-09-23')

# Create new model
# TODO: choose which model to load
from example_models import Null, Hold
from tcn_model import TCN_
from combine_example import Combine

# Evaluate the model
from utils.simulate import Simulation, evaluate
from utils.crossval import none
sim = Simulation([0.0], 1000)
model, args, kwargs = TCN_, {'filters': 32, 'ker_size': 8, 'window': 60}, {'alpha': 1e3, 'all_in': True}
for fold, index in none((stock, text), 0.8):
    totals = evaluate(model(sim.fees, **args), sim, fold, index, **kwargs)

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
