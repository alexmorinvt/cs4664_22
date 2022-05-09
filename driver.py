#!/usr/bin/env python3
"""CS4664: Data-Centric Computing - Final project.
Partly based on our work for MCM 2022 problem C.
Refer to README.md for installation instructions.
"""
from models import TCN
from utils.crossval import none, sliding
from utils.simulate import Simulation, evaluate
from statistics import mean, median


# NOTE: only NFLX (for now)
# NOTE: only specific section (for now)
from utils.data import load_data
names = ['NFLX']
stock, text = load_data(names, '2021-09-07', '2021-09-23')
sim = Simulation([0.0], 1000)

# Create new model
# TODO: command-line arguments - choose which model to load
model = TCN
cval = sliding
avg = mean

# Run automatic hyperparameter tuning
from utils.hyper import sweep
args, kwargs = sweep(model, avg, sim=sim, train_val=(stock, text), partition=cval, split=0.8, folds=5)
print(args, kwargs)

# Evaluate the model
for fold, index in none((stock, text), 0.8):
    totals, = evaluate(model(sim.fees, **args), sim, fold, index, [kwargs])

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
