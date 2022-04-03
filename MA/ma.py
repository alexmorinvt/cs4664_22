import pandas as pd

stonks = ['INTC', 'MMM', 'UAL', 'NFLX']

INTC = pd.read_csv("../DATA/%s_daily.csv" % (stonks[0]))

prices = INTC['close']

print(prices.head)