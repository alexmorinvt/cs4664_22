import pandas as pd
from math import isnan

stonks = ['INTC', 'MMM', 'UAL', 'NFLX']

for s in stonks:
    stock = pd.read_csv("../DATA/%s_daily.csv" % (s))

    def portfolio_value(p, price):
        return p[0] + p[1] * price

    prices = pd.DataFrame(stock['close'])
    prices['close'] = prices['close'].values[::-1]
    prices = prices['close']

    n_days = 50
    mas = prices.rolling(n_days).mean()

    portfolio = [1000.0, 0]
    x = 0
    for price, ma in zip(prices, mas):
        if isnan(ma) or isnan(price):
            continue

        if price > ma and portfolio[0] > 0:
            # print(portfolio_value(portfolio, price))
            portfolio[1] = portfolio[0] / price
            portfolio[0] = 0
            # print(portfolio_value(portfolio, price))
        elif portfolio[1] > 0:
            portfolio[0] = portfolio[1] * price
            portfolio[1] = 0
        
        x += 1

    print("CURRENT STOCK: %s" % (s))
    print('Hold portfolio:\t\t%.2f' % (prices[len(prices) - 1] / prices[0] * 1000))
    print('MA(%d) portfolio:\t%.2f' % (n_days, portfolio[0] + portfolio[1] * prices[len(prices) - 1]))