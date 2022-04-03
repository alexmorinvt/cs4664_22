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

    slow = 26
    fast = 12
    n_days_1 = 50
    n_days_2 = 100

    #calculating moving averages
    d = prices.ewm(span=slow, adjust=False, min_periods=26).mean()
    k = prices.ewm(span=fast, adjust=False, min_periods=12).mean()
    macd = d -k

    # #calculating 9 day average OF macd
    macd_s = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    macd_h = macd - macd_s

    long_ma = prices.ewm(span=slow, adjust=False, min_periods=n_days_2).mean()
    short_ma = prices.ewm(span=slow, adjust=False, min_periods=n_days_1).mean()

    portfolio = [1000.0, 0]
    x = 0
    for price, m, l, s in zip(prices, macd_h, long_ma, short_ma):
        if isnan(m) or isnan(price):
            continue

        if price > s and m > 0 and portfolio[0] > 0:
            # print(portfolio_value(portfolio, price))
            portfolio[1] = portfolio[0] / price
            portfolio[0] = 0
            # print(portfolio_value(portfolio, price))
        elif (price < s) and portfolio[1] > 0:
            portfolio[0] = portfolio[1] * price
            portfolio[1] = 0
        
        x += 1

    print("CURRENT STOCK: %s" % (s))
    print('Hold portfolio:\t\t%.2f' % (prices[len(prices) - 1] / prices[0] * 1000))
    print('MACD(%d, %d) portfolio:\t%.2f' % (slow, fast, portfolio[0] + portfolio[1] * prices[len(prices) - 1]))