import pandas as pd
import mplfinance as fplt

# Load data
stock = pd.read_csv("./DATA/NFLX_1min_2years.csv", parse_dates=True)
stock = stock.set_index(pd.DatetimeIndex(stock['time']))

# Create plot
fplt.plot(
    stock[100:25:-1],
    type='candle',
    style='charles',
    title='NFLX, April 01, 2022',
    ylabel='Price ($)',
    volume=True,
    ylabel_lower='Shares\nTraded',
)