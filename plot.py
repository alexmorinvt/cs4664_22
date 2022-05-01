from matplotlib.pyplot import margins, tight_layout
import pandas as pd
import mplfinance as fplt

DIR = "./DISPLAY/transform/"


# Load data
stock = pd.read_csv("./DATA/NFLX_1min_2years.csv", parse_dates=True)
stock.set_index(pd.DatetimeIndex(stock['time']), inplace=True)
stock.drop(columns=['Unnamed: 0', 'time'], inplace=True)

# Original data
def candlestick(data, type, name, file):
    fplt.plot(
        data,
        type='candle',
        style='charles',
        title=f'{type} series; March 30, 2022',
        ylabel=f'{name} ($)',
        volume=True,
        ylabel_lower='Shares\nTraded',
        savefig=f'{DIR}{file}.pdf',
        scale_padding={'right':1.5, 'top':1.0, 'left':0.5, 'bottom':0.5},
    )
candlestick(stock[1080:1020:-1], 'Original', 'NFLX', 'original')

# Logarithmic transform
import numpy as np
for name in ['open', 'high', 'low', 'close', 'volume']:
    stock[name].iloc[:-1] = np.diff(np.log(stock[name]))
stock = stock[:-1]
candlestick(stock[1079:1020:-1], 'Transformed', 'diff(log(NFLX))', 'transformed')

# Welch periodogram
import matplotlib.pyplot as plt
from scipy.signal import welch
_, Pxx_spec = welch(stock[['close']].values.reshape(-1), 1.0, 'flattop', 20000, scaling='spectrum')
plt.semilogy(np.sqrt(Pxx_spec))
plt.subplots_adjust(left=0.2)
plt.xlabel('Period')
plt.ylabel('Linear spectrum [V RMS]')
plt.savefig(f"{DIR}periodogram.pdf")
plt.show()
