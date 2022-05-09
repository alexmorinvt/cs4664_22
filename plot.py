import matplotlib.pyplot as plt
from matplotlib.pyplot import margins, tight_layout
import pandas as pd
import mplfinance as fplt
from pytz import all_timezones

DIR = "./DISPLAY/transform/"

if False:
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
    from scipy.signal import welch
    _, Pxx_spec = welch(stock[['close']].values.reshape(-1), 1.0, 'flattop', 20000, scaling='spectrum')
    plt.semilogy(np.sqrt(Pxx_spec))
    plt.subplots_adjust(left=0.2)
    plt.xlabel('Period')
    plt.ylabel('Linear spectrum [V RMS]')
    plt.savefig(f"{DIR}periodogram.pdf")
    plt.show()

# Alpha comparison
from json import load
with open('./SWEEPS/TCN_.json') as infile:
    for result in load(infile)['sweep']:
        if (result['filters'] == 16 and
            result['ker_size'] == 2 and
            result['window'] == 25):
            break
from statistics import stdev
mean_all_in, mean_some_in = [], []
std_all_in, std_some_in = [], []
for param in result['scores']:
    dest = [mean_all_in, std_all_in] if param['all_in'] else [mean_some_in, std_some_in]
    score = [(10**x - 1) * 100 for x in param['score']]
    mean, std = sum(score) / len(score), stdev(score)
    dest[0].append((param['alpha'], mean))
    dest[1].append((param['alpha'], mean-std, mean+std))
plt.xscale('log')
t = lambda l: zip(*l)
plt.plot(*t(mean_some_in), "C0", label="Gradual")
plt.plot(*t(mean_all_in), "C1", label="All in")
plt.fill_between(*t(std_some_in), color="C0", alpha=0.25)
plt.fill_between(*t(std_all_in), color="C1", alpha=0.25)
plt.legend()
plt.xlabel("Alpha")
plt.ylabel("Percent increase")
plt.savefig(f"./DISPLAY/alpha.pdf")
