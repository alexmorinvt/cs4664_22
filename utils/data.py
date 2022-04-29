import pandas as pd
from os.path import exists

from api_requests import download_data


between = lambda d, s, e: d[(s <= d.date_time) & (d.date_time <= e)]
text_match = lambda stock_data, text: [between(t, s.iloc[0].date_time, s.iloc[-1].date_time) for s, t in zip(stock_data, text)]


def segment(stock, text, start, end):
    """Get a partition of the data."""
    stock_train = [s[start:end] for s in stock]
    text_train = text_match(stock_train, text)
    return stock_train, text_train


def train(stock, text, index):
    """Get the training dataset."""
    return segment(stock, text, 0, index)


def valid(stock, text, index):
    """Get each validation point."""
    from tqdm import trange
    for i in trange(index, len(stock[0]) if stock else 0):
        yield segment(stock, text, 0, i+1)


def combine(stock, text, fillna=False):
    combined = [pd.merge(s, t, how='outer', on='date_time', sort=True) for s, t in zip(stock, text)]
    if fillna:
        combined = [c.fillna(method='ffill').dropna() for c in combined]
    else:
        columns = ['open', 'high', 'low', 'close', 'volume']
        for c in combined:
            c[columns] = c[columns].fillna(method='ffill')
    return combined


def load_data(names, date_start, date_end):
    """Load (and download as needed) data for requested stocks."""
    stock, text = [], []
    for name in names:

        # Download stock data
        filename = f"./DATA/{name}_1min_2years.csv"
        if not exists(filename):
            print(f"Downloading stock data for {name}...")
            download_data([name])

        # Load stock data
        try:
            s = pd.read_csv(filename)[::-1]
            s.columns.name = name
            s['date_time'] = pd.to_datetime(s.time)
            s.drop(columns=['Unnamed: 0', 'time'], inplace=True)
            s = between(s, date_start, date_end)
            stock.append(s)
        except:
            print(f"WARNING: {filename} not found; skipping")

        # Download text data
        filename = f"./DATA/TEXT/{name}_bert_sen.csv"
        if not exists(filename):
            print(f"WARNING: Don't know how to download text data")
            # print(f"Downloading text data for {name}...")
            # TODO: text data download / parse

        # Load text data
        try:
            t = pd.read_csv(filename, parse_dates=[['date', 'time']])[::-1]
            t.columns.name = name
            t.drop(columns=['Unnamed: 0'], inplace=True)
            t = between(t, date_start, date_end)
            text.append(t)
        except:
            print(f"WARNING: {filename} not found; skipping")
    
    return stock, text
