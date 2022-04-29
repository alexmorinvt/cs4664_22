import pandas as pd

def combine(stock, text, fillna=False):
    combined = [pd.merge(s, t, how='outer', on='date_time', sort=True) for s, t in zip(stock, text)]
    if fillna:
        combined = [c.fillna(method='ffill').dropna() for c in combined]
    else:
        columns = ['open', 'high', 'low', 'close', 'volume']
        for c in combined:
            c[columns] = c[columns].fillna(method='ffill')
    return combined