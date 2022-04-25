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
    for i in trange(index, len(stock[0])):
        yield segment(stock, text, 0, i+1)
        