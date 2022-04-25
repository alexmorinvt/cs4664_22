between = lambda d, s, e: d[(s <= d.date_time) & (d.date_time <= e)]
text_match = lambda stock_data, text: [between(t, s.iloc[0].date_time, s.iloc[-1].date_time) for s, t in zip(stock_data, text)]


def train(stock, text, split):
    """Get the training dataset."""
    idx = round(len(stock[0]) * split)
    stock_train = [s[:idx] for s in stock]
    text_train = text_match(stock_train, text)
    return stock_train, text_train


def valid(stock, text, split):
    """Get each validation point."""
    train_len = len(train(stock, text, split)[0][0])
    for i in range(train_len, len(stock[0])):
        stock_val = [s[:i+1] for s in stock]
        text_val = text_match(stock_val, text)
        yield stock_val, text_val
        