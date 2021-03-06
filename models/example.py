from .model import Model


class Null(Model):
    """Do nothing."""

    def train(self, stock, text):
        """Nothing to fit."""
        pass
    
    def test(self, stock, text, portfolio, **hyper):
        """Nothing to do."""
        return [0] * len(stock)


class Hold(Model):
    """Go all in and hold forever."""

    def __init__(self, fees, **hyper):
        """Handle conversion elsewhere."""
        super().__init__(fees, **hyper)
        self.convert = True

    def train(self, stock, text):
        """Nothing to fit."""
        pass
    
    def test(self, stock, text, portfolio, **hyper):
        """Buy equal amounts."""
        return [portfolio[-1] and 1 / len(stock)] * len(stock)


class Combine(Model):
    """NLP data combining example."""

    def train(self, stock, text):
        """Combine data and print."""
        from utils.data import combine
        combined = combine(stock, text) # Also try fillna=True
        print(combined)

    def test(self, stock, text, portfolio):
        """TODO: something."""
        return [0] * len(stock)