from model import Model

class Null(Model):
    """Do nothing."""

    def train(self, stocks, texts):
        """Nothing to fit."""
        pass
    
    def test(self, stock, text, portfolio):
        """Nothing to do."""
        return [0] * len(stock)


class Hold(Model):
    """Go all in and hold forever."""

    def __init__(self, fees):
        """Handle conversion elsewhere."""
        self.convert = True

    def train(self, stocks, texts):
        """Nothing to fit."""
        pass
    
    def test(self, stock, text, portfolio):
        """Buy equal amounts."""
        return [portfolio[-1] and 1 / len(stock)] * len(stock)