from model import Model
from DATA.utils import combine

class Combine(Model):
    """NLP data combining example."""

    def train(self, stock, text):
        """Combine data and print."""
        #combined = [s.dropna() for s in combine(stock, text, fillna=False)] # Also try fillna=True
        combined = combine(stock, text) # Also try fillna=True
        print(combined)
        return combined
    
    def test(self, stock, text, portfolio):
        """TODO: something."""
        return [0] * len(stock)