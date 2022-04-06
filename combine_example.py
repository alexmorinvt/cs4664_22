from model import Model
from DATA.utils import combine

class Combine(Model):
    """NLP data combining example."""

    def train(self, stock, text):
        """Combine data and print."""
        combined = combine(stock, text) # Also try fillna=True
        print(combined)
    
    def test(self, stock, text, portfolio):
        """TODO: something."""
        return [0] * len(stock)