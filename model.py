from typing import List

class Model:
    """Interface for implementing models.
    Requires defining the following methods:
    * `__init__`: create new model
    * `train`: find model parameters
    * `test`: make trading predictions
    """

    def __init__(self, fees: list):
        """Default constructor.
        
        Create a new model with fresh weights.
        In particular, global variables are disallowed.
        
        self.fees: transaction fees of each stock.
        self.convert: if `test` returns values from -1 to 1.
        """
        self.fees = fees
        self.convert = False

    def train(self, stocks: list, texts: list) -> None:
        """Train the model.

        Receive all preceding data and update model state.

        Args:
            stock: list of dataframes with the following values:
                'time', 'open', 'high', 'low', 'close', 'volume'
            text: list of TODO
        """
        raise NotImplementedError
    
    def test(self, stock: list, text: list, portfolio: List[float]) -> List[float]:
        """Test the model.

        Receive a single data point and make one prediction.

        Args:
            stock, text: same as in `train`.
                Only the last data point is new.
            portfolio: amounts of each resource available.
                0-n: stock amounts, n+1: cash amount.
                Uses latest 'close' as current exchange rate.
        
        Returns:
            Amount to buy/sell for each stock
        """
        raise NotImplementedError