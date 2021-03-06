from typing import List


class Model:
    """Interface for implementing models.

    Requires defining the following methods:
    * `__init__`: create new model
    * `train`: find model parameters
    * `test`: make trading predictions
    
    Requires defining the following variables:
    * `config`: hyperparameters for sweeps
    """
    config = {
        'null-train': {'min': 0, 'max': 0, 'by': 1, 'log': False, 'train': True},
        'null-test':  {'min': 0, 'max': 0, 'by': 1, 'log': False, 'train': False},
    }


    def __init__(self, fees: list, **hyper):
        """Default constructor.
        
        Create a new model with fresh weights.
        In particular, global variables are disallowed.
        
        self.fees: transaction fees of each stock.
        self.convert: if `test` returns values from -1 to 1.
        self.[param]: hyperparameters, for each param in hyper.
        """
        self.fees = fees
        self.convert = False
        self.__dict__.update(hyper)


    def train(self, stocks: list, texts: list) -> None:
        """Train the model.

        Receive all preceding data and update model state.

        Args:
            stock: list of dataframes with the following values:
                'date_time', 'open', 'high', 'low', 'close', 'volume'
            text: list of dataframes with the followming values:
                'date_time', 'headline', 'positive', 'neutral', 'negative'
        """
        raise NotImplementedError


    def test(self, stock: list, text: list, portfolio: List[float], **hyper) -> List[float]:
        """Test the model.

        Receive a single data point and make one prediction.

        Args:
            stock, text: same as in `train`.
                Only the last data point is new.
            portfolio: amounts of each resource available.
                0-n: stock amounts, n+1: cash amount.
                Uses latest 'close' as current exchange rate.
            hyper: inference hyperparameters.
        
        Returns:
            Amount to buy/sell for each stock
        """
        raise NotImplementedError


    def test_all(self, stock, text, index, **hyper):
        """Super secret function (test model in parallel).
        
        Receive all test data and do necessary preprocessing.
        Portfolios will still be visible through later calls to `test`.

        WARNING: this makes it really easy to introduce bias.
        Make sure you know what you're doing!

        Args:
            stock, text: same as in `test`.
                Only the last of the provided points.
            index: location of train-validation split.
            hyper: inference hyperparameters.
        """
        pass
