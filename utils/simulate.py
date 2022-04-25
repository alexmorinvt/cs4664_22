from model import Model


class Simulation:
    """Run inference based on the given scenario."""    

    def __init__(self, fees: list, principal: float):
        """Setup a simulation."""
        self.fees = fees
        self.principal = principal
        self.reset()


    def reset(self):
        """Load original configuration."""
        self.portfolio = [0]*len(self.fees) + [self.principal]


    def act(self, action, xchg, convert):
        """Perform the given action."""
        # TODO: conversion
        # TODO: better simulation
        amt = action[0]
        if convert:
            if amt > 0:
                amt *= self.portfolio[-1] / xchg
            elif amt < 0:
                amt *= self.portfolio[0]
        self.portfolio[0] += amt
        self.portfolio[1] -= amt * xchg
        assert(p >= 0 for p in self.portfolio)


    def value(self, xchg):
        """Value of the current portfolio."""
        return self.portfolio[0] * xchg + self.portfolio[1]


def evaluate(model: Model, sim: Simulation, train_val, index):
    """Train and validate a model."""
    from utils.data import train, valid
    
    # Train the model
    model.train(*train(*train_val, index))

    # Validate the model
    totals = [sim.portfolio[-1]]
    for stock_test, text_test in valid(*train_val, index):
        action = model.test(stock_test, text_test, sim.portfolio)
        xchg = stock_test[0].iloc[-1]['close']
        sim.act(action, xchg, model.convert)
        totals.append(sim.value(xchg))

    # Liquidate all assets
    print(f"[ {sim.portfolio[0]:.3f} NFLX,\t ${sim.portfolio[1]:.2f} ]\tTotal: ${totals[-1]:.2f}")
    sim.reset()
    return totals
