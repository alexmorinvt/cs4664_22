from math import log10
from utils.data import segment


def cross_validate(model_factory, baseline_factory, sim, train_val, partition, **part_args):
    """Run cross-validation with a given partition method.
    
    Args:
        model_factory: function producing model to evaluate.
        baseline_factory: model to compare against.
        sim: starting simulation scenario.
        train_val: data for training and validation.
        partition: method for partitioning the data.
        part_args: kwargs for partition function.
    
    Returns:
        Logarithm base 10 of ratio with baseline, else raw score.
    """
    from utils.simulate import evaluate
    scores = []
    for fold, index in partition(train_val, **part_args):
        score = evaluate(model_factory(sim.fees), sim, fold, index)[-1]
        if model_factory is not None:
            score /= evaluate(baseline_factory(sim.fees), sim, fold, index)[-1]
            score = log10(score)
        scores.append(score)
    return sum(scores) / len(scores)


length = lambda train_val: len(train_val[0][0])
index = lambda train_val, split: round(length(train_val) * split)


def none(train_val, split):
    """Simple validation (no cross-validation)."""
    yield train_val, index(train_val, split)


def _rolling(train_val, split, folds, func):
    """Helper function."""
    assert(folds > 1 and split > 0.5)
    total = length(train_val) 
    half = total // 2
    idx = index(train_val, split) - total + half
    for i in range(folds):
        shift = round((total - half) / (folds - 1) * i)
        start, id = func(idx, shift)
        yield segment(*train_val, start, half+shift), id


def expanding(train_val, split, folds):
    """Expanding window (from beginning of data)."""
    yield from _rolling(train_val, split, folds, lambda idx, shift: (0, idx+shift))


def sliding(train_val, split, folds):
    """Sliding window (equal length of data)."""
    yield from _rolling(train_val, split, folds, lambda idx, shift: (shift, idx))
