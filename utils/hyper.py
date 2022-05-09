from gettext import find
from typing import Type
from itertools import product
from click import confirm
from traceback import print_exc
from json import dump, load, JSONDecodeError
import os

from utils.crossval import cross_validate
from models import Model
from models import Hold


SWEEP_DIR = "./SWEEPS/"

def sweep(model: Type[Model], eval, **cross_args):
    """Grid-search over the hyperparameter space.

    Args:
        model: to evaluate according to its `config`.
        cross_args: kwargs for cross-validation.
        
    Returns:
        Tuple of best train, test hyperparameters.

    Produces:
        Validation results in a .json file.
    """
    results = gen_config(model, cross_args)
    filename = SWEEP_DIR + results['model'] + ".json"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    previous = check_previous(filename, results)

    # Try every combination of parameters
    best = (-float('inf'), {}, {})
    for params in all_possible(model.config, True):
        print(f"\nParameters: {params}")
        
        # Check if result already exists
        found = None
        def not_found(item):
            nonlocal found
            if 'scores' not in item:
                return False
            score = item['scores']
            del item['scores']
            if fnd := item == params:
                print("Found in previous sweep")
                found = score
            item['scores'] = score
            return not fnd
        previous = list(filter(not_found, previous))

        # Run cross-validation with the chosen hyperparameters
        try:
            if not found:
                found = list(all_possible(model.config, False))
                scores = cross_validate(lambda fees: model(fees, **params), model_hypers=found, **cross_args)
                for pars, score in zip(found, scores):
                    pars['score'] = score
            for pars in found:
                score = pars['score']
                del pars['score']
                best = max(best, (eval(score), dict(params), dict(pars)))
                pars['score'] = score
            params['scores'] = found
        except KeyboardInterrupt:
            break
        except Exception as e:
            params['error'] = str(e)
            print_exc()
        results['sweep'].append(params)
        
        # Save results to a file
        #with open(filename, 'w') as outfile:
        #    dump(results, outfile, indent=4)
    
    else:
        print("Finished sweep!")
        return best[1:]
    print("\nTerminated")


def all_possible(model_config, train=True):
    conf = {name: config for name, config in model_config.items() if config['train'] == train}
    names, configs = zip(*conf.items())
    for values in product(*(possible(config) for config in configs)):
        yield {name: value for name, value in zip(names, values)}


def possible(config):
    """Get possible values of a parameter based on its configuration."""
    
    # Validate config
    for name in ['min', 'max', 'by', 'log']:
        assert(name in config)
    
    # Generate values
    from operator import iadd, imul
    op = imul if config['log'] else iadd
    curr = config['min']
    while curr <= config['max']:
        yield curr
        curr = op(curr, config['by'])


def gen_config(model: Type[Model], cross_args):
    """Generate sweep configuration for logging."""
    stock, text = cross_args['train_val']
    assert(len(stock) == len(text))
    for s, t in zip(stock, text):
        assert(s.columns.name == t.columns.name)
    if 'baseline_factory' not in cross_args:
        cross_args['baseline_factory'] = lambda fees: Hold(fees)
    return {
        'model': model.__name__,
        'simulation': {
            'fees': cross_args['sim'].fees,
            'principal': cross_args['sim'].principal,
        },
        'validation': {
            'method': cross_args['partition'].__name__,
            'split': cross_args['split'],
            'folds': cross_args['folds'],
        },
        'data': {
            'stock': [s.columns.name for s in stock],
            'len_stock': [len(s) for s in stock],
            'len_text': [len(t) for t in text],
            'begin': str(min((s.iloc[0].date_time for s in stock), default=None)),
            'end': str(max((s.iloc[-1].date_time for s in stock), default=None)),
        },
        'baseline': cross_args['baseline_factory'](cross_args['sim'].fees).__class__.__name__
            if cross_args['baseline_factory'] is not None else 'None',
        'sweep': list(),
    }


def check_previous(filename, results):
    """Check for existing sweep and its validity."""
    previous = []
    if os.path.exists(filename):
        print(f"Found existing sweep {filename}")
        prev_results = None
        with open(filename, 'r') as infile:
            try:
                prev_results = load(infile)
            except JSONDecodeError:
                print(f"ERROR: could not load sweep")
        if prev_results:
            previous = prev_results['sweep']
            prev_results['sweep'] = list()
            if prev_results != results:
                print(f"WARNING: does not match current configuration")
                previous = []
        if not previous:
            if not confirm(f"Delete {filename}?"):
                print(f"Exiting...")
                raise SystemExit
    return previous
