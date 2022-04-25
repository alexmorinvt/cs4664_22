from typing import Type
from model import Model
from itertools import product
from utils.crossval import cross_validate
import os
from traceback import print_exc
from json import dump


SWEEP_DIR = "./SWEEPS/"

def sweep(model: Type[Model], **cross_args):
    """Grid-search over the hyperparameter space.

    Args:
        model: to evaluate according to its `config`.
        cross_args: kwargs for cross-validation.
        

    Produces:
        Validation results in a .json file.
    """
    results = {
        'model': model.__name__,
        'simulation': {
            'fees': cross_args['sim'].fees,
            'principal': cross_args['sim'].principal
        },
        'validation': {
            'method': cross_args['partition'].__name__,
            'split': cross_args['split'],
            'folds': cross_args['folds']
        },
        'sweep': list()
    }
    filename = SWEEP_DIR + results['model'] + ".json"
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Try every combination of parameters
    names, configs = zip(*model.config.items())
    for values in product(*(possible(config) for config in configs)):
        params = {name: value for name, value in zip(names, values)}
        print(f"\nParameters: {params}")
        
        # Run cross-validation with the chosen hyperparameters
        try:
            params['score'] = cross_validate(lambda fees: model(fees, **params), **cross_args)
            print(f"Result: ${params['score']:.2f}")
        except KeyboardInterrupt:
            break
        except Exception as e:
            params['error'] = str(e)
            print_exc()
        results['sweep'].append(params)
        
        # Save results to a file
        with open(filename, 'w') as outfile:
            dump(results, outfile, indent=4)
    
    else:
        print("Finished sweep!")
        return
    print("\nTerminated")


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
