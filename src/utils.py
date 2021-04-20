import os
from functools import partial
import optuna
from optuna.samplers import TPESampler
os.environ['PYTHONHASHSEED'] = '0'

'''
optuna.trial.Trial:
https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html
e.g. objective_args.
objective_args = {
    'num_leaves': {
        'type': 'int',
        'suggest_args': {
            'name': 'num_leaves',
            'low': 2,
            'high': 128,
        }
    },
    'max_depth': {
        'type': 'int',
        'suggest_args': {
            'name': 'max_depth',
            'low': 3,
            'high': 8,
        }
    },
    'min_data_in_leaf': {
        'type': 'int',
        'suggest_args': {
            'name': 'min_data_in_leaf',
            'low': 5,
            'high': 90,
        }
    },
    'n_estimators': {
        'type': 'int',
        'suggest_args': {
            'name': 'n_estimators',
            'low': 100,
            'high': 500,
        }
    },
    'learning_rate': {
        'type': 'uniform',
        'suggest_args': {
            'name': 'learning_rate',
            'low': 0.0001,
            'high': 0.1
        }
    },
    'bagging_fraction': {
        'type': 'uniform',
        'suggest_args': {
            'name': 'bagging_fraction',
            'low': 0.0001,
            'high': 1.0,
        }
    },
    'feature_fraction': {
        'type': 'uniform',
        'suggest_args': {
            'name': 'feature_fraction',
            'low': 0.0001,
            'high': 1.0,
        }
    },
    'random_seed': {
        'type': 'default',
        'value': 0
    }
}
e.g. pipeline_args.
pipeline_args = {
    'fit_attr': 'fit',
    'pred_attr': 'predict_proba',
    'fit_args': {'X': x_train, 'y': y_train},
    'pred_args': {'X': x_test},
    'metric': lambda true, pred: -log_loss(true, pred[:, 1]),
    'metric_args': {'true': y_test},
    'model': lightgbm.LGBMClassifier,
}
'''

class Optuna_for_LGB:
    def __init__(self):
        self.best_params = dict()
        self.best_iterations = dict()
        self.early_stop = False
    
    def create_model(self, trial, model, objective_args):
        params = dict()
        # params['random_state'] = seed
        for name, param in objective_args.items():
            if param['type'] == 'default':
                params[name] = param['value']
            else:
                params[name] = getattr(trial, 'suggest_{0}'.format(param['type']))(**param['suggest_args'])
        return model(**params)

    def objective(self, trial, fit_attr, pred_attr, fit_args, pred_args, metric, metric_args, model, objective_args):
        model = self.create_model(trial, model, objective_args)
        # fit
        getattr(model, fit_attr)(**fit_args)
        if self.early_stop:
            best_iteration = model.best_iteration_
            pred_args['num_iteration'] = best_iteration
            self.best_iterations[trial.number] = best_iteration
        # predict
        pred = getattr(model, pred_attr)(**pred_args)
        metric_args['pred'] = pred
        return metric(**metric_args)    

    def parameter_tuning(self, pipeline_args, objective_args, n_trials, n_jobs, seed):
        if 'early_stopping_rounds' in pipeline_args['fit_args'].keys():
            self.early_stop = True
        pipeline_args['objective_args'] = objective_args
        sampler = TPESampler(seed=seed)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(partial(self.objective, **pipeline_args), n_trials=n_trials, n_jobs=n_jobs)
        self.best_params = study.best_params.copy()
        for name, param in objective_args.items():
            if param['type'] == 'default':
                self.best_params[name] = param['value']
        if self.early_stop:
            self.best_params['best_iteration_'] = self.best_iterations[study.best_trial.number]
        return self.best_params