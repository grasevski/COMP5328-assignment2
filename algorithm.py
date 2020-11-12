#!/usr/bin/env python3
"""Build transition matrix estimator and classifiers."""
import argparse
import copy
import csv
import datetime
import functools
import itertools
import os
import random
import sys
from typing import Callable, Dict, List, Optional, Tuple
import jsonlines
import lightgbm
import optuna
from optuna.trial import Trial
import optuna.integration.lightgbm as lgb
from optuna.integration import PyTorchLightningPruningCallback
import numpy as np
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import torch
from torch import nn, Tensor, from_numpy, no_grad
import torch.nn.functional as F
from torch.optim import Optimizer, Adam
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

Model = Callable[[np.ndarray], np.ndarray]
Params = Dict[str, any]
Net = Callable[[int, int, Params], nn.Module]
Transform = Callable[[Tensor], Tensor]
Tuner = Callable[[int, int, Trial], Params]


class Backward:
    """Use inverse transition matrix to denoise."""
    def __init__(self, algorithm):
        """Training and tuning interface."""
        self._algorithm = algorithm

    def train(self,
              params: Params,
              X: np.ndarray,
              y: np.ndarray,
              X_val: np.ndarray,
              y_val: np.ndarray,
              _: Optional[np.ndarray] = None) -> None:
        """Train and validate using given hyperparams."""
        self._model = self._algorithm.train(params, X, y, X_val, y_val)

    def tune(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray,
             y_val: np.ndarray) -> Params:
        """Find optimal hyperparams for given train/val split."""
        return self._algorithm.tune(X, y, X_val, y_val)

    def __call__(self,
                 X: np.ndarray,
                 T: Optional[np.ndarray] = None,
                 denoise: bool = False) -> np.ndarray:
        """Predict, with flag to indicate whether to denoise."""
        ret = self._model(X)
        if denoise:
            ret = softmax(np.linalg.pinv(T) @ ret.T, axis=0).T
        return ret


class Lgbm:
    """Interface for training and tuning lightgbm."""
    @staticmethod
    def train(params: Params, X: np.ndarray, y: np.ndarray, X_val: np.ndarray,
              y_val: np.ndarray) -> Model:
        """Training with early stopping on validation set."""
        p, dtrain, dval = Lgbm._init(X, y, X_val, y_val)
        params.update(p)
        return lightgbm.train(params,
                              dtrain,
                              valid_sets=dval,
                              verbose_eval=False).predict

    def tune(X: np.ndarray, y: np.ndarray, X_val: np.ndarray,
             y_val: np.ndarray) -> Params:
        """Hyperparam tuning using optuna lightgbm integration."""
        params, dtrain, dval = Lgbm._init(X, y, X_val, y_val)
        return lgb.train(params, dtrain, valid_sets=dval,
                         verbose_eval=False).params

    @staticmethod
    def _init(X: np.ndarray, y: np.ndarray, X_val: np.ndarray,
              y_val: np.ndarray) -> Tuple[Params, lgb.Dataset, lgb.Dataset]:
        params = {
            'objective': 'softmax',
            'metric': 'softmax',
            'verbosity': -1,
            'force_row_wise': True,
            'early_stopping_round': 1,
            'num_class': int(max(y) + 1)
        }
        dtrain, dval = lgb.Dataset(X, y), lgb.Dataset(X_val, y_val)
        return params, dtrain, dval


class LR:
    """Interface for training and tuning logistic regression."""
    @staticmethod
    def train(params: Params, X: np.ndarray, y: np.ndarray, X_val: np.ndarray,
              y_val: np.ndarray) -> Model:
        """Wrap sklearn training interface."""
        return LogisticRegression(n_jobs=-1, **params).fit(X, y).predict_proba

    @staticmethod
    def tune(X: np.ndarray, y: np.ndarray, X_val: np.ndarray,
             y_val: np.ndarray) -> Params:
        """Hyperparam tuning using generic optuna integration."""
        study = optuna.create_study(direction='maximize')
        f = functools.partial(LR._objective, X, y, X_val, y_val)
        study.optimize(f, n_trials=100, n_jobs=-1)
        return study.best_params

    @staticmethod
    def _objective(X: np.ndarray, y: np.ndarray, X_val: np.ndarray,
                   y_val: np.ndarray, trial: Trial) -> float:
        C = trial.suggest_loguniform('C', 1e-9, 1e9)
        solvers = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
        solver = trial.suggest_categorical('solver', solvers)
        multi_class = trial.suggest_categorical('multi_class', ['auto', 'ovr'])
        model = LogisticRegression(C=C,
                                   solver=solver,
                                   multi_class=multi_class,
                                   n_jobs=-1)
        model.fit(X, y)
        return top1_accuracy(model.predict_proba(X_val), y_val)


class NeuralNet:
    """For use as a black box classifier."""
    def __init__(self, build: Net, tuner: Optional[Tuner] = None):
        """Initialize neural network configuration."""
        self._build, self._tune = build, tuner

    def train(self,
              params: Params,
              X: np.ndarray,
              y: np.ndarray,
              X_val: np.ndarray,
              y_val: np.ndarray,
              callbacks: List[pl.Callback] = []) -> Model:
        """Train using the backwards method."""
        model = NeuralNet.build(self._build, params, X, y)
        NeuralNet.do_training(model, X, y, X_val, y_val, callbacks=callbacks)

        def predict(X: np.ndarray) -> np.ndarray:
            with no_grad():
                return softmax(model(from_numpy(X)).numpy(), axis=1)

        return predict

    def tune(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray,
             y_val: np.ndarray) -> Params:
        """Optuna hyperparam tuning."""
        if not self._tune:
            return {}
        pruner = optuna.pruners.MedianPruner()
        study = optuna.create_study(direction='maximize', pruner=pruner)
        f = functools.partial(self._objective, X, y, X_val, y_val)
        study.optimize(f, n_trials=100, n_jobs=-1)
        return study.best_params

    def _objective(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray,
                   y_val: np.ndarray, trial: Trial) -> float:
        metrics_callback = MetricsCallback()
        callbacks = [
            metrics_callback,
            PyTorchLightningPruningCallback(trial, monitor='val_acc')
        ]
        in_dim, out_dim = NeuralNet._dimensions(X, y)
        model = self.train(self._tune(in_dim, out_dim, trial),
                           X,
                           y,
                           X_val,
                           y_val,
                           callbacks=callbacks)
        return metrics_callback.metrics[-1]['val_acc'].item()

    @staticmethod
    def build(builder: Net, params: Params, X: np.ndarray,
              y: np.ndarray) -> nn.Module:
        """Construct neural network according to required dimensions."""
        in_dim, out_dim = NeuralNet._dimensions(X, y)
        return builder(in_dim, out_dim, params)

    @staticmethod
    def _dimensions(X: np.ndarray, y: np.ndarray) -> Tuple[int, int]:
        return X.shape[1], int(max(y)) + 1

    @staticmethod
    def do_training(model: nn.Module,
                    X: np.ndarray,
                    y: np.ndarray,
                    X_val: np.ndarray,
                    y_val: np.ndarray,
                    transform: Optional[Transform] = None,
                    callbacks: List[pl.Callback] = []) -> None:
        """Main neural network training loop."""
        params = {
            'callbacks': callbacks + [EarlyStopping('val_acc')],
            'checkpoint_callback': False,
            'deterministic': True,
            'fast_dev_run': FAST_DEV_RUN,
            'logger': False,
            'progress_bar_refresh_rate': 0,
            'weights_summary': None,
        }
        if DEVICE == 'cuda':
            params['accelerator'] = 'ddp'
            params['auto_select_gpus'] = True
            params['gpus'] = -1
            params['precision'] = 16
        elif DEVICE != 'cpu':
            params['accelerator'] = 'ddp'
            params['precision'] = 16
            params['tpu_cores'] = TPU_CORES
        trainer = pl.Trainer(**params)
        train_dl = NeuralNet._data_loader(X, y)
        val_dl = NeuralNet._data_loader(X_val, y_val)
        trainer.fit(NeuralNetWrapper(model, transform), train_dl, val_dl)

    @staticmethod
    def _data_loader(X: np.ndarray, y: np.ndarray) -> DataLoader:
        return DataLoader(TensorDataset(from_numpy(X), from_numpy(y)),
                          batch_size=1024)


class Forward:
    """Append transition matrix to neural network during training."""
    def __init__(self, build: Net, tuner: Optional[Tuner] = None):
        """Wrap neural net architecture in generic interface."""
        self._build, self._tune = build, tuner

    def backward(self) -> NeuralNet:
        """Convert to backwards method."""
        return NeuralNet(self._build, self._tune)

    def train(self,
              params: Params,
              X: np.ndarray,
              y: np.ndarray,
              X_val: np.ndarray,
              y_val: np.ndarray,
              T: Optional[np.ndarray] = None,
              reuse: bool = False) -> None:
        """Train with or without noise, depending on params."""
        if not reuse:
            self._model = NeuralNet.build(self._build, params, X, y)
        if T is None:
            NeuralNet.do_training(self._model, X, y, X_val, y_val)
            return
        T = from_numpy(T).to(DEVICE)
        sm = nn.Softmax(dim=1)

        def transform(x: Tensor, T: Tensor = T) -> Tensor:
            return sm(T @ sm(x).T).T

        NeuralNet.do_training(self._model, X, y, X_val, y_val, transform)

    def tune(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray,
             y_val: np.ndarray) -> Params:
        """Tune according to noisy classification accuracy."""
        return self.backward().tune(X, y, X_val, y_val)

    def __call__(self,
                 X: np.ndarray,
                 T: Optional[np.ndarray] = None,
                 denoise: bool = False) -> np.ndarray:
        """Predict, using transition matrix as necessary."""
        with no_grad():
            ret = softmax(self._model(from_numpy(X)).numpy(), axis=1)
        if T is not None and not denoise:
            ret = softmax(T @ ret.T, axis=0).T
        return ret


class MetricsCallback(pl.Callback):
    """PyTorch Lightning metric callback."""
    def __init__(self):
        """Record metrics in array."""
        super().__init__()
        self.metrics = []

    def on_validation_end(self, trainer, pl_module):
        """Append metrics."""
        self.metrics.append(trainer.callback_metrics)


class NeuralNetWrapper(pl.LightningModule):
    """Use pytorch lightning interface for multicore training."""
    def __init__(self, model: nn.Module, transform: Optional[Transform]):
        super().__init__()
        """Wrap pytorch model in lightning interface."""
        self._model, self._transform = model, transform

    def forward(self, x: Tensor) -> Tensor:
        """Apply transition matrix if necessary."""
        ret = self._model(x)
        if self._transform:
            ret = self._transform(ret)
        return ret

    def configure_optimizers(self) -> Optimizer:
        """Just use plain Adam for now."""
        return Adam(self.parameters())

    def training_step(self, batch: Tuple[Tensor, Tensor], _) -> Tensor:
        """Minimize cross entropy."""
        return F.cross_entropy(self(batch[0]), batch[1])

    def validation_step(self, batch, batch_nb):
        """Early stopping based on validation accuracy."""
        self.log('val_acc', top1_accuracy(self(batch[0]), batch[1]))


def linear(in_dim: int, out_dim: int, _: Params) -> nn.Module:
    """Multinomial logistic regression."""
    return nn.Linear(in_dim, out_dim)


class ThreeLayer:
    """The simplest possible universal function approximator."""
    @staticmethod
    def build(in_dim: int, out_dim: int, params: Params) -> nn.Module:
        """Create the network, ready to be trained."""
        d = params['hidden_dim']
        return nn.Sequential(nn.Linear(in_dim, d), nn.ReLU(), nn.Linear(d, d),
                             nn.ReLU(), nn.Linear(d, out_dim))

    @staticmethod
    def tune(in_dim: int, out_dim: int, trial: Trial) -> Params:
        """Tune the dimension of the hidden layer."""
        low, high = min(in_dim, out_dim), max(in_dim, out_dim)
        hidden_dim = trial.suggest_int('hidden_dim', low, high, log=True)
        return {'hidden_dim': hidden_dim}


def top1_accuracy(pred, y):
    """Main evaluation metric."""
    return sum(pred.argmax(axis=1) == y) / len(y)


def estimate_transition_matrix(model, X: np.ndarray) -> np.ndarray:
    """Estimate anchor points to generate transition matrix."""
    p = model(X)
    return np.hstack([p[i][np.newaxis].T for i in p.argmax(axis=0)])


def evaluate(model, params: Dict[str, any]) -> Tuple[float, float]:
    """Run one evaluation round."""
    if isinstance(model, Forward):
        model = copy.copy(model)
    else:
        model = Backward(model)
    Xtr, Str, Xtr_val, Str_val, T, Xts, Yts = load(params['dataset'])
    model.train(params['params'], Xtr, Str, Xtr_val, Str_val, T)
    ret = {}
    ret['acc_val'] = top1_accuracy(model(Xtr_val, T), Str_val)
    ret['acc'] = top1_accuracy(model(Xts, T, True), Yts)
    if isinstance(model, Forward):
        model.train(params['params'], Xtr, Str, Xtr_val, Str_val)
    ret['T_hat'] = estimate_transition_matrix(model, Xtr)
    if isinstance(model, Forward):
        model.train(params['params'], Xtr, Str, Xtr_val, Str_val, ret['T_hat'],
                    True)
    ret['T_hat_err'] = np.linalg.norm(T - ret['T_hat'])
    ret['acc_val_hat'] = top1_accuracy(model(Xtr_val, ret['T_hat']), Str_val)
    ret['acc_hat'] = top1_accuracy(model(Xts, ret['T_hat'], True), Yts)
    return ret


def evaluate_batch(model, params: Dict[str, any]) -> Dict[str, any]:
    """Run ten evaluation rounds and get the mean and stdev."""
    results = [evaluate(model, params) for _ in range(10)]
    u = {k: np.mean([r[k] for r in results], axis=0) for k in KEYS}
    x = {f'{k}_std': np.std([r[k] for r in results], axis=0) for k in KEYS}
    return {'dataset': params['dataset'], **u, **x}


def train() -> None:
    """Run training and output evaluation results in csv format."""
    headers = ['ts', 'dataset', 'model'] + KEYS + [f'{k}_std' for k in KEYS]
    w = csv.DictWriter(sys.stdout, headers)
    w.writeheader()
    for params in PARAMS:
        pl.seed_everything(0)
        model = MODEL[params['model']]
        w.writerow({
            'ts': str(datetime.datetime.now()),
            'model': params['model'],
            **evaluate_batch(model, params)
        })
        if isinstance(model, Forward):
            w.writerow({
                'ts': str(datetime.datetime.now()),
                'model': f'{params["model"]}_backward',
                **evaluate_batch(model.backward(), params)
            })


def load(
    dataset: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray]:
    """Data preprocessing."""
    SCALE = 255
    with np.load(f'data/{dataset}.npz') as data:
        Xtr = data['Xtr'].reshape((len(data['Xtr']), -1)).astype(np.float32)
        Xtr /= SCALE
        Xts = data['Xts'].reshape((len(data['Xts']), -1)).astype(np.float32)
        Xts /= SCALE
        Str = data['Str'].astype(np.int64)
        Xtr, Xtr_val, Str, Str_val = train_test_split(Xtr, Str, test_size=0.2)
        Yts = data['Yts'].astype(np.int64)
    T = np.array(DATA[dataset]).astype(np.float32)
    return Xtr, Str, Xtr_val, Str_val, T, Xts, Yts


def tune() -> None:
    """Run hyperparam tuning and output params in jsonl format."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    w = jsonlines.Writer(sys.stdout, flush=True)
    for dataset, (name, model) in itertools.product(DATA, MODEL.items()):
        pl.seed_everything(0)
        Xtr, Str, Xtr_val, Str_val, _, _, _ = load(dataset)
        w.write({
            'ts': str(datetime.datetime.now()),
            'dataset': dataset,
            'model': name,
            'params': model.tune(Xtr, Str, Xtr_val, Str_val)
        })


def main() -> None:
    """Run all training and evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-t', '--tune', help=tune.__doc__, action='store_true')
    if parser.parse_args().tune:
        tune()
        return
    train()

if 'TPU_NAME' in os.environ:
    DEVICE = 'xla'
elif torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'

TPU_CORES = 1
FAST_DEV_RUN = True
KEYS = ['acc_val', 'acc', 'acc_val_hat', 'acc_hat', 'T_hat_err', 'T_hat']
DATA = {
    'FashionMNIST0.5': [[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]],
    'FashionMNIST0.6': [[0.4, 0.3, 0.3], [0.3, 0.4, 0.3], [0.3, 0.3, 0.4]],
    'CIFAR': [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
}
MODEL = {
    'linear': Forward(linear),
    'three_layer': Forward(ThreeLayer.build, ThreeLayer.tune),
    'lgb': Lgbm,
    'logistic': LR,
}
PARAMS = [
    {
        "ts": "2020-11-09 21:27:04.115906",
        "dataset": "FashionMNIST0.5",
        "model": "linear",
        "params": {}
    },
    {
        "ts": "2020-11-09 21:27:04.184178",
        "dataset": "FashionMNIST0.5",
        "model": "three_layer",
        "params": {
            "hidden_dim": 3
        }
    },
    {
        "ts": "2020-11-09 21:27:04.339063",
        "dataset": "FashionMNIST0.5",
        "model": "lgb",
        "params": {
            "objective": "softmax",
            "metric": "multi_logloss",
            "verbosity": -1,
            "force_row_wise": True,
            "num_class": 3,
            "feature_pre_filter": False,
            "lambda_l1": 0.00010308565871016856,
            "lambda_l2": 9.262909768209701e-05,
            "num_leaves": 9,
            "feature_fraction": 0.9520000000000001,
            "bagging_fraction": 0.9478399337923673,
            "bagging_freq": 1,
            "min_child_samples": 20,
            "num_iterations": 1000,
            "early_stopping_round": 1
        }
    },
    {
        "ts": "2020-11-09 21:30:08.050028",
        "dataset": "FashionMNIST0.5",
        "model": "logistic",
        "params": {
            "C": 0.000757356598665866,
            "solver": "liblinear",
            "multi_class": "ovr"
        }
    },
    {
        "ts": "2020-11-09 22:22:57.713244",
        "dataset": "FashionMNIST0.6",
        "model": "linear",
        "params": {}
    },
    {
        "ts": "2020-11-09 22:22:57.784873",
        "dataset": "FashionMNIST0.6",
        "model": "three_layer",
        "params": {
            "hidden_dim": 3
        }
    },
    {
        "ts": "2020-11-09 22:22:57.958981",
        "dataset": "FashionMNIST0.6",
        "model": "lgb",
        "params": {
            "objective": "softmax",
            "metric": "multi_logloss",
            "verbosity": -1,
            "force_row_wise": True,
            "num_class": 3,
            "feature_pre_filter": False,
            "lambda_l1": 0.0,
            "lambda_l2": 0.0,
            "num_leaves": 3,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.9054039912511501,
            "bagging_freq": 3,
            "min_child_samples": 20,
            "num_iterations": 1000,
            "early_stopping_round": 1
        }
    },
    {
        "ts": "2020-11-09 22:24:15.015987",
        "dataset": "FashionMNIST0.6",
        "model": "logistic",
        "params": {
            "C": 0.0005710289362229364,
            "solver": "saga",
            "multi_class": "auto"
        }
    },
    {
        'dataset': 'CIFAR',
        'model': 'linear',
        'params': {}
    },
    {
        'dataset': 'CIFAR',
        'model': 'three_layer',
        "params": {
            "hidden_dim": 3
        }
    },
    {
        "ts": "2020-11-11 12:33:43.689590",
        "dataset": "CIFAR",
        "model": "lgb",
        "params": {
            "objective": "softmax",
            "metric": "multi_logloss",
            "verbosity": -1,
            "force_row_wise": True,
            "num_class": 3,
            "feature_pre_filter": False,
            "lambda_l1": 3.7327508171383625e-05,
            "lambda_l2": 0.005723295583781243,
            "num_leaves": 8,
            "feature_fraction": 1.0,
            "bagging_fraction": 0.9714542036015783,
            "bagging_freq": 3,
            "min_child_samples": 20,
            "num_iterations": 1000,
            "early_stopping_round": 1
        }
    },
    {
        "ts": "2020-11-11 12:49:39.544802",
        "dataset": "CIFAR",
        "model": "logistic",
        "params": {
            "C": 3987250.7131022774,
            "solver": "saga",
            "multi_class": "ovr"
        }
    },
]

if __name__ == '__main__':
    main()
