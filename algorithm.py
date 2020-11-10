#!/usr/bin/env python3
"""Build transition matrix estimator and classifiers."""
import argparse
import csv
import datetime
import functools
import itertools
import os
import random
import sys
from typing import Callable, Dict, Tuple
import jsonlines
import lightgbm
import optuna
import optuna.integration.lightgbm as lgb
import numpy as np
from scipy.special import softmax
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn

if 'COLAB_TPU_ADDR' in os.environ:
    import torch_xla
    import torch_xla.core.xla_model as xm
    device = xm.xla_device()
elif torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

Model = Callable[[np.ndarray], np.ndarray]
Trainer = Callable[
    [Dict[str, any], np.ndarray, np.ndarray, np.ndarray, np.ndarray], Model]
Tuner = Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], Dict[str,
                                                                        any]]
Net = Callable[[int, int, Dict[str, any]], nn.Module]
NNTuner = Callable[
    [np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray], Dict[str,
                                                                       any]]


class Backward:
    """Use inverse transition matrix to denoise."""
    def __init__(self, trainer: Trainer, tuner: Tuner = None):
        """Training and tuning interface."""
        self._train, self._tune = trainer, tuner

    def train(self, params: Dict[str, any], X: np.ndarray, y: np.ndarray,
              _: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> None:
        """Train and validate using given hyperparams."""
        self._model = self._train(params, X, y, X_val, y_val)

    def tune(self, X: np.ndarray, y: np.ndarray, _: np.ndarray,
             X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, any]:
        """Find optimal hyperparams for given train/val split."""
        return self._tune(X, y, X_val, y_val) if self._tune else {}

    def __call__(self,
                 X: np.ndarray,
                 T: np.ndarray,
                 denoise: bool = False) -> np.ndarray:
        """Predict, with flag to indicate whether to denoise."""
        ret = self._model(X)
        if denoise:
            ret = softmax(np.linalg.inv(T) @ ret.T, axis=0).T
        return ret


def lgbm_init(
        X: np.ndarray, y: np.ndarray, X_val: np.ndarray,
        y_val: np.ndarray) -> Tuple[Dict[str, any], lgb.Dataset, lgb.Dataset]:
    """Configure lightgbm model."""
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


def lgbm(params: Dict[str, any], X: np.ndarray, y: np.ndarray,
         X_val: np.ndarray, y_val: np.ndarray) -> Model:
    """Training with early stopping on validation set."""
    p, dtrain, dval = lgbm_init(X, y, X_val, y_val)
    params.update(p)
    return lightgbm.train(params, dtrain, valid_sets=dval,
                          verbose_eval=False).predict


def lgbm_tune(X: np.ndarray, y: np.ndarray, X_val: np.ndarray,
              y_val: np.ndarray) -> Dict[str, any]:
    """Hyperparam tuning using optuna lightgbm integration."""
    params, dtrain, dval = lgbm_init(X, y, X_val, y_val)
    return lgb.train(params, dtrain, valid_sets=dval,
                     verbose_eval=False).params


def logistic_regression_objective(X: np.ndarray, y: np.ndarray,
                                  X_val: np.ndarray, y_val: np.ndarray,
                                  trial) -> float:
    """Logistic regression tuning code."""
    C = trial.suggest_loguniform('C', 1e-9, 1e9)
    solver = trial.suggest_categorical(
        'solver', ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'])
    multi_class = trial.suggest_categorical('multi_class', ['auto', 'ovr'])
    model = LogisticRegression(C=C, solver=solver, multi_class=multi_class)
    model.fit(X, y)
    return top1_accuracy(model.predict_proba(X_val), y_val)


def logistic_regression(params: Dict[str, any], X: np.ndarray, y: np.ndarray,
                        X_val: np.ndarray, y_val: np.ndarray) -> Model:
    """Multiclass logistic regression classifier."""
    return LogisticRegression(**params).fit(X, y).predict_proba


def logistic_regression_tune(X: np.ndarray, y: np.ndarray, X_val: np.ndarray,
                             y_val: np.ndarray) -> Dict[str, any]:
    """Hyperparam tuning using generic optuna integration."""
    study = optuna.create_study(direction='maximize')
    f = functools.partial(logistic_regression_objective, X, y, X_val, y_val)
    study.optimize(f, n_trials=100)
    return study.best_params


class Forward:
    """Append transition matrix to neural network during training."""
    def __init__(self, build: Net, tuner: NNTuner = None):
        """Wrap neural net architecture in generic interface."""
        self._build, self._tune = build, tuner

    def train(self, params: Dict[str, any], X: np.ndarray, y: np.ndarray,
              T: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> None:
        """Train with hyperparams and early stopping on validation set."""
        T = torch.from_numpy(T).to(device)
        sm = nn.Softmax(dim=1).to(device)
        self._model = train_nn(self._build, params, X, y,
                               lambda x, T=T: sm(T @ sm(x).T).T, X_val, y_val)

    def tune(self, X: np.ndarray, y: np.ndarray, T: np.ndarray,
             X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, any]:
        """Find optimal hyperparams for neural net."""
        return self._tune(X, y, T, X_val, y_val) if self._tune else {}

    def __call__(self,
                 X: np.ndarray,
                 T: np.ndarray,
                 denoise: bool = False) -> np.ndarray:
        """Predict, using transition matrix as necessary."""
        with torch.no_grad():
            ret = softmax(self._model(torch.from_numpy(X).to(device)).cpu().numpy(), axis=1)
        if not denoise:
            ret = softmax(T @ ret.T, axis=0).T
        return ret


def train_nn(build: Net, params: Dict[str, any], X: np.ndarray, y: np.ndarray,
             transform: Callable[[torch.Tensor], torch.Tensor],
             X_val: np.ndarray, y_val: np.ndarray) -> nn.Module:
    """SGD with early stopping on validation set."""
    device = xm.xla_device(
    ) if 'COLAB_TPU_ADDR' in os.environ else 'cuda' if torch.cuda.is_available(
    ) else 'cpu'
    model = build(X.shape[1], max(y) + 1, params).to(device)
    X = torch.from_numpy(X).to(device)
    y = torch.from_numpy(y).to(device)
    X_val = torch.from_numpy(X_val).to(device)
    y_val = torch.from_numpy(y_val).to(device)
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=1e-1,
                                weight_decay=1e-5,
                                momentum=0.9)
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
        X, y),
                                               batch_size=256,
                                               shuffle=True)
    criterion = nn.CrossEntropyLoss()
    best = 0
    for epoch in range(10):
        model.train()
        for X, y in train_loader:
            optimizer.zero_grad()
            pred = transform(model(X))
            criterion(pred, y).backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            pred = transform(model(X_val))
        acc = top1_accuracy(pred, y_val)
        if acc < best:
            break
        best = acc
    return model


class NeuralNet:
    """For use as black box classifier."""
    def __init__(self, model: nn.Module):
        """Wrap pytorch model."""
        self._model = model

    def __call__(self, X: np.ndarray) -> np.ndarray:
        """Numpy prediction interface."""
        with torch.no_grad():
            return softmax(self._model(torch.from_numpy(X).to(device)).cpu().numpy(), axis=1)


def neural_net(
    build: Net
) -> Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], NeuralNet]:
    """Wrap pytorch network in generic train function."""
    return lambda params, X, y, X_val, y_val: NeuralNet(
        train_nn(build, params, X, y, lambda x: x, X_val, y_val))


def linear(in_dim: int, out_dim: int, _: Dict[str, any]) -> nn.Module:
    """Multinomial logistic regression."""
    return nn.Linear(in_dim, out_dim)


def three_layer(in_dim: int, out_dim: int, _: Dict[str, any]) -> nn.Module:
    """The simplest possible universal function approximator."""
    return nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(),
                         nn.Linear(out_dim, out_dim), nn.ReLU(),
                         nn.Linear(out_dim, out_dim))


def top1_accuracy(pred: np.ndarray, y: np.ndarray) -> float:
    """Main evaluation metric."""
    return sum(pred.argmax(axis=1) == y) / len(y)


def reset_seed(seed: int = 0):
    """Fix all random seeds for repeating the experiment result."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # If multi-GPUs are used.
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def evaluate(params: Dict[str, any]) -> Tuple[float, float]:
    """Run one evaluation round."""
    Xtr, Str, T, Xtr_val, Str_val, Xts, Yts = load(params['dataset'])
    model = MODEL[params['model']]
    model.train(params['params'], Xtr, Str, T, Xtr_val, Str_val)
    acc_val = top1_accuracy(model(Xtr_val, T), Str_val)
    acc = top1_accuracy(model(Xts, T, True), Yts)
    return acc_val, acc


def train() -> None:
    """Output evaluation results in csv format."""
    w = csv.DictWriter(
        sys.stdout,
        ['ts', 'dataset', 'model', 'acc_val', 'acc_val_std', 'acc', 'acc_std'])
    w.writeheader()
    for params in PARAMS:
        reset_seed()
        acc_val, acc = [], []
        for i in range(10):
            v, a = evaluate(params)
            acc_val.append(v)
            acc.append(a)
        w.writerow({
            'ts': str(datetime.datetime.now()),
            'dataset': params['dataset'],
            'model': params['model'],
            'acc_val': np.mean(acc_val),
            'acc_val_std': np.std(acc_val),
            'acc': np.mean(acc),
            'acc_std': np.std(acc),
        })


def load(
    dataset: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray]:
    """Data preprocessing."""
    SCALE = 255
    with np.load(f'data/{dataset}.npz') as data:
        Xtr = data['Xtr'].reshape(
            (len(data['Xtr']), -1)).astype(np.float32) / SCALE
        Xts = data['Xts'].reshape(
            (len(data['Xts']), -1)).astype(np.float32) / SCALE
        Xtr, Xtr_val, Str, Str_val = train_test_split(Xtr,
                                                      data['Str'].astype(
                                                          np.int64),
                                                      test_size=0.2)
        Yts = data['Yts'].astype(np.int64)
    T = np.array(DATA[dataset], dtype=np.float32)
    return Xtr, Str, T, Xtr_val, Str_val, Xts, Yts


def tune() -> None:
    """Output optimal hyperparams in jsonl format."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    w = jsonlines.Writer(sys.stdout, flush=True)
    for dataset, (name, model) in itertools.product(DATA, MODEL.items()):
        reset_seed()
        Xtr, Str, T, Xtr_val, Str_val, _, _ = load(dataset)
        w.write({
            'ts': str(datetime.datetime.now()),
            'dataset': dataset,
            'model': name,
            'params': model.tune(Xtr, Str, T, Xtr_val, Str_val)
        })


def main() -> None:
    """Run all training and evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-t', '--tune', help=tune.__doc__, action='store_true')
    if parser.parse_args().tune:
        tune()
        return
    train()


DATA = {
    'FashionMNIST0.5': [[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]],
    'FashionMNIST0.6': [[0.4, 0.3, 0.3], [0.3, 0.4, 0.3], [0.3, 0.3, 0.4]],
}
MODEL = {
    'forward_linear': Forward(linear),
    'backward_linear': Backward(neural_net(linear)),
    'forward_three_layer': Forward(three_layer),
    'backward_three_layer': Backward(neural_net(three_layer)),
    'lgb': Backward(lgbm, lgbm_tune),
    'logistic': Backward(logistic_regression, logistic_regression_tune),
}
PARAMS = [
    {
        "ts": "2020-11-09 21:27:04.115906",
        "dataset": "FashionMNIST0.5",
        "model": "forward_linear",
        "params": {}
    },
    {
        "ts": "2020-11-09 21:27:04.184178",
        "dataset": "FashionMNIST0.5",
        "model": "backward_linear",
        "params": {}
    },
    {
        "ts": "2020-11-09 21:27:04.235472",
        "dataset": "FashionMNIST0.5",
        "model": "forward_three_layer",
        "params": {}
    },
    {
        "ts": "2020-11-09 21:27:04.288037",
        "dataset": "FashionMNIST0.5",
        "model": "backward_three_layer",
        "params": {}
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
        "model": "forward_linear",
        "params": {}
    },
    {
        "ts": "2020-11-09 22:22:57.784873",
        "dataset": "FashionMNIST0.6",
        "model": "backward_linear",
        "params": {}
    },
    {
        "ts": "2020-11-09 22:22:57.843417",
        "dataset": "FashionMNIST0.6",
        "model": "forward_three_layer",
        "params": {}
    },
    {
        "ts": "2020-11-09 22:22:57.899029",
        "dataset": "FashionMNIST0.6",
        "model": "backward_three_layer",
        "params": {}
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
]

if __name__ == '__main__':
    main()
