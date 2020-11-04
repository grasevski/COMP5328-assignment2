#!/usr/bin/env python3
"""Build transition matrix estimators"""
import csv
import os
import random
import sys
from typing import Callable, List, Tuple
import lightgbm as lgb
import numpy as np
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn


class Backward:
    def __init__(self, model):
        self._model = model

    def train(self, X: np.ndarray, y: np.ndarray, _: np.ndarray) -> None:
        self._model.fit(X, y)

    def __call__(self,
                 X: np.ndarray,
                 T: np.ndarray,
                 denoise: bool = False) -> np.ndarray:
        ret = self._model.predict_proba(X)
        if denoise:
            ret = softmax(np.linalg.inv(T) @ ret.T, axis=0).T
        return ret


Model = Callable[[int, int], nn.Module]


class Forward:
    def __init__(self, build: Model):
        self._build = build

    def train(self, X: np.ndarray, y: np.ndarray, T: np.ndarray) -> None:
        T = torch.from_numpy(T.astype(np.float32))
        sm = nn.Softmax(dim=1)
        self._model = train(self._build, X, y, lambda x: sm(T @ sm(x).T).T)

    def __call__(self,
                 X: np.ndarray,
                 T: np.ndarray,
                 denoise: bool = False) -> np.ndarray:
        with torch.no_grad():
            ret = softmax(self._model(torch.from_numpy(X.astype(
                np.float32))).numpy(),
                          axis=1)
        if not denoise:
            ret = softmax(T @ ret.T, axis=0).T
        return ret


def train(build: Model, X: np.ndarray, y: np.ndarray,
          transform: Callable[[torch.Tensor], torch.Tensor]) -> nn.Module:
    model = build(X.shape[1], max(y) + 1)
    X = torch.from_numpy(X.astype(np.float32))
    y = torch.from_numpy(y.astype(np.int64))
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=1e-1,
                                weight_decay=1e-5,
                                momentum=0.9)
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
        X, y),
                                               batch_size=256,
                                               shuffle=True)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(10):
        for X, y in train_loader:
            optimizer.zero_grad()
            pred = transform(model(X))
            criterion(pred, y).backward()
            optimizer.step()
    model.eval()
    return model


class NeuralNet:
    def __init__(self, build: Model):
        self._build = build

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self._model = train(self._build, X, y, lambda x: x)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return softmax(self._model(torch.from_numpy(X.astype(
                np.float32))).numpy(),
                           axis=1)


def evaluate(dataset: str, T: List[List[float]], model) -> Tuple[float, float]:
    with np.load(f'data/{dataset}.npz') as data:
        Xtr = data['Xtr'].reshape((len(data['Xtr']), -1))
        Xts = data['Xts'].reshape((len(data['Xts']), -1))
        Xtr, Xtr_val, Str, Str_val = train_test_split(Xtr,
                                                      data['Str'],
                                                      test_size=0.2)
        Yts = data['Yts']
    T = np.array(T)
    model.train(Xtr, Str, T)
    acc_val = top1_accuracy(model(Xtr_val, T), Str_val)
    acc = top1_accuracy(model(Xts, T, True), Yts)
    return acc_val, acc


def linear(in_dim: int, out_dim: int) -> nn.Module:
    return nn.Linear(in_dim, out_dim)


def three_layer(in_dim: int, out_dim: int) -> nn.Module:
    return nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(),
                         nn.Linear(out_dim, out_dim), nn.ReLU(),
                         nn.Linear(out_dim, out_dim))


def top1_accuracy(pred: np.ndarray, y: np.ndarray) -> float:
    return sum(pred.argmax(axis=1) == y) / len(y)


def reset_seed(seed: int = 0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # If multi-GPUs are used.
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main() -> None:
    """Run all training and evaluation"""
    w = csv.DictWriter(
        sys.stdout,
        ['dataset', 'model', 'acc_val', 'acc_val_std', 'acc', 'acc_std'])
    w.writeheader()
    for dataset, T in DATA.items():
        for name, model in MODEL.items():
            reset_seed()
            acc_val, acc = [], []
            for i in range(10):
                v, a = evaluate(dataset, T, model)
                acc_val.append(v)
                acc.append(a)
            w.writerow({
                'dataset': dataset,
                'model': name,
                'acc_val': np.mean(acc_val),
                'acc_val_std': np.std(acc_val),
                'acc': np.mean(acc),
                'acc_std': np.std(acc)
            })


DATA = {
    'FashionMNIST0.5': [[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]],
    'FashionMNIST0.6': [[0.4, 0.3, 0.3], [0.3, 0.4, 0.3], [0.3, 0.3, 0.4]],
}
MODEL = {
    'forward_linear': Forward(linear),
    'backward_linear': Backward(NeuralNet(linear)),
    'forward_three_layer': Forward(three_layer),
    'backward_three_layer': Backward(NeuralNet(three_layer)),
    'LGB': Backward(lgb.LGBMClassifier()),
    'logistic': Backward(LogisticRegression()),
}

if __name__ == '__main__':
    main()
