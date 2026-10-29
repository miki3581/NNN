import numpy as np
from model import Model
from layers import Dense
from activations import Tanh
from numpy.typing import NDArray

def make_model(task: str, rng: np.random.Generator, l2: float=0.0) -> Model:

    if task == "multiclass":
        k = 3
    else:
        k = 1
    layers = [Dense(4, 5, rng, init="xavier", l2=l2), Tanh(), Dense(5, k, rng, init="xavier", l2=l2)]

    return Model(layers, task)

def make_data(task: str, rng: np.random.Generator, n: int=6) -> tuple[NDArray[np.floating], NDArray[np.floating]]:

    k = 3
    if task == "multiclass":
        y = np.eye(k)[rng.integers(0, k, size=n)]
    elif task == "regression":
        y = rng.normal(size=(n, 1))
    else:
        y = rng.integers(0, 2, size=(n, 1)).astype(float)
    X = rng.normal(size=(n, 4))
    
    return X, y