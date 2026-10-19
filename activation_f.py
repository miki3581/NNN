import numpy as np

def ReLU(x) -> np.ndarray:
    return np.maximum(0.0, x)

def LeakyReLU (x, alpha = 0.01) -> np.ndarray:
    return np.where(x > 0, x, alpha * x)

def tanh (x) -> np.ndarray:
    return np.tanh(x)

def sigmoid(x) -> np.ndarray:
    return 1/(1 + np.exp(-x))
