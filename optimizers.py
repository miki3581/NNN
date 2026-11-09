import numpy as np


class SGD:

    def __init__(self, lr: float, momentum: float=0.0) -> None:

        self.lr = lr
        self.momentum = momentum

        if lr <= 0:
            raise ValueError("Learning rate lower than or equal to 0")

        if momentum < 0 or momentum >= 1:
            raise ValueError("Momentum lower than 0 or higher or equal 1")

        self.v = None


    def step(self, params: list) -> None:

        if self.v is None:
            self.v = [np.zeros_like(p) for p, _ in params]
        for i, (p, g) in enumerate(params):
            if self.momentum == 0:
                p -= self.lr * g
            else:
                self.v[i] = self.momentum * self.v[i] + g
                p -= self.lr * self.v[i]


class Adam:

    def __init__(self, lr: float=1e-3, beta1: float=0.9, beta2: float=0.999, eps: float=1e-8) -> None:

        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        if self.lr <= 0:
            raise ValueError("Learning rate lower or equal than 0")

        if self.beta1 < 0 or self.beta1 >= 1:
            raise ValueError("beta1 lower tahn 0 or higher or equal than 1")

        if self.beta2 < 0 or self.beta2 >= 1:
            raise ValueError("beta2 lower tahn 0 or higher or equal than 1")

        if self.eps <= 0:
            raise ValueError("eps lower or equal than 0")

        self.m = None
        self.v = None
        self.t = 0


    def step(self, params: list) -> None:

        if self.v is None:
            self.m = [np.zeros_like(p) for p, _ in params]
        if self.v is None:
            self.v = [np.zeros_like(p) for p, _ in params]
        
        self.t += 1
        for i, (p, g) in enumerate(params):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g ** 2
            m = self.m[i] / (1 - self.beta1 ** self.t)
            v = self.v[i] / (1 - self.beta2 ** self.t)
            p -= self.lr * m / (np.sqrt(v) + self.eps)