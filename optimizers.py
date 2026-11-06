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


    def step(self, params:list) -> None:

        if self.v is None:
            self.v = [np.zeros_like(p) for p, _ in params]
        for i, (p, g) in enumerate(params):
            if self.momentum == 0:
                p -= self.lr * g
            else:
                self.v[i] = self.momentum * self.v[i] + g
                p -= self.lr * self.v[i]