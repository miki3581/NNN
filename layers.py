import numpy as np
from typing import Optional
from numpy.typing import NDArray


class Dense:
    """
    Dense layer
    Calculates Z = X @ W + b
    Stores X and gradients to update weights
    """

    def __init__(self, d_in: int, d_out: int, rng: np.random.Generator, init: str="he", l2: float=0.0) -> None:
        """
        Stores hyperparameters, randomizes weight matrix, zeroes b and prepares X, dW and db for next pass
        """
        
        self.l2 = l2
        self.d_in = d_in
        self.d_out = d_out

        if init.lower() == "he":
            std = np.sqrt(2 / d_in)
        elif init.lower() == "xavier":
            std = np.sqrt(2 / (d_in + d_out))
        else:
            raise ValueError("Unknown weight initialization, choose He or Xavier")

        self.W = rng.normal(0.0, std, size=(d_in, d_out))

        self.b = np.zeros((1, d_out))

        self.X: Optional[NDArray[np.floating]] = None
        self.dW: Optional[NDArray[np.floating]] = None
        self.db: Optional[NDArray[np.floating]] = None

    def forward(self, X:NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Stores X, calculates and returns Z
        """

        self.X = X
        Z = self.X @ self.W + self.b
        return Z

    def backward(self, dZ:NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Checks for call order, stores dW, db and calculates input gradient dX
        """

        if self.X is None:
            raise RuntimeError("X is empty, backward called before forward")

        self.dW = self.X.T @ dZ + self.l2 * self.W
        self.db = np.sum(dZ, axis=0, keepdims=True)
        dX = dZ @ self.W.T
        return dX
