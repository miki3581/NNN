import numpy as np
from typing import Optional
from numpy.typing import NDArray
class Dense:
    """
    Dense layer
    Calculates Z = X @ W.T + b
    Stores X and gradients to update weights
    """

    def __init__(self, d_in: int, d_out: int, rng, init: str="He", l2: float=0.0) -> None:
        
        self.l2 = l2
        self.d_in = d_in
        self.d_out = d_out

        if init == "He":
            std = np.sqrt(2 / d_in)
        elif init == "Xavier":
            std = np.sqrt(2 / (d_in + d_out))
        else:
            raise ValueError("Unknown weight innitialization, chose He or Xavier")

        self.W = rng.normal(0.0, std, size=(d_in, d_out))

        self.b = np.zeros((1, d_out))

        self.X = None
        self.dW: Optional[NDArray[np.floating]] = None
        self.db: Optional[NDArray[np.floating]] = None

    def forward(self, X:NDArray[np.floating]) -> NDArray[np.floating]:

        self.X = X
        return np.dot(self.X, self.W) + self.b

    def backward(self, dZ:NDArray[np.floating]) -> NDArray[np.floating]:

        if self.X is None:
            raise RuntimeError("Mask is empty, backward called before forward")

        self.dW = np.dot(np.transpose(self.X), dZ) + self.l2* self.W
        self.db = np.sum(dZ, axis=0, keepdims=True)
        dX = np.dot(dZ, np.transpose(self.W))
        return dX, self.dW, self.db
