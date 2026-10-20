import numpy as np
from numpy.typing import NDArray
from typing import Optional

class ReLU:
    """
    ReLU activation layer with forward and backward passes
    """
    def __init__(self) -> None:
        """
        Initializes mask to store activated enurons during forward pass
        """
        self.mask: Optional[NDArray[np.bool_]] = None

    def forward(self, Z: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Forward pass, assigns mask value for neurons and calculates max
        """
        self.mask = (Z > 0.0)
        return np.maximum(0.0, Z)

    def backward(self, dA:NDArray[np.floating]) -> NDArray[np.flaoting]:
        """
        Backward pass, computes gradient for backprop
        """
        if self.mask is None:
            raise RuntimeError("Mask is empty, backward called before forward")

        return dA * self.mask


class LeakyReLU:
    """
    Leaky ReLU activation layer with forward and backward passes
    """

    def __init__(self, alpha: float = 0.01) -> None:
        """
        Initializes mask to store activated enurons during forward pass
        """
        self.alpha = alpha
        self.mask: Optional[NDArray[np.bool_]] = None

    def forward(self, Z: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Forward pass, assigns mask value for neurons and calculates max with alpha
        """
        self.mask = (Z > 0)
        return np.where(self.mask, Z, Z * self.alpha)

    def backward(self, dA: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Backward pass, computes gradient for backprop
        """
        return np.where(self.mask, dA, self.alpha * dA)
        

class tanh:
    """"
    Hyperbolic tangent (tanh) activation function, stores A for backprop
    """
    def __init__(self):
        self.A: Optional[NDArray] = None

    def forward(self, Z: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Computes and stores output value A
        """
        self.A = np.tanh(Z)
        return self.A

    def backward(self, dA: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Backward pass, computes gradient
        Derivative for tanh(x) is 1 - tanh(x)^2
        If Z positive A ~ 1 -> 1 - A^2 ~ 0
        If Z negative A ~ -1 -> 1 - A^2 ~ 0
        If Z = 0, A = 0 -> max gradient
        """
        return dA * (1.0 - self.A ** 2)

    def sigmoid(x) -> np.ndarray:
        return 1/(1 + np.exp(-x))

    def softmax(x) -> np.ndarray:
        return np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x)))