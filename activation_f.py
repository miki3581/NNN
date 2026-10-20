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



    def LeakyReLU (x, alpha = 0.01) -> np.ndarray:
        return np.where(x > 0, x, alpha * x)

    def tanh (x) -> np.ndarray:
        return np.tanh(x)

    def sigmoid(x) -> np.ndarray:
        return 1/(1 + np.exp(-x))

    def softmax(x) -> np.ndarray:
        return np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x)))