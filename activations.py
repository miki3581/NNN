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

    def backward(self, dA:NDArray[np.floating]) -> NDArray[np.floating]:
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
        if self.mask is None:
            raise RuntimeError("Mask is empty, backward called before forward")
        
        return np.where(self.mask, dA, self.alpha * dA)
        

class Tanh:
    """"
    Hyperbolic tangent (tanh) activation function, stores A for backprop
    """
    def __init__(self) -> None:
        self.A: Optional[NDArray[np.floating]] = None

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
        if self.A is None:
            raise RuntimeError("A is empty, backward called before forward")
        
        return dA * (1.0 - self.A ** 2)


class Sigmoid:
    """
    Sigmoid activation function, stores A for backprop
    """

    def __init__(self) -> None:
        self.A: Optional[NDArray[np.floating]] = None

    def forward(self, Z: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Computes and stores output value A
        """
        self.A = np.exp(Z)/(np.exp(Z) + 1)
        return np.where(Z < 0, np.exp(Z)/(np.exp(Z) + 1), 1/(1 + np.exp(-Z)))

    def backward(self, dA: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Backward pass, computes gradient
        f'(x) = A(1 - A)
        """
        if self.A is None:
            raise RuntimeError("A is empty, backward called before forward")

        return dA * self.A * (1.0 - self.A)

class Softmax:
    def __init__(self, axis: int = -1) -> None:
        """
        Storing output probabilities A
        """
        self.A: Optional[NDArray[np.floating]] = None
        self.axis = axis

    def forward(self, Z: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Calculating probability distribution A minding the overflow for large Z
        """
        max_Z = np.max(Z, axis=self.axis, keepdims=True)
        shifted_Z = Z - max_Z
        exp_Z = np.exp(shifted_Z)
        sum_exp_Z = np.sum(exp_Z, axis = self.axis, keepdims=True)

        self.A = exp_Z / sum_exp_Z
        return self.A

    def backward(self, dA: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Calculating gradient
        """
        if self.A is None:
            raise RuntimeError("A is empty, backward called before forward")

        sum_dA_A = np.sum(dA * self.A, axis=self.axis, keepdims=True)
        dZ = self.A * (dA - sum_dA_A)
        return dZ
