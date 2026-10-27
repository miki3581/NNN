import numpy as np
from typing import Optional
from numpy.typing import NDArray
from activations import Sigmoid, Softmax

class MSE:
    """
    Calculating MSE
    L = 1/2N to keep gradients the same for all loss functions and simplify math
    """
    
    def __init__(self) -> None:
        """
        Initializes Z and y
        """
        
        self.Z: Optional[NDArray[np.floating]] = None
        self.y: Optional[NDArray[np.floating]] = None


    def forward(self, Z: NDArray[np.floating], y: NDArray[np.floating]) -> float:
        """
        Checks array sizes, calculates and returns MSE as float
        """

        if Z.shape != y.shape:
            raise ValueError("Z and y shapes don't match")

        self.Z = Z
        self.y = y

        L = (1 / (2 * Z.shape[0])) * np.sum((self.Z - self.y) ** 2)
        return float(L)


    def backward(self) -> NDArray[np.floating]:
        """
        Checks if Z is presesnt, returns gradient dZ
        """

        if self.Z is None:
            raise RuntimeError("Z is missing")
        else:
            return (self.Z - self.y) / self.Z.shape[0]


class BCE:
    """
    Calculating Binary cross-entropy from logits, used with sigmoid output and 0/1 labels
    BCE counted using stable softplus logit to avoid exp overflow
    Using previously created Sigmoid to calculate gradient
    """
    
    def __init__(self) -> None:
        """
        Initializes Z and y
        """
        
        self.Z: Optional[NDArray[np.floating]] = None
        self.y: Optional[NDArray[np.floating]] = None


    def forward(self, Z: NDArray[np.floating], y: NDArray[np.floating]) -> float:
        """
        Checks array sizes, calculates and returns BCE as float
        """

        if Z.shape != y.shape:
            raise ValueError("Z and y shapes don't match")

        self.Z = Z
        self.y = y

        L = np.sum(np.maximum(self.Z, 0) + np.log1p(np.exp(-np.abs(self.Z))) - self.y * self.Z) / self.Z.shape[0]
        return float(L)


    def backward(self) -> NDArray[np.floating]:
        """
        Checks if Z is presesnt, returns gradient dZ
        """

        if self.Z is None:
            raise RuntimeError("Z is missing")
        else:
            return (Sigmoid().forward(self.Z) - self.y) / self.Z.shape[0]


class CE:
    """
    Calculationg cross-entropy from logits, used with softmax output and one-hot labels
    L = 1/n · Σ (LSE(z) - Σ y·z)
    """

    def __init__(self) -> None:
        """
        Initializes Z and y
        """
        
        self.Z: Optional[NDArray[np.floating]] = None
        self.y: Optional[NDArray[np.floating]] = None


    def forward(self, Z: NDArray[np.floating], y: NDArray[np.floating]) -> float:
        """
        Checks array sizes, calculates and returns CE as float
        """

        if Z.shape != y.shape:
            raise ValueError("Z and y shapes don't match")
        
        self.Z = Z
        self.y = y

        m = np.max(self.Z, axis=1, keepdims=True)
        LSE = m + np.log(np.sum(np.exp(self.Z - m),  axis=1, keepdims=True))
        sumZy = np.sum(self.Z * self.y, axis=1, keepdims=True)
        L = np.sum(LSE - sumZy) / self.Z.shape[0]
        return float(L)


    def backward(self) -> NDArray[np.floating]:
        """
        Checks if Z is present, returns gradient dZ
        """

        if self.Z is None:
            raise RuntimeError("Z is missing")
        else:
            return (Softmax().forward(self.Z) - self.y) / self.Z.shape[0]
        