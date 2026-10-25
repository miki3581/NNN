import numpy as np
from typing import Optional
from numpy.typing import NDArray

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
