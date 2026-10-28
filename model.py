import numpy as np
from numpy.typing import NDArray 
from layers import Dense
from activations import Sigmoid, Softmax

class Model:
    """
    Building neural network from list of layers (Dense and activations)
    Output layer returns logit, Sigmoid or Softmax from activations applied for classification
    """
    
    def __init__(self, layers: list, task: str = None) -> None:
        """
        Stores layers and tasks with check for concrete task
        """

        self.layers = layers
        self.task = task

        if task not in ["regression", "binary", "multiclass"]:
            raise ValueError("Unknown task, pick one from regression, binary and multiclass")


    def forward(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Passes through layers in order, returns logit
        """

        for layer in self.layers:
            X = layer.forward(X)
        return X
            
    
    def backward(self, dZ: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Passes through layers in reverse order, returns input gradient
        """
        
        for layer in reversed(self.layers):
            dZ = layer.backward(dZ)
        return dZ


    def l2_penalty(self) -> float:
        """
        Returns sum of (l2 / 2) * ||W||^2 over Dense layers
        """
        
        pen = 0.0
        for layer in self.layers:
            if isinstance(layer, Dense):
                pen += (layer.l2 / 2) * np.sum(layer.W ** 2)
        return float(pen)


    def predict_proba(self, X: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Returns class probabilities: sigmoid for binary, softmax for multiclass
        Raises ValueError for regression
        """

        Z = self.forward(X)
        if self.task == "binary":
            proba = Sigmoid().forward(Z)
        elif self.task == "multiclass":
            proba = Softmax().forward(Z)
        elif self.task == "regression":
            raise ValueError("Can't calculate probability for regression")
        return proba


    def predict(self, X: NDArray[np.floating]) -> NDArray[np.floating] | NDArray[np.integer]:
        """
        Returns predictions: values for regression (float),
        0/1 labels (threshold 0.5) for binary, class indices (argmax) for multiclass (int)
        """

        if self.task == "binary":
            P = self.predict_proba(X)
            return (P >= 0.5).astype(int)
        elif self.task == "multiclass":
            P = self.predict_proba(X)
            return np.argmax(P, axis=1, keepdims=True)
        elif self.task == "regression":
            return self.forward(X)