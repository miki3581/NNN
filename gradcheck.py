"""
Storing helper functions and variables for testing gradients
"""
import numpy as np


EPS = 1e-6
TOLERANCE = 1e-6


def num_grad(f, P):
    """
    Central-difference estimate of df/dP. P is perturbed in place and restored after each step.
    """

    grad = np.zeros_like(P)
    for idx in np.ndindex(P.shape):
        original = P[idx]
        P[idx] = original + EPS
        L_plus = f()
        P[idx] = original - EPS
        L_minus = f()
        P[idx] = original
        grad[idx] = (L_plus - L_minus) / (2 * EPS)
    return grad


def rel_error(a, b):
    """
    Relative error ||a - b|| / (||a|| + ||b||); returns 0.0 when both are zero.
    """
    norm_sum = np.linalg.norm(a) + np.linalg.norm(b)
    if norm_sum == 0.0:
        return 0.0
    return np.linalg.norm(a - b) / norm_sum