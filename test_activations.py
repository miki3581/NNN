import numpy as np
from activations import ReLU, LeakyReLU, Tanh, Sigmoid, Softmax

EPS = 1e-6
TOLERANCE = 1e-6


def grad_check(layer, Z, rng):
    """
    Compare backward(dA) with an estimate of dL/dZ, where L = sum(dA * forward(Z)).
    """
    dA = rng.normal(size=Z.shape)

    layer.forward(Z)
    analytic = layer.backward(dA)

    numeric = np.zeros_like(Z)
    for idx in np.ndindex(Z.shape):
        Z_plus, Z_minus = Z.copy(), Z.copy()
        Z_plus[idx] += EPS
        Z_minus[idx] -= EPS
        L_plus = np.sum(dA * layer.forward(Z_plus))
        L_minus = np.sum(dA * layer.forward(Z_minus))
        numeric[idx] = (L_plus - L_minus) / (2 * EPS)

    norm_sum = np.linalg.norm(analytic) + np.linalg.norm(numeric)
    if norm_sum == 0.0:
        return 0.0
    return np.linalg.norm(analytic - numeric) / norm_sum


def main():
    rng = np.random.default_rng(42)
    layers = [ReLU(), LeakyReLU(), Tanh(), Sigmoid(), Softmax()]
    shape = (4, 5)

    all_passed = True
    for layer in layers:
        Z = rng.normal(size=shape)
        Z[np.abs(Z) < 1e-3] = 0.5
        error = grad_check(layer, Z, rng)
        passed = error < TOLERANCE
        all_passed &= passed
        status = "PASS" if passed else "FAIL"
        print(f"{status}  {type(layer).__name__:<10} relative error = {error:.2e}")

    print("\nAll checks passed." if all_passed else "\nSome checks FAILED.")
    raise SystemExit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
