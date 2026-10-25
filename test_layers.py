"""
Tests for the Dense layer: shapes, call order, init, L2, extreme inputs and gradient check.
"""
from layers import Dense
from gradcheck import num_grad, rel_error, TOLERANCE
import numpy as np


def test_shapes():
    """
    Z, dX, dW and db have the expected shapes for a single sample (N=1) and a batch (N=8).
    """

    d_in, d_out = 7, 3

    for N in (1, 8):
        rng = np.random.default_rng(42)
        layer = Dense(d_in, d_out, rng)
        X = np.ones((N, d_in))
        Z = layer.forward(X)
        dZ = np.ones(Z.shape)
        dX = layer.backward(dZ)

        assert Z.shape == (N, d_out), f"Wrong Z shape : {Z.shape}"
        assert dX.shape == (N, d_in), f"Wrong dX shape: {dX.shape}"
        assert layer.dW.shape == (d_in, d_out), f"Wrong dW shape: {layer.dW.shape}"
        assert layer.db.shape == (1, d_out), f"Wrong db shape: {layer.db.shape}"


def test_order():
    """
    backward called before forward raises RuntimeError.
    """

    d_in, d_out = 7, 3
    rng = np.random.default_rng(42)    
    layer = Dense(d_in, d_out, rng)
    dZ = np.ones((1, d_out))

    try:
        layer.backward(dZ)
    except RuntimeError:
        pass
    else:
        assert False, "Backward before forward did not raise RuntimeError"


def test_invalid_init():
    """
    An unknown init name raises ValueError.
    """

    d_in, d_out = 7, 3
    rng = np.random.default_rng(42)

    try:
        layer = Dense(d_in, d_out, rng, init = 'a')
    except ValueError:
        pass
    else:
        assert False, "init='a' did not raise ValueError"


def test_manual_example():
    """
    forward and backward match values computed by hand for a small 2 -> 3 layer.
    """

    rng = np.random.default_rng(42)
    d_in, d_out = 2, 3
    layer = Dense(d_in, d_out, rng, l2 = 0.0)
    layer.W = np.array([[1.0, 0.0, -1.0], [2.0, 1.0,  0.0]])
    layer.b = np.array([[0.5, 0.0, -0.5]])
    X = np.array([[1.0, 2.0], [0.0, 0.0]])
    expected_Z = np.array([[5.5, 2.0, -1.5],[0.5, 0.0, -0.5]])
    expected_dW = np.array([[1.0, 1.0 ,1.0], [2.0, 2.0, 2.0]])
    expected_db = np.array([[2.0, 2.0, 2.0]])
    expected_dX = np.array([[0.0, 3.0], [0.0, 3.0]])
    Z = layer.forward(X)
    dZ = np.ones(Z.shape)
    dX = layer.backward(dZ)
    assert np.allclose(Z, expected_Z), f"Wrong Z: {Z}"
    assert np.allclose(layer.dW, expected_dW), f"Wrong dW: {layer.dW}"
    assert np.allclose(layer.db, expected_db), f"Wrong db: {layer.db}"
    assert np.allclose(dX, expected_dX), f"Wrong dX: {dX}"


def test_init():
    """
    He and Xavier weights have mean close to 0 and std within 1% of the expected value; bias is zero.
    """

    d_in, d_out = 1000, 500
    expected_he = np.sqrt(2 / d_in)
    expected_xavier = np.sqrt(2 / (d_in + d_out))

    for init, expected_std in (("he", expected_he), ("xavier", expected_xavier)):
        rng = np.random.default_rng(42)
        layer = Dense(d_in, d_out, rng, init = init, l2 = 0.0)
        std = layer.W.std()
        mean = layer.W.mean()
        assert abs(std / expected_std - 1) < 0.01, f"Std diff too large {init}: std {std:.5f}, expected {expected_std:.5f}"
        assert abs(mean) < 0.01 * expected_std, f"Mean too large {init}: mean {mean:.5f}"
        assert np.all(layer.b == 0), f"{init} b not zeroed"


def test_repeatability():
    """
    The same seed gives identical weights; one generator gives different weights to consecutive layers.
    """

    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    d_in, d_out = 7, 3
    
    layer1 = Dense(d_in, d_out, rng1, l2 = 0.0)
    layer2 = Dense(d_in, d_out, rng2, l2 = 0.0)
    assert np.array_equal(layer1.W, layer2.W), f"Weights not equal: {layer1.W, layer2.W}"

    rng = np.random.default_rng(42)
    layer3 = Dense(d_in, d_out, rng, l2 = 0.0)
    layer4 = Dense(d_in, d_out, rng, l2 = 0.0)
    assert not np.array_equal(layer3.W, layer4.W), f"Weights are the same: {layer3.W, layer4.W}"


def test_L2():
    """
    L2 changes only dW (by exactly l2 * W); db and dX do not depend on l2.
    """

    N, d_in, d_out = 8, 7, 3
    rng = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    layer1 = Dense(d_in, d_out, rng, l2 = 0.0)
    layer2 = Dense(d_in, d_out, rng2, l2 = 0.1)

    rng3 = np.random.default_rng(40)
    X = rng3.normal(size=(N, d_in))
    dZ = rng3.normal(size=(N, d_out))

    layer1.forward(X)
    dX1 = layer1.backward(dZ)
    layer2.forward(X)
    dX2 = layer2.backward(dZ)
    data_grad = X.T @ dZ

    assert np.array_equal(layer1.dW, data_grad), "dW with L2=0 differs from X.T @ dZ"
    assert np.allclose(layer2.dW - data_grad, 0.1 * layer2.W), "dW with L2 differs from 0.1 weight"
    assert np.array_equal(layer1.db, layer2.db), "db differs between layers"
    assert np.array_equal(dX1, dX2), "dX differs between layers"


def test_X_all_zeros():
    """
    For X = 0 the output equals the bias and dW reduces to the L2 term.
    """

    N, d_in, d_out = 8, 7, 3
    rng = np.random.default_rng(42)
    layer = Dense(d_in, d_out, rng, l2 = 0.1)
    layer.b = rng.normal(size = (1, d_out))
    X = np.zeros((N, d_in))
    Z = layer.forward(X)
    dZ = rng.normal(size = (Z.shape))
    dX = layer.backward(dZ)
    assert np.allclose(Z, layer.b), "Z differs from b while X are zeros"
    assert np.allclose(layer.dW, 0.1 * layer.W), "dW differs from 0.1 W while X are zeros"

def test_X_very_large():
    """
    Very large inputs (~1e6) give finite Z, dX, dW and db.
    """
    N, d_in, d_out = 8, 7, 3
    rng = np.random.default_rng(42)
    layer = Dense(d_in, d_out, rng, l2 = 0.1)
    X = 1e6 * rng.normal(size = (N, d_in))
    Z = layer.forward(X)
    dZ = rng.normal(size = (Z.shape))
    dX = layer.backward(dZ)
    assert np.all(np.isfinite(Z)), "Z not finite for large X"
    assert np.all(np.isfinite(dX)), "dX not finite for large X"
    assert np.all(np.isfinite(layer.dW)), "dW not finite for large X"
    assert np.all(np.isfinite(layer.db)), "db not finite for large X"


def test_gradcheck_dense():
    """
    Compare dX, dW and db from backward with numerical gradients of
    L = sum(dA * forward(X)) + (l2 / 2) * sum(W ** 2), for He and Xavier init.
    """

    N, d_in, d_out = 4, 5, 3
    rng = np.random.default_rng(42)

    for init in ("he", "xavier"):
        layer = Dense(d_in, d_out, rng, init=init, l2=0.1)
        X = rng.normal(size=(N, d_in))
        dA = rng.normal(size=(N, d_out))

        def loss():
            return np.sum(dA * layer.forward(X)) + 0.5 * layer.l2 * np.sum(layer.W ** 2)

        layer.forward(X)
        dX = layer.backward(dA)
        dW = layer.dW.copy()
        db = layer.db.copy()

        num_dX = num_grad(loss, X)
        num_dW = num_grad(loss, layer.W)
        num_db = num_grad(loss, layer.b)

        for name, analytic, numeric in (("dX", dX, num_dX), ("dW", dW, num_dW), ("db", db, num_db)):
            error = rel_error(analytic, numeric)
            assert error < TOLERANCE, f"{init}: {name} relative error {error:.2e}"


def main():
    """
    Run all tests, print PASS/FAIL for each and exit with code 1 if any test failed.
    """
    tests = [test_shapes, test_order, test_invalid_init, 
             test_manual_example, test_init, test_repeatability, 
             test_L2, test_X_all_zeros, test_X_very_large, test_gradcheck_dense]

    all_passed = True
    for test in tests:
        try:
            test()
        except Exception as e:
            all_passed = False
            print(f"FAIL  {test.__name__:<25} {type(e).__name__}: {e}")
        else:
            print(f"PASS  {test.__name__}")

    print("\nAll checks passed." if all_passed else "\nSome checks FAILED.")
    raise SystemExit(0 if all_passed else 1)


if __name__ == "__main__":
    main()