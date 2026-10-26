import numpy as np
from gradcheck import num_grad, rel_error, TOLERANCE
from losses import MSE, BCE


def test_mse_order():
    """
    backward called before forward raises RuntimeError
    """
    
    loss = MSE()

    try:
        loss.backward()
    except RuntimeError:
        pass
    else:
        assert False, "Backward before forward did not raise RuntimeError"


def test_mse_manual():
    """
    L and dZ match values computed by hand
    """

    loss = MSE()
    Z = np.array([[1.0], [3.0]])
    y = np.zeros((2, 1))
    L_exp = 2.5
    dZ_exp = [[0.5], [1.5]]

    L = loss.forward(Z, y)
    dZ = loss.backward()

    assert np.isclose(L_exp, L), f"Calculated L different from theoretical: {L}"
    assert np.allclose(dZ_exp, dZ), f"Calculated dZ different from theoretical: {dZ}"


def test_mse_shapes():
    """
    Z and y have the expected shapes
    """

    Z = np.array([[1.0], [3.0]])
    y = np.zeros((1, ))
    loss = MSE()
    try:
        L = loss.forward(Z, y)
    except ValueError:
        pass
    else:
        assert False, "Z and y shapes doesn't match"

    Z = np.array([[1.0], [3.0]])
    y = np.zeros((2, 1))
    L = loss.forward(Z, y)
    dZ = loss.backward()
    assert Z.shape == dZ.shape, "Z shape doesn't match"
    assert isinstance(L, float), "L isn't float"


def test_mse_gradcheck():
    """
    Compare calculated Z with numerical gradient
    """

    rng = np.random.default_rng(42)
    Z = rng.normal(size=(2, 1))
    y = rng.normal(size=(2, 1))
    
    loss = MSE()
    L = loss.forward(Z, y)
    dZ = loss.backward().copy()
    grad = num_grad(lambda: loss.forward(Z, y), Z)
    assert rel_error(dZ, grad) < TOLERANCE, "Difference between calculations and estimate larger than tolerance"


def test_bce_order():
    """
    backward called before forward raises RuntimeError
    """
    
    loss = BCE()

    try:
        loss.backward()
    except RuntimeError:
        pass
    else:
        assert False, "Backward before forward did not raise RuntimeError"


def test_bce_manual():
    """
    L and dZ match values computed by hand
    """

    loss = BCE()
    Z = np.zeros((2, 1))
    y = np.array([[0.0], [1.0]])
    L_exp = np.log(2)
    dZ_exp = [[0.25], [-0.25]]

    L = loss.forward(Z, y)
    dZ = loss.backward()

    assert np.isclose(L_exp, L), f"Calculated L different from theoretical: {L}"
    assert np.allclose(dZ_exp, dZ), f"Calculated dZ different from theoretical: {dZ}"


def test_bce_shapes():
    """
    Checking if Z and y have the expected shapes
    dZ and Z have the same shapes, y is float
    """

    Z = np.array([[1.0], [3.0]])
    y = np.zeros((1, ))
    loss = BCE()
    try:
        L = loss.forward(Z, y)
    except ValueError:
        pass
    else:
        assert False, "Z and y shapes doesn't match"

    Z = np.array([[1.0], [3.0]])
    y = np.zeros((2, 1))
    L = loss.forward(Z, y)
    dZ = loss.backward()
    assert Z.shape == dZ.shape, "Z shape doesn't match"
    assert isinstance(L, float), "L isn't float"


def test_bce_gradcheck():
    """
    Compare analytic dZ with numerical gradient of L
    rng.integers in y to randomize 0/1 label
    """

    rng = np.random.default_rng(42)
    Z = rng.normal(size=(2, 1))
    y = rng.integers(0, 2, size=(2, 1)).astype(float)
    
    loss = BCE()
    L = loss.forward(Z, y)
    dZ = loss.backward().copy()
    grad = num_grad(lambda: loss.forward(Z, y), Z)
    assert rel_error(dZ, grad) < TOLERANCE, "Difference between calculations and estimate larger than tolerance"


def test_bce_extreme_value():
    """
    
    """

    Z = np.array([[1000.0], [-1000.0], [1000.0], [-1000.0]])
    y = np.array([[1.0], [0.0], [0.0], [1.0]])
    loss = BCE()
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        L = loss.forward(Z, y)
        dZ = loss.backward()
    assert np.isclose(L, 500.0), "L doesn't equal 500"
    assert np.all(np.isfinite(dZ)), "dZ is infinite"

def main():
    """
    Run all tests, print PASS/FAIL for each and exit with code 1 if any test failed
    """

    tests = [test_mse_order, test_mse_manual, test_mse_shapes, test_mse_gradcheck,
             test_bce_order, test_bce_manual, test_bce_shapes, test_bce_gradcheck, test_bce_extreme_value]
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