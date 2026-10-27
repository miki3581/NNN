import numpy as np
from gradcheck import num_grad, rel_error, TOLERANCE
from losses import MSE, BCE, CE


def test_order(loss_cls: type):
    """
    backward called before forward raises RuntimeError
    """
    
    loss = loss_cls()

    try:
        loss.backward()
    except RuntimeError:
        pass
    else:
        assert False, "Backward before forward did not raise RuntimeError"


def test_shapes(loss_cls: type):
    """
    Checking if Z and y have the expected shapes
    dZ and Z have the same shapes, L is float
    """

    Z = np.array([[1.0], [3.0]])
    y = np.zeros((1, ))
    loss = loss_cls()
    try:
        L = loss.forward(Z, y)
    except ValueError:
        pass
    else:
        assert False, "Z and y shapes doesn't match"

    Z = np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 6.0]])
    y = np.zeros((2, 3))
    L = loss.forward(Z, y)
    dZ = loss.backward()
    assert Z.shape == dZ.shape, "Z shape doesn't match"
    assert isinstance(L, float), "L isn't float"


def test_gradcheck(loss_cls: type, y: np.typing.NDArray[np.floating]):
    """
    Compare analytic dZ with numerical gradient
    """

    rng = np.random.default_rng(42)
    Z = rng.normal(size=(y.shape))
    
    loss = loss_cls()
    L = loss.forward(Z, y)
    dZ = loss.backward().copy()
    grad = num_grad(lambda: loss.forward(Z, y), Z)
    assert rel_error(dZ, grad) < TOLERANCE, "Difference between calculations and estimate larger than tolerance"


def test_mse_manual():
    """
    Computed L and dZ match values calculated by hand
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


def test_bce_manual():
    """
    Computed L and dZ match values calculated by hand
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


def test_ce_manual():
    """
    Computed L and dZ match values calculated by hand
    """

    loss = CE()
    Z = np.zeros((2, 3))
    y = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    L_exp = np.log(3)
    dZ_exp = [[-1/3, 1/6, 1/6], [1/6, 1/6, -1/3]]

    L = loss.forward(Z, y)
    dZ = loss.backward()

    assert np.isclose(L_exp, L), f"Calculated L different from theoretical: {L}"
    assert np.allclose(dZ_exp, dZ), f"Calculated dZ different from theoretical: {dZ}"


def test_bce_extreme_value():
    """
    Checks finite L and dZ for very large and small logits
    """

    Z = np.array([[1000.0], [-1000.0], [1000.0], [-1000.0]])
    y = np.array([[1.0], [0.0], [0.0], [1.0]])
    loss = BCE()
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        L = loss.forward(Z, y)
        dZ = loss.backward()
    assert np.isclose(L, 500.0), "L doesn't equal 500"
    assert np.all(np.isfinite(dZ)), "dZ is infinite"


def test_ce_extreme_value():
    """
    Checks finite L and dZ for very large and small logits
    """

    Z = np.array([[1000.0, 0.0, 0.0], [0.0, 1000.0, 0.0]])
    y = np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    loss = CE()
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        L = loss.forward(Z, y)
        dZ = loss.backward()
    assert np.isclose(L, 500.0), "L doesn't equal 500"
    assert np.all(np.isfinite(dZ)), "dZ is infinite"


def main():
    """
    Run all tests, print PASS/FAIL for each and exit with code 1 if any test failed
    """
    n, k = 2, 3
    rng = np.random.default_rng(42)

    tests = [(test_order, (MSE,)), (test_order,(BCE,)), (test_order,(CE,)),
             (test_shapes, (MSE,)), (test_shapes, (BCE,)), (test_shapes, (CE,)),
             (test_gradcheck, (MSE, rng.normal(size=(n, 1)))), 
             (test_gradcheck, (BCE, rng.integers(0, 2, size=(n, 1)).astype(float))), 
             (test_gradcheck, (CE, np.eye(k)[rng.integers(0, k, size=n)])),
             (test_mse_manual, ()), (test_bce_manual, ()), (test_ce_manual, ()),
             (test_bce_extreme_value, ()), (test_ce_extreme_value, ())]
    all_passed = True

    for test, args in tests:
        try:
            test(*args)
        except Exception as e:
            all_passed = False
            if args:
                print(f"FAIL  {test.__name__} {args[0].__name__} {type(e).__name__}: {e}")
            else:
                print(f"FAIL  {test.__name__} {type(e).__name__}: {e}")
        else:
            if args:
                print(f"PASS  {test.__name__} {args[0].__name__}")
            else:
                print(f"PASS  {test.__name__}")

    print("\nAll checks passed." if all_passed else "\nSome checks FAILED.")
    raise SystemExit(0 if all_passed else 1)
    

if __name__ == "__main__":
    main()