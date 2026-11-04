import numpy as np
from model import Model
from layers import Dense
from activations import Tanh
from gradcheck import num_grad, rel_error, TOLERANCE
from losses import MSE, BCE, CE
from numpy.typing import NDArray

def make_model(task: str, rng: np.random.Generator, l2: float=0.0) -> Model:
    """
    Helper function building a small network (Dense(4, 5) -> Tanh -> Dense(5, k))
    k = 3 for multiclass, 1 otherwise; Xavier init, same l2 in both Dense layers
    """

    if task == "multiclass":
        k = 3
    else:
        k = 1
    layers = [Dense(4, 5, rng, init="xavier", l2=l2), Tanh(), Dense(5, k, rng, init="xavier", l2=l2)]

    return Model(layers, task)

def make_data(task: str, rng: np.random.Generator, n: int=6) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """
    Helper function creating X and y arrays
    float (n, 1) for regression, 0/1 (n, 1) for binary, one-hot (n, 3) for multiclass
    """

    k = 3
    if task == "multiclass":
        y = np.eye(k)[rng.integers(0, k, size=n)]
    elif task == "regression":
        y = rng.normal(size=(n, 1))
    else:
        y = rng.integers(0, 2, size=(n, 1)).astype(float)
    X = rng.normal(size=(n, 4))
    
    return X, y


def test_shapes(task: str, rng: np.random.Generator):
    """
    Checking shape of dX, dW and db against X, W and b
    """
    
    model = make_model(task=task, rng=rng)
    X, y = make_data(task=task, rng=rng)

    Z = model.forward(X)
    assert Z.shape == y.shape, f"Z [{Z.shape}] and y [{y.shape}] shape don't match"

    dZ = np.ones_like(Z)
    dX = model.backward(dZ)

    for layer in model.layers:
        if isinstance(layer, Dense):
            assert layer.dW.shape == layer.W.shape, f"W [{layer.W.shape}] and dW [{layer.dW.shape}] shape don't match"
            assert layer.db.shape == layer.b.shape, f"b [{layer.b.shape}] and db [{layer.db.shape}] shape don't match"
            assert dX.shape == X.shape, f"X [{X.shape}] and dX [{dX.shape}] shape don't match"


def test_order(task: str, rng: np.random.Generator):
    """
    backward called before forward raises RuntimeError
    """
    
    model = make_model(task=task, rng=rng)
    dZ = np.array([[1.0], [3.0]])

    try:
        model.backward(dZ)
    except RuntimeError:
        pass
    else:
        assert False, "Backward before forward did not raise RuntimeError"


def test_bad_task(task: str, rng: np.random.Generator):
    """
    Unknown task returns ValueError
    """

    try:
        model = make_model(task=task, rng=rng)
    except ValueError:
        pass
    else:
        assert False, "task='unknown' did not raise ValueError"


def test_l2_penalty(rng: np.random.Generator):
    """
    Computed L2 penalty value matches value calculated by hand
    2 examples 1st with different l2 value for each layer, 2nd with l2=0
    """
    
    task = "binary"
    model = make_model(task=task, rng=rng)
    model2 = make_model(task=task, rng=rng)
    model.layers[0].l2 = 0.1
    model.layers[2].l2 = 0.2
    for layer in model.layers:
        if isinstance(layer, Dense):
            layer.W = np.ones_like(layer.W)
            layer.b = np.ones_like(layer.b)
    for layer in model2.layers:
        if isinstance(layer, Dense):
            layer.W = np.ones_like(layer.W)
            layer.b = np.ones_like(layer.b)

    assert np.isclose(model.l2_penalty(), 1.5), "l2 penalty value not as expected for l2 = 0.1/0.2"
    assert np.isclose(model2.l2_penalty(), 0.0), "l2 penalty value not as expected for l2 = 0.0"
    

def test_gradcheck(task: str, loss_cls: type, rng: np.random.Generator):
    """
    Compare analytic dW and db with numerical gradient of loss + L2 
    """

    model = make_model(task=task, rng=rng, l2 = 0.1)
    X, y = make_data(task=task, rng=rng)
    loss = loss_cls()

    Z = model.forward(X)
    L = loss.forward(Z, y)
    dZ = model.backward(loss.backward())
    for layer in model.layers:
        if isinstance(layer, Dense):
            J = lambda: loss.forward(model.forward(X), y) + model.l2_penalty()
            grad = num_grad(J, layer.W)
            assert rel_error(layer.dW, grad) < TOLERANCE, "Difference between calculations and estimate larger than tolerance"
            grad = num_grad(J, layer.b)
            assert rel_error(layer.db, grad) < TOLERANCE, "Difference between calculations and estimate larger than tolerance"


def test_predict(task: str, rng: np.random.Generator):
    """
    regression: predict returns logits, predict_proba raises ValueError
    binary: shape, probabilities in (0, 1), integer 0/1 labels consistent with Z >= 0
    multiclass: shape, probability rows sum to 1, integer labels 0..2 consistent with argmax of Z
    """
    
    n = 6
    model = make_model(task=task, rng=rng)
    X, _ = make_data(task=task, rng=rng)
    
    Z = model.forward(X)

    if task == "regression":
        pred = model.predict(X)
        assert pred.shape == (n, 1), "Wrong shape"
        try:
            model.predict_proba(X)
        except ValueError:
            pass
        else:
            assert False, ""
        assert np.allclose(pred, Z), "pred =/= Z"

    elif task == "binary":
        proba = model.predict_proba(X)
        assert proba.shape == (n, 1), "proba is wrong shape"
        assert np.all((proba > 0) & (proba < 1)), "Probability not between 0 and 1"

        pred = model.predict(X)
        assert pred.shape == (n, 1), "pred is wrong shape"
        assert np.all((pred == 0) | (pred == 1)), "pred not 0 or 1"
        assert np.issubdtype(pred.dtype, np.integer), "Pred isn't integer"
        assert np.array_equal((Z >= 0).astype(int), pred), "pred isn't equal logit"

    elif task == "multiclass":
        proba = model.predict_proba(X)
        assert proba.shape == (n, 3), "proba is wrong shape"
        assert np.all((proba > 0) & (proba < 1)), "Probability not between 0 and 1"
        assert np.allclose(proba.sum(axis=1), 1.0), "Probability doesn't sum to 1"
        
        pred = model.predict(X)
        assert pred.shape == (n, 1), "pred is wrong shape"
        assert np.issubdtype(pred.dtype, np.integer), "pred isn't integer"
        assert np.all((pred >= 0) & (pred < 3)), "pred not 0, 1 or 2"
        assert np.array_equal(np.argmax(Z, axis=1, keepdims=True), pred), "pred isn't equal logit"

def main():
    """
    Run all tests, print PASS/FAIL for each and exit with code 1 if any test failed
    """

    rng = np.random.default_rng(42)
    tests = [(test_shapes, ("regression", rng)), (test_shapes, ("binary", rng)), (test_shapes, ("multiclass", rng)),
             (test_order, ("regression", rng)), (test_order, ("binary", rng)), (test_order, ("multiclass", rng)),
             (test_bad_task, ("unknown", rng)), (test_l2_penalty, (rng,)),
             (test_gradcheck, ("regression", MSE, rng)), (test_gradcheck, ("binary", BCE, rng)), (test_gradcheck, ("multiclass", CE, rng)),
             (test_predict, ("regression", rng)), (test_predict, ("binary", rng)), (test_predict, ("multiclass", rng))]

    all_passed = True
    for test, args in tests:
        try:
            test(*args)
        except Exception as e:
            all_passed = False
            if args and isinstance(args[0], str):
                print(f"FAIL  {test.__name__} {args[0]} {type(e).__name__}: {e}")
            else:
                print(f"FAIL  {test.__name__} {type(e).__name__}: {e}")
        else:
            if args and isinstance(args[0], str):
                print(f"PASS  {test.__name__} {args[0]}")
            else:
                print(f"PASS  {test.__name__}")

    print("\nAll checks passed." if all_passed else "\nSome checks FAILED.")
    raise SystemExit(0 if all_passed else 1)

if __name__ == "__main__":
    main()