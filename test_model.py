import numpy as np
from model import Model
from layers import Dense
from activations import Tanh
from numpy.typing import NDArray

def make_model(task: str, rng: np.random.Generator, l2: float=0.0) -> Model:

    if task == "multiclass":
        k = 3
    else:
        k = 1
    layers = [Dense(4, 5, rng, init="xavier", l2=l2), Tanh(), Dense(5, k, rng, init="xavier", l2=l2)]

    return Model(layers, task)

def make_data(task: str, rng: np.random.Generator, n: int=6) -> tuple[NDArray[np.floating], NDArray[np.floating]]:

    k = 3
    if task == "multiclass":
        y = np.eye(k)[rng.integers(0, k, size=n)]
    elif task == "regression":
        y = rng.normal(size=(n, 1))
    else:
        y = rng.integers(0, 2, size=(n, 1)).astype(float)
    X = rng.normal(size=(n, 4))
    
    return X, y


def test_shapes(task: str, rng: np.random):

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


def test_order(task: str, rng: np.random):
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


def main():

    rng = np.random.default_rng(42)
    tests = [(test_shapes, ('regression', rng)), (test_shapes, ("binary", rng)), (test_shapes, ("multiclass", rng)),
             (test_order, ('regression', rng)), (test_order, ("binary", rng)), (test_order, ("multiclass", rng))]

    all_passed = True
    for test, args in tests:
        try:
            test(*args)
        except Exception as e:
            all_passed = False
            if args:
                print(f"FAIL  {test.__name__} {args[0]} {type(e).__name__}: {e}")
            else:
                print(f"FAIL  {test.__name__} {type(e).__name__}: {e}")
        else:
            if args:
                print(f"PASS  {test.__name__} {args[0]}")
            else:
                print(f"PASS  {test.__name__}")

    print("\nAll checks passed." if all_passed else "\nSome checks FAILED.")
    raise SystemExit(0 if all_passed else 1)

if __name__ == "__main__":
    main()