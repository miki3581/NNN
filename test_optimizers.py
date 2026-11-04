import numpy as np
from model import Model
from layers import Dense
from losses import MSE
from test_model import make_model, make_data


def test_params_num(rng: np.random.Generator):
    model = make_model(task="regression", rng=rng)
    X, y = make_data(task="regression", rng=rng)
    Z = model.forward(X)
    loss = MSE()
    L = loss.forward(Z, y)
    dZ = loss.backward()
    dX = model.backward(dZ)
    assert len(model.params()) == 4, "Wrong number of parameters"

    P = model.params()
    assert P[0][0] is model.layers[0].W, "params[0][0] is not layers[0].W"
    assert P[0][1] is model.layers[0].dW, "params[0][1] is not layers[0].dW"
    assert P[1][0] is model.layers[0].b, "params[1][0] is not layers[0].b"
    assert P[1][1] is model.layers[0].db, "params[1][1] is not layers[0].db"
    assert P[2][0] is model.layers[2].W, "params[2][0] is not layers[2].W"
    assert P[2][1] is model.layers[2].dW, "params[2][1] is not layers[2].dW"
    assert P[3][0] is model.layers[2].b, "params[3][0] is not layers[2].b"
    assert P[3][1] is model.layers[2].db, "params[3][1] is not layers[2].db"


def test_order(rng: np.random.Generator):

    model = make_model(task="regression", rng=rng)
    try:
        model.params()
    except RuntimeError:
        pass
    else:
        assert False, "params before backward did not raise RuntimeError"


def main():

    rng = np.random.default_rng(42)
    tests = [(test_params_num, (rng,)), (test_order, (rng,))]

    all_passed = True
    for test, args in tests:
        try:
            test(*args)
        except Exception as e:
            all_passed = False
            print(f"FAIL  {test.__name__:<25} {type(e)}: {e}")
        else:
            print(f"PASS  {test.__name__}")

    print("\nAll checks passed." if all_passed else "\nSome checks FAILED.")
    raise SystemExit(0 if all_passed else 1)

if __name__ == "__main__":
    main()