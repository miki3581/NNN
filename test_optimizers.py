import numpy as np
from model import Model
from layers import Dense
from losses import MSE
from test_model import make_model, make_data
from optimizers import SGD, Adam


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


def test_step(task: str):

    p1 = np.array([[1.0]])
    p2 = np.array([[2.0]])
    g1 = np.array([[0.5]])
    g2 = np.array([[-1.0]])
    exp_p_SGD = [0.95, 2.1]
    exp_p_Adam = [0.9, 2.1]
    if task == "SGD":
        opt = SGD(lr=0.1)
        opt.step([(p1, g1), (p2, g2)])
        assert np.allclose(exp_p_SGD[0], p1), "Calculated p1 differs from theoretical"
        assert np.allclose(exp_p_SGD[1], p2), "Calculated p2 differs from theoretical"
    elif task == "Adam":
        opt = Adam(lr=0.1)
        opt.step([(p1, g1), (p2, g2)])
        assert np.allclose(exp_p_Adam[0], p1), "Calculated p1 differs from theoretical"
        assert np.allclose(exp_p_Adam[1], p2), "Calculated p2 differs from theoretical"


def test_momentum():

    p1 = np.array([[1.0], [1.0]])
    p2 = np.array([1.0, 1.0])
    g1 = np.ones_like(p1)
    g2 = np.ones_like(p2)
    lr, momentum = 0.1, 0.9
    exp_p = 0.439
    opt = SGD(lr=lr, momentum=momentum)
    opt.step([(p1, g1), (p2, g2)])
    opt.step([(p1, g1), (p2, g2)])
    opt.step([(p1, g1), (p2, g2)])

    assert np.allclose(exp_p, p1), "Calculated p1 differs from theoretical"
    assert np.allclose(exp_p, p2), "Calculated p2 differs from theoretical"


def test_validation(task: str):

    if task == "SGD":
        try:
            SGD(lr=0.0)
        except ValueError:
            pass
        else:
            assert False, "lr <= 0 didn't return error"

        try:
            SGD(lr=0.1, momentum=1.0)
        except ValueError:
            pass
        else:
            assert False, "momentum = 1 didn't return error"
            
        SGD(lr=0.1, momentum=0.0)

        try:
            SGD(lr=0.1, momentum=-0.1)
        except ValueError:
            pass
        else:
            assert False, "momentum = -0.1 didn't return error"
    
    elif task == "Adam":
        try:
            Adam(lr=0.0)
        except ValueError:
            pass
        else:
            assert False, "lr <= 0 didn't return error"
        
        try:
            Adam(beta1=1.0)
        except ValueError:
            pass
        else:
            assert False, "beta1 >= 1 didn't return error"
        
        Adam(beta1=0.0)

        try:
            Adam(beta2=1.0)
        except ValueError:
            pass
        else:
            assert False, "beta2 >= 1 didn't return error"
        
        Adam(beta2=0.0)

        try:
            Adam(eps=0.0)
        except ValueError:
            pass
        else:
            assert False, "eps <= 0 didn't return error"


def test_parabola(task: str):

    p = np.array([1.0 , -2.0])
    if task == "SGD":
        opt = SGD(lr=0.1)
    elif task == "Adam":
        opt = Adam(lr=0.1)
    for _ in range(500):
        opt.step([(p, p.copy())])
    assert np.max(np.abs(p)) < 1e-3, "Parabola is divergent"


def main():

    rng = np.random.default_rng(42)
    tests = [(test_params_num, (rng,)), (test_order, (rng,)), (test_step, ("SGD",)),
             (test_step, ("Adam",)), (test_momentum, ()), (test_validation, ("SGD",)),
             (test_validation, ("Adam",)), (test_parabola, ("SGD", )), (test_parabola, ("Adam",))]

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