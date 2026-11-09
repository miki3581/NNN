import numpy as np
from model import Model
from layers import Dense
from losses import MSE
from test_model import make_model, make_data
from optimizers import SGD, Adam


def test_params_num(rng: np.random.Generator):
    """
    Checking that model.params() returns 4 pairs that are references (not copies) 
    to W, dW, b, db of both Dense layers
    """
    
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
    """
    Checking if params before backward raises RuntimeError
    """
    
    model = make_model(task="regression", rng=rng)
    try:
        model.params()
    except RuntimeError:
        pass
    else:
        assert False, "params before backward did not raise RuntimeError"


def test_step(task: str):
    """
    Checking value of two parameters agains hand calculated values
    SGD: p1 = 0.95, p2 = 2.1; Adam: p1 ~ 0.9, p2 ~ 2.1
    """
    
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
    """
    Checking if momentum correctly influences final parameter value
    lr, momentum = 0.1, 0.9, p = 0.439
    """
    
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
    """
    Checking that invalid hyperparameters raise ValueError
    and boundary values that are allowed (momentum=0, beta=0) do not
    """
    
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
    """
    Minimizing convex function for 500 steps, final max gradient p must be below 1e-3
    Gradient is recomputed from the current p in every step (p.copy(), because p is
    updated in place)
    """
    
    p = np.array([1.0 , -2.0])
    if task == "SGD":
        opt = SGD(lr=0.1)
    elif task == "Adam":
        opt = Adam(lr=0.1)
    for _ in range(500):
        opt.step([(p, p.copy())])
    assert np.max(np.abs(p)) < 1e-3, "Parabola is divergent"


def test_model_w_opt(rng: np.random.Generator):
    """
    Testing small regression model with MSE and Adam for 200 steps
    Checking if loss is lowering fast enough by comparing loss form first and last iteration
    """
    
    model = make_model(task="regression", rng=rng)
    X, y = make_data(task="regression", rng=rng, n=20)
    opt = Adam(lr=0.01)
    loss = MSE()
    L0 = None
    for _ in range(200):
        Z = model.forward(X)
        L = loss.forward(Z, y)
        if L0 is None:
            L0 = L
        dZ = loss.backward()
        dX = model.backward(dZ)
        opt.step(model.params())
    assert L <= L0 / 2, "Loss after 200 steps not small enough"



def main():
    """
    Runs all tests and prints PASS/FAIL for each
    """
    
    rng = np.random.default_rng(42)
    tests = [(test_params_num, (rng,)), (test_order, (rng,)), (test_step, ("SGD",)),
             (test_step, ("Adam",)), (test_momentum, ()), (test_validation, ("SGD",)),
             (test_validation, ("Adam",)), (test_parabola, ("SGD", )), (test_parabola, ("Adam",)),
             (test_model_w_opt, (rng,))]

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