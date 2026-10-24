from layers import Dense
import numpy as np


def test_shapes():
    
    d_in, d_out = 7, 3

    for N in (1, 8):
        rng = np.random.default_rng(42)
        layer = Dense(d_in, d_out, rng)
        X = np.ones((N, d_in))
        Z = layer.forward(X)
        dZ = np.ones((np.shape(Z)))
        dX = layer.backward(dZ)

        assert Z.shape == (N, d_out), f"Wrong Z shape : {Z.shape}"
        assert dX.shape == (N, d_in), f"Wrong dX shape: {dX.shape}"
        assert layer.dW.shape == (d_in, d_out), f"Wrong dW shape: {layer.dW.shape}"
        assert layer.db.shape == (1, d_out), f"Wrong db shape: {layer.db.shape}"


def test_order():
    
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
    
    d_in, d_out = 7, 3
    rng = np.random.default_rng(42)

    try:
        layer = Dense(d_in, d_out, rng, init = 'a')
    except ValueError:
        pass
    else:
        assert False, "init='a' did not raise ValueError"


