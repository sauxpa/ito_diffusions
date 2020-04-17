import numpy as np
from ito_diffusions import BM_multi_d, GBM_multi_d,\
    Vasicek_multi_d, BlackKarasinski_multi_d,\
    SABR, SABR_AS_lognorm, SABR_AS_loglogistic, SABR_tanh


def test_BM_multi_d(seed: int = 0):
    np.random.seed(seed)
    x0 = [0.0, 0.0]
    X = BM_multi_d(x0=x0)  # Test with default arguments
    df = X.simulate()
    for key in X.keys:
        msg = 'Generated BM_multi_d({:s}) path is not stochastic.'\
            .format(key)
        assert df.std()[key] > 0.0, msg


def test_GBM_multi_d(seed: int = 0):
    np.random.seed(seed)
    x0 = [1.0, 1.0]
    X = GBM_multi_d(x0=x0)  # Test with default arguments
    df = X.simulate()
    for key in X.keys:
        msg = 'Generated GBM_multi_d({:s}) path is not stochastic.'\
            .format(key)
        assert df.std()[key] > 0.0, msg


def test_Vasicek_multi_d(seed: int = 0):
    np.random.seed(seed)
    x0 = [0.0, 0.0]
    X = Vasicek_multi_d(x0=x0)  # Test with default arguments
    df = X.simulate()
    for key in X.keys:
        msg = 'Generated Vasicek_multi_d({:s}) path is not stochastic.'\
            .format(key)
        assert df.std()[key] > 0.0, msg


def test_BlackKarasinski_multi_d(seed: int = 0):
    np.random.seed(seed)
    x0 = [0.5, 0.5]
    X = BlackKarasinski_multi_d(x0=x0)  # Test with default arguments
    df = X.simulate()
    for key in X.keys:
        msg = 'Generated BK_multi_d({:s}) path is not stochastic.'\
            .format(key)
        assert df.std()[key] > 0.0, msg


def test_SABR(seed: int = 0):
    np.random.seed(seed)
    x0 = [0.05, 0.4]
    X = SABR(x0=x0)  # Test with default arguments
    df = X.simulate()
    for key in X.keys:
        msg = 'Generated SABR({:s}) path is not stochastic.'\
            .format(key)
        assert df.std()[key] > 0.0, msg


def test_SABR_AS_lognorm(seed: int = 0):
    np.random.seed(seed)
    x0 = [0.05, 0.4]
    X = SABR_AS_lognorm(x0=x0)  # Test with default arguments
    df = X.simulate()
    for key in X.keys:
        msg = 'Generated SABR_AS_lognorm({:s}) path is not stochastic.'\
            .format(key)
        assert df.std()[key] > 0.0, msg


def test_SABR_AS_loglogistic(seed: int = 0):
    np.random.seed(seed)
    x0 = [0.05, 0.4]
    X = SABR_AS_loglogistic(x0=x0)  # Test with default arguments
    df = X.simulate()
    for key in X.keys:
        msg = 'Generated SABR_AS_loglogistic({:s}) path is not stochastic.'\
            .format(key)
        assert df.std()[key] > 0.0, msg


def test_SABR_tanh(seed: int = 0):
    np.random.seed(seed)
    x0 = [0.05, 0.4]
    X = SABR_tanh(x0=x0)  # Test with default arguments
    df = X.simulate()
    for key in X.keys:
        msg = 'Generated SABR_tanh({:s}) path is not stochastic.'\
            .format(key)
        assert df.std()[key] > 0.0, msg
