import numpy as np
from scipy.stats import arcsine
from ito_diffusions import BM, GBM, BMPeriodic, SLN,\
    Vasicek, CIR, BlackKarasinski, pseudo_GBM, Alpha_pinned_BM, F_pinned_BM,\
    FBM, Levy, Lognormal_multifractal


def test_BM(seed: int = 0):
    np.random.seed(seed)
    X = BM()  # Test with default arguments
    df = X.simulate()
    msg = 'Generated BM path is not stochastic.'
    assert df.std()['spot'] > 0.0, msg


def test_GBM(seed: int = 0):
    np.random.seed(seed)
    X = GBM()  # Test with default arguments
    df = X.simulate()
    msg = 'Generated GBM path is not stochastic.'
    assert df.std()['spot'] > 0.0, msg


def test_BMPeriodic(seed: int = 0):
    np.random.seed(seed)
    X = BMPeriodic()  # Test with default arguments
    df = X.simulate()
    msg = 'Generated BMPeriodic path is not stochastic.'
    assert df.std()['spot'] > 0.0, msg


def test_SLN(seed: int = 0):
    np.random.seed(seed)
    X = SLN()  # Test with default arguments
    df = X.simulate()
    msg = 'Generated SLN path is not stochastic.'
    assert df.std()['spot'] > 0.0, msg


def test_Vasicek(seed: int = 0):
    np.random.seed(seed)
    X = Vasicek()  # Test with default arguments
    df = X.simulate()
    msg = 'Generated Vasicek path is not stochastic.'
    assert df.std()['spot'] > 0.0, msg


def test_CIR(seed: int = 0):
    np.random.seed(seed)
    X = CIR()  # Test with default arguments
    df = X.simulate()
    msg = 'Generated CIR path is not stochastic.'
    assert df.std()['spot'] > 0.0, msg


def test_BlackKarasinski(seed: int = 0):
    np.random.seed(seed)
    X = BlackKarasinski()  # Test with default arguments
    df = X.simulate()
    msg = 'Generated BlackKarasinski path is not stochastic.'
    assert df.std()['spot'] > 0.0, msg


def test_pseudo_GBM(seed: int = 0):
    np.random.seed(seed)
    X = pseudo_GBM()  # Test with default arguments
    df = X.simulate()
    msg = 'Generated pseudo_GBM path is not stochastic.'
    assert df.std()['spot'] > 0.0, msg


def test_Alpha_pinned_BM(seed: int = 0):
    np.random.seed(seed)
    X = Alpha_pinned_BM()  # Test with default arguments
    df = X.simulate()
    msg = 'Generated Alpha_pinned_BM path is not stochastic.'
    assert df.std()['spot'] > 0.0, msg


def test_F_pinned_BM(seed: int = 0):
    np.random.seed(seed)
    distr = arcsine(loc=0.0, scale=1.0)
    X = F_pinned_BM(x0=0.0, distr=distr, pin=1.0)
    df = X.simulate()
    msg = 'Generated F_pinned_BM path is not stochastic.'
    assert df.std()['spot'] > 0.0, msg


def test_FBM(seed: int = 0):
    np.random.seed(seed)
    X = FBM(H=0.4)
    df = X.simulate()
    msg = 'Generated fBM path is not stochastic.'
    assert df.std()['spot'] > 0.0, msg


def test_Levy(seed: int = 0):
    np.random.seed(seed)
    X = Levy(jump_intensity=1.0, jump_size_distr=1.0)
    df = X.simulate()
    msg = 'Generated Levy path is not stochastic.'
    assert df.std()['spot'] > 0.0, msg


def test_Lognormal_multifractal(seed: int = 0):
    np.random.seed(seed)
    X = Lognormal_multifractal(scheme_step= 0.1)
    df = X.simulate()
    msg = 'Generated Lognormal_multifractal path is not stochastic.'
    assert df.std()['MRW'] > 0.0, msg
