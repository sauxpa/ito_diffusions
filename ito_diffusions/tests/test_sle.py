import numpy as np
from ito_diffusions import FastChordalSLEHalfPlane


def test_FastChordalSLEHalfPlane(seed: int = 0):
    np.random.seed(seed)
    X = FastChordalSLEHalfPlane(kappa=8/3)
    df = X.simulate()
    msg = 'Generated SLE path is not stochastic.'
    assert df['x'].std() > 0.0 and df['y'].std() > 0, msg
