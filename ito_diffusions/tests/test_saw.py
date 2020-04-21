import numpy as np
from ito_diffusions import SAW_2D


def test_SAW_2D(seed: int = 0):
    np.random.seed(seed)
    X = SAW_2D()  # Test with default arguments
    df = X.simulate()
    msg = 'Generated SAW path is not stochastic.'
    assert df['x'].std() > 0.0 and df['y'].std() > 0, msg
