import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from functools import lru_cache
from tqdm import tqdm


def rotation_matrix_2d(theta: float, round: int = 12) -> np.ndarray:
    """Returns the 2D matrix associated with the linear
    map rotation of angle theta.
    Round the resulting entries to ensure exact rotation
    at special angles e.g a rotation of angle pi is
    [
        [-1, 0],
        [0, 1],
    ],
    and not ~1e-16 instead of 0.
    """
    return np.round(
        np.array(
            [
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)],
            ]
        ),
        round,
    )


def v_dot(a: np.ndarray) -> np.ndarray:
    """Define a dot product function used for the rotate operation."""
    return lambda b: np.dot(a, b)


class SAW_2D:
    """Simulate Self-Avoiding Walks on regular 2D lattices
    using the pivot algorithm
    """

    def __init__(
        self,
        scheme_steps: int = 1000,
        n_success: int = 500,
        p: int = 4,
        verbose: bool = False,
        rng: np.random._generator.Generator = np.random.default_rng(),
    ):
        # Length of the SAW
        self._scheme_steps = scheme_steps
        # Degree of the lattice (e.g p=4 corresponds to the square lattice Z^2)
        self._p = p
        # Number of succesful symmetries in the pivot algorithm
        self._n_success = n_success

        # Switch tqdm on and off
        self._verbose = verbose

        self._rng = rng

    @property
    def rng(self):
        return self._rng

    @rng.setter
    def rng(self, new_rng):
        self._rng = new_rng

    @property
    def scheme_steps(self) -> int:
        return self._scheme_steps

    @scheme_steps.setter
    def N(self, new_scheme_steps: int) -> None:
        self._scheme_steps = new_scheme_steps

    @property
    def p(self) -> int:
        return self._p

    @p.setter
    def p(self, new_p: int) -> None:
        type(self).rotation_matrices.fget.cache_clear()
        self._p = new_p

    @property
    def n_success(self) -> int:
        return self._n_success

    @n_success.setter
    def n_success(self, new_n_success: int) -> None:
        self._n_success = new_n_success

    @property
    def verbose(self) -> bool:
        return self._verbose

    @verbose.setter
    def verbose(self, new_verbose: bool) -> None:
        self._verbose = new_verbose

    @property
    def _chain(self) -> np.ndarray:
        """Returns a chain walk.
        Useful to initialize the pivot algorithm.
        """
        return np.stack(
            (np.arange(self.scheme_steps), np.zeros(self.scheme_steps)), axis=1
        )

    @property
    @lru_cache(maxsize=None)
    def rotation_matrices(self) -> list:
        thetas = 2 * np.pi * np.linspace(1 / self.p, 1, self.p - 1, endpoint=False)
        return [rotation_matrix_2d(theta) for theta in thetas]

    def simulate(self):
        # Counter for succesful flip
        t = 0
        # Initialize walk to a chain
        walk = self._chain

        # while loop until the number of successful step up to t
        with tqdm(total=self.n_success, disable=not self.verbose) as pbar:
            while t <= self.n_success:
                # Pick a pivot uniformly on the walk (excluding the edges)
                pivot = self.rng.integers(1, self.scheme_steps - 1)
                # Choose which side to twist
                side = self.rng.random() < 0.5
                if side:
                    old_walk = walk[0 : pivot + 1]
                    temp_walk = walk[pivot + 1 :]
                else:
                    old_walk = walk[pivot:]
                    temp_walk = walk[0:pivot]

                # Pick a symmetry operator
                symtry_oprtr = self.rotation_matrices[
                    self.rng.integers(len(self.rotation_matrices))
                ]
                # Apply the symmetry
                new_walk = (
                    np.apply_along_axis(v_dot(symtry_oprtr), 1, temp_walk - walk[pivot])
                    + walk[pivot]
                )

                # Use cdist function of scipy package to calculate
                # the pair-pair distance between old_chain and new_chain
                overlap = cdist(new_walk, old_walk).flatten()
                # Determine whether the proposed walk intersects itself.
                # This is exact for the square lattice; for the other regular
                # lattices, consider points should be separated by at least 1
                # unit length (edges are assumed to be of size 1), and
                # therefore reject if there are less than 0.5 apart.
                if np.min(overlap) < 0.5:
                    continue
                else:
                    if side:
                        walk = np.concatenate((old_walk, new_walk), axis=0)
                    else:
                        walk = np.concatenate((new_walk, old_walk), axis=0)
                    t += 1
                    pbar.update(1)
        df = pd.DataFrame({"x": walk[:, 0], "y": walk[:, 1]})
        df.index = np.arange(1, self.scheme_steps + 1)
        return df
