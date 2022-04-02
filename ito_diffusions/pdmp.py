import numpy as np
import abc
from collections import defaultdict
from typing import List


class PDMP(abc.ABC):
    """Generic class for Piecewise Deterministic Markov Process.
    dX_t = drift_m(t,X_t)dt
    where m is the mode, subject to boundary and Poisson mode jumps.
    """

    def __init__(
        self,
        x0: float = 0.0,
        m0: int = 0,
        T: float = 1.0,
        t0: float = 0.0,
        scheme_steps: int = 100,
        barrier_params: defaultdict = defaultdict(list),
        jump_params: defaultdict = defaultdict(list),
        rng: np.random._generator.Generator = np.random.default_rng(),
        verbose: bool = False,
        **kwargs,
    ) -> None:
        self._x0 = x0
        self._m0 = m0
        self._T = T
        self._t0 = t0
        self._scheme_steps = scheme_steps
        self._barrier_params = barrier_params
        self._jump_params = jump_params

        self._rng = rng

        # Technical: numerical tolerance for barrier crossing
        self._barrier_tol = 1e-10

        # For tqdm
        self._verbose = verbose

    @property
    def x0(self) -> float:
        return self._x0

    @x0.setter
    def x0(self, new_x0: float) -> None:
        self._x0 = new_x0

    @property
    def m0(self) -> float:
        return self._m0

    @m0.setter
    def m0(self, new_m0: float) -> None:
        self._m0 = new_m0

    @property
    def T(self) -> float:
        return self._T

    @T.setter
    def T(self, new_T) -> None:
        self._T = new_T

    @property
    def t0(self) -> float:
        return self._t0

    @t0.setter
    def t0(self, new_t0) -> None:
        self._t0 = new_t0

    @property
    def scheme_steps(self) -> int:
        return self._scheme_steps

    @scheme_steps.setter
    def scheme_steps(self, new_scheme_steps) -> None:
        self._scheme_steps = new_scheme_steps

    @property
    def rng(self):
        return self._rng

    @rng.setter
    def rng(self, new_rng):
        self._rng = new_rng

    @property
    def barrier_params(self):
        return self._barrier_params

    @barrier_params.setter
    def barrier_params(self, new_barrier_params):
        self._barrier_params = new_barrier_params

    @property
    def jump_params(self) -> defaultdict:
        return self._jump_params

    @jump_params.setter
    def jump_params(self, new_jump_params) -> None:
        self._jump_params = new_jump_params

    @property
    def verbose(self) -> bool:
        return self._verbose

    @verbose.setter
    def verbose(self, new_verbose: bool) -> None:
        self._verbose = new_verbose

    @property
    def barriers(self) -> List[float]:
        return self.barrier_params["barriers"]

    @property
    def barrier_jump_mode_func(self):
        """scipy.stats distribution"""
        return self.barrier_params["jump_mode_func"]

    @property
    def natural_jump_intensity_func(self):
        """scipy.stats distribution"""
        return self.jump_params["jump_intensity_func"]

    @property
    def natural_jump_mode_func(self):
        """scipy.stats distribution"""
        return self.jump_params["jump_mode_func"]

    @property
    def scheme_step(self) -> float:
        return (self.T - self.t0) / self.scheme_steps

    @property
    def time_steps(self) -> np.ndarray:
        return np.linspace(self.t0, self.T, self.scheme_steps + 1)

    def barrier_crossed(self, x, y, barrier) -> bool:
        """barrier is crossed if x and y are on each side of the barrier"""
        return (
            x < barrier - self._barrier_tol and y >= barrier - self._barrier_tol
        ) or (x > barrier + self._barrier_tol and y <= barrier + self._barrier_tol)

    @abc.abstractmethod
    def drift(self, t, x, m):
        pass

    @abc.abstractmethod
    def simulate(self):
        pass
