# Author: Patrick Saux <patrick.saux@inria.fr>

import numpy as np
import abc
from collections import defaultdict
from .noise import Fractional_Gaussian_Noise
from typing import List, Union


class Ito_diffusion(abc.ABC):
    """Generic class for Ito diffusion
    dX_t = b(t,X_t)dt + sigma(t,X_t)*dW_t
    with a potential boundary condition at barrier.
    Supports standard gaussian noise and fractional gaussian noise
    Typical example : barrier=0, barrier_condition='absorb'
    (only this one is supported for now)
    """

    def __init__(
        self,
        x0: Union[float, List] = 0.0,
        T: float = 1.0,
        scheme_steps: int = 100,
        barrier: None = None,
        barrier_condition: None = None,
        noise_params: defaultdict = defaultdict(float),
        jump_params: defaultdict = defaultdict(float),
        rng: np.random._generator.Generator = np.random.default_rng(),
        verbose: bool = False,
        **kwargs,
    ) -> None:
        self._x0 = x0
        self._T = T
        self._scheme_steps = scheme_steps
        self._barrier = barrier
        self._barrier_condition = barrier_condition
        self._noise_params = noise_params
        self._jump_params = jump_params
        self._rng = rng

        noise_type = self._noise_params["type"]
        # if a Hurst index is specified but is equal to 0.5
        # then simply use the gaussian noise
        H = self._noise_params.get("H", 0.5)
        if not noise_type or H == 0.5:
            noise_type = "gaussian"

        if noise_type == "fgaussian":
            self._noise = Fractional_Gaussian_Noise(
                T=self._T,
                scheme_steps=self._scheme_steps,
                H=self._noise_params["H"],
                n_kl=self._noise_params.get("n_kl", 100),
                method=self._noise_params.get("method", "vector"),
                rng=self.rng,
            )
        else:
            self._noise = None

        # For tqdm
        self._verbose = verbose

    @property
    def x0(self) -> Union[float, List]:
        return self._x0

    @x0.setter
    def x0(self, new_x0: Union[float, List]) -> None:
        self._x0 = new_x0

    @property
    def len_x0(self) -> int:
        if hasattr(self.x0, "__len__"):
            return len(self.x0)
        else:
            return 1

    @property
    def T(self) -> float:
        return self._T

    @T.setter
    def T(self, new_T) -> None:
        self._T = new_T

    @property
    def scheme_steps(self) -> int:
        return self._scheme_steps

    @scheme_steps.setter
    def scheme_steps(self, new_scheme_steps) -> None:
        self._scheme_steps = new_scheme_steps
        # if a Hurst index is specified but is equal to 0.5
        # then simply use the gaussian noise
        if self.noise_type == "fgaussian":
            self._noise = Fractional_Gaussian_Noise(
                T=self.T,
                scheme_steps=new_scheme_steps,
                H=self._noise_params["H"],
                n_kl=self._noise_params.get("n_kl", 100),
                method=self._noise_params.get("method", "vector"),
                rng=self.rng,
            )

    @property
    def rng(self):
        return self._rng

    @rng.setter
    def rng(self, new_rng):
        self._rng = new_rng

    @property
    def barrier(self):
        return self._barrier

    @barrier.setter
    def barrier(self, new_barrier):
        self._barrier = new_barrier

    @property
    def barrier_condition(self):
        if self._barrier_condition not in [None, "absorb"]:
            raise NameError(
                "Unsupported barrier condition : {}".format(self._barrier_condition)
            )
        else:
            return self._barrier_condition

    @barrier_condition.setter
    def barrier_condition(self, new_barrier_condition):
        self._barrier_condition = new_barrier_condition

    @property
    def noise_params(self) -> defaultdict:
        return self._noise_params

    @noise_params.setter
    def noise_params(self, new_noise_params) -> None:
        self._noise_params = new_noise_params
        noise_type = self._noise_params["type"]
        # if a Hurst index is specified but is equal to 0.5
        # then simply use the gaussian noise
        H = self._noise_params.get("H", 0.5)
        if not noise_type or H == 0.5:
            noise_type = "gaussian"

        if noise_type == "fgaussian":
            self._noise = Fractional_Gaussian_Noise(
                T=self.T,
                scheme_steps=self.scheme_steps,
                H=self._noise_params["H"],
                n_kl=self._noise_params.get("n_kl", 100),
                method=self._noise_params.get("method", "vector"),
                rng=self.rng,
            )
        else:
            self._noise = None

    @property
    def noise_type(self) -> str:
        noise_type = self.noise_params["type"]
        # if a Hurst index is specified but is equal to 0.5
        # then simply use the gaussian noise
        H = self.noise_params.get("H", 0.5)
        if noise_type and H != 0.5:
            return noise_type
        else:
            return "gaussian"

    @property
    def noise(self):
        return self._noise

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
    def has_jumps(self):
        return len(self.jump_params) > 0

    @property
    def jump_intensity_func(self):
        """scipy.stats distribution"""
        return self.jump_params["jump_intensity_func"]

    @property
    def jump_size_distr(self):
        """scipy.stats distribution"""
        return self.jump_params["jump_size_distr"]

    @property
    def scheme_step(self) -> float:
        return self.T / self.scheme_steps

    @property
    def scheme_step_sqrt(self) -> float:
        return np.sqrt(self.scheme_step)

    @property
    def time_steps(self) -> List:
        return np.linspace(0, self.T, self.scheme_steps + 1)

    def barrier_crossed(self, x, y, barrier) -> bool:
        """barrier is crossed if x and y are on each side of the barrier"""
        return (x <= barrier and y >= barrier) or (x >= barrier and y <= barrier)

    @abc.abstractmethod
    def drift(self, t: float, x: Union[float, List]):
        pass

    @abc.abstractmethod
    def vol(self, t: float, x: Union[float, List]):
        pass

    @abc.abstractmethod
    def simulate(self):
        pass
