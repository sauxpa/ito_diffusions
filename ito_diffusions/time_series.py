# Author: Patrick Saux <patrick.jr.saux@gmail.com>

import numpy as np
import pandas as pd
import abc
from collections import defaultdict
from typing import List
from tqdm import tqdm
from .ito_diffusion import Ito_diffusion


class Time_series(Ito_diffusion):
    """Generic class for a time series process
    X_t = F_t(X_s, Z_s, s<t),
    where Z is a noise process, with a potential
    boundary condition at barrier.
    """

    def __init__(
        self,
        x0: List = [0.0],
        T: float = 100.0,
        barrier: None = None,
        barrier_condition: None = None,
        noise_params: defaultdict = defaultdict(int),
        rng: np.random._generator.Generator = np.random.default_rng(),
        verbose: bool = False,
        **kwargs,
    ) -> None:

        super().__init__(
            x0=x0,
            T=T,
            scheme_steps=T,  # only allow integer time steps for time series 
            barrier=barrier,
            barrier_condition=barrier_condition,
            noise_params=noise_params,
            rng=rng,
            verbose=verbose,
            **kwargs,
        )

    @abc.abstractmethod
    def drift(self, t: float, x: List, z: List):
        pass

    @abc.abstractmethod
    def vol(self, t: float, x: List, z: List):
        pass

    def simulate(self) -> pd.DataFrame:
        """Iterative scheme"""
        # Main process
        last_step = self.x0[-1]
        x = np.empty(self.scheme_steps + 1)
        x[: self.len_x0] = self.x0

        # Noise process
        z = np.zeros(self.scheme_steps + 1)

        with tqdm(total=self.scheme_steps, disable=not self.verbose) as pbar:
            for i, t in enumerate(self.time_steps[self.len_x0 :]):
                # for regular gaussian noise, generate them sequentially
                if self.noise_type == "gaussian":
                    z[i + self.len_x0] = self.rng.normal()

                previous_step = last_step

                # Naming is borrowed from diffusion models
                # (note the difference with Ito_diffusion: the process is
                # simulated in integrated, not differential form).
                last_step = (
                    self.drift(t, x[: i + self.len_x0], z[: i + self.len_x0 + 1])
                    + self.vol(t, x[: i + self.len_x0], z[: i + self.len_x0 + 1])
                )

                if (
                    self.barrier_condition == "absorb"
                    and self.barrier is not None
                    and self.barrier_crossed(previous_step, last_step, self.barrier)
                ):
                    last_step = self.barrier

                x[i + self.len_x0] = last_step
                pbar.update(1)

        df = pd.DataFrame({"spot": x})
        df.index = self.time_steps
        return df


class AR(Time_series):
    """Instantiate Time_series to simulate an autoregressive model AR(p)
    X_t = mu + sum_{i=1}^p a_i * X_{t-i} + vol * Z_t
    where mu and (a_i)_{i=1}^p are real numbers.
    """

    def __init__(
        self,
        x0: List = [0.0],
        T: float = 100.0,
        mu: float = 0.0,
        a: List[float] = [],
        vol: float = 1.0,
        barrier: None = None,
        barrier_condition: None = None,
        noise_params: defaultdict = defaultdict(int),
        verbose: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            x0=x0,
            T=T,
            barrier=barrier,
            barrier_condition=barrier_condition,
            noise_params=noise_params,
            verbose=verbose,
            **kwargs,
        )
        self._mu = float(mu)
        self._a = np.array(a)
        self._vol_double = float(vol)

    @property
    def mu(self) -> float:
        return self._mu

    @mu.setter
    def mu(self, new_mu: float) -> None:
        self._mu = float(new_mu)

    @property
    def a(self) -> List[float]:
        return self._a

    @a.setter
    def a(self, new_a: List[float]) -> None:
        self._a = np.array(new_a)

    @property
    def p(self) -> int:
        return len(self.a)

    @property
    def vol_double(self) -> float:
        return self._vol_double

    @vol_double.setter
    def vol_double(self, new_vol: float) -> None:
        self._vol_double = float(new_vol)

    def drift(self, t, x, z) -> float:
        return self.mu + self.a @ x[-1 : -self.p - 1 : -1]

    def vol(self, t, x, z) -> float:
        return self.vol_double * z[-1]


class MA(Time_series):
    """Instantiate Time_series to simulate an moving-average model MA(q)
    X_t = mu + vol * Z_t sum_{i=1}^q b_i * Z_{t-i}
    where mu and (b_i)_{i=1}^p are real numbers.
    """

    def __init__(
        self,
        x0: List = [0.0],
        T: float = 100.0,
        mu: float = 0.0,
        b: List[float] = [0.0],
        vol: float = 1.0,
        barrier: None = None,
        barrier_condition: None = None,
        noise_params: defaultdict = defaultdict(int),
        verbose: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            x0=x0,
            T=T,
            barrier=barrier,
            barrier_condition=barrier_condition,
            noise_params=noise_params,
            verbose=verbose,
            **kwargs,
        )
        self._mu = float(mu)
        self._b = np.array(b)
        self._vol_double = float(vol)

    @property
    def mu(self) -> float:
        return self._mu

    @mu.setter
    def mu(self, new_mu: float) -> None:
        self._mu = float(new_mu)

    @property
    def b(self) -> List[float]:
        return self._b

    @b.setter
    def b(self, new_b: List[float]) -> None:
        self._b = np.array(new_b)

    @property
    def q(self) -> int:
        return len(self.b)

    @property
    def vol_double(self) -> float:
        return self._vol_double

    @vol_double.setter
    def vol_double(self, new_vol: float) -> None:
        self._vol_double = float(new_vol)

    def drift(self, t, x, z) -> float:
        return self.mu + self.vol_double * self.b @ z[-2 : -self.q - 2 : -1]

    def vol(self, t, x, z) -> float:
        return self.vol_double * z[-1]
