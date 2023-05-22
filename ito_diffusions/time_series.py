# Author: Patrick Saux <patrick.jr.saux@gmail.com>

import numpy as np
import pandas as pd
import abc
from collections import defaultdict
from typing import List
from tqdm import tqdm
import logging
from .ito_diffusion import Ito_diffusion


class Time_series_1d(Ito_diffusion):
    """Generic class for a time series process
    X_t = F_t(X_s, Z_s, s<t),
    where Z is a noise process, with a potential
    boundary condition at barrier.
    """

    def __init__(
        self,
        x0: List = [0.0],
        T: float = 100.0,
        max_lag: int = -1,
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

        # To avoid unnecessary passing of large arrays, when simulating only
        # pass the last max_lag entries (or the whole array if max_lag == -1).
        self._max_lag = max_lag

    @abc.abstractmethod
    def drift(self, t: float, x: List[float], z: List[float]):
        pass

    @abc.abstractmethod
    def vol(self, t: float, x: List[float], z: List[float]):
        pass

    def simulate(self) -> pd.DataFrame:
        """Iterative scheme"""
        # Main process
        last_step = self.x0[-1]
        x = np.empty(self.scheme_steps + 1)
        x[: self.len_x0] = self.x0

        # Noise process
        z = np.zeros(self.scheme_steps + 1)

        max_lag = np.maximum(self.max_lag, 0)

        with tqdm(total=self.scheme_steps, disable=not self.verbose) as pbar:
            for i, t in enumerate(self.time_steps[self.len_x0 :]):
                if self.noise_type == "gaussian":
                    last_noise = self.rng.normal()

                z[i + self.len_x0] = last_noise

                previous_step = last_step

                # Naming is borrowed from diffusion models
                # (note the difference with Ito_diffusion: the process is
                # simulated in integrated, not differential form).
                last_step = (
                    self.drift(
                        t,
                        x[i + self.len_x0 - max_lag : i + self.len_x0],
                        z[i + self.len_x0 - max_lag : i + self.len_x0 + 1],
                    )
                    + self.vol(
                        t,
                        x[i + self.len_x0 - max_lag : i + self.len_x0],
                        z[i + self.len_x0 - max_lag : i + self.len_x0 + 1],
                    )
                    * last_noise
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


class AR(Time_series_1d):
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
    def max_lag(self) -> int:
        return self.p

    @property
    def vol_double(self) -> float:
        return self._vol_double

    @vol_double.setter
    def vol_double(self, new_vol: float) -> None:
        self._vol_double = float(new_vol)

    def drift(self, t, x, z) -> float:
        return self.mu + self.a @ x[-1 : -self.p - 1 : -1]

    def vol(self, t, x, z) -> float:
        return self.vol_double


class MA(Time_series_1d):
    """Instantiate Time_series to simulate an moving-average model MA(q)
    X_t = mu + vol * Z_t + vol * sum_{j=1}^q b_j * Z_{t-j}
    where mu and (b_j)_{j=1}^p are real numbers.
    """

    def __init__(
        self,
        x0: List = [0.0],
        T: float = 100.0,
        mu: float = 0.0,
        b: List[float] = [],
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
    def max_lag(self) -> int:
        return self.q

    @property
    def vol_double(self) -> float:
        return self._vol_double

    @vol_double.setter
    def vol_double(self, new_vol: float) -> None:
        self._vol_double = float(new_vol)

    def drift(self, t, x, z) -> float:
        return self.mu + self.vol_double * self.b @ z[-2 : -self.q - 2 : -1]

    def vol(self, t, x, z) -> float:
        return self.vol_double


class ARMA(Time_series_1d):
    """Instantiate Time_series to simulate an autoregressive moving-average
    model ARMA(p, q)
    X_t = mu + vol * Z_t
             + sum_{i=1}^p a_i X_{t-i}
             + vol * sum_{j=1}^q b_j * Z_{t-j}
    where mu, (a_i)_{i=1}^p and (b_j)_{j=1}^p are real numbers.
    """

    def __init__(
        self,
        x0: List = [0.0],
        T: float = 100.0,
        mu: float = 0.0,
        a: List[float] = [],
        b: List[float] = [],
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
        self._b = np.array(b)
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
    def b(self) -> List[float]:
        return self._b

    @b.setter
    def b(self, new_b: List[float]) -> None:
        self._b = np.array(new_b)

    @property
    def q(self) -> int:
        return len(self.b)

    @property
    def max_lag(self) -> int:
        return np.maximum(self.p, self.q)

    @property
    def vol_double(self) -> float:
        return self._vol_double

    @vol_double.setter
    def vol_double(self, new_vol: float) -> None:
        self._vol_double = float(new_vol)

    def drift(self, t, x, z) -> float:
        return (
            self.mu
            + self.a @ x[-1 : -self.p - 1 : -1]
            + self.vol_double * self.b @ z[-2 : -self.q - 2 : -1]
        )

    def vol(self, t, x, z) -> float:
        return self.vol_double


class Time_series_CH(Ito_diffusion):
    """Generic class for a conditionally heteroskedastic time series process
    (X_t, sigma_t) = F_t(X_s, sigma_t, Z_s, s<t),
    where Z is a noise process, with a potential boundary condition at barrier.
    """

    def __init__(
        self,
        x0: List = [0.0],
        sigma0: List = [1.0],
        T: float = 100.0,
        max_lag: int = -1,
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

        # To avoid unnecessary passing of large arrays, when simulating only
        # pass the last max_lag entries (or the whole array if max_lag == -1).
        self._max_lag = max_lag

        self.sigma0 = sigma0

    def check_sigma0(self, new_sigma0) -> bool:
        msg = "Same number of initial values for  x and sigma is required!"
        assert len(new_sigma0) == self.len_x0, msg

    @property
    def sigma0(self) -> List[float]:
        return self._sigma0

    @sigma0.setter
    def sigma0(self, new_sigma0: List[float]) -> None:
        self.check_sigma0(new_sigma0)
        self._sigma0 = new_sigma0

    @abc.abstractmethod
    def drift(self, t: float, x: List[float], sigma: List[float], z: List[float]):
        pass

    @abc.abstractmethod
    def vol(self, t: float, x: List[float], sigma: List[float], z: List[float]):
        pass

    def simulate(self) -> pd.DataFrame:
        """Iterative scheme"""
        # Main process
        last_step = self.x0[-1]
        last_sigma = self.sigma0[-1]
        x = np.empty(self.scheme_steps + 1)
        sigma = np.empty(self.scheme_steps + 1)
        x[: self.len_x0] = self.x0
        sigma[: self.len_x0] = self.sigma0

        # Noise process
        z = np.zeros(self.scheme_steps + 1)

        max_lag = np.maximum(self.max_lag, 0)

        with tqdm(total=self.scheme_steps, disable=not self.verbose) as pbar:
            for i, t in enumerate(self.time_steps[self.len_x0 :]):
                if self.noise_type == "gaussian":
                    last_noise = self.rng.normal()

                z[i + self.len_x0] = last_noise

                previous_step = last_step

                # Naming is borrowed from diffusion models
                # (note the difference with Ito_diffusion: the process is
                # simulated in integrated, not differential form).
                last_sigma = self.vol(
                    t,
                    x[i + self.len_x0 - max_lag : i + self.len_x0],
                    sigma[i + self.len_x0 - max_lag : i + self.len_x0],
                    z[i + self.len_x0 - max_lag : i + self.len_x0 + 1],
                )
                last_step = (
                    self.drift(
                        t,
                        x[i + self.len_x0 - max_lag : i + self.len_x0],
                        sigma[i + self.len_x0 - max_lag : i + self.len_x0],
                        z[i + self.len_x0 - max_lag : i + self.len_x0 + 1],
                    )
                    + last_sigma * last_noise
                )

                if (
                    self.barrier_condition == "absorb"
                    and self.barrier is not None
                    and self.barrier_crossed(previous_step, last_step, self.barrier)
                ):
                    last_step = self.barrier

                x[i + self.len_x0] = last_step
                sigma[i + self.len_x0] = last_sigma
                pbar.update(1)

        df = pd.DataFrame({"spot": x, "vol": sigma})
        df.index = self.time_steps
        return df


class ARCH(Time_series_CH):
    """Instantiate Time_series_CH to simulate an autoregressive conditional
    heteroskedasticity model ARCH(p)
    X_t = sigma_t Z_t and sigma^2_t = v + sum_{i=1}^p a_i * X^2_{t-i}
    where v > 0 and (a_i)_{i=1}^p are real numbers.
    """

    def __init__(
        self,
        x0: List = [0.0],
        sigma0: List = [1.0],
        T: float = 100.0,
        v: float = 1.0,
        a: List[float] = [],
        drift: float = 0.0,
        vol: float = 1.0,
        barrier: None = None,
        barrier_condition: None = None,
        noise_params: defaultdict = defaultdict(int),
        verbose: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            x0=x0,
            sigma0=sigma0,
            T=T,
            barrier=barrier,
            barrier_condition=barrier_condition,
            noise_params=noise_params,
            verbose=verbose,
            **kwargs,
        )
        self._v = float(v)
        self._a = np.array(a)
        self._drift_double = float(drift)
        self._vol_double = float(vol)

    @property
    def v(self) -> float:
        return self._v

    @v.setter
    def v(self, new_v: float) -> None:
        self._v = float(new_v)

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
    def max_lag(self) -> int:
        return self.p

    @property
    def drift_double(self) -> float:
        return self._drift_double

    @drift_double.setter
    def drift_double(self, new_drift: float) -> None:
        self._drift_double = float(new_drift)

    @property
    def vol_double(self) -> float:
        return self._vol_double

    @vol_double.setter
    def vol_double(self, new_vol: float) -> None:
        self._vol_double = float(new_vol)

    def drift(self, t, x, sigma, z) -> float:
        return self.drift_double

    def vol(self, t, x, sigma, z) -> float:
        return self.vol_double * np.sqrt(
            self.v + self.a @ x[-1 : -self.p - 1 : -1] ** 2
        )


class GARCH(Time_series_CH):
    """Instantiate Time_series_CH to simulate a generalized autoregressive
    conditional heteroskedasticity model ARCH(p)
    X_t = sigma_t Z_t and
    sigma^2_t = v + sum_{i=1}^p a_i * X^2_{t-i} + sum_{j=1}^q b_j sigma^2_{t-j}
    where v > 0, (a_i)_{i=1}^p and (b_j)_{j=1}^q are real numbers.
    """

    def __init__(
        self,
        x0: List = [0.0],
        sigma0: List = [1.0],
        T: float = 100.0,
        v: float = 1.0,
        a: List[float] = [],
        b: List[float] = [],
        drift: float = 0.0,
        vol: float = 1.0,
        barrier: None = None,
        barrier_condition: None = None,
        noise_params: defaultdict = defaultdict(int),
        verbose: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            x0=x0,
            sigma0=sigma0,
            T=T,
            barrier=barrier,
            barrier_condition=barrier_condition,
            noise_params=noise_params,
            verbose=verbose,
            **kwargs,
        )
        self._v = float(v)
        self._a = np.array(a)
        self._b = np.array(b)
        self._drift_double = float(drift)
        self._vol_double = float(vol)

    @property
    def v(self) -> float:
        return self._v

    @v.setter
    def v(self, new_v: float) -> None:
        self._v = float(new_v)

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
    def b(self) -> List[float]:
        return self._b

    @b.setter
    def b(self, new_b: List[float]) -> None:
        self._b = np.array(new_b)

    @property
    def q(self) -> int:
        return len(self.b)

    @property
    def max_lag(self) -> int:
        return np.maximum(self.p, self.q)

    @property
    def drift_double(self) -> float:
        return self._drift_double

    @drift_double.setter
    def drift_double(self, new_drift: float) -> None:
        self._drift_double = float(new_drift)

    @property
    def vol_double(self) -> float:
        return self._vol_double

    @vol_double.setter
    def vol_double(self, new_vol: float) -> None:
        self._vol_double = float(new_vol)

    def drift(self, t, x, sigma, z) -> float:
        return self.drift_double

    def vol(self, t, x, sigma, z) -> float:
        return self.vol_double * np.sqrt(
            self.v
            + self.a @ x[-1 : -self.p - 1 : -1] ** 2
            + self.b @ sigma[-1 : -self.q - 1 : -1] ** 2
        )


class NAGARCH(Time_series_CH):
    """Instantiate Time_series_CH to simulate a nonlinear asymmetric
    generalized autoregressive conditional heteroskedasticity model ARCH(p)
    X_t = sigma_t Z_t and
    sigma^2_t = v + a * (X_{t-1} - theta * sigma_{t-1})^2 + b sigma^2_{t-1}
    where v > 0, a, b and theta are real numbers.
    """

    def __init__(
        self,
        x0: List = [0.0],
        sigma0: List = [1.0],
        T: float = 100.0,
        v: float = 1.0,
        a: float = 0.0,
        b: float = 0.0,
        theta: float = 0.0,
        drift: float = 0.0,
        vol: float = 1.0,
        barrier: None = None,
        barrier_condition: None = None,
        noise_params: defaultdict = defaultdict(int),
        verbose: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            x0=x0,
            sigma0=sigma0,
            T=T,
            barrier=barrier,
            barrier_condition=barrier_condition,
            noise_params=noise_params,
            verbose=verbose,
            **kwargs,
        )
        self._v = float(v)
        self._a = float(a)
        self._b = float(b)
        self._theta = float(theta)
        self._drift_double = float(drift)
        self._vol_double = float(vol)

        self.check_params(self.a, self.b, self.theta)

    def check_params(self, a, b, theta):
        if a * (1 + theta**2) + b >= 1:
            msg = "Process may fail to be stationary or nonnegative (a * (1 + theta ** 2) + b >= 1)."
            logging.warning(msg)

    @property
    def v(self) -> float:
        return self._v

    @v.setter
    def v(self, new_v: float) -> None:
        self._v = float(new_v)

    @property
    def a(self) -> float:
        return self._a

    @a.setter
    def a(self, new_a: float) -> None:
        self._a = float(new_a)
        self.check_params(self.a, self.b, self.theta)

    @property
    def theta(self) -> float:
        return self._theta

    @theta.setter
    def theta(self, new_theta: float) -> None:
        self._theta = float(new_theta)
        self.check_params(self.a, self.b, self.theta)

    @property
    def p(self) -> int:
        return 1

    @property
    def b(self) -> float:
        return self._b

    @b.setter
    def b(self, new_b: float) -> None:
        self._b = float(new_b)
        self.check_params(self.a, self.b, self.theta)

    @property
    def q(self) -> int:
        return 1

    @property
    def max_lag(self) -> int:
        return 1

    @property
    def drift_double(self) -> float:
        return self._drift_double

    @drift_double.setter
    def drift_double(self, new_drift: float) -> None:
        self._drift_double = float(new_drift)

    @property
    def vol_double(self) -> float:
        return self._vol_double

    @vol_double.setter
    def vol_double(self, new_vol: float) -> None:
        self._vol_double = float(new_vol)

    def drift(self, t, x, sigma, z) -> float:
        return self.drift_double

    def vol(self, t, x, sigma, z) -> float:
        return self.vol_double * np.sqrt(
            self.v
            + self.a * (x[-1] - self.theta * sigma[-1]) ** 2
            + self.b * sigma[-1] ** 2
        )
