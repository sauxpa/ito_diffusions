# Author: Patrick Saux <patrick.jr.saux@gmail.com>

import numpy as np
from scipy.special import gamma, beta
import mpmath as mp
from functools import lru_cache
from typing import List, Union
import logging


class Gaussian_Noise:
    def __init__(
        self,
        rng: np.random._generator.Generator = np.random.default_rng(),
    ) -> None:
        self._rng = rng

    @property
    def rng(self):
        return self._rng

    @rng.setter
    def rng(self, new_rng):
        self._rng = new_rng

    def simulate(self, size: int = None) -> Union[float, List[float]]:
        if size:
            return self.rng.normal(size=size)
        else:
            return self.rng.normal()


class Student_Noise:
    def __init__(
        self,
        df: float = 4.0,
        rng: np.random._generator.Generator = np.random.default_rng(),
    ) -> None:
        self.df = df
        self._rng = rng

    @property
    def rng(self):
        return self._rng

    @rng.setter
    def rng(self, new_rng):
        self._rng = new_rng

    @property
    def df(self) -> float:
        return self._df

    @df.setter
    def df(self, new_df: float) -> None:
        if new_df <= 2:
            msg = "Student distribution with degree of freedom {} (<= 2) does not have finite variance.".format(
                new_df
            )
            logging.warning(msg)
        self._df = new_df

    def simulate(self, size: int = None) -> Union[float, List[float]]:
        if size:
            return np.sqrt((self.df - 2) / self.df) * self.rng.standard_t(self.df, size=size)
        else:
            return np.sqrt((self.df - 2) / self.df) * self.rng.standard_t(self.df)


class Fractional_Gaussian_Noise:
    """Fractional Gaussian noise

    Covariance function of fractional Brownian motion :

    E[B_H(t) B_H (s)]=1/2*(|t|^{2H}+|s|^{2H}-|t-s|^{2H})

    2 implementations :
        * Slow implementation where the full trajectory of the fBM is generated
            as a correlated gaussian draw (requires manipulation of
            scheme_steps*scheme_steps matrix).
        * Faster KL-like expansion method
        (https://projecteuclid.org/download/pdf_1/euclid.ejp/1464816842)
        -- only works for H>0.5, other methods exist
        (e.g https://arxiv.org/pdf/1810.11850.pdf) but are more complicated
        to make numerically robust.
    """

    def __init__(
        self,
        H: float = 0.5,
        T: float = 1.0,
        scheme_steps: int = 100,
        method: str = "vector",
        n_kl: int = 100,
        rng: np.random._generator.Generator = np.random.default_rng(),
    ) -> None:
        self._T = T
        self._scheme_steps = scheme_steps
        self._H = H
        self._method = method
        self._n_kl = n_kl
        self._rng = rng

    @property
    def scheme_steps(self) -> int:
        return self._scheme_steps

    @scheme_steps.setter
    def scheme_steps(self, new_scheme_steps) -> None:
        self._scheme_steps = new_scheme_steps

    @property
    def scheme_step(self) -> float:
        return self.T / self.scheme_steps

    @property
    def time_steps(self) -> list:
        return [step * self.scheme_step for step in range(self.scheme_steps + 1)]

    @property
    def T(self) -> float:
        return self._T

    @T.setter
    def T(self, new_T) -> None:
        self._T = new_T

    @property
    def rng(self):
        return self._rng

    @rng.setter
    def rng(self, new_rng):
        self._rng = new_rng

    @property
    def H(self) -> float:
        return self._H

    @H.setter
    def H(self, new_H) -> None:
        # coeff_kl depends on self.H, blow cache if it changes
        self.coeff_kl.cache_clear()
        self._H = new_H

    @property
    def n_kl(self) -> float:
        return self._n_kl

    @n_kl.setter
    def n_kl(self, new_n_kl) -> None:
        # coeff_kl depends on self.n_kl, blow cache if it changes
        self.coeff_kl.cache_clear()
        self._n_kl = new_n_kl

    @property
    def method(self) -> str:
        return self._method

    @method.setter
    def method(self, new_method) -> None:
        self._method = new_method

    def covariance(self, s, t) -> float:
        return 0.5 * (
            t ** (2 * self.H) + s ** (2 * self.H) - np.abs(t - s) ** (2 * self.H)
        )

    def covariance_matrix(self) -> List[float]:
        cov = np.zeros((self.scheme_steps + 1, self.scheme_steps + 1))
        for i in range(self.scheme_steps + 1):
            cov[i][i] = (i * self.scheme_step) ** (2 * self.H)
            for j in range(i):
                cov[i][j] = self.covariance(i * self.scheme_step, j * self.scheme_step)
                cov[j][i] = cov[i][j]
        return cov

    def simulate_vector_method(self) -> List[float]:
        """fBM samples are simulated as a correlated gaussian vector."""
        cum_noise = self.rng.multivariate_normal(
            np.zeros((self.scheme_steps + 1,)), self.covariance_matrix()
        )
        return cum_noise[1:] - cum_noise[:-1]

    def g_lo(self, t: float) -> float:
        """Helper function for KL simulation of fBM with H<1/2"""
        return t ** (2 * self.H)

    def g_hi(self, t: float) -> float:
        """Helper function for KL simulation of fBM with H>1/2"""
        return -2 * self.H * (2 * self.H - 1) * (t ** (2 * self.H - 2))

    @lru_cache(maxsize=None)
    def coeff_kl(self):
        """Series coefficients in the KL expansion.
        Remark: gammainc exists in a vectorized form in scipy but
        does not support complex arguments; mpmath handles this but
        is not vectorized.
        """
        gb_ratio = np.sqrt(gamma(2 - 2 * self.H) / beta(self.H - 0.5, 1.5 - self.H))
        a = [gb_ratio / np.sqrt(2 * self.H - 1)]
        for k in range(1, self.n_kl + 1):
            a.append(
                gb_ratio
                * np.sqrt(
                    2
                    * float(
                        mp.re(
                            1j
                            * np.exp(-self.H * np.pi * 1j)
                            * mp.gammainc(2 * self.H - 1, 0, k * np.pi * 1j)
                        )
                    )
                )
                * (k * np.pi) ** (-self.H - 0.5)
            )
        return a

    def simulate_kl_method(self) -> List[float]:
        """fBM samples are simulated using a Karhunen-Loeve expansion
        (see https://projecteuclid.org/download/pdf_1/euclid.ejp/1464816842).
        """
        # cum_noise = []
        cum_noise = np.empty(self.scheme_steps)
        if self.H <= 0.5:
            raise ValueError("H<=0.5 is not supported for KL method.")
        elif self.H > 0.5:
            gauss_x = self.rng.normal(size=self.n_kl)
            gauss_y = self.rng.normal(size=self.n_kl)
            gauss_0 = self.rng.normal()
            # the below is memoized
            a = self.coeff_kl()
            for i, t in enumerate(self.time_steps):
                i_range = np.array(range(1, self.n_kl + 1)) * np.pi * t / self.T
                sx = np.multiply((1 - np.cos(i_range)), gauss_x)
                sy = np.multiply(np.sin(i_range), gauss_y)
                s = a[0] * t / self.T * gauss_0 + np.dot(a[1:], sx + sy)
                cum_noise[i] = s
        cum_noise = cum_noise * self.T**self.H
        return cum_noise[1:] - cum_noise[:-1]

    def simulate(self) -> List[float]:
        """Returns increments of the fBM."""
        if self.method == "vector":
            return self.simulate_vector_method()
        elif self.method == "kl":
            return self.simulate_kl_method()
        else:
            raise NameError("Unsupported simulation method : {}".format(self.method))
