import numpy as np
from numpy import random as rd
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from .ito_diffusion import Ito_diffusion


class Ito_diffusion_multi_d(Ito_diffusion):
    """Generic class for multidimensional Ito diffusion
    x0, drift and vol can be supplied as list/np.array...
    they will be casted to np.array
    x0 : initial vector, the dimension d of which is used to infer the
            dimension of the diffusion
    keys: optional, list of string of size d to name each of the dimension
    n_factors : number of factors i.e of Brownian motion driving the diffusion
    The covariance function has to return a matrix of dimension d*n_factors
    Potential boundary condition at barrier=(x1,...,xd).
    Example syntax : barrier=(0, None) means the boundary condition is on the
    first coordinate only, at 0.
    """

    def __init__(
        self,
        x0: np.ndarray = np.zeros(1),
        T: float = 1.0,
        scheme_steps: int = 100,
        n_factors: int = 1,
        keys: None = None,
        barrier: np.ndarray = np.full(1, None),
        barrier_condition: np.ndarray = np.full(1, None),
        jump_params: defaultdict = defaultdict(int),
        verbose: bool = False,
        **kwargs,
    ) -> None:

        x0 = np.array(x0)
        super().__init__(
            x0=x0,
            T=T,
            scheme_steps=scheme_steps,
            barrier=barrier,
            barrier_condition=barrier_condition,
            jump_params=jump_params,
            **kwargs,
        )
        self._keys = self._check_keys(keys)
        self._n_factors = n_factors

    @property
    def d(self) -> int:
        return len(self.x0)

    def _check_keys(self, keys: list) -> list:
        if not keys:
            keys = ["dim_{}".format(i) for i in range(self.d)]
        return keys

    @property
    def keys(self) -> list:
        return self._keys

    @keys.setter
    def keys(self, new_keys: list) -> None:
        self._keys = self._check_keys(new_keys)

    def simulate(self) -> pd.DataFrame:
        """Euler-Maruyama scheme"""
        last_step = self.x0
        x = np.empty((self.scheme_steps + 1, self.d))
        x[0, :] = last_step

        with tqdm(total=self.scheme_steps, disable=not self.verbose) as pbar:
            for i, t in enumerate(self.time_steps[1:]):
                # z drawn for a N(0_d,1_d)
                previous_step = last_step
                z = self.rng.normal(size=self._n_factors)
                inc = self.drift(t, last_step) * self.scheme_step + np.dot(
                    self.vol(t, last_step), self.scheme_step_sqrt * z
                )
                last_step = last_step + inc

                if self.has_jumps:
                    intensities = self.jump_intensity_func(t, previous_step)
                    jump_sizes = self.jump_size_distr.rvs()
                    for j, intensity in enumerate(intensities):
                        N = rd.poisson(intensity * self.scheme_step)
                        last_step[j] += N * jump_sizes[j]

                if self.barrier_condition == "absorb":
                    for j, coord in enumerate(last_step):
                        if self.barrier[j] is not None and self.barrier_crossed(
                            previous_step[j], coord, self.barrier[j]
                        ):
                            last_step[j] = self.barrier[j]
                x[i + 1, :] = last_step
                pbar.update(1)

        df_dict = dict()
        for i, key in enumerate(self.keys):
            df_dict[key] = x[:, i]
        df = pd.DataFrame(df_dict)
        df.index = self.time_steps
        return df


class BM_multi_d(Ito_diffusion_multi_d):
    """Instantiate Ito_diffusion to simulate a drifted Brownian motion
    dX_t = drift*dt + vol*dW_t
    where drift and vol are real vector and matrix respectively
    """

    def __init__(
        self,
        x0: np.ndarray = np.zeros(1),
        T: float = 1.0,
        scheme_steps: int = 100,
        drift: np.ndarray = np.zeros(1),
        vol: np.ndarray = np.eye(1),
        keys: None = None,
        barrier: np.ndarray = np.full(1, None),
        barrier_condition: np.ndarray = np.full(1, None),
        verbose: bool = False,
        **kwargs,
    ) -> None:
        self._drift_vector = np.array(drift)
        # vol is actually a covariance matrix here
        self._vol_matrix = np.array(vol)
        n_factors = self._vol_matrix.shape[1]
        super().__init__(
            x0=x0,
            T=T,
            scheme_steps=scheme_steps,
            keys=keys,
            n_factors=n_factors,
            barrier=barrier,
            barrier_condition=barrier_condition,
            verbose=verbose,
            **kwargs,
        )

    @property
    def drift_vector(self) -> np.ndarray:
        return self._drift_vector

    @drift_vector.setter
    def drift_vector(self, new_drift: np.ndarray) -> None:
        self._drift_vector = np.array(new_drift)

    @property
    def vol_matrix(self) -> np.ndarray:
        return self._vol_matrix

    @vol_matrix.setter
    def vol_matrix(self, new_vol: np.ndarray) -> None:
        self._vol_matrix = np.array(new_vol)

    def drift(self, t, x) -> np.ndarray:
        return self.drift_vector

    def vol(self, t, x) -> np.ndarray:
        return self.vol_matrix


class GBM_multi_d(Ito_diffusion_multi_d):
    """Instantiate Ito_diffusion to simulate a geometric Brownian motion
    dX_t = drift*X_t*dt + vol*X_t*dW_t
    where drift and vol are real vector and matrix respectively
    """

    def __init__(
        self,
        x0: np.ndarray = np.ones(1),
        T: float = 1.0,
        scheme_steps: int = 100,
        drift: np.ndarray = np.zeros(1),
        vol: np.ndarray = np.eye(1),
        keys: None = None,
        barrier: np.ndarray = np.full(1, None),
        barrier_condition: np.ndarray = np.full(1, None),
        verbose: bool = False,
        **kwargs,
    ) -> None:
        self._drift_vector = np.array(drift)
        # vol is actually a covariance matrix here
        self._vol_matrix = np.array(vol)
        n_factors = self._vol_matrix.shape[1]
        super().__init__(
            x0=x0,
            T=T,
            scheme_steps=scheme_steps,
            n_factors=n_factors,
            keys=keys,
            barrier=barrier,
            barrier_condition=barrier_condition,
            verbose=verbose,
            **kwargs,
        )

    @property
    def drift_vector(self) -> np.ndarray:
        return self._drift_vector

    @drift_vector.setter
    def drift_vector(self, new_drift: np.ndarray) -> np.ndarray:
        self._drift_vector = np.array(new_drift)

    @property
    def vol_matrix(self) -> np.ndarray:
        return self._vol_matrix

    @vol_matrix.setter
    def vol_matrix(self, new_vol: np.ndarray) -> None:
        self._vol_matrix = np.array(new_vol)

    def drift(self, t, x: np.ndarray) -> np.ndarray:
        return np.multiply(x, self.drift_vector)

    def vol(self, t, x: np.ndarray) -> np.ndarray:
        return np.multiply(x, self.vol_matrix.T).T


class Vasicek_multi_d(Ito_diffusion_multi_d):
    """Instantiate Ito_diffusion to simulate a mean-reverting mutlivariate
    correlated Vasicek diffusion:
    dX_t = mean_reversion*(long_term-X_t)*dt + vol*dW_t
    where mean_reversion, long_term and vol are real numbers.
    """

    def __init__(
        self,
        x0: np.ndarray = np.ones(1),
        T: float = 1.0,
        scheme_steps: int = 100,
        mean_reversion: np.ndarray = np.zeros(1),
        long_term: np.ndarray = np.zeros(1),
        vol: np.ndarray = np.eye(1),
        keys: None = None,
        barrier: np.ndarray = np.full(1, None),
        barrier_condition: np.ndarray = np.full(1, None),
        verbose: bool = False,
        **kwargs,
    ) -> None:

        self._mean_reversion = np.array(mean_reversion)
        self._long_term = np.array(long_term)
        # vol is actually a covariance matrix here
        self._vol_matrix = np.array(vol)
        n_factors = self._vol_matrix.shape[1]
        super().__init__(
            x0=x0,
            T=T,
            scheme_steps=scheme_steps,
            n_factors=n_factors,
            keys=keys,
            barrier=barrier,
            barrier_condition=barrier_condition,
            verbose=verbose,
            **kwargs,
        )

    @property
    def mean_reversion(self) -> np.ndarray:
        return self._mean_reversion

    @mean_reversion.setter
    def mean_reversion(self, new_mean_reversion: np.ndarray) -> None:
        self._mean_reversion = new_mean_reversion

    @property
    def long_term(self) -> np.ndarray:
        return self._long_term

    @long_term.setter
    def long_term(self, new_long_term: np.ndarray) -> None:
        self._long_term = new_long_term

    @property
    def vol_matrix(self) -> np.ndarray:
        return self._vol_matrix

    @vol_matrix.setter
    def vol_matrix(self, new_vol: np.ndarray) -> None:
        self._vol_matrix = np.array(new_vol)

    def drift(self, t, x: np.ndarray) -> np.ndarray:
        return np.multiply(self.long_term - x, self.mean_reversion)

    def vol(self, t, x: np.ndarray) -> np.ndarray:
        return self.vol_matrix


class BlackKarasinski_multi_d(Vasicek_multi_d):
    """Instantiate Ito_diffusion to simulate a mean-reverting mutlivariate
    correlated Black-Karasinski diffusion:
    dlog(X_t) = mean_reversion*(long_term-log(X_t))*dt + vol*dW_t
    where mean_reversion, long_term and vol are real numbers.
    """

    def __init__(
        self,
        x0: np.ndarray = np.ones(1),
        T: float = 1.0,
        scheme_steps: int = 100,
        mean_reversion: np.ndarray = np.zeros(1),
        long_term: np.ndarray = np.ones(1),
        vol: np.ndarray = np.eye(1),
        keys: None = None,
        barrier: np.ndarray = np.full(1, None),
        barrier_condition: np.ndarray = np.full(1, None),
        verbose: bool = False,
        **kwargs,
    ) -> None:

        super().__init__(
            x0=np.log(x0),
            T=T,
            scheme_steps=scheme_steps,
            mean_reversion=mean_reversion,
            long_term=np.log(long_term),
            vol=vol,
            keys=keys,
            barrier=barrier,
            barrier_condition=barrier_condition,
            verbose=verbose,
            **kwargs,
        )

    @Vasicek_multi_d.long_term.setter
    def long_term(self, new_long_term) -> None:
        self._long_term = np.log(new_long_term)

    @Vasicek_multi_d.x0.setter
    def x0(self, new_x0: float) -> None:
        self._x0 = np.log(new_x0)

    def simulate(self):
        df = super().simulate()
        for key in self.keys:
            df[key] = np.exp(df[key])
        return df


class SABR(Ito_diffusion_multi_d):
    """Instantiate Ito_diffusion to simulate a SABR stochastic vol model
    dX_t = s_t*X_t^beta*dW_t
    ds_t = vov*s_t*dB_t
    d<W,B>_t = rho*dt
    where beta, vov, rho are real numbers
    """

    def __init__(
        self,
        x0: np.ndarray = np.array([1.0, 1.0]),
        T: float = 1.0,
        scheme_steps: int = 100,
        keys: None = None,
        beta: float = 1.0,
        vov: float = 1.0,
        rho: float = 0.0,
        barrier: np.ndarray = np.full(1, None),
        barrier_condition: np.ndarray = np.full(1, None),
        verbose: bool = False,
        **kwargs,
    ) -> None:
        self._beta = float(beta)
        self._vov = float(vov)
        self._rho = float(rho)
        n_factors = 2
        super().__init__(
            x0=x0,
            T=T,
            scheme_steps=scheme_steps,
            n_factors=n_factors,
            keys=keys,
            barrier=barrier,
            barrier_condition=barrier_condition,
            verbose=verbose,
            **kwargs,
        )

    @property
    def beta(self) -> float:
        return self._beta

    @beta.setter
    def beta(self, new_beta: float) -> None:
        self._beta = float(new_beta)

    @property
    def rho(self) -> float:
        return self._rho

    @rho.setter
    def rho(self, new_rho: float) -> None:
        self._rho = new_rho

    @property
    def vov(self) -> float:
        return self._vov

    @vov.setter
    def vov(self, new_vov: float) -> None:
        self._vov = new_vov

    @property
    def rho_dual(self) -> float:
        return np.sqrt(1 - self.rho**2)

    def drift(self, t, x) -> np.ndarray:
        return np.zeros_like(x)

    def vol(self, t, x: np.ndarray) -> np.ndarray:
        """Project dB onto dW and an orhtogonal white noise dZ
        dB_t = rho*dW_t + sqrt(1-rho^2)*dZ_t
        """
        return np.array(
            [
                [x[1] * (x[0]) ** self.beta, 0],
                [self.vov * x[1] * self.rho, self.vov * x[1] * self.rho_dual],
            ]
        )


class SABR_AS_lognorm(Ito_diffusion_multi_d):
    """Instantiate Ito_diffusion to simulate a modified SABR with local spiky vol
    dX_t = s_t*C(X_t)*dW_t
    ds_t = vov*s_t*dB_t
    d<W,B>_t = rho*dt
    C(x) = exp(-c*log((y+shift)/K_max)^2)
    where shift, K_max, c, vov, rho are real numbers
    """

    def __init__(
        self,
        x0: np.ndarray = np.array([1.0, 1.0]),
        T: float = 1.0,
        scheme_steps: int = 100,
        keys: None = None,
        shift: float = 0.0,
        K_max: float = 1.0,
        c: float = 1.0,
        vov: float = 1.0,
        rho: float = 0.0,
        barrier: np.ndarray = np.full(1, None),
        barrier_condition: np.ndarray = np.full(1, None),
        verbose: bool = False,
        **kwargs,
    ) -> None:
        self._shift = float(shift)
        self._K_max = float(K_max)
        self._c = float(c)
        self._vov = float(vov)
        self._rho = float(rho)
        n_factors = 2
        super().__init__(
            x0=x0,
            T=T,
            scheme_steps=scheme_steps,
            n_factors=n_factors,
            keys=keys,
            barrier=barrier,
            barrier_condition=barrier_condition,
            verbose=verbose,
            **kwargs,
        )

    @property
    def shift(self) -> float:
        return self._shift

    @shift.setter
    def shift(self, new_shift: float) -> None:
        self._shift = float(new_shift)

    @property
    def K_max(self) -> float:
        return self._K_max

    @K_max.setter
    def K_max(self, new_K_max: float) -> None:
        self._K_max = float(new_K_max)

    @property
    def c(self) -> float:
        return self._c

    @c.setter
    def c(self, new_c: float) -> None:
        self._c = float(new_c)

    @property
    def rho(self) -> float:
        return self._rho

    @rho.setter
    def rho(self, new_rho: float) -> None:
        self._rho = new_rho

    @property
    def vov(self) -> float:
        return self._vov

    @vov.setter
    def vov(self, new_vov: float) -> None:
        self._vov = new_vov

    @property
    def rho_dual(self) -> float:
        return np.sqrt(1 - self.rho**2)

    def drift(self, t, x: np.ndarray) -> np.ndarray:
        return np.zeros_like(x)

    def vol(self, t, x: np.ndarray) -> np.ndarray:
        """Project dB onto dW and an orhtogonal white noise dZ
        dB_t = rho*dW_t + sqrt(1-rho^2)*dZ_t
        """
        return np.array(
            [
                [
                    x[1]
                    * np.exp(-self.c * np.log((x[0] + self.shift) / self.K_max) ** 2),
                    0,
                ],
                [self.vov * x[1] * self.rho, self.vov * x[1] * self.rho_dual],
            ]
        )


class SABR_AS_loglogistic(Ito_diffusion_multi_d):
    """Instantiate Ito_diffusion to simulate a modified SABR with local spiky vol
    dX_t = s_t*C(X_t)*dW_t
    ds_t = vov*s_t*dB_t
    d<W,B>_t = rho*dt
    C(x) = beta/alpha*(xs/alpha)^(beta-1)/(1+(*xs/alpha)^beta)^2
    xs = x + shift
    mode = alpha*((beta-1)/(beta+1))^(1-beta)
    where shift, mode, beta, vov, rho are real numbers
    """

    def __init__(
        self,
        x0: np.ndarray = np.array([1.0, 1.0]),
        T: float = 1.0,
        scheme_steps: int = 100,
        keys: None = None,
        shift: float = 0.0,
        mode: float = 1.0,
        beta: float = 1.0,
        vov: float = 1.0,
        rho: float = 0.0,
        barrier: np.ndarray = np.full(1, None),
        barrier_condition: np.ndarray = np.full(1, None),
        verbose: bool = False,
        **kwargs,
    ) -> None:
        self._shift = float(shift)
        self._mode = float(mode)
        self._beta = float(beta)
        self._vov = float(vov)
        self._rho = float(rho)
        n_factors = 2
        super().__init__(
            x0=x0,
            T=T,
            scheme_steps=scheme_steps,
            n_factors=n_factors,
            keys=keys,
            barrier=barrier,
            barrier_condition=barrier_condition,
            verbose=verbose,
            **kwargs,
        )

    @property
    def shift(self) -> float:
        return self._shift

    @shift.setter
    def shift(self, new_shift: float) -> None:
        self._shift = float(new_shift)

    @property
    def mode(self) -> float:
        return self._mode

    @mode.setter
    def mode(self, new_mode: float) -> None:
        self._mode = float(new_mode)

    @property
    def beta(self) -> float:
        return self._beta

    @beta.setter
    def beta(self, new_beta: float) -> None:
        self._beta = float(new_beta)

    @property
    def rho(self) -> float:
        return self._rho

    @rho.setter
    def rho(self, new_rho: float) -> None:
        self._rho = new_rho

    @property
    def vov(self) -> float:
        return self._vov

    @vov.setter
    def vov(self, new_vov: float) -> None:
        self._vov = new_vov

    @property
    def rho_dual(self) -> float:
        return np.sqrt(1 - self.rho**2)

    def drift(self, t, x: np.ndarray) -> np.ndarray:
        return np.zeros_like(x)

    @property
    def alpha(self):
        return self.mode / (((self.beta - 1) / (self.beta + 1)) ** (1 - self.beta))

    def vol(self, t, x: np.ndarray) -> np.ndarray:
        """Project dB onto dW and an orhtogonal white noise dZ
        dB_t = rho*dW_t + sqrt(1-rho^2)*dZ_t
        """
        return np.array(
            [
                [
                    x[1]
                    * self.beta
                    / self.alpha
                    * ((x[0] + self.shift) / self.alpha) ** (self.beta - 1)
                    / (1 + ((x[0] + self.shift) / self.alpha) ** self.beta) ** 2,
                    0,
                ],
                [self.vov * x[1] * self.rho, self.vov * x[1] * self.rho_dual],
            ]
        )


class SABR_tanh(Ito_diffusion_multi_d):
    """Instantiate Ito_diffusion to simulate a modified SABR with tanh local
    vol model:
    dX_t = s_t*C(X_t)*dW_t
    ds_t = vov*s_t*dB_t
    d<W,B>_t = rho*dt
    C(x) = tanh((x+shift)/l)
    where shift, l, vov, rho are real numbers
    """

    def __init__(
        self,
        x0: np.ndarray = np.array([1.0, 1.0]),
        T: float = 1.0,
        scheme_steps: int = 100,
        keys: None = None,
        shift: float = 0.0,
        l: float = 1.0,
        vov: float = 1.0,
        rho: float = 0.0,
        barrier: np.ndarray = np.full(1, None),
        barrier_condition: np.ndarray = np.full(1, None),
        verbose: bool = False,
        **kwargs,
    ) -> None:
        self._shift = float(shift)
        self._l = float(l)
        self._vov = float(vov)
        self._rho = float(rho)
        n_factors = 2
        super().__init__(
            x0=x0,
            T=T,
            scheme_steps=scheme_steps,
            n_factors=n_factors,
            keys=keys,
            barrier=barrier,
            barrier_condition=barrier_condition,
            verbose=verbose,
            **kwargs,
        )

    @property
    def shift(self) -> float:
        return self._shift

    @shift.setter
    def shift(self, new_shift: np.ndarray) -> None:
        self._shift = float(new_shift)

    @property
    def l(self) -> float:  # noqa
        return self._l

    @l.setter
    def l(self, new_l: float) -> None:  # noqa
        self._l = float(new_l)

    @property
    def rho(self) -> float:
        return self._rho

    @rho.setter
    def rho(self, new_rho: float) -> None:
        self._rho = new_rho

    @property
    def vov(self) -> float:
        return self._vov

    @vov.setter
    def vov(self, new_vov: float) -> None:
        self._vov = new_vov

    @property
    def rho_dual(self) -> float:
        return np.sqrt(1 - self.rho**2)

    def drift(self, t, x: np.ndarray) -> np.ndarray:
        return np.zeros_like(x)

    def vol(self, t, x: np.ndarray) -> np.ndarray:
        """Project dB onto dW and an orhtogonal white noise dZ
        dB_t = rho*dW_t + sqrt(1-rho^2)*dZ_t
        """
        return np.array(
            [
                [x[1] * np.tanh((x[0] + self.shift) / self.l), 0],
                [self.vov * x[1] * self.rho, self.vov * x[1] * self.rho_dual],
            ]
        )
