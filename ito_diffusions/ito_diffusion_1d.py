import numpy as np
from numpy import random as rd
import pandas as pd
import abc
from functools import lru_cache
from collections import defaultdict
from scipy.stats import uniform
from typing import Callable, Union
from tqdm import tqdm
import logging
from .ito_diffusion import Ito_diffusion


class Ito_diffusion_1d(Ito_diffusion):
    def __init__(
        self,
        x0: float = 0.0,
        T: float = 1.0,
        scheme_steps: int = 100,
        barrier: None = None,
        barrier_condition: None = None,
        noise_params: defaultdict = defaultdict(int),
        jump_params: defaultdict = defaultdict(int),
        rng: np.random._generator.Generator = np.random.default_rng(),
        verbose: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            x0=x0,
            T=T,
            scheme_steps=scheme_steps,
            barrier=barrier,
            barrier_condition=barrier_condition,
            noise_params=noise_params,
            jump_params=jump_params,
            rng=rng,
            verbose=verbose,
            **kwargs,
        )

    def simulate(self) -> pd.DataFrame:
        """Euler-Maruyama scheme"""
        last_step = self.x0
        x = np.empty(self.scheme_steps + 1)
        x[0] = last_step

        if self.noise_type == "fgaussian":
            noises = self.noise.simulate()

        with tqdm(total=self.scheme_steps, disable=not self.verbose) as pbar:
            for i, t in enumerate(self.time_steps[1:]):
                # for regular gaussian noise, generate them sequentially
                if self.noise_type == "gaussian":
                    z = self.scheme_step_sqrt * self.rng.normal()
                else:
                    z = noises[i]

                previous_step = last_step
                last_step += (
                    self.drift(t, last_step) * self.scheme_step
                    + self.vol(t, last_step) * z
                )

                if self.has_jumps:
                    intensity = self.jump_intensity_func(t, previous_step)
                    N = self.rng.poisson(intensity * self.scheme_step)
                    last_step += N * self.jump_size_distr.rvs()

                if (
                    self.barrier_condition == "absorb"
                    and self.barrier is not None
                    and self.barrier_crossed(previous_step, last_step, self.barrier)
                ):
                    last_step = self.barrier

                x[i + 1] = last_step
                pbar.update(1)

        df = pd.DataFrame({"spot": x})
        df.index = self.time_steps
        return df


class BM(Ito_diffusion_1d):
    """Instantiate Ito_diffusion to simulate a drifted Brownian motion
    dX_t = drift*dt + vol*dW_t
    where drift and vol are real numbers
    """

    def __init__(
        self,
        x0: float = 0.0,
        T: float = 1.0,
        scheme_steps: int = 100,
        drift: float = 0.0,
        vol: float = 1.0,
        barrier: None = None,
        barrier_condition: None = None,
        noise_params: defaultdict = defaultdict(int),
        jump_params: defaultdict = defaultdict(int),
        verbose: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            x0=x0,
            T=T,
            scheme_steps=scheme_steps,
            barrier=barrier,
            barrier_condition=barrier_condition,
            noise_params=noise_params,
            jump_params=jump_params,
            verbose=verbose,
            **kwargs,
        )
        self._drift_double = float(drift)
        self._vol_double = float(vol)

    @property
    def drift_double(self) -> float:
        return self._drift_double

    @drift_double.setter
    def drift_double(self, new_drift) -> None:
        self._drift_double = new_drift

    @property
    def vol_double(self) -> float:
        return self._vol_double

    @vol_double.setter
    def vol_double(self, new_vol) -> None:
        self._vol_double = new_vol

    def drift(self, t, x) -> float:
        return self.drift_double

    def vol(self, t, x) -> float:
        return self.vol_double


class GBM(Ito_diffusion_1d):
    """Instantiate Ito_diffusion to simulate a geometric Brownian motion
    dX_t = drift*X_t*dt + vol*X_t*dW_t
    where drift and vol are real numbers
    """

    def __init__(
        self,
        x0: float = 1.0,
        T: float = 1.0,
        scheme_steps: int = 100,
        drift: float = 0.0,
        vol: float = 1.0,
        barrier: None = None,
        barrier_condition: None = None,
        noise_params: defaultdict = defaultdict(int),
        jump_params: defaultdict = defaultdict(int),
        verbose: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            x0,
            T,
            scheme_steps,
            barrier=barrier,
            barrier_condition=barrier_condition,
            noise_params=noise_params,
            jump_params=jump_params,
            verbose=verbose,
            **kwargs,
        )
        self._drift_double = float(drift)
        self._vol_double = float(vol)

    @property
    def drift_double(self) -> float:
        return self._drift_double

    @drift_double.setter
    def drift_double(self, new_drift) -> None:
        self._drift_double = new_drift

    @property
    def vol_double(self) -> float:
        return self._vol_double

    @vol_double.setter
    def vol_double(self, new_vol) -> None:
        self._vol_double = new_vol

    def drift(self, t, x) -> float:
        return self.drift_double * x

    def vol(self, t, x) -> float:
        return self.vol_double * x


class Bessel(Ito_diffusion_1d):
    """Instantiate Ito_diffusion to simulate a Bessel process
    dX_t = (d-1)/2*dt + vol*dW_t
    where d and vol are real numbers.
    For vol=1 and d an integer, this corresponds to the law
    of ||W_t|| where W is a d-dimensional Brownian motion.
    """

    def __init__(
        self,
        x0: float = 1.0,
        T: float = 1.0,
        scheme_steps: int = 100,
        d: float = 0.0,
        vol: float = 1.0,
        barrier: None = None,
        barrier_condition: None = None,
        noise_params: defaultdict = defaultdict(int),
        jump_params: defaultdict = defaultdict(int),
        verbose: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            x0,
            T,
            scheme_steps,
            barrier=barrier,
            barrier_condition=barrier_condition,
            noise_params=noise_params,
            jump_params=jump_params,
            verbose=verbose,
            **kwargs,
        )
        self.check_d(d)
        self._d = d
        self._vol_double = float(vol)

    def check_d(self, d):
        if d == 1:
            msg = "Bessel with d=1 is equivalent to a Brownian motion.\
Using the BM class is more stable in this case."
            logging.warning(msg)

    @property
    def d(self) -> float:
        return self._d

    @d.setter
    def d(self, new_d) -> None:
        self.check_d(new_d)
        self._d = new_d

    @property
    def vol_double(self) -> float:
        return self._vol_double

    @vol_double.setter
    def vol_double(self, new_vol) -> None:
        self._vol_double = new_vol

    def drift(self, t, x) -> float:
        return 0.5 * (self.d - 1) / x

    def vol(self, t, x) -> float:
        return self.vol_double * x


class BMPeriodic(Ito_diffusion_1d):
    """Instantiate Ito_diffusion to simulate a drifted Brownian motion
    dX_t = drift_t*dt + vol_t*dW_t
    where t -> drift_t and t -> vol_t are periodic functions.
    """

    def __init__(
        self,
        x0: float = 0.0,
        T: float = 1.0,
        scheme_steps: int = 100,
        drift_freq: float = 1.0,
        drift_phase: float = 0.0,
        drift_amp: float = 1.0,
        vol_freq: float = 1.0,
        vol_phase: float = 0.0,
        vol_amp: float = 1.0,
        barrier: None = None,
        barrier_condition: None = None,
        noise_params: defaultdict = defaultdict(int),
        jump_params: defaultdict = defaultdict(int),
        verbose: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            x0=x0,
            T=T,
            scheme_steps=scheme_steps,
            barrier=barrier,
            barrier_condition=barrier_condition,
            noise_params=noise_params,
            jump_params=jump_params,
            verbose=verbose,
            **kwargs,
        )
        self._drift_freq = float(drift_freq)
        self._drift_phase = float(drift_phase)
        self._drift_amp = float(drift_amp)
        self._vol_freq = float(vol_freq)
        self._vol_phase = float(vol_phase)
        self._vol_amp = float(vol_amp)

    @property
    def drift_freq(self) -> float:
        return self._drift_freq

    @drift_freq.setter
    def drift_freq(self, new_drift_freq) -> None:
        self._drift_freq = new_drift_freq

    @property
    def drift_phase(self) -> float:
        return self._drift_phase

    @drift_phase.setter
    def drift_phase(self, new_drift_phase) -> None:
        self._drift_phase = new_drift_phase

    @property
    def drift_amp(self) -> float:
        return self._drift_amp

    @drift_amp.setter
    def drift_amp(self, new_drift_amp) -> None:
        self._drift_amp = new_drift_amp

    @property
    def vol_freq(self) -> float:
        return self._vol_freq

    @vol_freq.setter
    def vol_freq(self, new_vol_freq) -> None:
        self._vol_freq = new_vol_freq

    @property
    def vol_phase(self) -> float:
        return self._vol_phase

    @vol_phase.setter
    def vol_phase(self, new_vol_phase) -> None:
        self._vol_phase = new_vol_phase

    @property
    def vol_amp(self) -> float:
        return self._vol_amp

    @vol_amp.setter
    def vol_amp(self, new_vol_amp) -> None:
        self._vol_amp = new_vol_amp

    def drift(self, t, x) -> float:
        return self.drift_amp * np.sin(
            2 * np.pi * self.drift_freq * t + self.drift_phase
        )

    def vol(self, t, x) -> float:
        return self.vol_amp * np.sin(2 * np.pi * self.vol_freq * t + self.vol_phase)


class SLN(Ito_diffusion_1d):
    """Instantiate Ito_diffusion to simulate a shifted lognormal diffusion
    dX_t = drift*X_t*dt + sigma*(shift*X_t+(mixing-shift)*X_0+(1-m)*L)*dW_t
    where drift and vol are real numbers
    """

    def __init__(
        self,
        x0: float = 1.0,
        T: float = 1.0,
        scheme_steps: int = 100,
        drift: float = 0.0,
        sigma: float = 1.0,
        shift: float = 0.0,
        mixing: float = 0.0,
        L: float = 1.0,
        barrier: None = None,
        barrier_condition: None = None,
        noise_params: defaultdict = defaultdict(int),
        jump_params: defaultdict = defaultdict(int),
        verbose: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            x0,
            T,
            scheme_steps,
            barrier=barrier,
            barrier_condition=barrier_condition,
            noise_params=noise_params,
            jump_params=jump_params,
            verbose=verbose,
            **kwargs,
        )
        self._drift_double = drift
        self._sigma = sigma
        self._shift = shift
        self._mixing = mixing
        self._L = L

    @property
    def drift_double(self) -> float:
        return self._drift_double

    @drift_double.setter
    def drift_double(self, new_drift) -> None:
        self._drift_double = new_drift

    @property
    def sigma(self) -> float:
        return self._sigma

    @sigma.setter
    def sigma(self, new_sigma) -> None:
        self._sigma = float(new_sigma)

    @property
    def shift(self) -> float:
        return self._shift

    @shift.setter
    def shift(self, new_shift) -> None:
        self._shift = new_shift

    @property
    def mixing(self) -> float:
        return self._mixing

    @mixing.setter
    def mixing(self, new_mixing) -> None:
        self._mixing = new_mixing

    @property
    def L(self) -> float:
        return self._L

    @L.setter
    def L(self, new_L) -> None:
        self._L = new_L

    def drift(self, t, x) -> float:
        return self.drift_double * x

    def vol(self, t, x) -> float:
        return self.sigma * (
            self.shift * x
            + (self.mixing - self.shift) * self.x0
            + (1 - self.mixing) * self.L
        )


class Vasicek(Ito_diffusion_1d):
    """Instantiate Ito_diffusion to simulate a mean-reverting Vasicek diffusion
    dX_t = mean_reversion*(long_term-X_t)*dt + vol*dW_t
    where mean_reversion, long_term and vol are real numbers
    """

    def __init__(
        self,
        x0: float = 1.0,
        T: float = 1.0,
        scheme_steps: int = 100,
        mean_reversion: float = 1.0,
        long_term: float = 0.0,
        vol: float = 1.0,
        barrier: None = None,
        barrier_condition: None = None,
        noise_params: defaultdict = defaultdict(int),
        jump_params: defaultdict = defaultdict(int),
        verbose: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            x0,
            T,
            scheme_steps,
            barrier=barrier,
            barrier_condition=barrier_condition,
            noise_params=noise_params,
            jump_params=jump_params,
            verbose=verbose,
            **kwargs,
        )
        self._mean_reversion = float(mean_reversion)
        self._long_term = float(long_term)
        self._vol_double = float(vol)

    @property
    def mean_reversion(self) -> float:
        return self._mean_reversion

    @mean_reversion.setter
    def mean_reversion(self, new_mean_reversion) -> None:
        self._mean_reversion = new_mean_reversion

    @property
    def long_term(self) -> float:
        return self._long_term

    @long_term.setter
    def long_term(self, new_long_term) -> None:
        self._long_term = new_long_term

    @property
    def vol_double(self) -> float:
        return self._vol_double

    @vol_double.setter
    def vol_double(self, new_vol) -> None:
        self._vol_double = new_vol

    def drift(self, t, x) -> float:
        return self.mean_reversion * (self.long_term - x)

    def vol(self, t, x) -> float:
        return self.vol_double


class CIR(Ito_diffusion_1d):
    """Instantiate Ito_diffusion to simulate a mean-reverting CIR diffusion
    dX_t = mean_reversion*(long_term-X_t)*dt + vol*sqrt(X_t)*dW_t
    where mean_reversion, long_term and vol are real numbers
    """

    def __init__(
        self,
        x0: float = 1.0,
        T: float = 1.0,
        scheme_steps: int = 100,
        mean_reversion: float = 1.0,
        long_term: float = 0.0,
        vol: float = 1.0,
        barrier: None = None,
        barrier_condition: None = None,
        noise_params: defaultdict = defaultdict(int),
        jump_params: defaultdict = defaultdict(int),
        verbose: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            x0,
            T,
            scheme_steps,
            barrier=barrier,
            barrier_condition=barrier_condition,
            noise_params=noise_params,
            jump_params=jump_params,
            verbose=verbose,
            **kwargs,
        )
        self._mean_reversion = float(mean_reversion)
        self._long_term = float(long_term)
        self._vol_double = float(vol)

    @property
    def mean_reversion(self) -> float:
        return self._mean_reversion

    @mean_reversion.setter
    def mean_reversion(self, new_mean_reversion) -> None:
        self._mean_reversion = new_mean_reversion

    @property
    def long_term(self) -> float:
        return self._long_term

    @long_term.setter
    def long_term(self, new_long_term) -> None:
        self._long_term = new_long_term

    @property
    def vol_double(self) -> float:
        return self._vol_double

    @vol_double.setter
    def vol_double(self, new_vol) -> None:
        self._vol_double = new_vol

    @property
    def feller_condition(self) -> bool:
        """Returns whether mean_reversion * long_term > 0.5*vol^2"""
        return self.mean_reversion * self.long_term > 0.5 * self.vol_double**2

    def drift(self, t, x) -> float:
        return self.mean_reversion * (self.long_term - x)

    def vol(self, t, x) -> float:
        return self.vol_double * np.sqrt(x)


class BlackKarasinski(Vasicek):
    """Instantiate Ito_diffusion to simulate a mean-reverting Black-Karasinski
    diffusion:
    dlog(X_t) = mean_reversion*(long_term-log(X_t))*dt + vol*dW_t
    where mean_reversion, long_term and vol are real numbers
    """

    def __init__(
        self,
        x0: float = 1.0,
        T: float = 1.0,
        scheme_steps: int = 100,
        mean_reversion: float = 1.0,
        long_term: float = 1.0,
        vol: float = 1.0,
        barrier: None = None,
        barrier_condition: None = None,
        noise_params: defaultdict = defaultdict(int),
        jump_params: defaultdict = defaultdict(int),
        verbose: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            np.log(x0),
            T,
            scheme_steps,
            mean_reversion=mean_reversion,
            long_term=np.log(long_term),
            vol=vol,
            barrier=barrier,
            barrier_condition=barrier_condition,
            noise_params=noise_params,
            jump_params=jump_params,
            verbose=verbose,
            **kwargs,
        )

    @Vasicek.long_term.setter
    def long_term(self, new_long_term) -> None:
        self._long_term = np.log(new_long_term)

    @Vasicek.x0.setter
    def x0(self, new_x0: float) -> None:
        self._x0 = np.log(new_x0)

    def simulate(self):
        df = super().simulate()
        df["spot"] = np.exp(df["spot"])
        return df


class pseudo_GBM(Ito_diffusion_1d):
    """Instantiate Ito_diffusion to simulate
    dX_t = drift*dt + vol*X_t*dW_t
    where r and vol are real numbers
    """

    def __init__(
        self,
        x0: float = 1.0,
        T: float = 1.0,
        scheme_steps: int = 100,
        drift: float = 0.0,
        vol: float = 1.0,
        barrier: None = None,
        barrier_condition: None = None,
        noise_params: defaultdict = defaultdict(int),
        jump_params: defaultdict = defaultdict(int),
        verbose: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            x0,
            T,
            scheme_steps,
            barrier=barrier,
            barrier_condition=barrier_condition,
            noise_params=noise_params,
            verbose=verbose,
            **kwargs,
        )
        self._drift_double = float(drift)
        self._vol_double = float(vol)

    @property
    def drift_double(self) -> float:
        return self._drift_double

    @drift_double.setter
    def drift_double(self, new_drift) -> None:
        self._drift_double = new_drift

    @property
    def vol_double(self) -> float:
        return self._vol_double

    @vol_double.setter
    def vol_double(self, new_vol) -> None:
        self._vol_double = new_vol

    def drift(self, t, x) -> float:
        return self.drift_double

    def vol(self, t, x) -> float:
        return self.vol_double * x


class ContainedBM(Ito_diffusion_1d):
    """Simulate a process with the covariance structure as the Brownian motion
    but a.s contained within bounds. Because the vol is constrained
    to be sigma*dW_t, only the drifts remains to ensure the process stays
    within the prescribed boundaries. A suggestion is :
    dX_t = kappa * (1/(X_t-lo)^beta - 1/(hi-X_t)^beta) * dt + sigma * dW_t.
    Beware, for beta>1 only do weak solutions exist (cf. Feller criterion).
    This translates in practice by numerical instabilities should this
    condition not hold.
    """

    def __init__(
        self,
        x0: float = 0.5,
        T: float = 1.0,
        scheme_steps: int = 100,
        beta: float = 1.0,
        kappa: float = 1.0,
        sigma: float = 1.0,
        lo: float = 0.0,
        hi: float = 1.0,
        barrier: None = None,
        barrier_condition: None = None,
        verbose: bool = False,
        **kwargs,
    ) -> None:

        super().__init__(
            x0=x0,
            T=T,
            scheme_steps=scheme_steps,
            barrier=barrier,
            barrier_condition=barrier_condition,
            verbose=verbose,
            **kwargs,
        )
        self._beta = float(beta)
        self._kappa = float(kappa)
        self._sigma = float(sigma)
        self._lo = float(lo)
        self._hi = float(hi)

    @property
    def kappa(self) -> float:
        return self._kappa

    @kappa.setter
    def kappa(self, new_kappa) -> None:
        self._kappa = new_kappa

    @property
    def beta(self) -> float:
        return self._beta

    @beta.setter
    def beta(self, new_beta) -> None:
        self._beta = new_beta

    @property
    def sigma(self) -> float:
        return self._sigma

    @sigma.setter
    def sigma(self, new_sigma) -> None:
        self._sigma = new_sigma

    @property
    def lo(self) -> float:
        return self._lo

    @lo.setter
    def lo(self, new_lo) -> None:
        self._lo = new_lo

    @property
    def hi(self) -> float:
        return self._hi

    @hi.setter
    def hi(self, new_hi) -> None:
        self._hi = new_hi

    def drift(self, t, x) -> float:
        return self.kappa * (
            1 / (x - self.lo) ** self.beta - 1 / (self.hi - x) ** self.beta
        )

    def vol(self, t, x) -> float:
        return self.sigma


class Pinned_diffusion(Ito_diffusion_1d):
    """Generic class for pinned diffusions, i.e diffusions which are
    constrained to arrive at a given point at the terminal date.
    """

    def __init__(
        self,
        x0: float = 0.0,
        T: float = 1.0,
        scheme_steps: int = 100,
        alpha: float = 1.0,
        vol: float = 1.0,
        pin: float = 0.0,
        noise_params: defaultdict = defaultdict(int),
        jump_params: defaultdict = defaultdict(int),
        verbose: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            x0=x0,
            T=T,
            scheme_steps=scheme_steps,
            noise_params=noise_params,
            jump_params=jump_params,
            verbose=verbose,
            **kwargs,
        )
        self._pin = float(pin)

    @property
    def pin(self) -> float:
        return self._pin

    @pin.setter
    def pin(self, new_pin) -> None:
        self._pin = new_pin

    @abc.abstractmethod
    def _h(self, t):
        pass

    def drift(self, t, x) -> float:
        """In a pinned diffusion, drift is written as
        h(t)*(y-x)
        """
        if t == self.T:
            return 0.0
        else:
            return self._h(t) * (self.pin - x)

    def simulate(self, show_backbone=True) -> pd.DataFrame:
        """Euler-Maruyama scheme"""
        df = super().simulate()
        df.loc[self.T].spot = self.pin

        if show_backbone:
            # discretization scheme to estimate the backbone
            last_step = self.x0
            backbone = [last_step]
            for t in self.time_steps[1:]:
                last_step += self._h(t) * (self.pin - last_step) * self.scheme_step
                backbone.append(last_step)
            df["backbone"] = backbone

        return df


class Alpha_pinned_BM(Pinned_diffusion):
    """Instantiate Ito_diffusion to simulate an alpha-pinned Brownian motion
    dX_t = alpha*(y-X_t)/(T-t)*dt + vol*dW_t
    where alpha, y (pin) and vol are real numbers
    """

    def __init__(
        self,
        x0: float = 0.0,
        T: float = 1.0,
        scheme_steps: int = 100,
        alpha: float = 1.0,
        vol: float = 1.0,
        pin: float = 0.0,
        noise_params: defaultdict = defaultdict(int),
        jump_params: defaultdict = defaultdict(int),
        verbose: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            x0=x0,
            T=T,
            scheme_steps=scheme_steps,
            pin=pin,
            noise_params=noise_params,
            jump_params=jump_params,
            verbose=verbose,
            **kwargs,
        )
        self._alpha = float(alpha)
        self._vol_double = float(vol)

    @property
    def alpha(self) -> float:
        return self._alpha

    @alpha.setter
    def alpha(self, new_alpha) -> None:
        self._alpha = new_alpha

    @property
    def vol_double(self) -> float:
        return self._vol_double

    @vol_double.setter
    def vol_double(self, new_vol) -> None:
        self._vol_double = new_vol

    def _h(self, t) -> float:
        if t == self.T:
            return 0
        else:
            return self.alpha / (self.T - t)

    def vol(self, t, x) -> float:
        return self.vol_double


class F_pinned_BM(Pinned_diffusion):
    """Instantiate Ito_diffusion to simulate an F-pinned Brownian motion
    dX_t = f(t)*(y-X_t)/(1-F(t))*dt + sqrt(f(t))*dW_t
    where y (pin) is a real number, f and F respectively the pdf and cdf
    of a probability distribution over [0,T]
    """

    def __init__(
        self,
        x0: float = 0.0,
        T: float = 1.0,
        scheme_steps: int = 100,
        distr: None = None,
        pin: float = 0.0,
        noise_params: defaultdict = defaultdict(int),
        jump_params: defaultdict = defaultdict(int),
        verbose: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            x0=x0,
            T=T,
            scheme_steps=scheme_steps,
            pin=pin,
            noise_params=noise_params,
            jump_params=jump_params,
            verbose=verbose,
            **kwargs,
        )
        if not distr:
            raise NameError("Must specify a probability distribution")
        else:
            self._distr = distr
            self._f = distr.pdf
            self._F = distr.cdf

    def _h(self, t) -> float:
        if t == self._T:
            return 0
        else:
            return self._f(t) / (1 - self._F(t))

    def vol(self, t, x) -> float:
        return np.sqrt(self._f(t))


class FBM(BM):
    """Instantiate Ito_diffusion to simulate a drifted fractional Brownian motion
    dX_t = drift*dt + vol*dW^H_t,
    H: Hurst parameter
    where drift, vol are real numbers
    """

    def __init__(
        self,
        x0: float = 0.0,
        T: float = 1.0,
        scheme_steps: int = 100,
        drift: float = 0.0,
        vol: float = 1.0,
        H: float = 0.51,
        method: str = "vector",
        n_kl: int = 100,
        barrier: None = None,
        barrier_condition: None = None,
        jump_params: defaultdict = defaultdict(int),
        verbose: bool = False,
        **kwargs,
    ) -> None:

        noise_params = {
            "type": "fgaussian",
            "H": H,
            "method": method,
            "n_kl": n_kl,
        }
        super().__init__(
            x0=x0,
            T=T,
            scheme_steps=scheme_steps,
            drift=drift,
            vol=vol,
            barrier=barrier,
            barrier_condition=barrier_condition,
            noise_params=noise_params,
            verbose=verbose,
            **kwargs,
        )

    @property
    def H(self) -> float:
        return self.noise._H

    @H.setter
    def H(self, new_H) -> None:
        self.noise_params["H"] = new_H
        self.noise.H = new_H

    @property
    def n_kl(self) -> int:
        return self.noise._n_kl

    @n_kl.setter
    def n_kl(self, new_n_kl) -> None:
        self.noise_params["n_kl"] = new_n_kl
        self.noise.n_kl = new_n_kl

    @property
    def method(self) -> float:
        return self.noise._method

    @method.setter
    def method(self, new_method) -> None:
        self.noise_params["method"] = new_method
        self.noise.method = new_method


class Levy(Ito_diffusion_1d):
    """Instantiate Ito_diffusion to simulate a Levy process
    dX_t = drift*dt + vol*dW_t + dJ_t
    where drift and vol are real numbers
    """

    def __init__(
        self,
        x0: float = 0.0,
        T: float = 1.0,
        scheme_steps: int = 100,
        drift: float = 0.0,
        vol: float = 1.0,
        barrier: None = None,
        barrier_condition: None = None,
        noise_params: defaultdict = defaultdict(float),
        jump_intensity: float = 1.0,
        jump_size_distr: Union[float, Callable] = None,
        verbose: bool = False,
        **kwargs,
    ) -> None:

        self._jump_intensity = jump_intensity
        jump_size_distr = self._num_to_callable_distr(jump_size_distr)
        jump_params = {
            "jump_intensity_func": lambda t, x: jump_intensity,
            "jump_size_distr": jump_size_distr,
        }

        super().__init__(
            x0=x0,
            T=T,
            scheme_steps=scheme_steps,
            barrier=barrier,
            barrier_condition=barrier_condition,
            noise_params=noise_params,
            jump_params=jump_params,
            verbose=verbose,
            **kwargs,
        )
        self._drift_double = float(drift)
        self._vol_double = float(vol)

    @property
    def jump_intensity(self) -> float:
        return self._jump_intensity

    @jump_intensity.setter
    def jump_intensity(self, new_jump_int) -> None:
        self._jump_intensity = new_jump_int
        self.jump_params["jump_intensity_func"] = lambda t, x: new_jump_int

    def _num_to_callable_distr(self, x: Union[float, Callable]) -> Callable:
        """Jump size distribution is required to be a callable (cf. simulate in
        Ito_diffusion_1d). If the provided argument is numeric, return a Dirac
        mass at the argument.
        """
        if type(x) in [float, int]:
            return uniform(loc=x, scale=0.0)
        else:
            return x

    @property
    def jump_size_distr(self) -> float:
        return self.jump_params["jump_size_distr"]

    @jump_size_distr.setter
    def jump_size_distr(self, new_jump_size_distr: Union[float, Callable]) -> None:
        new_jump_size_distr = self._num_to_callable_distr(new_jump_size_distr)
        self.jump_params["jump_size_distr"] = new_jump_size_distr

    @property
    def drift_double(self) -> float:
        return self._drift_double

    @drift_double.setter
    def drift_double(self, new_drift) -> None:
        self._drift_double = new_drift

    @property
    def vol_double(self) -> float:
        return self._vol_double

    @vol_double.setter
    def vol_double(self, new_vol) -> None:
        self._vol_double = new_vol

    def drift(self, t, x) -> float:
        return self.drift_double

    def vol(self, t, x) -> float:
        return self.vol_double


class Lognormal_multifractal:
    """Simulate a lognormal multifractal cascade process
    X_t = lim_{l->0+} int_0^t exp(w_l,T(u))dW_u
    where w_l,T(u) is a gaussian process with known mean and covariance
    functions. This is approximated by discretization of
    dX_t = exp(w_l,T(t))dW_t for a small value of l.
    """

    def __init__(
        self,
        x0: float = 0.0,
        T: float = 1.0,
        scheme_step: float = 0.01,
        intermittency: float = 0.01,
        integral_scale: float = 1.0,
        l_param: None = None,
        rng: np.random._generator.Generator = np.random.default_rng(),
        verbose: bool = False,
    ) -> None:
        self._x0 = float(x0)
        self._T = float(T)
        self._scheme_step = float(scheme_step)
        self._intermittency = float(intermittency)
        self._integral_scale = float(integral_scale)

        self._rng = rng

        if l_param is None:
            l_param = self._scheme_step / 128
        self._l_param = l_param

    @property
    def x0(self) -> float:
        return self._x0

    @x0.setter
    def x0(self, new_x0) -> None:
        self.omega_simulate.cache_clear()
        self._x0 = new_x0

    @property
    def T(self) -> float:
        return self._T

    @T.setter
    def T(self, new_T) -> None:
        self.omega_simulate.cache_clear()
        self._T = new_T

    @property
    def scheme_step(self) -> float:
        return self._scheme_step

    @scheme_step.setter
    def scheme_step(self, new_scheme_step) -> None:
        self.omega_simulate.cache_clear()
        self.scheme_step = new_scheme_step

    @property
    def rng(self):
        return self._rng

    @rng.setter
    def rng(self, new_rng):
        self._rng = new_rng

    @property
    def intermittency(self) -> float:
        return self._intermittency

    @intermittency.setter
    def intermittency(self, new_intermittency) -> None:
        self.omega_simulate.cache_clear()
        self._intermittency = new_intermittency

    @property
    def integral_scale(self) -> float:
        return self._integral_scale

    @integral_scale.setter
    def integral_scale(self, new_integral_scale) -> None:
        self.omega_simulate.cache_clear()
        self._integral_scale = new_integral_scale

    @property
    def l_param(self):
        return self._l_param

    @l_param.setter
    def l_param(self, new_l):
        self.omega_simulate.cache_clear()
        self._l_param = new_l

    @property
    def verbose(self) -> bool:
        return self._verbose

    @verbose.setter
    def verbose(self, new_verbose: bool) -> None:
        self._verbose = new_verbose

    @property
    def l_sqrt(self):
        return np.sqrt(self.l_param)

    @property
    def time_steps(self) -> list:
        return [step * self.l_param for step in range(self.scheme_steps + 1)]

    @property
    def scheme_steps(self) -> float:
        return np.floor(self.T / self.l_param).astype("int")

    def expectation(self) -> float:
        return -self.intermittency**2 * (
            np.log(self.integral_scale / self.l_param) - 1
        )

    def covariance(self, tau) -> float:
        if self.integral_scale <= tau:
            return 0
        elif tau < self.integral_scale and tau >= self.l_param:
            return self.intermittency**2 * np.log(self.integral_scale / tau)
        else:
            return self.intermittency**2 * (
                np.log(self.integral_scale / self.l_param) + 1 - tau / self.l_param
            )

    def covariance_matrix(self) -> np.ndarray:
        cov = np.zeros((self.scheme_steps, self.scheme_steps))
        for i in range(self.scheme_steps):
            cov[i][i] = self.covariance(0)
            for j in range(i):
                cov[i][j] = self.covariance((i - j) * self._scheme_step)
                cov[j][i] = cov[i][j]
        return cov

    @lru_cache(maxsize=None)
    def omega_simulate(self) -> np.ndarray:
        return self.rng.multivariate_normal(
            self.expectation() * np.ones((self.scheme_steps,)), self.covariance_matrix()
        )

    def MRM(self) -> list:
        """Multifractal Random Measure"""
        last_step = 0
        x = [last_step]
        omega = self.omega_simulate()
        for n in range(1, self.scheme_steps):
            last_step += self.l_param * np.exp(2 * omega[n - 1])
            x.append(last_step)

    def simulate(self) -> pd.DataFrame:
        """Multifractal Random Walk"""
        last_step = self.x0
        x = [last_step]
        last_brownian = self.x0
        BM = [last_brownian]
        omega = self.omega_simulate()
        for om in omega:
            noise = self.rng.normal() * self.l_sqrt
            last_step += np.exp(om) * noise
            x.append(last_step)
            last_brownian += noise
            BM.append(last_brownian)

        df = pd.DataFrame({"MRW": x, "BM": BM})
        df.index = self.time_steps
        return df
