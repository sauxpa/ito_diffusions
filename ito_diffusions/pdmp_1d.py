import numpy as np
import pandas as pd
from collections import defaultdict
from typing import List
from tqdm import tqdm
from .pdmp import PDMP


class PDMP_1d(PDMP):
    def __init__(
        self,
        x0: float = 0.0,
        m0: int = 0,
        T: float = 1.0,
        t0: float = 0.0,
        scheme_steps: int = 100,
        barrier_params: defaultdict = defaultdict(list),
        jump_params: defaultdict = defaultdict(list),
        verbose: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            x0=x0,
            m0=m0,
            T=T,
            t0=t0,
            scheme_steps=scheme_steps,
            barrier_params=barrier_params,
            jump_params=jump_params,
            verbose=verbose,
            **kwargs,
        )

    def simulate(self) -> pd.DataFrame:
        """Euler scheme"""
        last_mode = self.m0
        last_step = self.x0
        x = np.empty(self.scheme_steps + 1)
        mode = np.empty(self.scheme_steps + 1, dtype=int)
        natural_jump = np.zeros(self.scheme_steps + 1, dtype=bool)
        boundary_jump = np.zeros(self.scheme_steps + 1, dtype=bool)
        x[0] = last_step
        mode[0] = last_mode
        natural_jump[0] = False
        boundary_jump[0] = False

        with tqdm(total=self.scheme_steps, disable=not self.verbose) as pbar:
            for i, t in enumerate(self.time_steps[1:]):
                previous_step = last_step
                last_step += self.drift(t, last_step, last_mode) * self.scheme_step
                intensity = self.natural_jump_intensity_func(
                    t, previous_step, last_mode
                )
                if self.rng.poisson(intensity * self.scheme_step) > 0:
                    last_mode = int(
                        self.natural_jump_mode_func(t, previous_step, last_mode).rvs(random_state=self.rng)
                    )
                    natural_jump[i + 1] = True

                for barrier_idx, barrier in enumerate(self.barriers):
                    if self.barrier_crossed(previous_step, last_step, barrier):
                        last_mode = int(
                            self.barrier_jump_mode_func[barrier_idx](
                                t, previous_step, last_mode
                            ).rvs(random_state=self.rng)
                        )
                        last_step = barrier
                        boundary_jump[i + 1] = True
                        break
                x[i + 1] = last_step
                mode[i + 1] = last_mode
                pbar.update(1)
        df = pd.DataFrame(
            {
                "position": x,
                "mode": mode,
                "natural_jump": natural_jump,
                "boundary_jump": boundary_jump,
            }
        )
        df.index = self.time_steps
        return df


class PDMP_1d_linear(PDMP_1d):
    def __init__(
        self,
        x0: float = 0.0,
        m0: int = 0,
        T: float = 1.0,
        t0: float = 0.0,
        scheme_steps: int = 100,
        drifts: List[float] = [],
        barrier_params: defaultdict = defaultdict(list),
        jump_params: defaultdict = defaultdict(list),
        verbose: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            x0=x0,
            m0=m0,
            T=T,
            t0=t0,
            scheme_steps=scheme_steps,
            barrier_params=barrier_params,
            jump_params=jump_params,
            verbose=verbose,
            **kwargs,
        )
        self._drifts = drifts

    @property
    def drifts(self) -> List[float]:
        return self._drifts

    @drifts.setter
    def drifts(self, new_drifts: List[float]) -> None:
        self._drifts = new_drifts

    def drift(self, t, x, m) -> float:
        return self.drifts[m]
