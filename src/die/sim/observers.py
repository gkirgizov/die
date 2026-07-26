"""Observer interface and core (presentation-free) observers."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from die.sim.simulation import RunResult, Simulation, StepEvent


class SimObserver:
    """No-op base class for simulation observers: override what you need.

    Observers receive live (aliased) env arrays in events
    and must not mutate them.
    """

    def on_run_start(self, sim: Simulation, num_steps: int) -> None:
        pass

    def on_step(self, sim: Simulation, event: StepEvent) -> None:
        pass

    def on_run_end(self, sim: Simulation, result: RunResult) -> None:
        pass

    def close(self) -> None:
        """Release resources; idempotent, called in a ``finally`` block."""


class TqdmObserver(SimObserver):
    """Progress bar with running reward and selected env stats."""

    def __init__(self, postfix_keys: Sequence[str] = ('num_agents', 'mean_reward')):
        self._postfix_keys = tuple(postfix_keys)
        self._bar = None
        self._total_reward = 0.0

    def on_run_start(self, sim: Simulation, num_steps: int) -> None:
        from tqdm import tqdm
        self._total_reward = 0.0
        self._bar = tqdm(total=num_steps)

    def on_step(self, sim: Simulation, event: StepEvent) -> None:
        if self._bar is None:
            return
        self._total_reward += event.reward
        postfix = {key: event.info[key]
                   for key in self._postfix_keys if key in event.info}
        self._bar.update(1)
        self._bar.set_postfix(total_reward=round(self._total_reward, 3), **postfix)

    def on_run_end(self, sim: Simulation, result: RunResult) -> None:
        self.close()

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None


class MetricsObserver(SimObserver):
    """Collects per-step metrics in memory; optionally logs every N steps."""

    def __init__(self, log_every: int = 0, logger: logging.Logger | None = None):
        self._log_every = log_every
        self._logger = logger or logging.getLogger('die.sim')
        self.history: list[dict[str, float]] = []

    @property
    def total_reward(self) -> float:
        return float(sum(record['reward'] for record in self.history))

    def on_step(self, sim: Simulation, event: StepEvent) -> None:
        # NB: env info may carry its own (rounded) 'reward' entry;
        # the exact per-step reward must win.
        record = {**event.info, 'step': event.step, 'reward': event.reward}
        self.history.append(record)
        if self._log_every and event.step % self._log_every == 0:
            self._logger.info('step %d: %s', event.step, record)
