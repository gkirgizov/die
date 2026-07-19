"""Evolutionary training on the JAX core.

Reuses everything structural from :class:`~die.train.trainer.EvolutionTrainer`
-- the generation loop, statistics, loggers, checkpoint and rollout cadence --
and swaps only the evaluation strategy: instead of simulating candidates one
at a time, the whole population is evaluated in a single vectorized call.

The xarray env is still held (and still used for the periodic rollout
animations), so recording and replay keep working exactly as before.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from die.env import Env
from die.jx.agent import JaxNCAAgent
from die.jx.env import init_state
from die.jx.rollout import make_evaluate_population
from die.jx.state import EnvConfig, EnvState
from die.train.searcher import BatchedSearcherBackend
from die.train.trainer import EvolutionTrainer, TrainConfig

if TYPE_CHECKING:
    from die.train.loggers import TrainLogger


class JaxEvolutionTrainer(EvolutionTrainer):
    """Trains a :class:`~die.jx.agent.JaxNCAAgent` with a batched backend."""

    def __init__(self,
                 env: Env,
                 agent: JaxNCAAgent,
                 backend: BatchedSearcherBackend,
                 config: TrainConfig,
                 jx_config: EnvConfig,
                 loggers: Sequence[TrainLogger] = (),
                 run_context: dict | None = None):
        super().__init__(env, agent, backend, config, loggers, run_context)
        self.jx_config = jx_config
        self._evaluate_population = None
        # Building the initial world (perlin noise) is the expensive part of a
        # seeded reset, so states are cached per seed -- the JAX-side twin of
        # Env's init snapshot cache.
        self._state_cache: dict[int | None, EnvState] = {}

    def describe(self) -> dict:
        described = super().describe()
        described.update({f'jx.{key}': str(value)
                          for key, value in vars(self.jx_config).items()})
        return described

    def _setup_backend(self) -> None:
        self._evaluate_population = make_evaluate_population(
            self.agent.model, self.jx_config,
            self.config.episode_steps, self.agent.action_coefs)
        self.backend.setup_batched(self.agent, self._evaluate_fitnesses,
                                   popsize=self.config.popsize, seed=self.config.seed)

    def _initial_state(self, seed: int | None) -> EnvState:
        if seed not in self._state_cache:
            self._state_cache[seed] = init_state(seed, self.jx_config)
        return self._state_cache[seed]

    def _evaluate_fitnesses(self, population) -> np.ndarray:
        """Total episode reward per candidate, all on the same initial world."""
        initial_state = self._initial_state(self._current_eval_seed)
        return np.asarray(self._evaluate_population(population, initial_state))
