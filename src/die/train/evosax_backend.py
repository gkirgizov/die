"""Evosax-based searcher backend (PGPE with ClipUp / separable CMA-ES).

The JAX counterpart of :mod:`die.train.evotorch_backend`, and the default
training path. Requires the ``train`` extra.

Hyperparameters are translated from the evotorch defaults so both backends
start from comparable settings -- see :meth:`EvosaxBackend.setup_batched`.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np

try:
    import jax
    import jax.numpy as jnp
    from evosax.algorithms import PGPE, Sep_CMA_ES
    from evosax.core.optimizer import clipup
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "The training stack requires the 'train' extra: "
        "uv sync --extra train / pip install 'die[train]'") from exc

from die.agents.base import Agent
from die.train.searcher import BatchedSearcherBackend, GenStats, PopulationEvalFn


class EvosaxBackend(BatchedSearcherBackend):
    """Evolution strategies over a flax parameter pytree.

    NB: evosax minimizes, while fitness here is reward to be maximized, so
    fitness is negated on the way in and back out.
    """

    def __init__(self,
                 algorithm: Literal['pgpe', 'cmaes'] = 'pgpe',
                 radius_init: float = 1.5,
                 stdev_init: float = 0.1,
                 stdev_learning_rate: float = 0.1,
                 momentum: float = 0.9,
                 **algo_kwargs):
        self._algorithm = algorithm
        self._radius_init = radius_init
        self._stdev_init = stdev_init
        self._stdev_lr = stdev_learning_rate
        self._momentum = momentum
        self._algo_kwargs = algo_kwargs

        self._agent: Agent | None = None
        self._strategy = None
        self._state = None
        self._params = None
        self._evaluate_population: PopulationEvalFn | None = None
        self._key = None
        self._generation = 0

    @property
    def params(self) -> dict:
        return {
            'backend': 'evosax',
            'algorithm': self._algorithm,
            'radius_init': self._radius_init,
            'stdev_init': self._stdev_init,
            'stdev_learning_rate': self._stdev_lr,
            'momentum': self._momentum,
            **self._algo_kwargs,
        }

    def setup_batched(self,
                      agent: Agent,
                      evaluate_population: PopulationEvalFn,
                      popsize: int,
                      seed: int | None = None) -> None:
        self._agent = agent
        self._evaluate_population = evaluate_population
        self._key = jax.random.key(seed if seed is not None else 0)
        template = agent.params

        if self._algorithm == 'pgpe':
            if popsize % 2 != 0:
                raise ValueError(f'PGPE samples antithetically and needs an even '
                                 f'popsize, got {popsize}')
            # ClipUp rules-of-thumb from the ClipUp paper, matching the
            # evotorch backend's configuration.
            max_speed = self._radius_init / 15.
            center_learning_rate = max_speed / 2.
            self._strategy = PGPE(
                population_size=popsize,
                solution=template,
                optimizer=clipup(learning_rate=center_learning_rate,
                                 max_velocity=max_speed,
                                 momentum=self._momentum),
                **self._algo_kwargs,
            )
            # evotorch parameterizes the search by the *radius* (norm) of the
            # initial stdev vector; evosax by its per-dimension scale.
            std_init = self._radius_init / math.sqrt(self._strategy.num_dims)
            self._params = self._strategy.default_params.replace(
                std_init=std_init, std_lr=self._stdev_lr)
        elif self._algorithm == 'cmaes':
            self._strategy = Sep_CMA_ES(
                population_size=popsize,
                solution=template,
                **self._algo_kwargs,
            )
            self._params = self._strategy.default_params.replace(std_init=self._stdev_init)
        else:
            raise ValueError(f'Unknown evosax algorithm: {self._algorithm!r}')

        self._key, init_key = jax.random.split(self._key)
        self._state = self._strategy.init(init_key, template, self._params)
        self._generation = 0

    def run_generation(self) -> GenStats:
        if self._strategy is None:
            raise RuntimeError('Backend not set up: call setup_batched() first')

        self._key, ask_key, tell_key = jax.random.split(self._key, 3)
        population, self._state = self._strategy.ask(ask_key, self._state, self._params)

        rewards = np.asarray(self._evaluate_population(population), dtype=np.float64)
        self._state, _ = self._strategy.tell(
            tell_key, population, jnp.asarray(-rewards), self._state, self._params)

        self._generation += 1
        return GenStats(
            generation=self._generation,
            best_fitness=float(rewards.max()),
            mean_fitness=float(rewards.mean()),
            best_ever_fitness=float(-self._state.best_fitness),
            extra={'median_fitness': float(np.median(rewards))},
        )

    def best_agent(self) -> Agent:
        if self._strategy is None:
            raise RuntimeError('Backend not set up: call setup_batched() first')
        best_params = self._strategy._unravel_solution(self._state.best_solution)
        return self._agent.with_params(best_params)
