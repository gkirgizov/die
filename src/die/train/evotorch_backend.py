"""Evotorch-based searcher backend (PGPE / CMA-ES).

Requires the ``train`` extra (evotorch). Evotorch is pinned: it is
semi-dormant upstream, so this module deliberately stays the only place
that imports it -- see docs/NOTES.md for the replacement roadmap.
"""

from typing import Literal

import torch as th

try:
    from evotorch import Solution
    from evotorch.algorithms import CMAES, PGPE
    from evotorch.neuroevolution import NEProblem
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "The training stack requires the 'train' extra: "
        "uv sync --extra train / pip install 'die[train]'") from exc

from die.agents.neural import TorchAgent
from die.train.searcher import EvalFn, GenStats, SearcherBackend


def make_net(problem: NEProblem, solution: Solution) -> th.nn.Module:
    """No-copying version of ``evotorch.NEProblem.make_net()``."""
    parameters = solution.access_values(keep_evals=True)
    with th.no_grad():
        net = problem.parameterize_net(parameters)
    return net


class EvotorchBackend(SearcherBackend):
    def __init__(self,
                 algorithm: Literal['pgpe', 'cmaes'] = 'pgpe',
                 radius_init: float = 1.5,
                 stdev_init: float = 0.1,
                 num_actors: int = 1,
                 **algo_kwargs):
        self._algorithm = algorithm
        self._radius_init = radius_init
        self._stdev_init = stdev_init
        self._num_actors = num_actors
        self._algo_kwargs = algo_kwargs
        self._problem: NEProblem | None = None
        self._searcher = None
        self._generation = 0

    @property
    def params(self) -> dict:
        return {
            'backend': 'evotorch',
            'algorithm': self._algorithm,
            'radius_init': self._radius_init,
            'stdev_init': self._stdev_init,
            'num_actors': self._num_actors,
            **self._algo_kwargs,
        }

    def setup(self,
              agent: TorchAgent,
              evaluate: EvalFn,
              popsize: int,
              seed: int | None = None) -> None:
        # NB: explicit seed gives the problem its own torch generator,
        # insulating evotorch's sampling from global RNG re-seeding
        # performed by seeded evaluation rollouts.
        self._problem = NEProblem(
            'max',
            network=agent,
            network_eval_func=evaluate,
            initial_bounds=[-0.5, 0.5],
            num_actors=self._num_actors,
            seed=seed,
        )
        if self._algorithm == 'pgpe':
            # ClipUp configuration and rules-of-thumb from the ClipUp paper,
            # carried over from the original experiment script
            radius_init = self._radius_init
            max_speed = radius_init / 15.
            center_learning_rate = max_speed / 2.
            self._searcher = PGPE(
                self._problem,
                popsize=popsize,
                radius_init=radius_init,
                center_learning_rate=center_learning_rate,
                stdev_learning_rate=0.1,
                optimizer='clipup',
                optimizer_config={'max_speed': max_speed, 'momentum': 0.9},
                **self._algo_kwargs,
            )
        elif self._algorithm == 'cmaes':
            self._searcher = CMAES(
                self._problem,
                stdev_init=self._stdev_init,
                popsize=popsize,
                separable=True,
                **self._algo_kwargs,
            )
        else:
            raise ValueError(f'Unknown evotorch algorithm: {self._algorithm!r}')
        self._generation = 0

    def run_generation(self) -> GenStats:
        if self._searcher is None:
            raise RuntimeError('Backend not set up: call setup() first')
        self._searcher.step()
        status = self._searcher.status
        self._generation += 1
        return GenStats(
            generation=self._generation,
            best_fitness=float(status['pop_best_eval']),
            mean_fitness=float(status['mean_eval']),
            best_ever_fitness=float(status['best_eval']),
            extra={'median_fitness': float(status['median_eval'])}
            if 'median_eval' in status else {},
        )

    def best_agent(self) -> TorchAgent:
        if self._searcher is None:
            raise RuntimeError('Backend not set up: call setup() first')
        best = make_net(self._problem, self._searcher.status['best'])
        assert isinstance(best, TorchAgent)
        return best
