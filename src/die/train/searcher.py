"""The swappable seam between the trainer and evolutionary-search libraries.

The interface is at generation granularity so that both problem-driven
libraries (evotorch) and plain ask-tell libraries (evosax, pycma, pyribs)
can implement it.

Two evaluation shapes are supported. Per-candidate backends (evotorch) call
``evaluate(agent)`` once per solution; batched backends (evosax over the JAX
core) hand the whole population to one vectorized call, which is where the
JAX path gets its speedup.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from die.agents.base import Agent

#: Fitness of a single candidate agent (maximization).
EvalFn = Callable[[Agent], float]
#: Fitness of a whole population, given its stacked parameter pytree.
PopulationEvalFn = Callable[[Any], np.ndarray]


@dataclass
class GenStats:
    """Per-generation statistics reported by a searcher backend."""
    generation: int
    best_fitness: float       # best of this generation's population
    mean_fitness: float
    best_ever_fitness: float
    extra: dict[str, float] = field(default_factory=dict)  # backend-specific


class SearcherBackend(ABC):
    """One evolutionary-search library behind a uniform interface."""

    @abstractmethod
    def setup(self,
              agent: Agent,
              evaluate: EvalFn,
              popsize: int,
              seed: int | None = None) -> None:
        """Binds the search to a template network and a fitness function."""

    @abstractmethod
    def run_generation(self) -> GenStats:
        """Runs one ask-evaluate-tell cycle."""

    @abstractmethod
    def best_agent(self) -> Agent:
        """Returns the best-so-far solution as a parameterized agent."""


class BatchedSearcherBackend(SearcherBackend):
    """A backend that evaluates the whole population in one vectorized call."""

    @abstractmethod
    def setup_batched(self,
                      agent: Agent,
                      evaluate_population: PopulationEvalFn,
                      popsize: int,
                      seed: int | None = None) -> None:
        """Binds the search to a template agent and a population fitness function."""

    def setup(self, agent: Agent, evaluate: EvalFn,
              popsize: int, seed: int | None = None) -> None:
        raise TypeError(f'{type(self).__name__} evaluates whole populations: '
                        'drive it with JaxEvolutionTrainer, which calls setup_batched()')
