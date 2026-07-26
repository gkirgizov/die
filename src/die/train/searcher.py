"""The swappable seam between the trainer and evolutionary-search libraries.

The interface is at generation granularity so that both problem-driven
libraries (evotorch) and plain ask-tell libraries (evosax, pycma, pyribs)
can implement it. See docs/NOTES.md for the intended JAX/evosax follow-up.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field

from die.agents.neural import TorchAgent

EvalFn = Callable[[TorchAgent], float]


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
              agent: TorchAgent,
              evaluate: EvalFn,
              popsize: int,
              seed: int | None = None) -> None:
        """Binds the search to a template network and a fitness function."""

    @abstractmethod
    def run_generation(self) -> GenStats:
        """Runs one ask-evaluate-tell cycle."""

    @abstractmethod
    def best_agent(self) -> TorchAgent:
        """Returns the best-so-far solution as a parameterized agent."""
