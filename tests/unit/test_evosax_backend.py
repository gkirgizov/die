"""The evosax searcher backend, exercised on a synthetic fitness landscape.

Keeping the fitness analytic separates "is the search wired up correctly"
(sign conventions, ask/tell plumbing, best-so-far tracking) from "is the env
physics right", which ``test_jx_env.py`` covers.
"""

import numpy as np
import pytest

jax = pytest.importorskip('jax')
jnp = jax.numpy
pytest.importorskip('evosax')

from die.train.evosax_backend import EvosaxBackend  # noqa: E402

TARGET = jnp.array([1.0, -2.0, 3.0])


class StubAgent:
    """Minimal stand-in for the parts of the Agent API the backend touches."""

    def __init__(self, params):
        self.params = params

    def with_params(self, params) -> 'StubAgent':
        return StubAgent(params)


@pytest.fixture
def agent() -> StubAgent:
    return StubAgent({'w': jnp.zeros(3)})


def evaluate_population(population) -> np.ndarray:
    """Maximized fitness: negative squared distance to the target."""
    return np.asarray(-jnp.sum((population['w'] - TARGET) ** 2, axis=-1))


@pytest.mark.parametrize('algorithm', ['pgpe', 'cmaes'])
class TestSearch:
    def test_improves_towards_target(self, agent, algorithm):
        backend = EvosaxBackend(algorithm=algorithm)
        backend.setup_batched(agent, evaluate_population, popsize=8, seed=0)
        history = [backend.run_generation() for _ in range(40)]

        assert history[-1].best_ever_fitness > history[0].best_fitness
        recovered = np.asarray(backend.best_agent().params['w'])
        np.testing.assert_allclose(recovered, np.asarray(TARGET), atol=0.5)

    def test_best_ever_is_monotone(self, agent, algorithm):
        backend = EvosaxBackend(algorithm=algorithm)
        backend.setup_batched(agent, evaluate_population, popsize=8, seed=0)
        history = [backend.run_generation() for _ in range(10)]

        best_ever = [stats.best_ever_fitness for stats in history]
        assert all(later >= earlier - 1e-9
                   for earlier, later in zip(best_ever, best_ever[1:]))

    def test_stats_are_consistent(self, agent, algorithm):
        backend = EvosaxBackend(algorithm=algorithm)
        backend.setup_batched(agent, evaluate_population, popsize=8, seed=0)
        stats = backend.run_generation()

        assert stats.generation == 1
        assert stats.best_fitness >= stats.mean_fitness
        assert stats.best_ever_fitness >= stats.best_fitness - 1e-9
        # Fitness is a maximized reward: negated on the way into evosax and back.
        assert stats.best_fitness <= 0.
        assert 'median_fitness' in stats.extra

    def test_is_reproducible(self, agent, algorithm):
        def run():
            backend = EvosaxBackend(algorithm=algorithm)
            backend.setup_batched(agent, evaluate_population, popsize=8, seed=11)
            return [backend.run_generation().best_fitness for _ in range(5)]

        assert run() == run()


class TestValidation:
    def test_pgpe_rejects_odd_popsize(self, agent):
        with pytest.raises(ValueError, match='even popsize'):
            EvosaxBackend(algorithm='pgpe').setup_batched(
                agent, evaluate_population, popsize=7, seed=0)

    def test_unknown_algorithm(self, agent):
        with pytest.raises(ValueError, match='Unknown evosax algorithm'):
            EvosaxBackend(algorithm='nope').setup_batched(
                agent, evaluate_population, popsize=8, seed=0)

    def test_requires_setup(self):
        with pytest.raises(RuntimeError, match='not set up'):
            EvosaxBackend().run_generation()

    def test_rejects_per_candidate_setup(self, agent):
        """Batched backends must be driven by the JAX trainer."""
        with pytest.raises(TypeError, match='whole populations'):
            EvosaxBackend().setup(agent, lambda a: 0., popsize=8)

    def test_params_describe_the_run(self):
        params = EvosaxBackend(algorithm='pgpe', radius_init=2.0).params
        assert params['backend'] == 'evosax'
        assert params['algorithm'] == 'pgpe'
        assert params['radius_init'] == 2.0
