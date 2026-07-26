"""Tests for the benchmark harness itself.

The harness is the thing every reported number passes through, so its own
correctness cannot be taken on faith: a scoring bug would move every agent
together and so would not look like a bug at all. These pin it against figures
established independently, in ``docs/JOURNAL.md`` and in the training script.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip('jax')
jnp = jax.numpy

from die.bench.canonical import EVAL_SEEDS, EVAL_STEPS, canonical_params  # noqa: E402
from die.bench.family import FAMILY_SEEDS, build_family  # noqa: E402
from die.bench.policies import NoState  # noqa: E402
from die.bench.runner import score_scenario  # noqa: E402
from die.jx.env import init_state  # noqa: E402
from die.jx.state import MEDIUM_FOOD  # noqa: E402


class NullPolicy:
    """Inlined rather than imported from die.jx.scripted, so a regression in
    the scripted ports cannot silently change the baseline these tests pin."""

    def init(self, key, params):
        return NoState()

    def act(self, key, state, obs, params):
        return jnp.zeros((3, params.max_agents), jnp.float32), state


class TestScoring:
    def test_do_nothing_matches_the_published_baseline(self):
        """The bar every agent is measured against.

        docs/JOURNAL.md and the rejax cross-check both put inaction on this
        config at 2.68-2.76. This harness is a separate implementation, so
        agreement is evidence it scores the same quantity.
        """
        metrics = score_scenario(NullPolicy(), canonical_params(), EVAL_SEEDS,
                                 EVAL_STEPS, jax.random.key(0))
        assert 2.6 <= float(metrics['total_return'].mean()) <= 2.85

    def test_doing_nothing_consumes_no_food(self):
        metrics = score_scenario(NullPolicy(), canonical_params(), EVAL_SEEDS,
                                 EVAL_STEPS, jax.random.key(0))
        # Agents still feed from the cell they sit on, so this is not exactly
        # zero -- but nothing forages, so it must stay small.
        assert float(metrics['food_consumed'].mean()) < 5.

    def test_scores_are_deterministic(self):
        """Same key, same score -- otherwise seed spreads are unreadable."""
        args = (NullPolicy(), canonical_params(), EVAL_SEEDS, EVAL_STEPS)
        first = score_scenario(*args, jax.random.key(0))['total_return']
        second = score_scenario(*args, jax.random.key(0))['total_return']
        np.testing.assert_allclose(first, second)

    def test_every_seed_is_a_distinct_world(self):
        """A silently-shared world would collapse the seed spread to zero and
        make every result look far more certain than it is."""
        metrics = score_scenario(NullPolicy(), canonical_params(), EVAL_SEEDS,
                                 EVAL_STEPS, jax.random.key(0))
        assert len(set(np.round(metrics['total_return'], 6))) > 1


class TestFamily:
    def test_food_scale_is_exactly_multiplicative(self):
        """The abundance axis must not disturb the spatial pattern, or a
        response along it could not be attributed to abundance."""
        base = init_state(FAMILY_SEEDS[0], canonical_params())
        scaled = init_state(FAMILY_SEEDS[0], canonical_params(food_scale=0.25))
        np.testing.assert_allclose(np.asarray(scaled.medium[MEDIUM_FOOD]),
                                   0.25 * np.asarray(base.medium[MEDIUM_FOOD]),
                                   rtol=1e-6)

    def test_noise_scenarios_enable_the_static_gate(self):
        """food_noise/comm_noise do nothing unless stochastic_medium is set --
        a silent no-op that would turn the noise axis into duplicated
        baselines."""
        for scenario in build_family():
            params = scenario.params()
            if params.food_noise or params.comm_noise:
                assert params.stochastic_medium, scenario.name

    def test_family_scenario_names_are_unique(self):
        names = [s.name for s in build_family()]
        assert len(names) == len(set(names))

    def test_family_seeds_are_disjoint_from_evaluation_seeds(self):
        """Otherwise the trained agents' family scores would not be
        out-of-sample."""
        assert not set(FAMILY_SEEDS) & set(EVAL_SEEDS)
