"""Running observation normalization.

The channels this normalizes are genuinely unbounded, so these tests double as
the record of *why* the module exists -- see ``test_chem_really_is_unbounded``.
"""

import numpy as np
import pytest

jax = pytest.importorskip('jax')
jnp = jax.numpy

from die.jx.interface import make_init_bank, reset, step  # noqa: E402
from die.jx.state import EnvParams  # noqa: E402
from die.rl.obs import OBS_AGENT_ALIVE, OBS_AGENT_FOOD, ObsNormalizer, RunningStats  # noqa: E402

FIELD = (16, 16)
MAX_AGENTS = FIELD[0] * FIELD[1]


def _repeat(obs, times: int):
    """Stacks an observation `times` over, whatever fields Obs happens to have."""
    return jax.tree.map(lambda leaf: jnp.stack([leaf] * times), obs)


def make_params(**overrides) -> EnvParams:
    params = dict(field_size=FIELD, max_agents=MAX_AGENTS, init_agent_ratio=0.2)
    return EnvParams(**{**params, **overrides})


def rollout(num_steps: int = 20, deposit: float = 0.3, **param_overrides):
    """Collects observations from a real rollout, not synthetic arrays."""
    params = make_params(**param_overrides)
    bank = make_init_bank([1, 2, 3], params)
    _, state = reset(jax.random.key(0), bank, params)
    action = jnp.zeros((3, MAX_AGENTS), jnp.float32).at[2].set(deposit)

    observations = []
    for i in range(num_steps):
        timestep, state = step(jax.random.key(i), state, action, bank, params)
        observations.append(timestep.obs)
    return observations


class TestRunningStats:
    def test_streaming_matches_batch_moments(self):
        """Chan's update must be exact, not merely close -- errors compound."""
        rng = np.random.default_rng(0)
        batches = [rng.normal(3., 2., size=(4, 50)).astype(np.float32) for _ in range(6)]

        stats = RunningStats.create(4)
        for batch in batches:
            stats = stats.merge(jnp.asarray(batch.mean(1)),
                                jnp.asarray(batch.var(1)),
                                jnp.asarray(float(batch.shape[1])))
        combined = np.concatenate(batches, axis=1)
        np.testing.assert_allclose(np.asarray(stats.mean), combined.mean(1), rtol=1e-4)
        np.testing.assert_allclose(np.asarray(stats.var), combined.var(1), rtol=1e-3)

    def test_normalize_standardizes(self):
        rng = np.random.default_rng(0)
        data = rng.normal(5., 3., size=(2, 4096)).astype(np.float32)
        stats = RunningStats.create(2).merge(jnp.asarray(data.mean(1)),
                                             jnp.asarray(data.var(1)),
                                             jnp.asarray(float(data.shape[1])))
        out = np.asarray(stats.normalize(jnp.asarray(data), feature_axis=-2))
        np.testing.assert_allclose(out.mean(1), [0., 0.], atol=1e-2)
        np.testing.assert_allclose(out.std(1), [1., 1.], atol=1e-2)

    def test_clipping_bounds_outliers(self):
        """Early in training the variance estimate is poor; clipping is the guard."""
        stats = RunningStats.create(1)
        out = stats.normalize(jnp.asarray([[1e6, -1e6]]), feature_axis=-2, clip=10.)
        np.testing.assert_array_equal(np.asarray(out), [[10., -10.]])


class TestChannelsAreUnbounded:
    def test_chem_really_is_unbounded(self):
        """The premise of this module: concentration accumulates past 1."""
        observations = rollout(num_steps=20, deposit=0.3)
        chem_max = max(float(obs.medium[2].max()) for obs in observations)
        assert chem_max > 1.0

    def test_normalization_bounds_it(self):
        observations = rollout(num_steps=20, deposit=0.3)
        normalizer = ObsNormalizer.create()
        for obs in observations:
            normalizer = normalizer.update(obs)
        out = normalizer.apply(observations[-1])
        assert np.all(np.abs(np.asarray(out.medium)) <= 10.)
        assert np.all(np.isfinite(np.asarray(out.medium)))


class TestObsNormalizer:
    def test_tracks_the_true_moments(self):
        observations = rollout()
        normalizer = ObsNormalizer.create()
        for obs in observations:
            normalizer = normalizer.update(obs)

        stacked = np.concatenate([np.asarray(obs.medium).reshape(3, -1)
                                  for obs in observations], axis=1)
        np.testing.assert_allclose(np.asarray(normalizer.medium.mean),
                                   stacked.mean(1), rtol=1e-3)
        np.testing.assert_allclose(np.sqrt(np.asarray(normalizer.medium.var)),
                                   stacked.std(1), rtol=1e-3)

    def test_agent_stats_ignore_dead_slots(self):
        """Dead slots carry zeros; unmasked they would track the death rate.

        This is the phantom-participation trap AGENTS.md flags for the physics,
        reappearing in the normalizer.
        """
        obs = rollout(num_steps=1)[0]
        normalizer = ObsNormalizer.create().update(obs)

        alive = np.asarray(obs.agent[OBS_AGENT_ALIVE]).astype(bool)
        food = np.asarray(obs.agent[OBS_AGENT_FOOD])
        assert alive.sum() < alive.size          # there really are dead slots

        masked_mean = food[alive].mean()
        assert float(normalizer.agent.mean[OBS_AGENT_FOOD]) == pytest.approx(
            masked_mean, rel=1e-3)
        # and the unmasked mean is far away, so this is not a vacuous assertion
        assert abs(food.mean() - masked_mean) > 0.1 * abs(masked_mean)

    def test_alive_row_passes_through(self):
        """It is a mask, not a feature: within-mask variance is exactly zero."""
        obs = rollout(num_steps=1)[0]
        normalizer = ObsNormalizer.create().update(obs)
        out = normalizer.apply(obs)
        np.testing.assert_array_equal(np.asarray(out.agent[OBS_AGENT_ALIVE]),
                                      np.asarray(obs.agent[OBS_AGENT_ALIVE]))
        assert np.all(np.isfinite(np.asarray(out.agent)))

    def test_handles_batched_observations(self):
        obs = rollout(num_steps=1)[0]
        batched = _repeat(obs, 4)

        single = ObsNormalizer.create().update(obs)
        batch = ObsNormalizer.create().update(batched)
        # same data four times over: identical moments, four times the count
        np.testing.assert_allclose(np.asarray(batch.medium.mean),
                                   np.asarray(single.medium.mean), rtol=1e-5)
        assert float(batch.medium.count) == pytest.approx(4 * float(single.medium.count))

        out = batch.apply(batched)
        assert out.medium.shape == batched.medium.shape
        assert out.agent.shape == batched.agent.shape

    def test_survives_jit_and_vmap(self):
        obs = rollout(num_steps=1)[0]

        updated = jax.jit(lambda n, o: n.update(o))(ObsNormalizer.create(), obs)
        assert np.all(np.isfinite(np.asarray(updated.medium.mean)))

        stacked = _repeat(obs, 3)
        out = jax.vmap(lambda o: updated.apply(o))(stacked)
        assert out.medium.shape == (3, 3, *FIELD)
