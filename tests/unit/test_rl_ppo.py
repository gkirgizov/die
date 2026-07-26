"""PPO internals.

The GAE and masking tests carry the weight here. Dead slots outnumber live
agents roughly nine to one at default settings, so a missing mask does not
crash -- it quietly averages the objective toward zero.
"""

import numpy as np
import pytest

jax = pytest.importorskip('jax')
jnp = jax.numpy

from die.jx.interface import make_init_bank  # noqa: E402
from die.jx.state import EnvParams  # noqa: E402
from die.rl.config import HyperParams, TrainStatic  # noqa: E402
from die.rl.ppo import (  # noqa: E402
    compute_gae,
    make_evaluate,
    make_train,
    masked_mean,
    masked_normalize,
)

FIELD = (12, 12)
MAX_AGENTS = FIELD[0] * FIELD[1]


def make_env_params(**overrides) -> EnvParams:
    params = dict(field_size=FIELD, max_agents=MAX_AGENTS, init_agent_ratio=0.2,
                  food_infinite=True, rate_decay_chem=0.025, diffuse_sigma=0.8)
    return EnvParams(**{**params, **overrides})


def tiny_static(**overrides) -> TrainStatic:
    static = dict(num_envs=2, num_steps=4, num_iterations=2, num_epochs=1,
                  num_minibatches=2, features=8, num_blocks=1, hidden=16, bank_size=2)
    return TrainStatic(**{**static, **overrides})


def reference_gae(reward, value, done, last_value, gamma, lam):
    """Straightforward backwards loop, written independently of the jax version."""
    advantages = np.zeros_like(reward)
    running, next_value = 0., last_value
    for t in reversed(range(len(reward))):
        continues = 1. - done[t]
        delta = reward[t] + gamma * next_value * continues - value[t]
        running = delta + gamma * lam * continues * running
        advantages[t] = running
        next_value = value[t]
    return advantages


class TestMasking:
    def test_masked_mean_ignores_masked_entries(self):
        x = jnp.asarray([1., 2., 100., 200.])
        mask = jnp.asarray([1., 1., 0., 0.])
        assert float(masked_mean(x, mask)) == pytest.approx(1.5)

    def test_masked_mean_survives_an_empty_mask(self):
        """Whole-population extinction must not divide by zero."""
        value = masked_mean(jnp.asarray([1., 2.]), jnp.zeros(2))
        assert float(value) == 0.
        assert bool(jnp.isfinite(value))

    def test_masked_normalize_uses_only_live_entries(self):
        x = jnp.asarray([1., 2., 3., 999.])
        mask = jnp.asarray([1., 1., 1., 0.])
        out = np.asarray(masked_normalize(x, mask))
        assert out[:3].mean() == pytest.approx(0., abs=1e-5)
        assert out[:3].std() == pytest.approx(1., abs=1e-3)

    def test_dead_slots_would_shift_the_statistics(self):
        """Guards the reason the mask exists, not just that it is applied."""
        x = jnp.asarray([1., 2., 3., 0., 0., 0.])
        mask = jnp.asarray([1., 1., 1., 0., 0., 0.])
        assert float(masked_mean(x, mask)) == pytest.approx(2.)
        assert float(x.mean()) == pytest.approx(1.)      # what no mask would give


class TestGae:
    def test_matches_a_reference_implementation(self):
        rng = np.random.default_rng(0)
        reward = rng.normal(size=(8, 3)).astype(np.float32)
        value = rng.normal(size=(8, 3)).astype(np.float32)
        done = (rng.random((8, 3)) < 0.2).astype(np.float32)
        last_value = rng.normal(size=3).astype(np.float32)

        advantages, targets = compute_gae(
            jnp.asarray(reward), jnp.asarray(value), jnp.asarray(done),
            jnp.asarray(last_value), 0.99, 0.95)
        expected = reference_gae(reward, value, done, last_value, 0.99, 0.95)

        np.testing.assert_allclose(np.asarray(advantages), expected, rtol=1e-4, atol=1e-5)
        np.testing.assert_allclose(np.asarray(targets), expected + value, rtol=1e-4, atol=1e-5)

    def test_death_cuts_the_bootstrap(self):
        """A terminated agent has no continuation value to bootstrap from."""
        reward = jnp.asarray([[1.], [1.]])
        value = jnp.asarray([[10.], [10.]])
        done = jnp.asarray([[0.], [1.]])          # dies on the second step
        advantages, _ = compute_gae(reward, value, done,
                                    jnp.asarray([10.]), 0.99, 0.95)
        # last step: r - V, with no gamma * V_next term
        assert float(advantages[1, 0]) == pytest.approx(1. - 10.)

    def test_death_cuts_the_recursion(self):
        """Advantage must not flow backwards across a terminal step."""
        value = jnp.zeros((3, 1))
        rewards = jnp.asarray([[0.], [0.], [100.]])
        terminal = jnp.asarray([[0.], [1.], [0.]])       # episode ends at t=1
        no_terminal = jnp.zeros((3, 1))

        cut = compute_gae(rewards, value, terminal, jnp.zeros(1), .99, .95)[0]
        uncut = compute_gae(rewards, value, no_terminal, jnp.zeros(1), .99, .95)[0]

        assert float(cut[0, 0]) == pytest.approx(0.)     # reward at t=2 cannot reach t=0
        assert float(uncut[0, 0]) > 50.                  # ...but does without the terminal

    def test_zero_reward_and_value_gives_zero_advantage(self):
        zeros = jnp.zeros((5, 2))
        advantages, targets = compute_gae(zeros, zeros, zeros, jnp.zeros(2), .99, .95)
        assert float(jnp.abs(advantages).max()) == 0.
        assert float(jnp.abs(targets).max()) == 0.


class TestTrain:
    def test_runs_and_stays_finite(self):
        env_params = make_env_params()
        static = tiny_static()
        bank = make_init_bank(range(static.bank_size), env_params)
        runner, metrics = jax.jit(make_train(static, bank))(
            jax.random.key(0), HyperParams(env=env_params))

        for leaf in jax.tree.leaves(metrics):
            assert np.all(np.isfinite(np.asarray(leaf)))
        assert metrics.raw_return.shape == (static.num_iterations,)
        for leaf in jax.tree.leaves(runner.params):
            assert np.all(np.isfinite(np.asarray(leaf)))

    def test_is_reproducible(self):
        env_params = make_env_params()
        static = tiny_static()
        bank = make_init_bank(range(static.bank_size), env_params)
        train = jax.jit(make_train(static, bank))
        first = train(jax.random.key(0), HyperParams(env=env_params))[1]
        second = train(jax.random.key(0), HyperParams(env=env_params))[1]
        np.testing.assert_array_equal(np.asarray(first.raw_return),
                                      np.asarray(second.raw_return))

    def test_joint_factorization_runs(self):
        """The evolution-equivalent objective: one advantage per world step."""
        env_params = make_env_params()
        static = tiny_static(factorization='joint')
        bank = make_init_bank(range(static.bank_size), env_params)
        _, metrics = jax.jit(make_train(static, bank))(
            jax.random.key(0), HyperParams(env=env_params))
        assert np.all(np.isfinite(np.asarray(metrics.raw_return)))

    def test_hyperparameter_sweep_costs_one_compilation(self):
        """The property the meta-learning phase is built on.

        purejaxrl's design cannot do this -- it closes over config as Python
        values, so only the seed can be mapped.
        """
        env_params = make_env_params()
        static = tiny_static()
        bank = make_init_bank(range(static.bank_size), env_params)
        train = jax.jit(make_train(static, bank))

        base = HyperParams(env=env_params)
        for learning_rate in (1e-3, 3e-3, 1e-2, 3e-2):
            train(jax.random.key(0), base.replace(learning_rate=learning_rate))
        assert train._cache_size() == 1

    def test_vmaps_over_a_population_of_configs(self):
        env_params = make_env_params()
        static = tiny_static()
        bank = make_init_bank(range(static.bank_size), env_params)

        base = HyperParams(env=env_params)
        population = jax.tree.map(
            lambda *leaves: jnp.stack(leaves),
            *[base.replace(learning_rate=lr) for lr in (1e-3, 1e-2, 1e-1)])
        keys = jax.random.split(jax.random.key(0), 3)

        _, metrics = jax.jit(jax.vmap(make_train(static, bank)))(keys, population)
        assert metrics.raw_return.shape == (3, static.num_iterations)
        # different rates must actually produce different runs
        finals = np.asarray(metrics.raw_return)[:, -1]
        assert len(np.unique(finals.round(4))) == 3


class TestEvaluate:
    def test_is_deterministic(self):
        env_params = make_env_params()
        static = tiny_static()
        bank = make_init_bank(range(static.bank_size), env_params)
        runner, _ = jax.jit(make_train(static, bank))(
            jax.random.key(0), HyperParams(env=env_params))

        evaluate = jax.jit(make_evaluate(static, bank), static_argnums=(4, 5))
        hypers = HyperParams(env=env_params)
        first = evaluate(runner.params, runner.normalizer, jax.random.key(1), hypers, 5, 2)
        second = evaluate(runner.params, runner.normalizer, jax.random.key(1), hypers, 5, 2)
        assert float(first) == float(second)
        assert np.isfinite(float(first))
