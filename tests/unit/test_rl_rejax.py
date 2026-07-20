"""The gymnax adapter: does the DIE core survive being flattened into one agent?

These pin the *contract* rejax relies on -- shapes, spaces, the observation
round-trip, and the action layout. The layout is the one that would fail
silently: a transposed action vector still has the right shape and still trains,
it just steers the deposit channel with the dx values.
"""

import pytest

jax = pytest.importorskip('jax')
pytest.importorskip('rejax')
jnp = jax.numpy

import numpy as np  # noqa: E402

from die.jx.interface import make_init_bank, reset  # noqa: E402
from die.jx.state import MEDIUM_CHEM, EnvParams  # noqa: E402
from die.rl.rejax_backend import (  # noqa: E402
    ConvActor,
    ConvCritic,
    DiePopulationEnv,
    RejaxParams,
    action_size,
    flatten_obs,
    make_rejax_ppo,
    obs_size,
    train_rejax,
    unflatten_obs,
)

FIELD = (8, 8)
MAX_AGENTS = FIELD[0] * FIELD[1]


def env_params(**overrides) -> EnvParams:
    params = dict(field_size=FIELD, max_agents=MAX_AGENTS, init_agent_ratio=0.2,
                  agents_die=False, food_infinite=False)
    return EnvParams(**{**params, **overrides})


def make_env(**overrides):
    params = RejaxParams(env=env_params(), max_steps_in_episode=10, **overrides)
    return DiePopulationEnv(make_init_bank([0, 1], params.env), params), params


class TestObservationCodec:
    def test_round_trips_exactly(self):
        params = env_params()
        obs, _ = reset(jax.random.key(0), make_init_bank([0], params), params)
        restored = unflatten_obs(flatten_obs(obs), params)

        np.testing.assert_array_equal(np.asarray(restored.medium), np.asarray(obs.medium))
        np.testing.assert_array_equal(np.asarray(restored.agent), np.asarray(obs.agent))
        np.testing.assert_array_equal(np.asarray(restored.cells), np.asarray(obs.cells))

    def test_cell_indices_survive_the_float_detour(self):
        """They are carried as float32 only because gymnax wants one array."""
        params = env_params()
        obs, _ = reset(jax.random.key(3), make_init_bank([7], params), params)
        restored = unflatten_obs(flatten_obs(obs), params)

        assert restored.cells.dtype == jnp.int32
        np.testing.assert_array_equal(np.asarray(restored.cells), np.asarray(obs.cells))

    def test_size_matches_the_declared_space(self):
        env, params = make_env()
        obs, _ = env.reset(jax.random.key(0), params)
        assert obs.shape == env.observation_space(params).shape
        assert obs.shape == (obs_size(params.env),)


class TestSpaces:
    def test_action_space_is_the_policy_support(self):
        """[-1, 1] exactly -- the env owns the scaling, so bounds cannot drift."""
        env, params = make_env()
        space = env.action_space(params)

        assert space.shape == (action_size(params.env),)
        assert float(space.low) == -1. and float(space.high) == 1.

    def test_observation_space_is_unbounded(self):
        env, params = make_env()
        space = env.observation_space(params)
        assert not np.isfinite(np.asarray(space.low)).all()   # chem grows past 1


class TestStep:
    def test_returns_the_gymnax_five_tuple(self):
        env, params = make_env()
        _, state = env.reset(jax.random.key(0), params)
        obs, _, reward, done, info = env.step(
            jax.random.key(1), state, jnp.zeros(action_size(params.env)), params)

        assert obs.shape == (obs_size(params.env),)
        assert jnp.shape(reward) == () and jnp.shape(done) == ()
        assert info == {}

    def test_reward_is_the_alive_masked_total_gain(self):
        """The same scalar evolution maximizes, so returns stay comparable."""
        from die.jx.interface import step_env

        env, params = make_env()
        _, state = env.reset(jax.random.key(0), params)
        action = jnp.zeros(action_size(params.env))

        _, _, reward, _, _ = env.step_env(jax.random.key(1), state, action, params)
        timestep, _ = step_env(jax.random.key(1), state, jnp.zeros((3, MAX_AGENTS)),
                               params.env)
        expected = (timestep.reward * timestep.alive).sum()
        np.testing.assert_allclose(float(reward), float(expected), rtol=1e-6)

    def test_action_layout_is_channel_major(self):
        """``[3 * A]`` unpacks as ``[3, A]``: dx, dy, then deposit.

        Getting this backwards is invisible -- same shape, still trains -- so it
        is pinned by driving *only* the deposit channel and checking that chem
        moves while positions do not.
        """
        env, params = make_env()
        _, state = env.reset(jax.random.key(0), params)

        action = jnp.zeros((3, MAX_AGENTS)).at[2].set(1.).ravel()
        _, deposited, _, _, _ = env.step_env(jax.random.key(1), state, action, params)
        _, still, _, _, _ = env.step_env(
            jax.random.key(1), state, jnp.zeros(action_size(params.env)), params)

        assert float(still.medium[MEDIUM_CHEM].sum()) == 0.
        assert float(deposited.medium[MEDIUM_CHEM].sum()) > 0.
        # ...and it was the deposit channel, not a movement one, that did it.
        np.testing.assert_allclose(np.asarray(deposited.agents[:2]),
                                   np.asarray(still.agents[:2]), atol=1e-6)

    def test_does_not_terminate_when_nothing_can_die(self):
        """``agents_die=False`` plus ``max_steps=None`` means no bootstrap cuts.

        rejax collapses truncation into termination, so the only safe episode
        shape here is one that never ends inside a rollout.
        """
        env, params = make_env()
        _, state = env.reset(jax.random.key(0), params)
        for i in range(5):
            _, state, _, done, _ = env.step(
                jax.random.key(i), state, jnp.zeros(action_size(params.env)), params)
            assert not bool(done)


class TestNetworks:
    def test_actor_emits_one_action_per_agent_channel(self):
        env, params = make_env()
        obs, _ = env.reset(jax.random.key(0), params)
        batch = obs[None]

        actor = ConvActor(params.env, features=8, num_blocks=1, hidden=16)
        variables = actor.init(jax.random.key(0), batch, jax.random.key(0))
        action, log_prob, entropy = actor.apply(variables, batch, jax.random.key(1))

        assert action.shape == (1, action_size(params.env))
        assert log_prob.shape == (1,) and entropy.shape == (1,)

    def test_actor_evaluation_is_deterministic(self):
        """``act`` uses the mean, matching die.rl.ppo.make_evaluate."""
        env, params = make_env()
        batch = env.reset(jax.random.key(0), params)[0][None]
        actor = ConvActor(params.env, features=8, num_blocks=1, hidden=16)
        variables = actor.init(jax.random.key(0), batch, jax.random.key(0))

        first = actor.apply(variables, batch, jax.random.key(1), method='act')
        second = actor.apply(variables, batch, jax.random.key(2), method='act')
        np.testing.assert_array_equal(np.asarray(first), np.asarray(second))
        assert np.abs(np.asarray(first)).max() <= 1.

    def test_critic_returns_one_joint_value_per_world(self):
        env, params = make_env()
        batch = jnp.stack([env.reset(jax.random.key(s), params)[0] for s in range(3)])

        critic = ConvCritic(params.env, features=8, num_blocks=1, hidden=16)
        variables = critic.init(jax.random.key(0), batch)
        assert critic.apply(variables, batch).shape == (3,)


class TestTraining:
    def test_a_short_run_completes_and_stays_finite(self):
        """Mechanical check that rejax's loop accepts this env at all."""
        params = RejaxParams(env=env_params(), max_steps_in_episode=8)
        bank = make_init_bank([0, 1], params.env)
        total = 4 * 8 * 2
        algorithm = make_rejax_ppo(bank, params, conv=True, total_timesteps=total,
                                   num_envs=4, num_steps=8, num_epochs=1,
                                   num_minibatches=2, eval_freq=total)
        _, (lengths, returns) = train_rejax(algorithm, jax.random.key(0))

        assert np.all(np.isfinite(np.asarray(returns)))
        assert np.all(np.asarray(lengths) == params.max_steps_in_episode)

    def test_the_stock_mlp_path_also_runs(self):
        """rejax untouched, as the architecture control."""
        params = RejaxParams(env=env_params(), max_steps_in_episode=8)
        bank = make_init_bank([0], params.env)
        total = 4 * 8 * 2
        algorithm = make_rejax_ppo(bank, params, conv=False, total_timesteps=total,
                                   num_envs=4, num_steps=8, num_epochs=1,
                                   num_minibatches=2, eval_freq=total,
                                   agent_kwargs={'hidden_layer_sizes': (16, 16)})
        _, (_, returns) = train_rejax(algorithm, jax.random.key(0))
        assert np.all(np.isfinite(np.asarray(returns)))

    def test_held_out_evaluation_uses_the_other_bank(self):
        """Otherwise the reported return measures drift, not generalization.

        Asserted behaviorally: the same policy, trained identically, scores
        differently when evaluated on disjoint worlds. Identity checks would not
        work here -- ``rejax.Algorithm.create`` deep-copies its config, so the
        env (and the bank inside it) is a copy, not the object handed in.
        """
        params = RejaxParams(env=env_params(), max_steps_in_episode=8)
        train_bank = make_init_bank([0, 1], params.env)
        eval_bank = make_init_bank([50, 51], params.env)
        total = 4 * 8

        def run(**extra):
            algorithm = make_rejax_ppo(train_bank, params, conv=True, total_timesteps=total,
                                       num_envs=4, num_steps=8, num_epochs=1,
                                       num_minibatches=2, eval_freq=total, **extra)
            return np.asarray(train_rejax(algorithm, jax.random.key(0))[1][1])

        on_train, on_held_out = run(), run(eval_bank=eval_bank)
        assert np.all(np.isfinite(on_train)) and np.all(np.isfinite(on_held_out))
        assert not np.allclose(on_train.mean(axis=-1), on_held_out.mean(axis=-1))
