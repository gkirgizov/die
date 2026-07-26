"""The RL-shaped interface over the JAX env: auto-reset, terminals, init bank.

The physics is pinned elsewhere (``test_jx_env.py``); what matters here is the
episode bookkeeping RL depends on, where mistakes are silent rather than loud.
"""

import numpy as np
import pytest

jax = pytest.importorskip('jax')
jnp = jax.numpy

from die.jx.env import step_with_aux  # noqa: E402
from die.jx.interface import (  # noqa: E402
    get_obs,
    make_init_bank,
    perturb_medium,
    reset,
    step,
    step_env,
)
from die.jx.state import (  # noqa: E402
    AGENT_ALIVE,
    MEDIUM_CHEM,
    MEDIUM_FOOD,
    EnvParams,
)

FIELD = (16, 16)
MAX_AGENTS = FIELD[0] * FIELD[1]
SEEDS = (1, 2, 3)


def make_params(**overrides) -> EnvParams:
    params = dict(field_size=FIELD, max_agents=MAX_AGENTS, init_agent_ratio=0.2)
    return EnvParams(**{**params, **overrides})


def deposit_action(amount: float = 0.05) -> jax.Array:
    return jnp.zeros((3, MAX_AGENTS), jnp.float32).at[2].set(amount)


class TestInitBank:
    def test_bank_stacks_worlds_on_a_leading_axis(self):
        params = make_params()
        bank = make_init_bank(SEEDS, params)
        assert bank.medium.shape == (len(SEEDS), 3, *FIELD)
        assert bank.agents.shape == (len(SEEDS), 4, MAX_AGENTS)

    def test_bank_worlds_differ(self):
        """A bank of identical worlds would silently remove episode diversity."""
        bank = make_init_bank(SEEDS, make_params())
        assert not np.allclose(np.asarray(bank.medium[0]), np.asarray(bank.medium[1]))

    def test_empty_bank_is_rejected(self):
        with pytest.raises(ValueError, match='at least one seed'):
            make_init_bank([], make_params())

    def test_reset_draws_a_bank_world(self):
        params = make_params()
        bank = make_init_bank(SEEDS, params)
        _, state = reset(jax.random.key(0), bank, params)
        drawn = np.asarray(state.medium)
        assert any(np.allclose(drawn, np.asarray(bank.medium[i])) for i in range(len(SEEDS)))

    def test_reset_samples_more_than_one_world(self):
        params = make_params()
        bank = make_init_bank(SEEDS, params)
        seen = {float(reset(jax.random.key(k), bank, params)[1].medium.sum())
                for k in range(20)}
        assert len(seen) > 1

    def test_obs_shapes(self):
        params = make_params()
        bank = make_init_bank(SEEDS, params)
        obs, state = reset(jax.random.key(0), bank, params)
        assert obs.medium.shape == (3, *FIELD)
        assert obs.agent.shape == (2, MAX_AGENTS)     # (agent_food, alive)
        np.testing.assert_array_equal(np.asarray(obs.agent[1]),
                                      np.asarray(state.agents[AGENT_ALIVE]))


class TestStepEnv:
    def test_reward_is_the_raw_per_agent_gain(self):
        """Unshaped: sharing and survival bonuses belong to die.rl, not here."""
        params = make_params()
        bank = make_init_bank(SEEDS, params)
        _, state = reset(jax.random.key(0), bank, params)
        action = deposit_action()

        timestep, _ = step_env(jax.random.key(0), state, action, params)
        _, _, aux = step_with_aux(state, action, params)
        np.testing.assert_array_equal(np.asarray(timestep.reward), np.asarray(aux.gained))
        assert timestep.reward.shape == (MAX_AGENTS,)

    def test_alive_is_the_pre_step_mask(self):
        params = make_params()
        bank = make_init_bank(SEEDS, params)
        _, state = reset(jax.random.key(0), bank, params)
        timestep, _ = step_env(jax.random.key(0), state, deposit_action(), params)
        np.testing.assert_array_equal(np.asarray(timestep.alive),
                                      np.asarray(state.agents[AGENT_ALIVE]))

    def test_death_terminates_without_truncating(self):
        """Terminated and truncated must stay distinguishable -- GAE treats them
        differently, and only truncation bootstraps."""
        params = make_params(agents_die=True, cost_weights=(10., 0.), max_steps=None)
        bank = make_init_bank([1], params)
        _, state = reset(jax.random.key(0), bank, params)

        timestep, _ = step_env(jax.random.key(0), state, deposit_action(1.), params)
        assert float(timestep.terminated.sum()) == float(timestep.alive.sum()) > 0
        # ...though wiping out the population does truncate the world
        assert bool(timestep.truncated)

    def test_no_truncation_without_a_step_limit(self):
        params = make_params(max_steps=None)
        bank = make_init_bank([1], params)
        _, state = reset(jax.random.key(0), bank, params)
        for _ in range(5):
            timestep, state = step_env(jax.random.key(0), state, deposit_action(), params)
            assert not bool(timestep.truncated)

    def test_truncates_at_the_step_limit(self):
        params = make_params(max_steps=3)
        bank = make_init_bank([1], params)
        _, state = reset(jax.random.key(0), bank, params)
        flags = []
        for _ in range(3):
            timestep, state = step_env(jax.random.key(0), state, deposit_action(), params)
            flags.append(bool(timestep.truncated))
        assert flags == [False, False, True]


class TestAutoReset:
    def test_world_restarts_at_the_step_limit(self):
        params = make_params(max_steps=3)
        bank = make_init_bank(SEEDS, params)
        _, state = reset(jax.random.key(0), bank, params)

        steps = []
        for i in range(7):
            _, state = step(jax.random.key(i), state, deposit_action(), bank, params)
            steps.append(int(state.step))
        assert steps == [1, 2, 0, 1, 2, 0, 1]

    def test_returned_obs_always_describes_the_returned_state(self):
        """The auto-reset contract: obs is post-reset, reward is pre-reset.

        Rollout code relies on this to avoid branching on episode boundaries --
        and must use `truncated` to cut bootstrapping, precisely because obs no
        longer belongs to the trajectory that earned the reward.
        """
        params = make_params(max_steps=2)
        bank = make_init_bank(SEEDS, params)
        _, state = reset(jax.random.key(0), bank, params)

        saw_reset = False
        for i in range(6):
            timestep, state = step(jax.random.key(i), state, deposit_action(), bank, params)
            expected = get_obs(state, params)
            np.testing.assert_array_equal(np.asarray(timestep.obs.medium),
                                          np.asarray(expected.medium))
            saw_reset |= bool(timestep.truncated)
        assert saw_reset

    def test_reward_still_describes_the_completed_step(self):
        params = make_params(max_steps=1)
        bank = make_init_bank([1], params)
        _, state = reset(jax.random.key(0), bank, params)
        timestep, _ = step(jax.random.key(0), state, deposit_action(), bank, params)
        assert bool(timestep.truncated)
        assert float(timestep.reward.sum()) != 0.       # not zeroed by the reset

    def test_extinction_restarts_the_world(self):
        """An extinct world would otherwise burn rollout budget producing nothing."""
        params = make_params(agents_die=True, cost_weights=(10., 0.), max_steps=None)
        bank = make_init_bank([1], params)
        _, state = reset(jax.random.key(0), bank, params)

        timestep, state = step(jax.random.key(0), state, deposit_action(1.), bank, params)
        assert bool(timestep.truncated)
        assert float(state.agents[AGENT_ALIVE].sum()) > 0.      # repopulated
        assert int(state.step) == 0


class TestMediumNoise:
    """Per-step medium noise, and the bias trap that shapes its design."""

    def test_off_by_default_and_rng_free(self):
        """A clean run must be bit-identical regardless of key -- the physics
        core stays deterministic, which is what keeps both engines comparable."""
        params = make_params()
        bank = make_init_bank([1], params)
        _, state = reset(jax.random.key(0), bank, params)
        first, _ = step_env(jax.random.key(0), state, deposit_action(.2), params)
        second, _ = step_env(jax.random.key(999), state, deposit_action(.2), params)
        np.testing.assert_array_equal(np.asarray(first.obs.medium),
                                      np.asarray(second.obs.medium))

    def test_on_makes_the_world_stochastic(self):
        params = make_params(stochastic_medium=True, food_noise=.25, comm_noise=.05)
        bank = make_init_bank([1], params)
        _, state = reset(jax.random.key(0), bank, params)
        first, _ = step_env(jax.random.key(0), state, deposit_action(.2), params)
        second, _ = step_env(jax.random.key(999), state, deposit_action(.2), params)
        assert not np.allclose(np.asarray(first.obs.medium), np.asarray(second.obs.medium))

    def test_food_noise_is_mean_preserving(self):
        """The trap: additive-then-clipped noise *creates food from nothing*.

        On an empty cell it turns negative draws into 0 and positive draws into
        free resource, so a "harder" world is really a richer one. This inverted
        a whole robustness experiment before being caught, which is why food
        noise is multiplicative.
        """
        params = make_params(stochastic_medium=True, food_noise=.25, comm_noise=0.)
        bank = make_init_bank([1], params)
        _, state = reset(jax.random.key(0), bank, params)

        clean = np.asarray(state.medium[MEDIUM_FOOD])
        samples = [np.asarray(perturb_medium(jax.random.key(k), state.medium,
                                             params)[MEDIUM_FOOD])
                   for k in range(200)]
        assert np.mean(samples) == pytest.approx(clean.mean(), rel=2e-3)

    def test_food_noise_cannot_create_food(self):
        params = make_params(stochastic_medium=True, food_noise=.5, comm_noise=0.)
        bank = make_init_bank([1], params)
        _, state = reset(jax.random.key(0), bank, params)
        medium = state.medium.at[MEDIUM_FOOD].set(0.)

        perturbed = perturb_medium(jax.random.key(0), medium, params)
        assert float(perturbed[MEDIUM_FOOD].max()) == 0.

    def test_comm_noise_can_fabricate_signal(self):
        """Deliberately *not* multiplicative: a corrupted channel has to be able
        to produce false positives, not merely attenuate real messages."""
        params = make_params(stochastic_medium=True, food_noise=0., comm_noise=.1)
        bank = make_init_bank([1], params)
        _, state = reset(jax.random.key(0), bank, params)
        medium = state.medium.at[MEDIUM_CHEM].set(0.)

        perturbed = perturb_medium(jax.random.key(0), medium, params)
        assert float(perturbed[MEDIUM_CHEM].max()) > 0.

    def test_channels_stay_physical(self):
        params = make_params(stochastic_medium=True, food_noise=.5, comm_noise=.5)
        bank = make_init_bank([1], params)
        _, state = reset(jax.random.key(0), bank, params)
        perturbed = perturb_medium(jax.random.key(0), state.medium, params)

        # concentrations, so never negative -- feeding on debt would be worse
        assert float(perturbed[MEDIUM_FOOD].min()) >= 0.
        assert float(perturbed[MEDIUM_CHEM].min()) >= 0.
        # the agent-presence channel is bookkeeping, not a measurement
        np.testing.assert_array_equal(np.asarray(perturbed[0]),
                                      np.asarray(state.medium[0]))


class TestCompiles:
    def _rollout(self):
        def rollout(key, params, bank, num_steps=6):
            _, state = reset(key, bank, params)

            def one(carry, step_key):
                timestep, carry = step(step_key, carry, deposit_action(), bank, params)
                return carry, timestep.reward.sum()

            return jax.lax.scan(one, state, jax.random.split(key, num_steps))[1]
        return rollout

    def test_scans_under_jit(self):
        params = make_params(max_steps=3)
        bank = make_init_bank(SEEDS, params)
        rewards = jax.jit(self._rollout(), static_argnums=3)(jax.random.key(0), params, bank)
        assert rewards.shape == (6,)
        assert np.all(np.isfinite(np.asarray(rewards)))

    def test_vmaps_over_parallel_worlds(self):
        params = make_params(max_steps=3)
        bank = make_init_bank(SEEDS, params)
        keys = jax.random.split(jax.random.key(0), 4)
        rollout = jax.jit(jax.vmap(self._rollout(), in_axes=(0, None, None)), static_argnums=3)
        rewards = np.asarray(rollout(keys, params, bank))
        assert rewards.shape == (4, 6)
        assert len({tuple(row.round(5)) for row in rewards}) > 1     # worlds differ

    def test_traced_param_change_does_not_recompile(self):
        params = make_params(max_steps=3)
        bank = make_init_bank(SEEDS, params)
        rollout = jax.jit(self._rollout(), static_argnums=3)
        for rate_feed in (0.05, 0.1, 0.2, 0.4):
            rollout(jax.random.key(0), params.replace(rate_feed=rate_feed), bank)
        assert rollout._cache_size() == 1
