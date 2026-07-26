"""Renewable and reappearing resource, pinned across both engines.

Three mechanisms, and the reason there are three rather than one is the whole
content of this file:

- **Wave flow** moves resource around. It does not create any, and before it
  was clipped it could drive food *negative*, which silently inverts every
  quantity downstream.
- **Logistic regrowth** renews resource in place. It provably cannot revive a
  cell foraged to exactly zero, because the term is proportional to what is
  already there.
- **Stochastic respawn** creates resource where there is none. It is the only
  one of the three that can, and it is the only one that needs an RNG -- which
  is why it lives in the interface layer rather than the physics core.

Getting these confused would make "dynamic environment" mean a world that
merely gets richer, which is the failure mode ``perturb_medium``'s docstring
already records for food noise.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip('jax')
jnp = jax.numpy

from die.data_init import DataInitializer  # noqa: E402
from die.env import BoundaryCondition, Dynamics, Env  # noqa: E402
from die.jx.env import init_state, regrow_food, step  # noqa: E402
from die.jx.interface import get_obs, step_env  # noqa: E402
from die.jx.state import MEDIUM_FOOD, EnvConfig  # noqa: E402

FIELD = (16, 16)
MAX_AGENTS = FIELD[0] * FIELD[1]

#: Calibrated so renewal offsets consumption instead of flooding the world.
#: Measured under physarum over 40 steps: the non-renewable baseline runs
#: 16.15 -> 10.18, these land within 1% of where they started.
BALANCED_REGROW = dict(regrow_rate=0.02, food_capacity=0.5)
BALANCED_RESPAWN = dict(stochastic_respawn=True, respawn_rate=0.005,
                        respawn_amount=0.15)


def make_config(**overrides) -> EnvConfig:
    params = dict(field_size=FIELD, max_agents=MAX_AGENTS, init_agent_ratio=0.2)
    return EnvConfig(**{**params, **overrides})


def const_action(deposit: float = 0.) -> jax.Array:
    action = np.zeros((3, MAX_AGENTS), dtype=np.float32)
    action[2] = deposit
    return jnp.asarray(action)


def food_of(state) -> np.ndarray:
    return np.asarray(state.medium[MEDIUM_FOOD])


class TestWaveIsClipped:
    """Food is a resource, so it may not go negative."""

    def test_wave_never_drives_food_negative(self):
        """Regression on measured behaviour: unclipped, this world reached
        -9.6 total and -1.24 per cell within five steps."""
        config = make_config(food_flow='wave', wave_scale=0.5, wave_decay=0.5)
        state = init_state(7, config)
        for _ in range(30):
            state, _ = step(state, const_action(), config)
            assert food_of(state).min() >= 0.

    def test_both_engines_clip(self):
        """The clip lives in two places (WaveSequence and jx.env) and must
        agree, or 'two engines, one physics' is already broken."""
        from die.data_init import WaveSequence
        sequence = WaveSequence(FIELD, dt=0.01, t_bounds=(0., 10.))
        flow = sequence.get_flow_operator(scale=0.5, decay=0.5)
        current = np.full(FIELD, 0.01)
        assert flow(current).min() >= 0.

    def test_wave_still_moves_food(self):
        """Guards the clip from having flattened the mechanism entirely."""
        config = make_config(food_flow='wave')
        state = init_state(7, config)
        before = food_of(state).copy()
        state, _ = step(state, const_action(), config)
        assert not np.allclose(food_of(state), before)


class TestLogisticRegrowth:
    def test_is_a_noop_when_disabled(self):
        """The default has to leave every earlier result untouched."""
        food = jnp.asarray(np.linspace(0., 1., 64).reshape(8, 8), jnp.float32)
        np.testing.assert_array_equal(
            np.asarray(regrow_food(food, make_config())), np.asarray(food))

    def test_cannot_revive_an_empty_cell(self):
        """The documented reason stochastic respawn has to exist separately."""
        food = jnp.zeros((4, 4), jnp.float32).at[0, 0].set(0.2)
        grown = np.asarray(regrow_food(food, make_config(**BALANCED_REGROW)))
        assert grown[0, 0] > 0.2
        assert np.count_nonzero(grown) == 1

    def test_saturates_at_capacity(self):
        """Logistic, not exponential: growth stops at the carrying capacity."""
        config = make_config(**BALANCED_REGROW)
        food = jnp.full((4, 4), config.food_capacity, jnp.float32)
        np.testing.assert_allclose(np.asarray(regrow_food(food, config)),
                                   np.asarray(food), atol=1e-6)

    def test_matches_the_xarray_engine(self):
        """Cross-engine parity on a scripted rollout, the same shape as
        TestRolloutParity in test_jx_env.py -- stationary agents under 'limit'
        so deposit collisions cannot muddy the comparison."""
        seed, steps, deposit = 7, 15, 0.1
        env = Env(FIELD, Dynamics(init_agent_ratio=0.2,
                                  boundary=BoundaryCondition.limit,
                                  **BALANCED_REGROW))
        env.reset(seed=seed)
        config = make_config(boundary='limit', **BALANCED_REGROW)
        state = init_state(seed, config)

        action_np = np.zeros((3, MAX_AGENTS))
        action_np[2] = deposit
        for _ in range(steps):
            action = DataInitializer.init_action_for(env.agents, action_np)
            _, reward_ref, *_ = env.step(action)
            state, reward = step(state, const_action(deposit), config)
            assert float(reward) == pytest.approx(reward_ref, rel=1e-5)

        np.testing.assert_allclose(food_of(state),
                                   env.medium.sel(channel='env_food').values,
                                   atol=1e-5)

    def test_offsets_consumption_at_the_calibrated_rate(self):
        """Pins the calibration: a renewable world should stay near where it
        started, not ratchet toward empty and not flood."""
        config = make_config(**BALANCED_REGROW)
        state = init_state(200, config)
        start = food_of(state).sum()
        for _ in range(40):
            state, _ = step(state, const_action(), config)
        assert 0.8 * start <= food_of(state).sum() <= 1.3 * start


class TestStochasticRespawn:
    def test_is_a_noop_when_the_gate_is_off(self):
        """Setting a rate without the static gate must do nothing.

        ``food_infinite`` is on to isolate the mechanism: ``step_env`` runs the
        full physics, so agents feed and the food field would change for
        reasons that have nothing to do with respawn.
        """
        config = make_config(respawn_rate=0.5, respawn_amount=1.0,
                             food_infinite=True)
        state = init_state(7, config)
        before = food_of(state).copy()
        _, state = step_env(jax.random.key(0), state, const_action(), config)
        np.testing.assert_allclose(food_of(state), before, atol=1e-6)

    def test_creates_food_where_there_was_none(self):
        """The capability logistic regrowth provably lacks."""
        config = make_config(stochastic_respawn=True, respawn_rate=0.1,
                             respawn_amount=0.5)
        state = init_state(7, config)
        empty_before = food_of(state) <= 1e-6
        assert empty_before.any(), 'test is vacuous without empty cells'

        _, state = step_env(jax.random.key(0), state, const_action(), config)
        assert (food_of(state)[empty_before] > 0).any()

    def test_arrival_rate_is_about_right(self):
        """Bernoulli per cell per step, at the stated rate and not some other.

        Averaged over many keys rather than asserted on one draw. Arrivals are
        Binomial(121, 0.1) here, sd ~3.5 against a mean of ~12, so a
        single-draw check is either vacuous or flaky depending on the tolerance
        -- an earlier version of this test picked a key that happened to be the
        minimum of 200 and failed at a mechanism that was correct.
        """
        rate, draws = 0.1, 64
        config = make_config(stochastic_respawn=True, respawn_rate=rate,
                             respawn_amount=0.5, food_infinite=True)
        state = init_state(7, config)
        empty = food_of(state) <= 1e-6

        filled = []
        for i in range(draws):
            _, stepped = step_env(jax.random.key(i), state, const_action(), config)
            filled.append(int((food_of(stepped)[empty] > 0).sum()))

        expected = rate * empty.sum()
        # Standard error of the mean over `draws` binomial samples.
        tolerance = 4 * np.sqrt(expected * (1 - rate) / draws)
        assert abs(np.mean(filled) - expected) < tolerance

    def test_regrowth_and_respawn_differ_in_reach(self):
        """The distinction the two mechanisms exist for, measured directly:
        regrowth keeps the occupied-cell count fixed, respawn grows it."""
        steps = 30
        grow_config = make_config(**BALANCED_REGROW)
        state = init_state(200, grow_config)
        occupied_before = int((food_of(state) > 1e-6).sum())
        for _ in range(steps):
            state, _ = step(state, const_action(), grow_config)
        assert int((food_of(state) > 1e-6).sum()) == occupied_before

        spawn_config = make_config(**BALANCED_RESPAWN)
        state = init_state(200, spawn_config)
        for i in range(steps):
            _, state = step_env(jax.random.key(i), state, const_action(), spawn_config)
        assert int((food_of(state) > 1e-6).sum()) > occupied_before


class TestObservableThroughTheInterface:
    def test_respawned_food_reaches_the_observation(self):
        """A resource a policy cannot see is not a resource."""
        config = make_config(stochastic_respawn=True, respawn_rate=0.3,
                             respawn_amount=0.5, food_infinite=True)
        state = init_state(7, config)
        before = np.asarray(get_obs(state, config).medium[MEDIUM_FOOD]).sum()
        _, state = step_env(jax.random.key(0), state, const_action(), config)
        assert np.asarray(get_obs(state, config).medium[MEDIUM_FOOD]).sum() > before
