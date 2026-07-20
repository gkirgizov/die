"""Physics of the JAX env, pinned against the xarray reference implementation."""

import numpy as np
import pytest

jax = pytest.importorskip('jax')
jnp = jax.numpy

from skimage import filters  # noqa: E402

from die.data_init import DataInitializer  # noqa: E402
from die.env import BoundaryCondition, Dynamics, Env  # noqa: E402
from die.jx.env import gaussian_blur_wrap, init_state, step, step_with_aux  # noqa: E402
from die.jx.state import (  # noqa: E402
    AGENT_ALIVE,
    AGENT_FOOD,
    MEDIUM_CHEM,
    MEDIUM_FOOD,
    EnvConfig,
    diffuse_radius_for,
)

FIELD = (16, 16)
MAX_AGENTS = FIELD[0] * FIELD[1]


def make_config(**overrides) -> EnvConfig:
    params = dict(field_size=FIELD, max_agents=MAX_AGENTS, init_agent_ratio=0.2)
    return EnvConfig(**{**params, **overrides})


def const_action(dx: float = 0., dy: float = 0., deposit: float = 0.) -> jax.Array:
    action = np.zeros((3, MAX_AGENTS), dtype=np.float32)
    action[0], action[1], action[2] = dx, dy, deposit
    return jnp.asarray(action)


def run_steps(state, action, config, num_steps: int):
    rewards = []
    for _ in range(num_steps):
        state, reward = step(state, action, config)
        rewards.append(float(reward))
    return state, rewards


class TestGaussianBlur:
    """The hand-rolled wrap-mode blur replaces skimage inside the JAX core."""

    @pytest.mark.parametrize('sigma', [0.5, 0.8, 1.0, 2.0])
    def test_matches_skimage_wrap(self, sigma):
        rng = np.random.default_rng(0)
        field = rng.random(FIELD).astype(np.float32)
        expected = filters.gaussian(field, sigma=sigma, mode='wrap', preserve_range=True)
        got = np.asarray(gaussian_blur_wrap(jnp.asarray(field), sigma))
        np.testing.assert_allclose(got, expected, atol=1e-6)


class TestInitParity:
    """Same seed must build the same world as Env.reset, up to the f32 cast."""

    @pytest.mark.parametrize('seed', [7, 42])
    def test_init_matches_xarray_env(self, seed):
        env = Env(FIELD, Dynamics(init_agent_ratio=0.2))
        env.reset(seed=seed)
        state = init_state(seed, make_config())

        np.testing.assert_allclose(np.asarray(state.medium), env.medium.values, atol=1e-6)
        np.testing.assert_allclose(np.asarray(state.agents), env.agents.values, atol=1e-6)

    def test_different_seeds_differ(self):
        config = make_config()
        assert not np.allclose(np.asarray(init_state(1, config).medium),
                               np.asarray(init_state(2, config).medium))


class TestRolloutParity:
    """Cross-implementation parity on a scripted rollout.

    Stationary agents under the 'limit' boundary keep the layout collision-free,
    which isolates the physics (deposit, feed, cost, diffusion, decay) from the
    one deliberate divergence -- see :class:`TestDepositCollisions`.
    """

    def test_scripted_rollout_matches_xarray_env(self):
        seed, steps, deposit = 7, 20, 0.1
        env = Env(FIELD, Dynamics(init_agent_ratio=0.2,
                                  boundary=BoundaryCondition.limit))
        env.reset(seed=seed)
        config = make_config(boundary='limit')
        state = init_state(seed, config)

        action_np = np.zeros((3, MAX_AGENTS))
        action_np[2] = deposit
        action_jx = const_action(deposit=deposit)

        for _ in range(steps):
            action = DataInitializer.init_action_for(env.agents, action_np)
            _, reward_ref, *_ = env.step(action)
            state, reward = step(state, action_jx, config)
            assert float(reward) == pytest.approx(reward_ref, rel=1e-5)

        np.testing.assert_allclose(np.asarray(state.medium), env.medium.values, atol=1e-5)
        np.testing.assert_allclose(np.asarray(state.agents), env.agents.values, atol=1e-5)


class TestDepositCollisions:
    """The one deliberate physics divergence from the xarray env.

    When several agents share a cell, numpy fancy-index ``+=`` keeps only the
    last write, while the JAX core scatter-adds. Accumulating is the intended
    semantics (docs/NOTES.md logs the last-write behavior as a collision bug).
    """

    def test_collided_deposits_accumulate(self):
        config = make_config(boundary='limit', rate_decay_chem=0., diffuse_sigma=1e-6)
        state = init_state(7, config)
        # Park two alive agents on the same cell, kill the rest.
        agents = np.zeros((4, MAX_AGENTS), dtype=np.float32)
        agents[:, :2] = np.array([[0.5], [0.5], [1.], [1.]])
        state = state.replace(agents=jnp.asarray(agents),
                              medium=state.medium.at[MEDIUM_CHEM].set(0.))

        state, _ = step(state, const_action(deposit=0.3), config)
        chem_total = float(jnp.sum(state.medium[MEDIUM_CHEM]))
        assert chem_total == pytest.approx(0.6, rel=1e-4)


class TestFoodConservation:
    def test_food_depletes(self):
        config = make_config(food_infinite=False, cost_weights=(0., 0.))
        state = init_state(7, config)
        before = float(jnp.sum(state.medium[MEDIUM_FOOD]))
        state, _ = run_steps(state, const_action(), config, 10)
        assert float(jnp.sum(state.medium[MEDIUM_FOOD])) < before

    def test_food_monotonically_depletes(self):
        config = make_config(food_infinite=False, cost_weights=(0., 0.))
        state = init_state(7, config)
        totals = [float(jnp.sum(state.medium[MEDIUM_FOOD]))]
        for _ in range(5):
            state, _ = step(state, const_action(), config)
            totals.append(float(jnp.sum(state.medium[MEDIUM_FOOD])))
        assert all(later < earlier for earlier, later in zip(totals, totals[1:]))

    def test_food_infinite_is_conserved(self):
        config = make_config(food_infinite=True, cost_weights=(0., 0.))
        state = init_state(7, config)
        before = float(jnp.sum(state.medium[MEDIUM_FOOD]))
        state, _ = run_steps(state, const_action(), config, 10)
        assert float(jnp.sum(state.medium[MEDIUM_FOOD])) == pytest.approx(before, rel=1e-5)


class TestBoundary:
    def test_wrap_keeps_coords_in_bounds(self):
        config = make_config(boundary='wrap')
        state = init_state(7, config)
        state, _ = run_steps(state, const_action(dx=0.2, dy=0.2), config, 10)
        alive = np.asarray(state.agents[AGENT_ALIVE]) > 0
        coords = np.asarray(state.agents[:2])[:, alive]
        assert np.all(coords >= 0.) and np.all(coords < 1.)

    def test_wrap_is_modular(self):
        config = make_config(boundary='wrap')
        state = init_state(7, config)
        start = np.asarray(state.agents[:2]).copy()
        state, _ = step(state, const_action(dx=0.2, dy=0.2), config)
        np.testing.assert_allclose(np.asarray(state.agents[:2]), (start + 0.2) % 1., atol=1e-6)

    def test_limit_clips_at_bounds(self):
        config = make_config(boundary='limit')
        state = init_state(7, config)
        # 10 * 0.2 = 2.0 >> 1.0: every agent ends up pinned at the corner.
        state, _ = run_steps(state, const_action(dx=0.2, dy=0.2), config, 10)
        np.testing.assert_allclose(np.asarray(state.agents[:2]), 1., atol=1e-6)


class TestLifecycle:
    def test_starvation_kills_agents(self):
        # Deposit cost dominates any possible feeding gain.
        config = make_config(agents_die=True, cost_weights=(10., 0.))
        state = init_state(7, config)
        state, _ = run_steps(state, const_action(deposit=1.0), config, 3)
        assert float(jnp.sum(state.agents[AGENT_ALIVE])) == 0.

    def test_alive_mask_is_monotone(self):
        config = make_config(agents_die=True, cost_weights=(1., 0.))
        state = init_state(7, config)
        counts = [float(jnp.sum(state.agents[AGENT_ALIVE]))]
        for _ in range(4):
            state, _ = step(state, const_action(deposit=0.5), config)
            counts.append(float(jnp.sum(state.agents[AGENT_ALIVE])))
        assert all(later <= earlier for earlier, later in zip(counts, counts[1:]))

    def test_agents_survive_without_death_pressure(self):
        config = make_config(agents_die=False, cost_weights=(10., 0.))
        state = init_state(7, config)
        before = float(jnp.sum(state.agents[AGENT_ALIVE]))
        state, _ = run_steps(state, const_action(deposit=1.0), config, 5)
        assert float(jnp.sum(state.agents[AGENT_ALIVE])) == before
        assert float(jnp.min(state.agents[AGENT_FOOD])) < 0.  # went into debt, still alive


class TestReward:
    def test_first_reward_equals_feeding_gain(self):
        # 'limit' boundary: under 'wrap' the x=1.0 seam relocates edge agents
        # on the first step, changing the set of feeding cells mid-step.
        config = make_config(boundary='limit', cost_weights=(0., 0.))
        state = init_state(7, config)
        agent_mask = np.asarray(state.medium[0]) > 0
        expected = float((config.rate_feed * np.asarray(state.medium[MEDIUM_FOOD])
                          * agent_mask).sum())

        _, reward = step(state, const_action(), config)
        assert float(reward) == pytest.approx(expected, rel=1e-5)

    def test_action_cost_reduces_reward(self):
        state_free = init_state(7, make_config(cost_weights=(0., 0.)))
        _, reward_free = step(state_free, const_action(dx=0.01, deposit=0.5),
                              make_config(cost_weights=(0., 0.)))
        config_costly = make_config()
        state_costly = init_state(7, config_costly)
        _, reward_costly = step(state_costly, const_action(dx=0.01, deposit=0.5), config_costly)
        assert float(reward_costly) < float(reward_free)


class TestChemical:
    def test_chem_decays_without_deposit(self):
        config = make_config()
        state = init_state(7, config)
        state = state.replace(medium=state.medium.at[MEDIUM_CHEM].set(1.0))
        before = float(jnp.sum(state.medium[MEDIUM_CHEM]))
        state, _ = step(state, const_action(), config)
        assert float(jnp.sum(state.medium[MEDIUM_CHEM])) < before

    def test_deposit_adds_chem(self):
        config = make_config(rate_decay_chem=0.)
        state = init_state(7, config)
        before = float(jnp.sum(state.medium[MEDIUM_CHEM]))
        state, _ = step(state, const_action(deposit=1.0), config)
        assert float(jnp.sum(state.medium[MEDIUM_CHEM])) > before


class TestFoodFlow:
    def test_wave_flow_changes_food(self):
        config = make_config(food_flow='wave', food_infinite=True)
        state = init_state(7, config)
        before = np.asarray(state.medium[MEDIUM_FOOD]).copy()
        state, _ = step(state, const_action(), config)
        assert not np.allclose(np.asarray(state.medium[MEDIUM_FOOD]), before)

    def test_wave_flow_is_deterministic(self):
        config = make_config(food_flow='wave')
        first, _ = run_steps(init_state(7, config), const_action(), config, 3)
        second, _ = run_steps(init_state(7, config), const_action(), config, 3)
        np.testing.assert_array_equal(np.asarray(first.medium), np.asarray(second.medium))


class TestStepAux:
    """The per-agent breakdown behind the scalar reward.

    This is the signal RL consumes; ``gained.sum()`` is what evolution sees.
    The two must never disagree, or the engines are optimizing different things.
    """

    def test_gained_sums_to_scalar_reward(self):
        config = make_config()
        state = init_state(7, config)
        _, reward, aux = step_with_aux(state, const_action(deposit=.1), config)
        assert float(aux.gained.sum()) == pytest.approx(float(reward), rel=1e-6)

    def test_step_matches_step_with_aux(self):
        """The scalar path must stay byte-identical -- evolution depends on it."""
        config = make_config()
        state, action = init_state(7, config), const_action(deposit=.1)
        plain_state, plain_reward = step(state, action, config)
        aux_state, aux_reward, _ = step_with_aux(state, action, config)
        assert float(plain_reward) == float(aux_reward)
        np.testing.assert_array_equal(np.asarray(plain_state.medium),
                                      np.asarray(aux_state.medium))

    def test_dead_slots_contribute_nothing(self):
        """Dead slots are parked at (0, 0); unmasked they would phantom-forage."""
        config = make_config()
        state = init_state(7, config)
        _, _, aux = step_with_aux(state, const_action(deposit=.1), config)
        assert np.all(np.asarray(aux.gained)[np.asarray(aux.alive) == 0] == 0.)
        # and the mask really does exclude someone, or this asserts nothing
        assert 0 < float(aux.alive.sum()) < MAX_AGENTS

    def test_died_marks_this_steps_terminals(self):
        """`died` is the per-agent terminal signal; GAE must not bootstrap across it."""
        config = make_config(agents_die=True, cost_weights=(10., 0.))
        state = init_state(7, config)
        alive_before = float(state.agents[AGENT_ALIVE].sum())

        next_state, _, aux = step_with_aux(state, const_action(deposit=1.), config)

        # burning 10x their stock kills everyone who acted, in one step
        assert float(aux.alive.sum()) == alive_before
        assert float(aux.died.sum()) == alive_before
        assert float(next_state.agents[AGENT_ALIVE].sum()) == 0.

    def test_nobody_dies_without_death_pressure(self):
        config = make_config(agents_die=False, cost_weights=(10., 0.))
        state = init_state(7, config)
        _, _, aux = step_with_aux(state, const_action(deposit=1.), config)
        assert float(aux.died.sum()) == 0.


class TestParamsSplit:
    """Static/traced discipline -- the prerequisite for sweeping configurations.

    A shape-affecting field left traced does not fail here; it fails much later
    inside ``jit`` with an opaque tracer error. These tests are the tripwire.
    """

    def _rollout_fn(self):
        action = const_action(deposit=.1)

        @jax.jit
        def rollout(config, state):
            def one(carry, _):
                carry, reward, _ = step_with_aux(carry, action, config)
                return carry, reward
            return jax.lax.scan(one, state, length=3)[1].sum()

        return rollout

    def test_only_continuous_scalars_are_traced(self):
        """Named rather than counted, so adding a field is a deliberate choice.

        A shape-affecting field left traced does not fail here -- it fails much
        later inside ``jit``, with a tracer error that points nowhere useful.
        """
        import dataclasses

        config = make_config()
        traced = [field.name for field in dataclasses.fields(config)
                  if field.metadata.get('pytree_node', True)]
        assert traced == ['rate_feed', 'rate_decay_chem', 'diffuse_sigma',
                          'cost_weights', 'wave_scale', 'wave_decay',
                          # Resource renewal. Traced, and the classification
                          # was earned: regrow_rate is applied inside the
                          # jitted step, so an earlier version that branched on
                          # it in Python raised TracerBoolConversionError the
                          # moment params arrived as a jit argument. The fix
                          # was to drop the branch, not to make it static --
                          # these have to sweep under vmap like the rest.
                          'regrow_rate', 'food_capacity',
                          'respawn_rate', 'respawn_amount',
                          'food_noise', 'comm_noise']
        # cost_weights is one field but two leaves
        assert len(jax.tree.leaves(config)) == len(traced) + 1
        for static in (config.field_size, config.max_agents, config.boundary):
            assert static not in jax.tree.leaves(config)

    def test_traced_change_does_not_recompile(self):
        """The whole point: a population of configs costs one compilation."""
        config, rollout = make_config(), self._rollout_fn()
        state = init_state(7, config)
        for rate_feed in (0.05, 0.1, 0.2, 0.4):
            rollout(config.replace(rate_feed=rate_feed), state)
        assert rollout._cache_size() == 1

    def test_static_change_does_recompile(self):
        config, rollout = make_config(), self._rollout_fn()
        state = init_state(7, config)
        rollout(config, state)
        rollout(config.replace(agents_die=not config.agents_die), state)
        assert rollout._cache_size() == 2

    def test_vmap_over_configs(self):
        config = make_config(diffuse_radius=diffuse_radius_for(.5))
        state = init_state(7, config)
        batch = jax.tree.map(lambda *leaves: jnp.stack(leaves),
                             *[config.replace(rate_feed=rf) for rf in (.05, .1, .2)])
        rewards = np.asarray(jax.vmap(self._rollout_fn(), in_axes=(0, None))(batch, state))
        assert rewards.shape == (3,)
        assert len(np.unique(rewards)) == 3      # configs actually took effect
        assert np.all(np.diff(rewards) > 0)      # more feeding -> more reward

    def test_radius_is_pinned_at_construction(self):
        """Resolved once, while sigma is concrete, then carried on the treedef.

        Without this, passing params *as an argument* to a jitted function --
        not closing over them -- would make sigma a tracer and leave the kernel
        with no derivable shape.
        """
        config = make_config(diffuse_sigma=.8)
        assert config.diffuse_radius == diffuse_radius_for(.8)

        @jax.jit
        def peek(params):
            return params.diffuse_sigma * 0 + params.blur_radius

        assert float(peek(config)) == diffuse_radius_for(.8)

    def test_sweeping_sigma_keeps_the_constructing_window(self):
        """Sigma sweeps *within* a fixed truncation window, not across windows.

        The radius sets a shape, so it cannot vary along a vmap axis. Sweeping
        sigma upward from a narrow-constructed params therefore truncates
        silently -- construct with the widest sigma (or pin diffuse_radius) to
        cover the whole sweep range.
        """
        narrow = make_config(diffuse_sigma=.5)
        assert narrow.diffuse_radius == 2

        @jax.jit
        def peek(params):
            return params.diffuse_sigma * 0 + params.blur_radius

        batch = jax.tree.map(lambda *leaves: jnp.stack(leaves),
                             narrow.replace(diffuse_sigma=.5),
                             narrow.replace(diffuse_sigma=2.))
        assert np.all(np.asarray(jax.vmap(peek)(batch)) == 2)
        assert diffuse_radius_for(2.) == 8      # what an unswept sigma=2 would use

        wide = make_config(diffuse_radius=diffuse_radius_for(2.))
        assert wide.blur_radius == 8            # the way to cover the range

    def test_explicit_radius_matches_the_derived_one(self):
        config = make_config(diffuse_sigma=.8)
        state = init_state(7, config)
        pinned = config.replace(diffuse_radius=diffuse_radius_for(.8))
        derived, _ = step(state, const_action(deposit=.1), config)
        explicit, _ = step(state, const_action(deposit=.1), pinned)
        np.testing.assert_array_equal(np.asarray(derived.medium),
                                      np.asarray(explicit.medium))
