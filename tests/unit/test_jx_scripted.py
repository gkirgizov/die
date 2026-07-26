"""JAX ports of the scripted agents, pinned against the xarray originals.

The point of this file is parity, and parity of *stateful* agents specifically:
:class:`GradientAgent` and :class:`PhysarumAgent` mutate their momentum and
heading inside ``forward``, so matching them on step 1 proves nothing. Every
comparison here therefore runs several consecutive steps, driving both engines
off the same world sequence -- the xarray env is stepped once with the xarray
agent's action, and the JAX observation is rebuilt from those same arrays, so
neither implementation can drift by seeing a different medium.
"""

import numpy as np
import pytest

jax = pytest.importorskip('jax')
jnp = jax.numpy

from die.agents.gradient import GradientAgent, PhysarumAgent  # noqa: E402
from die.agents.static import BrownianAgent, ConstAgent  # noqa: E402
from die.data_init import DataInitializer  # noqa: E402
from die.env import Dynamics, Env  # noqa: E402
from die.jx.env import cells_at  # noqa: E402
from die.jx.interface import Obs  # noqa: E402
from die.jx.scripted import (  # noqa: E402
    BrownianPolicy,
    ConstPolicy,
    GradientPolicy,
    GradientState,
    NullPolicy,
    PhysarumPolicy,
    gather_by_agents,
    polar2xy,
    xy2polar,
)
from die.jx.state import EnvParams  # noqa: E402

FIELD = (16, 16)
MAX_AGENTS = FIELD[0] * FIELD[1]
SEED = 7
RATIO = 0.2

#: Tuned defaults from die.registry.make_agent, with the noise turned off so
#: the gradient agents become deterministic and can be compared step for step.
GRADIENT_KWARGS = dict(sense_offset=0.03, inertia=0.95, scale=0.01,
                       deposit=4.5, noise_scale=0.0, normalized_grad=True)
PHYSARUM_KWARGS = dict(turn_angle=35, sense_angle=120, sense_offset=0.03,
                       turn_tolerance=0.05, inertia=0., scale=0.0075,
                       deposit=4.5, noise_scale=0.0, normalized_grad=True)


def make_params(**overrides) -> EnvParams:
    params = dict(field_size=FIELD, max_agents=MAX_AGENTS, init_agent_ratio=RATIO)
    return EnvParams(**{**params, **overrides})


def make_env() -> Env:
    env = Env(FIELD, Dynamics(init_agent_ratio=RATIO))
    env.reset(seed=SEED)
    return env


def to_obs(agents, medium, params: EnvParams) -> Obs:
    """Builds the JAX observation from the xarray env's own arrays.

    Both engines then see byte-identical input (up to the f32 cast), which is
    what makes a step-for-step comparison meaningful rather than a comparison of
    two separately-evolving worlds.
    """
    coords = jnp.asarray(agents.sel(channel=['x', 'y']).values, jnp.float32)
    ix, iy = cells_at(coords, params)
    return Obs(
        medium=jnp.asarray(medium.values, jnp.float32),
        agent=jnp.asarray(agents.sel(channel=['agent_food', 'alive']).values, jnp.float32),
        cells=jnp.stack([ix, iy]),
        coords=coords)


class FixedTurns:
    """Feeds the numpy agent the +/-1 turn draws the JAX policy is about to make.

    Physarum's only randomness is the turn taken by agents whose gradient does
    not determine one, and numpy's and JAX's streams are unrelated. Rather than
    excluding those agents from the comparison -- which would leave the turn
    logic half-tested and would anyway diverge the carried heading a step later
    -- the test computes the draws on the JAX side and hands the same array to
    the numpy agent, making the whole trajectory comparable.

    ``normal`` covers ``_get_some_noise``, whose value is irrelevant at
    ``noise_scale=0`` but whose shape is not.
    """

    def __init__(self, turns: np.ndarray):
        self._turns = turns

    def integers(self, low, high, shape) -> np.ndarray:
        return self._turns

    def normal(self, loc, scale, size) -> np.ndarray:
        return np.zeros(size)


def carry_from(agent) -> GradientState:
    """Seeds the JAX carry from the numpy agent's own initial state.

    Both draw their starting momentum from a normal, but from different
    generators; copying makes the two trajectories comparable from step 0
    instead of from wherever they happen to converge.
    """
    return GradientState(grad=jnp.asarray(agent._prev_grad, jnp.float32),
                         direction=jnp.asarray(agent._direction_rads, jnp.float32))


def parity_steps(agent, policy, state, num_steps: int, share_turns: bool = False):
    """Runs both engines over one world sequence, yielding what each produced.

    Yields ``(action_jax, action_xarray, alive)`` per step. The env is advanced
    with the *xarray* action, so a divergence in the JAX action cannot feed back
    into the world and mask itself.
    """
    env = make_env()
    params = make_params()
    key = jax.random.key(0)

    for _ in range(num_steps):
        agents, medium = env.observe()
        alive = agents.sel(channel='alive').values > 0
        if share_turns:
            # Mirrors the documented `act` split: (turn_key, noise_key).
            turn_key = jax.random.split(key)[0]
            agent._rng = FixedTurns(
                np.asarray(jax.random.randint(turn_key, (MAX_AGENTS,), 0, 2)))

        action_xr = agent.forward((agents, medium)).values
        action_jx, state = policy.act(key, state, to_obs(agents, medium, params), params)

        yield np.asarray(action_jx), action_xr, alive
        env.step(DataInitializer.init_action_for(agents, action_xr))


def assert_per_channel_close(action_jx, action_xr, alive, rtol=1e-5, atol=1e-7):
    """Compares each action channel on its own scale.

    A single max-abs check over the whole action is dominated by ``deposit``,
    which is ~250x larger than ``dx``/``dy`` -- the movement channels would pass
    while being completely wrong.

    ``atol`` is what actually binds on ``dx``/``dy``, whose true values pass
    through zero; at 1e-7 it is ~1e-5 of those channels' full scale, against a
    measured worst case of 6e-9. ``rtol`` binds on ``deposit``. Both are float32
    rounding room and nothing more: a single misclassified Physarum turn moves
    ``dx`` by a full 2 * turn_angle and misses by ~1e-3.
    """
    for row, channel in enumerate(('dx', 'dy', 'deposit1')):
        np.testing.assert_allclose(action_jx[row][alive], action_xr[row][alive],
                                   rtol=rtol, atol=atol,
                                   err_msg=f'channel {channel} diverged')


class TestSignedZeroConvention:
    """Pins a deliberate bug-compatibility choice, so it fails legibly.

    :func:`die.utils.xy2polar` takes the angle of ``x + np.multiply(1j, y)``,
    which is **not** ``arctan2(y, x)``: the complex multiply contributes
    ``copysign(0, y)`` to the real part, so ``(-0.0, +0.0)`` collapses to
    ``(+0.0, +0.0)`` and reports angle 0 where IEEE reports pi.

    That is reachable rather than theoretical. ``grad_clip`` discards weak
    gradients by multiplying by a boolean, which preserves the sign bits of the
    direction it just threw away, and Physarum's ``undetermined_grad`` fires on
    angle 0 but not on pi. Measured on the 16x16 world at seed 7: at step 1, 28
    cells sit in ``0 < norm < grad_clip``, 22 carry a negative zero, and 13
    disagree between the two conventions -- so a clipped gradient that happened
    to point left is read as a decided heading while one pointing right is not.

    :func:`die.jx.scripted._angle` reproduces the quirk on purpose: diverging
    would make the JAX Physarum a *different agent* from the one the xarray
    benchmark numbers describe. This test exists so that "cleaning up"
    ``xy2polar`` to ``arctan2`` fails here, naming the coupling, rather than
    surfacing as an unexplained multi-step parity drift.
    """

    @pytest.mark.parametrize('x,y', [(-0.0, 0.0), (-0.0, -0.0), (0.0, 0.0),
                                     (-1.0, 0.0), (-1.0, -0.0), (1.0, 0.0)])
    def test_matches_the_xarray_helper(self, x, y):
        from die.jx.scripted import _angle
        from die.utils import xy2polar as reference
        np.testing.assert_allclose(float(_angle(jnp.float32(x), jnp.float32(y))),
                                   reference(np.float64(x), np.float64(y))[1],
                                   atol=1e-6)

    def test_the_quirk_is_still_a_quirk(self):
        """Guards the guard: if numpy ever made np.angle sign-neutral, the test
        above would pass vacuously and the port would need revisiting."""
        from die.utils import xy2polar as reference
        assert not np.isclose(reference(np.float64(-0.0), np.float64(0.0))[1],
                              np.arctan2(0.0, -0.0)), (
            'np.angle is now sign-neutral; _angle no longer needs to emulate it')


class TestNullPolicy:
    def test_action_is_exactly_zero(self):
        params = make_params()
        policy = NullPolicy()
        env = make_env()
        agents, medium = env.observe()
        action, _ = policy.act(jax.random.key(0), policy.init(jax.random.key(0), params),
                               to_obs(agents, medium, params), params)
        assert action.shape == (3, MAX_AGENTS)
        np.testing.assert_array_equal(np.asarray(action), 0.)


class TestConstPolicy:
    """Bit-exact equality -- there is no arithmetic here to lose precision in.

    Compared after casting the reference to float32 rather than with a
    tolerance: the only thing separating the two is that the JAX core is f32 and
    the xarray one f64, so ``-0.01`` differs in its last bits. Asserting equality
    *in the port's dtype* pins that nothing else differs, which a tolerance would
    not.
    """

    def test_matches_const_agent(self):
        delta_xy, deposit = (-0.01, 0.005), 0.1
        agent = ConstAgent(delta_xy=delta_xy, deposit=deposit)
        policy = ConstPolicy(delta_xy=delta_xy, deposit=deposit)
        state = policy.init(jax.random.key(0), make_params())

        for action_jx, action_xr, alive in parity_steps(agent, policy, state, 4):
            np.testing.assert_array_equal(action_jx[:, alive],
                                          action_xr.astype(np.float32)[:, alive])

    def test_masks_dead_slots_the_original_does_not(self):
        """Pins the one deliberate divergence from the xarray scripted agents.

        ``ConstAgent`` writes its constant into every slot including the dead
        ones, which then drift off the (0, 0) park. Nothing downstream reads
        those positions -- the env masks deposit, feeding and reward by ``alive``
        -- but AGENTS.md's rule is to mask every per-agent computation, and this
        port does.
        """
        agent = ConstAgent(delta_xy=(-0.01, 0.005), deposit=0.1)
        policy = ConstPolicy(delta_xy=(-0.01, 0.005), deposit=0.1)
        state = policy.init(jax.random.key(0), make_params())

        action_jx, action_xr, alive = next(parity_steps(agent, policy, state, 1))
        assert not alive.all(), 'test is vacuous without dead slots'
        np.testing.assert_array_equal(action_jx[:, ~alive], 0.)
        assert np.any(action_xr[:, ~alive] != 0.)


class TestGradientPolicy:
    """Deterministic at ``noise_scale=0``, so the carried state must match too."""

    def _pair(self):
        agent = GradientAgent(max_agents=MAX_AGENTS, seed=0, **GRADIENT_KWARGS)
        policy = GradientPolicy(scale=GRADIENT_KWARGS['scale'],
                                deposit=GRADIENT_KWARGS['deposit'],
                                inertia=GRADIENT_KWARGS['inertia'],
                                sense_offset=GRADIENT_KWARGS['sense_offset'],
                                noise_scale=0.)
        return agent, policy

    def test_matches_gradient_agent_over_multiple_steps(self):
        agent, policy = self._pair()
        steps = 0
        for action_jx, action_xr, alive in parity_steps(agent, policy, carry_from(agent), 8):
            assert_per_channel_close(action_jx, action_xr, alive)
            steps += 1
        assert steps == 8

    def test_momentum_actually_carries(self):
        """Guards the guard: with inertia 0.95 a stateless port would still pass
        step 0, so the test above is only meaningful if the carry does work."""
        agent, policy = self._pair()
        params = make_params()
        env = make_env()
        agents, medium = env.observe()
        obs = to_obs(agents, medium, params)

        state = carry_from(agent)
        first, next_state = policy.act(jax.random.key(0), state, obs, params)
        # Same observation, advanced carry -> a different action.
        second, _ = policy.act(jax.random.key(0), next_state, obs, params)
        assert not np.allclose(np.asarray(first), np.asarray(second))


class TestPhysarumPolicy:
    """Discrete turns, with the random branch pinned by sharing the draws."""

    def _pair(self):
        agent = PhysarumAgent(max_agents=MAX_AGENTS, seed=0, **PHYSARUM_KWARGS)
        policy = PhysarumPolicy(scale=PHYSARUM_KWARGS['scale'],
                                deposit=PHYSARUM_KWARGS['deposit'],
                                inertia=PHYSARUM_KWARGS['inertia'],
                                sense_offset=PHYSARUM_KWARGS['sense_offset'],
                                noise_scale=0.,
                                turn_angle=float(PHYSARUM_KWARGS['turn_angle']),
                                sense_angle=float(PHYSARUM_KWARGS['sense_angle']),
                                turn_tolerance=PHYSARUM_KWARGS['turn_tolerance'])
        return agent, policy

    def test_matches_physarum_agent_over_multiple_steps(self):
        agent, policy = self._pair()
        steps = 0
        for action_jx, action_xr, alive in parity_steps(agent, policy, carry_from(agent),
                                                        8, share_turns=True):
            assert_per_channel_close(action_jx, action_xr, alive)
            steps += 1
        assert steps == 8

    def test_deposit_mask_matches_exactly(self):
        """The mask is derived without any RNG, so it must agree agent for agent.

        Read off the numpy agent's own attribute rather than recomputed, so this
        compares implementations rather than two copies of one formula.
        """
        agent, policy = self._pair()
        params = make_params()
        env = make_env()
        state = carry_from(agent)
        key = jax.random.key(0)

        for _ in range(4):
            agents, medium = env.observe()
            turn_key = jax.random.split(key)[0]
            agent._rng = FixedTurns(
                np.asarray(jax.random.randint(turn_key, (MAX_AGENTS,), 0, 2)))
            action_xr = agent.forward((agents, medium)).values

            obs = to_obs(agents, medium, params)
            grad = policy.gradient_field(obs.medium[2])
            offset = jnp.stack(polar2xy(policy.sense_offset, state.direction))
            sensed = gather_by_agents(grad, obs.coords, params, offset)
            _, drads = xy2polar(sensed[0], sensed[1])
            decision = policy.choose_turn(turn_key, drads, state)

            np.testing.assert_array_equal(np.asarray(decision.deposit_mask),
                                          agent._deposit_mask)
            _, state = policy.act(key, state, obs, params)
            env.step(DataInitializer.init_action_for(agents, action_xr))

    def test_undetermined_is_not_degenerate(self):
        """Both branches must be exercised, or the parity test proves nothing.

        Step 0 is entirely undetermined (chem starts flat, so every gradient
        clips to zero); by a few steps in, most agents have a decided turn.
        """
        agent, policy = self._pair()
        params = make_params()
        env = make_env()
        state = carry_from(agent)
        key = jax.random.key(0)
        seen_determined = seen_undetermined = False

        for _ in range(5):
            agents, medium = env.observe()
            turn_key = jax.random.split(key)[0]
            agent._rng = FixedTurns(
                np.asarray(jax.random.randint(turn_key, (MAX_AGENTS,), 0, 2)))
            obs = to_obs(agents, medium, params)
            grad = policy.gradient_field(obs.medium[2])
            offset = jnp.stack(polar2xy(policy.sense_offset, state.direction))
            sensed = gather_by_agents(grad, obs.coords, params, offset)
            _, drads = xy2polar(sensed[0], sensed[1])
            undetermined = np.asarray(policy.choose_turn(turn_key, drads, state).undetermined)
            alive = agents.sel(channel='alive').values > 0
            seen_undetermined |= bool(undetermined[alive].any())
            seen_determined |= bool((~undetermined[alive]).any())

            action_xr = agent.forward((agents, medium)).values
            _, state = policy.act(key, state, obs, params)
            env.step(DataInitializer.init_action_for(agents, action_xr))

        assert seen_undetermined and seen_determined


class TestBrownianPolicy:
    """Distributional parity only -- numpy's and JAX's uniform streams differ."""

    def _samples(self, num_draws: int = 400):
        move_scale, deposit_scale = 0.01, 0.1
        agent = BrownianAgent(move_scale=move_scale, deposit_scale=deposit_scale)
        policy = BrownianPolicy(move_scale=move_scale, deposit_scale=deposit_scale)
        params = make_params()
        env = make_env()
        agents, medium = env.observe()
        obs = to_obs(agents, medium, params)
        alive = agents.sel(channel='alive').values > 0
        state = policy.init(jax.random.key(0), params)

        xr = np.stack([agent.forward((agents, medium)).values[:, alive]
                       for _ in range(num_draws)], axis=-1)
        keys = jax.random.split(jax.random.key(0), num_draws)
        jx = np.stack([np.asarray(policy.act(k, state, obs, params)[0])[:, alive]
                       for k in keys], axis=-1)
        return xr.reshape(3, -1), jx.reshape(3, -1), move_scale, deposit_scale

    def test_moments_match(self):
        xr, jx, move_scale, _ = self._samples()
        # Uniform on a symmetric interval: the mean's standard error is
        # s/sqrt(3n); 5x that is a comfortable band that still fails on a
        # genuinely wrong distribution.
        tolerance = 5 * move_scale / np.sqrt(3 * xr.shape[1])
        for row in range(3):
            assert abs(xr[row].mean() - jx[row].mean()) < max(tolerance, 0.05 * jx[row].std())
            np.testing.assert_allclose(jx[row].std(), xr[row].std(), rtol=0.05)

    def test_support_bounds(self):
        xr, jx, move_scale, deposit_scale = self._samples()
        for row in (0, 1):
            assert jx[row].min() >= -move_scale and jx[row].max() <= move_scale
            # The extremes of ~20k uniform draws land within a hair of the bound.
            np.testing.assert_allclose(jx[row].min(), -move_scale, atol=0.02 * move_scale)
            np.testing.assert_allclose(jx[row].max(), move_scale, atol=0.02 * move_scale)
        assert jx[2].min() >= 0. and jx[2].max() <= deposit_scale
        np.testing.assert_allclose(jx[2].max(), deposit_scale, atol=0.02 * deposit_scale)
        # and the original really does share those bounds
        assert xr[0].min() >= -move_scale and xr[2].min() >= 0.

    def test_draws_differ_between_keys(self):
        params = make_params()
        env = make_env()
        agents, medium = env.observe()
        obs = to_obs(agents, medium, params)
        policy = BrownianPolicy()
        state = policy.init(jax.random.key(0), params)
        first, _ = policy.act(jax.random.key(0), state, obs, params)
        second, _ = policy.act(jax.random.key(1), state, obs, params)
        assert not np.allclose(np.asarray(first), np.asarray(second))


def all_policies():
    return [NullPolicy(), ConstPolicy(), BrownianPolicy(),
            GradientPolicy(), PhysarumPolicy()]


def policy_ids():
    return [type(policy).__name__ for policy in all_policies()]


class TestJittable:
    """Everything must survive jit and scan -- that is the reason for the port."""

    @pytest.mark.parametrize('policy', all_policies(), ids=policy_ids())
    def test_act_jits(self, policy):
        params = make_params()
        env = make_env()
        agents, medium = env.observe()
        obs = to_obs(agents, medium, params)
        state = policy.init(jax.random.key(0), params)

        action, _ = jax.jit(policy.act)(jax.random.key(0), state, obs, params)
        assert action.shape == (3, MAX_AGENTS)
        assert jnp.isfinite(action).all()

    @pytest.mark.parametrize('policy', all_policies(), ids=policy_ids())
    def test_scans(self, policy):
        """The carry has to be a real pytree, not hidden attributes."""
        params = make_params()
        env = make_env()
        agents, medium = env.observe()
        obs = to_obs(agents, medium, params)

        @jax.jit
        def rollout(key):
            def one(carry, step_key):
                action, carry = policy.act(step_key, carry, obs, params)
                return carry, action
            init_key, scan_key = jax.random.split(key)
            return jax.lax.scan(one, policy.init(init_key, params),
                                jax.random.split(scan_key, 5))[1]

        actions = rollout(jax.random.key(0))
        assert actions.shape == (5, 3, MAX_AGENTS)
        assert jnp.isfinite(actions).all()

    @pytest.mark.parametrize('policy', all_policies(), ids=policy_ids())
    def test_dead_agents_get_zero_action(self, policy):
        """AGENTS.md: dead slots are parked at (0, 0) and must never act."""
        params = make_params()
        env = make_env()
        agents, medium = env.observe()
        obs = to_obs(agents, medium, params)
        alive = np.asarray(obs.agent[1]) > 0
        assert not alive.all(), 'test is vacuous without dead slots'

        state = policy.init(jax.random.key(0), params)
        action, _ = policy.act(jax.random.key(0), state, obs, params)
        np.testing.assert_array_equal(np.asarray(action)[:, ~alive], 0.)
