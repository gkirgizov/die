"""Policy/value networks.

The locality tests are the important ones: they pin the project's central
structural claim -- that information reaches an agent only through the medium --
as a property of the architecture rather than an aspiration.
"""

import numpy as np
import pytest

jax = pytest.importorskip('jax')
jnp = jax.numpy

from die.jx.interface import make_init_bank, reset  # noqa: E402
from die.jx.state import EnvParams  # noqa: E402
from die.rl.networks import (  # noqa: E402
    NUM_ACTIONS,
    ActorCritic,
    ConvTrunk,
    entropy,
    log_prob,
    sample_raw,
    squash,
    to_world_action,
)

FIELD = (16, 16)
MAX_AGENTS = FIELD[0] * FIELD[1]


def make_obs():
    params = EnvParams(field_size=FIELD, max_agents=MAX_AGENTS, init_agent_ratio=0.2)
    bank = make_init_bank([1], params)
    return reset(jax.random.key(0), bank, params)[0]


class TestConvTrunk:
    @pytest.mark.parametrize(('num_blocks', 'kernel_size', 'expected'),
                             [(1, 3, 5), (2, 3, 7), (3, 3, 9), (2, 5, 13)])
    def test_receptive_field(self, num_blocks, kernel_size, expected):
        """The agent's sensory horizon -- a scientific quantity, not a detail."""
        assert ConvTrunk(32, num_blocks, kernel_size).receptive_field == expected

    def test_output_shape_is_a_feature_field(self):
        trunk = ConvTrunk(features=8)
        medium = jnp.zeros((3, *FIELD))
        params = trunk.init(jax.random.key(0), medium)
        assert trunk.apply(params, medium).shape == (*FIELD, 8)

    def test_is_translation_equivariant(self):
        """Circular padding, matching the env's wrap boundary.

        A policy that saw an edge the physics does not have would learn
        artefacts along it.
        """
        trunk = ConvTrunk(features=8)
        rng = np.random.default_rng(0)
        medium = jnp.asarray(rng.random((3, *FIELD)), jnp.float32)
        params = trunk.init(jax.random.key(0), medium)

        shift = 5
        rolled_in = trunk.apply(params, jnp.roll(medium, shift, axis=1))
        rolled_out = jnp.roll(trunk.apply(params, medium), shift, axis=0)
        np.testing.assert_allclose(np.asarray(rolled_in), np.asarray(rolled_out), atol=1e-5)

    def test_distant_changes_cannot_reach(self):
        """Beyond the receptive field, information must travel via the medium.

        This is the architectural commitment that makes the pheromone channel
        the only route between distant locations -- and the reason global
        attention is an ablation here rather than the default.
        """
        trunk = ConvTrunk(features=8, num_blocks=2, kernel_size=3)
        radius = trunk.receptive_field // 2      # 3 cells
        rng = np.random.default_rng(0)
        medium = jnp.asarray(rng.random((3, *FIELD)), jnp.float32)
        params = trunk.init(jax.random.key(0), medium)
        baseline = trunk.apply(params, medium)

        probe = (8, 8)
        far = medium.at[:, 0, 0].add(100.)       # toroidal distance 8 > 3
        near = medium.at[:, probe[0] + radius - 1, probe[1]].add(100.)

        far_out = trunk.apply(params, far)[probe]
        near_out = trunk.apply(params, near)[probe]
        np.testing.assert_allclose(np.asarray(far_out),
                                   np.asarray(baseline[probe]), atol=1e-5)
        assert not np.allclose(np.asarray(near_out), np.asarray(baseline[probe]), atol=1e-5)


class TestActorCritic:
    def test_output_shapes(self):
        obs = make_obs()
        net = ActorCritic()
        params = net.init(jax.random.key(0), obs)
        mean, log_std, value = net.apply(params, obs)
        assert mean.shape == (MAX_AGENTS, NUM_ACTIONS)
        assert log_std.shape == (NUM_ACTIONS,)
        assert value.shape == (MAX_AGENTS,)

    def test_starts_near_its_mean(self):
        """Small final-layer gain -- the standard guard against a destructive
        first update."""
        obs = make_obs()
        net = ActorCritic()
        mean, _, _ = net.apply(net.init(jax.random.key(0), obs), obs)
        assert float(jnp.abs(mean).max()) < 0.1

    def test_agents_on_one_cell_share_a_prediction(self):
        """A shared policy gathered at a shared position is the same policy.

        Co-located agents can still act differently -- the distribution is
        sampled per agent -- but their parameters must be identical.
        """
        obs = make_obs()
        obs = obs.replace(cells=jnp.zeros_like(obs.cells))
        net = ActorCritic()
        mean, _, value = net.apply(net.init(jax.random.key(0), obs), obs)
        # proprioception still differs, so allow it to move the output slightly
        assert float(mean.std(axis=0).max()) < 0.5
        assert float(value.std()) < 1.0

    def test_vmaps_over_worlds(self):
        obs = make_obs()
        net = ActorCritic()
        params = net.init(jax.random.key(0), obs)
        batched = jax.tree.map(lambda leaf: jnp.stack([leaf] * 4), obs)
        mean, _, value = jax.jit(jax.vmap(lambda o: net.apply(params, o)))(batched)
        assert mean.shape == (4, MAX_AGENTS, NUM_ACTIONS)
        assert value.shape == (4, MAX_AGENTS)


class TestSquashedGaussian:
    def test_log_prob_matches_change_of_variables(self):
        rng = np.random.default_rng(0)
        mean = jnp.asarray(rng.normal(size=(64, NUM_ACTIONS)), jnp.float32)
        log_std = jnp.asarray(rng.normal(scale=.3, size=NUM_ACTIONS), jnp.float32)
        raw = sample_raw(jax.random.key(0), mean, log_std)

        u, m, s = np.asarray(raw), np.asarray(mean), np.exp(np.asarray(log_std))
        base = (-0.5 * ((u - m) / s) ** 2 - np.log(s) - 0.5 * np.log(2 * np.pi)).sum(-1)
        correction = np.log(1 - np.tanh(u) ** 2 + 1e-12).sum(-1)
        np.testing.assert_allclose(np.asarray(log_prob(raw, mean, log_std)),
                                   base - correction, atol=1e-3)

    def test_log_prob_is_finite_at_the_bounds(self):
        """The naive `log(1 - tanh^2)` underflows to -inf exactly where a
        converging policy spends its time."""
        raw = jnp.asarray([[-15., 0., 15.]])
        value = log_prob(raw, jnp.zeros((1, NUM_ACTIONS)), jnp.zeros(NUM_ACTIONS))
        assert bool(jnp.all(jnp.isfinite(value)))

    def test_squash_respects_the_action_space(self):
        raw = jnp.asarray([[-50., 0., 50.]])
        assert float(jnp.abs(squash(raw)).max()) <= 1.

    def test_world_action_shape_and_scaling(self):
        rng = np.random.default_rng(0)
        squashed = jnp.asarray(np.tanh(rng.normal(size=(MAX_AGENTS, NUM_ACTIONS))), jnp.float32)
        action = to_world_action(squashed, jnp.float32(.1), jnp.float32(2.))
        assert action.shape == (NUM_ACTIONS, MAX_AGENTS)
        assert float(jnp.abs(action[0]).max()) <= .1        # dx
        assert float(jnp.abs(action[1]).max()) <= .1        # dy
        assert float(jnp.abs(action[2]).max()) <= 2.        # deposit

    def test_entropy_grows_with_spread(self):
        wide = entropy(jnp.zeros(NUM_ACTIONS))
        narrow = entropy(jnp.full(NUM_ACTIONS, -2.))
        assert float(wide) > float(narrow)
