"""Display renormalization: rescaling unbounded channels instead of clipping."""

import numpy as np
import pytest

from die.registry import AgentId, DynamicsId, make_agent, make_env
from die.sim import Simulation
from die.sim.observers import SimObserver
from die.view.render import ChannelNormalizer, EnvRenderer, FrameSource


def heavy_tailed(shape=(32, 32), bulk=1.3, hot=4., n_hot=5, seed=0):
    """A field shaped like the real ones: a bulk topping out near ``bulk``
    plus a few far-out hot pixels (chem1 measures p99 ~1.3 against a max ~4).
    """
    rng = np.random.default_rng(seed)
    field = rng.uniform(0., bulk, shape)
    idx = rng.integers(0, shape[0], (n_hot, 2))
    field[idx[:, 0], idx[:, 1]] = hot
    return field


class TestChannelNormalizer:
    def test_brings_bulk_into_range(self):
        """The point of the feature: a field whose p99 is 4x its bulk should
        come back readable, not saturated to flat white.
        """
        field = heavy_tailed()
        out = ChannelNormalizer(decay=0.)(field, 'chem1')
        assert np.percentile(out, 99) == pytest.approx(1., abs=1e-6)
        assert out.max() > 1.  # hot pixels stay above; clipping catches them

    def test_brightens_a_dim_channel(self):
        """env_food peaks at ~0.26 and is currently near-invisible."""
        field = np.random.default_rng(1).uniform(0., 0.26, (32, 32))
        out = ChannelNormalizer(decay=0.)(field, 'env_food')
        assert out.max() / field.max() > 3.  # spans the display range now
        assert out.max() == pytest.approx(1., abs=0.05)

    def test_scale_is_smoothed_across_calls(self):
        """A step change in level must not snap the brightness in one frame."""
        normalizer = ChannelNormalizer(decay=0.9, min_scale=0.)
        steady = np.full((8, 8), 1.)
        normalizer(steady, 'c')
        jumped = normalizer(np.full((8, 8), 10.), 'c')
        # Without smoothing this would land at 1.0; the EMA lags well past it.
        assert jumped.mean() > 2.

    def test_channels_are_scaled_independently(self):
        normalizer = ChannelNormalizer(decay=0.)
        big = normalizer(np.full((8, 8), 4.), 'chem1')
        small = normalizer(np.full((8, 8), 0.1), 'env_food')
        assert big.max() == pytest.approx(small.max(), rel=1e-6)

    def test_near_empty_field_is_not_amplified_into_noise(self):
        """A dying run leaves a near-zero field; amplifying it by 1000x would
        turn numerical dust into a full-brightness image.
        """
        field = np.full((8, 8), 1e-5)
        out = ChannelNormalizer(decay=0., min_scale=0.05)(field, 'c')
        assert out.max() < 0.01

    def test_all_zero_field_stays_zero(self):
        out = ChannelNormalizer(decay=0.)(np.zeros((8, 8)), 'c')
        assert np.all(out == 0.)
        assert np.all(np.isfinite(out))


class TestRendererNormalization:
    """Only unbounded concentrations may be rescaled -- masks and signed
    fields carry meaning that per-channel scaling would destroy.
    """

    def _agents_pane(self, small_env, normalize):
        renderer = EnvRenderer(tuple(small_env.medium.shape[1:]), normalize=normalize)
        return renderer.render(small_env.medium, small_env.agents)[2]

    def test_alpha_mask_is_left_binary(self, small_env):
        alpha = self._agents_pane(small_env, normalize=True)[..., 3]
        assert set(np.unique(alpha)) <= {0., 1.}

    def test_all_zero_padding_channels_stay_zero(self, small_env):
        pane = self._agents_pane(small_env, normalize=True)
        assert np.all(pane[..., 0] == 0.)
        assert np.all(pane[..., 2] == 0.)

    def test_agent_owned_frames_are_not_rescaled(self, seeded):
        """The gradient pane is signed data shifted so 0.5 means 'no gradient';
        rescaling it would move the neutral point.
        """
        field_size = 32
        env = make_env(DynamicsId.st_perlin, field_size, agent_ratio=0.2)
        agent = make_agent(AgentId.physarum, field_size, seed=42)
        source = FrameSource(env, agent, normalize=True)
        env.reset(seed=42)
        agent.forward(env.observe())
        raw = agent.render()[0]
        assert np.allclose(source.frames()[-1], np.clip(raw, 0., 1.))

    def test_normalize_off_preserves_raw_values(self, small_env):
        renderer = EnvRenderer(tuple(small_env.medium.shape[1:]), normalize=False)
        pane = renderer.render(small_env.medium, small_env.agents)[0]
        expected = np.stack([small_env.medium.sel(channel=c) for c in
                             ('agents', 'env_food', 'chem1')], axis=-1)
        assert np.allclose(pane, expected)


class TestNormalizedRun:
    """End to end, asserted per channel.

    A whole-frame average would be meaningless here: the medium's 'agents'
    channel is a binary occupancy mask sitting at ~16% coverage, which
    normalization leaves alone (its p99 is already 1) and which swamps any
    aggregate.
    """

    def _medium_stats(self, normalize, steps=26):
        env = make_env(DynamicsId.st_perlin, 32, agent_ratio=0.2)
        agent = make_agent(AgentId.physarum, 32, seed=42)
        probe = _MediumProbe()
        Simulation(env, agent, observers=[probe], seed=42,
                   frame_source=FrameSource(env, agent, normalize=normalize)
                   ).run(steps)
        return probe

    def test_saturating_channel_stops_clipping(self, seeded):
        """chem1 climbs past 4.0 and pins several percent of the pane to white."""
        assert self._medium_stats(True).saturated['chem1'] < \
            0.5 * self._medium_stats(False).saturated['chem1']

    def test_dim_channel_gains_detail(self, seeded):
        """env_food never exceeds ~0.5, so it uses a fraction of the range."""
        assert self._medium_stats(True).levels['env_food'] > \
            1.3 * self._medium_stats(False).levels['env_food']

    def test_binary_occupancy_channel_is_left_alone(self, seeded):
        """It is a mask that happens to ride in the medium; rescaling a
        near-empty one by the min_scale floor would blow it out.
        """
        assert self._medium_stats(True).levels['agents'] == 2

    def test_frames_still_in_display_range(self, seeded):
        """Normalization must not reintroduce the imshow warning flood."""
        field_size = 32
        env = make_env(DynamicsId.st_perlin, field_size, agent_ratio=0.2)
        agent = make_agent(AgentId.physarum, field_size, seed=42)
        source = FrameSource(env, agent, normalize=True)
        probe = _RangeProbe()
        Simulation(env, agent, observers=[probe], seed=42,
                   frame_source=source).run(20)
        assert probe.low >= 0. and probe.high <= 1.


class _MediumProbe(SimObserver):
    """Per-channel saturation and distinct-level count of the medium pane,
    sampled on the final step of a run.
    """

    CHANNELS = ('agents', 'env_food', 'chem1')

    def __init__(self):
        self.saturated: dict[str, float] = {}
        self.levels: dict[str, int] = {}

    def on_step(self, sim, event) -> None:
        medium = sim.render_frames()[0]
        for index, name in enumerate(self.CHANNELS):
            channel = medium[..., index]
            self.saturated[name] = float(np.mean(channel >= 1.))
            self.levels[name] = len(np.unique(np.round(channel, 3)))


class _RangeProbe(SimObserver):
    def __init__(self):
        self.low, self.high = np.inf, -np.inf

    def on_step(self, sim, event) -> None:
        for frame in sim.render_frames():
            self.low = min(self.low, float(frame.min()))
            self.high = max(self.high, float(frame.max()))
