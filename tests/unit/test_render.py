"""Frame rendering tests: frames must be valid float RGB(A) images."""

import numpy as np
import pytest

from die.agents.base import Agent
from die.registry import AgentId, DynamicsId, make_agent, make_env
from die.sim import Simulation
from die.sim.observers import SimObserver
from die.view.render import FrameSource


class RangeProbe(SimObserver):
    """Records the value range of every rendered frame."""

    def __init__(self):
        self.ranges: list[tuple[float, float]] = []

    def on_step(self, sim, event) -> None:
        self.ranges += [(float(frame.min()), float(frame.max()))
                        for frame in sim.render_frames()]


class OutOfRangeAgent(Agent):
    """Agent whose self-render frame violates the [0, 1] float image range."""

    def forward(self, obs):
        raise NotImplementedError

    def render(self):
        return [np.full((4, 4, 3), 7.5)]

    @property
    def init_params(self):
        return {}


class TestDisplayRange:
    """Frames feed ``imshow``, which requires floats in [0, 1] and warns
    (once per frame -- flooding the progress bar) on anything else.
    """

    @pytest.mark.parametrize('agent_id', [AgentId.physarum, AgentId.gradient])
    def test_rendered_frames_stay_in_unit_range(self, agent_id, seeded):
        field_size = 32
        env = make_env(DynamicsId.st_perlin, field_size, agent_ratio=0.2)
        agent = make_agent(agent_id, field_size, seed=42)
        probe = RangeProbe()
        Simulation(env, agent, observers=[probe], seed=42).run(6)

        assert probe.ranges, 'no frames were rendered'
        assert min(low for low, _ in probe.ranges) >= 0.
        assert max(high for _, high in probe.ranges) <= 1.

    def test_agent_frames_are_clipped_too(self, small_env):
        frames = FrameSource(small_env, OutOfRangeAgent()).frames()
        assert frames[-1].max() == 1.

    def test_imshow_logs_no_clipping_warnings(self, seeded, caplog):
        """The symptom itself: one warning per frame per pane scrolls the
        progress bar off the screen. Drawn under Agg -- no display needed.
        """
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt

        field_size = 32
        env = make_env(DynamicsId.st_perlin, field_size, agent_ratio=0.2)
        agent = make_agent(AgentId.physarum, field_size, seed=42)

        class Shower(SimObserver):
            def on_step(self, sim, event) -> None:
                fig, ax = plt.subplots()
                for frame in sim.render_frames():
                    ax.imshow(frame)
                plt.close(fig)

        with caplog.at_level('WARNING', logger='matplotlib.image'):
            Simulation(env, agent, observers=[Shower()], seed=42).run(6)
        assert caplog.records == []
