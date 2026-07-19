"""The Agent-interface wrapper: serialization, dispatch, and replay."""

import numpy as np
import pytest

jax = pytest.importorskip('jax')

from die.env import Dynamics, Env  # noqa: E402
from die.jx.agent import JaxNCAAgent  # noqa: E402
from die.registry import load_agent  # noqa: E402
from die.sim import Simulation  # noqa: E402

FIELD = (16, 16)


@pytest.fixture
def agent() -> JaxNCAAgent:
    return JaxNCAAgent.create(FIELD, seed=0, kernel_sizes=(3, 3), scale=0.01, deposit=2.0)


@pytest.fixture
def env() -> Env:
    return Env(FIELD, Dynamics(init_agent_ratio=0.2))


class TestForward:
    def test_action_shape_and_channels(self, agent, env):
        env.reset(seed=7)
        action = agent.forward(env.observe())
        assert list(action.coords['channel'].values) == ['dx', 'dy', 'deposit1']
        assert action.shape[1] == env.agents.shape[1]
        assert np.all(np.isfinite(action.values))

    def test_runs_in_simulation(self, agent, env):
        result = Simulation(env, agent, seed=7).run(5)
        assert result.num_steps == 5
        assert np.isfinite(result.total_reward)

    def test_render_returns_image_after_step(self, agent, env):
        env.reset(seed=7)
        assert agent.render() == [None]
        agent.forward(env.observe())
        (image,) = agent.render()
        assert image.shape == (*FIELD, 3)  # channels-last, like the torch agent


class TestSerialization:
    def test_roundtrip_preserves_behavior(self, agent, env, tmp_path):
        path = tmp_path / 'best.msgpack'
        agent.save(path)
        restored = JaxNCAAgent.load(path)

        env.reset(seed=7)
        obs = env.observe()
        np.testing.assert_allclose(restored.forward(obs).values,
                                   agent.forward(obs).values, atol=1e-6)
        assert restored.init_params == agent.init_params

    def test_registry_dispatches_on_suffix(self, agent, tmp_path):
        path = tmp_path / 'best.msgpack'
        agent.save(path)
        assert isinstance(load_agent(path), JaxNCAAgent)

    def test_registry_dispatches_on_content(self, agent, tmp_path):
        """A renamed checkpoint still loads -- suffix is a hint, not a contract."""
        path = tmp_path / 'checkpoint.bin'
        agent.save(path)
        assert isinstance(load_agent(path), JaxNCAAgent)

    def test_torch_checkpoints_still_load(self, tmp_path):
        from die.agents.neural import NeuralAutomataAgent

        torch_agent = NeuralAutomataAgent(kernel_sizes=[3, 3], scale=0.01, deposit=2.0)
        path = tmp_path / 'best.pt'
        torch_agent.save(path)
        assert isinstance(load_agent(path), NeuralAutomataAgent)

    def test_rejects_foreign_checkpoint(self, tmp_path):
        from flax import serialization

        path = tmp_path / 'other.msgpack'
        path.write_bytes(serialization.msgpack_serialize({'format': 'something-else'}))
        with pytest.raises(ValueError, match='die-jx-nca'):
            JaxNCAAgent.load(path)

    def test_with_params_shares_architecture(self, agent):
        clone = agent.with_params(agent.params)
        assert clone.init_params == agent.init_params
        assert clone.model.kernel_sizes == agent.model.kernel_sizes


class TestReplayRecording:
    def test_records_animation_headless(self, agent, env, tmp_path):
        from die.view.record import AnimationRecorder

        path = tmp_path / 'rollout.gif'
        recorder = AnimationRecorder(path, fps=10)
        Simulation(env, agent, observers=[recorder], seed=7).run(3)
        assert path.exists() and path.stat().st_size > 0
