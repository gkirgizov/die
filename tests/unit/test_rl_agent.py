"""RLAgent: the bridge from a trained PPO policy back into the xarray env."""

import numpy as np
import pytest

jax = pytest.importorskip('jax')
jnp = jax.numpy

from die.env import Dynamics, Env  # noqa: E402
from die.jx.interface import make_init_bank, reset  # noqa: E402
from die.jx.state import EnvParams  # noqa: E402
from die.registry import load_agent  # noqa: E402
from die.rl.agent import RLAgent  # noqa: E402
from die.rl.networks import ActorCritic  # noqa: E402
from die.rl.obs import ObsNormalizer  # noqa: E402
from die.sim import Simulation  # noqa: E402

FIELD = (16, 16)
MAX_AGENTS = FIELD[0] * FIELD[1]


def make_agent(**overrides) -> RLAgent:
    kwargs = dict(features=8, num_blocks=1, kernel_size=3, hidden=16)
    kwargs.update(overrides)
    agent = RLAgent(**kwargs)

    env_params = EnvParams(field_size=FIELD, max_agents=MAX_AGENTS, init_agent_ratio=0.2)
    obs, _ = reset(jax.random.key(0), make_init_bank([1], env_params), env_params)
    network = ActorCritic(features=kwargs['features'], num_blocks=kwargs['num_blocks'],
                          kernel_size=kwargs['kernel_size'], hidden=kwargs['hidden'])
    agent.params = network.init(jax.random.key(0), obs)
    agent.normalizer = ObsNormalizer.create().update(obs)
    return agent


def make_xarray_env() -> Env:
    return Env(FIELD, Dynamics(init_agent_ratio=0.2))


class TestForward:
    def test_produces_a_valid_action_array(self):
        agent, env = make_agent(), make_xarray_env()
        obs, _ = env.reset(seed=7)
        action = agent.forward(obs)

        assert action.shape == (3, env.agents.shape[-1])
        assert np.all(np.isfinite(action.to_numpy()))

    def test_respects_the_action_scales(self):
        """The policy emits [-1, 1]; the world consumes scaled units."""
        agent = make_agent(action_scale=0.05, deposit_scale=2.)
        obs, _ = make_xarray_env().reset(seed=7)
        action = np.asarray(agent.forward(obs).to_numpy())

        assert np.abs(action[0]).max() <= 0.05
        assert np.abs(action[1]).max() <= 0.05
        assert np.abs(action[2]).max() <= 2.

    def test_is_deterministic(self):
        """Replay acts on the distribution mean, matching training evaluation."""
        agent, env = make_agent(), make_xarray_env()
        obs, _ = env.reset(seed=7)
        np.testing.assert_array_equal(np.asarray(agent.forward(obs).to_numpy()),
                                      np.asarray(agent.forward(obs).to_numpy()))

    def test_runs_inside_a_simulation(self):
        """The whole point of the wrapper: live view, recorder and replay work."""
        agent, env = make_agent(), make_xarray_env()
        result = Simulation(env, agent, seed=7).run(5)
        assert result.num_steps == 5
        assert np.isfinite(result.total_reward)


class TestCheckpoint:
    def test_roundtrips(self, tmp_path):
        agent, env = make_agent(), make_xarray_env()
        obs, _ = env.reset(seed=7)
        before = np.asarray(agent.forward(obs).to_numpy())

        path = tmp_path / f'policy{agent.checkpoint_suffix}'
        agent.save(path)
        loaded = RLAgent.load(path)

        assert loaded.init_params == agent.init_params
        np.testing.assert_allclose(np.asarray(loaded.forward(obs).to_numpy()),
                                   before, rtol=1e-5)

    def test_carries_observation_statistics(self, tmp_path):
        """Without them the network sees inputs on a scale it never trained on."""
        agent = make_agent()
        path = tmp_path / 'policy.msgpack'
        agent.save(path)
        loaded = RLAgent.load(path)

        assert loaded.normalizer is not None
        np.testing.assert_allclose(np.asarray(loaded.normalizer.medium.mean),
                                   np.asarray(agent.normalizer.medium.mean), rtol=1e-6)

    def test_rejects_a_foreign_checkpoint(self, tmp_path):
        jx_agent = pytest.importorskip('die.jx.agent')
        path = tmp_path / 'nca.msgpack'
        jx_agent.JaxNCAAgent.create(FIELD, seed=0).save(path)
        with pytest.raises(ValueError, match='die-rl-ppo'):
            RLAgent.load(path)

    def test_registry_dispatches_on_the_format_marker(self, tmp_path):
        """Both trainers write .msgpack, so the suffix cannot disambiguate."""
        jx_agent = pytest.importorskip('die.jx.agent')

        rl_path = tmp_path / 'ppo.msgpack'
        make_agent().save(rl_path)
        nca_path = tmp_path / 'nca.msgpack'
        jx_agent.JaxNCAAgent.create(FIELD, seed=0).save(nca_path)

        assert isinstance(load_agent(rl_path), RLAgent)
        assert isinstance(load_agent(nca_path), jx_agent.JaxNCAAgent)
