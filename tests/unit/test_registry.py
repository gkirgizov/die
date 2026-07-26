"""Registry construction tests: every id must build a working object."""

import pytest

from die.agents.base import Agent
from die.env import Dynamics, Env
from die.registry import AgentId, DynamicsId, make_agent, make_dynamics, make_env
from die.sim import Simulation


@pytest.mark.parametrize('agent_id', list(AgentId))
def test_every_agent_constructs(agent_id, small_field):
    agent = make_agent(agent_id, small_field, seed=1)
    assert isinstance(agent, Agent)


@pytest.mark.parametrize('dynamics_id', list(DynamicsId))
def test_every_dynamics_constructs(dynamics_id, small_field):
    dynamics = make_dynamics(dynamics_id, small_field, agent_ratio=0.15)
    assert isinstance(dynamics, Dynamics)
    assert dynamics.init_agent_ratio == 0.15


def test_make_env(small_field):
    env = make_env(DynamicsId.st_perlin, small_field)
    assert isinstance(env, Env)


def test_registry_accepts_string_ids(small_field):
    assert isinstance(make_agent('physarum', small_field), Agent)
    assert isinstance(make_dynamics('st-perlin-wide', small_field), Dynamics)


def test_overrides_are_applied(small_field):
    agent = make_agent(AgentId.brownian, small_field, move_scale=0.5)
    assert agent.init_params['move_scale'] == 0.5


@pytest.mark.parametrize('agent_id', list(AgentId))
def test_every_agent_runs_in_env(agent_id, small_field, seeded):
    """Every registry agent can act in every-day env for a few steps."""
    env = make_env(DynamicsId.st_perlin, small_field, agent_ratio=0.2)
    agent = make_agent(agent_id, small_field, seed=1)
    result = Simulation(env, agent, seed=42).run(3)
    assert result.num_steps == 3


@pytest.mark.parametrize('agent_id', [AgentId.const, AgentId.brownian,
                                      AgentId.gradient, AgentId.physarum])
def test_scripted_agent_json_roundtrip(agent_id, small_field, tmp_path):
    """Non-neural agents serialize their init params to JSON and reconstruct."""
    agent = make_agent(agent_id, small_field, seed=3)
    path = tmp_path / f'{agent_id}.json'
    agent.save(path)
    restored = type(agent).load(path)
    assert restored.init_params == agent.init_params
