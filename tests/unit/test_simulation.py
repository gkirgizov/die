"""Tests of the Simulation runner and observer protocol."""

import numpy as np
import pytest

from die.agents.static import ConstAgent
from die.env import Dynamics, Env
from die.sim import MetricsObserver, SimObserver, Simulation


class SpyObserver(SimObserver):
    def __init__(self):
        self.calls: list[str] = []
        self.events = []
        self.result = None

    def on_run_start(self, sim, num_steps):
        self.calls.append('on_run_start')

    def on_step(self, sim, event):
        self.calls.append('on_step')
        self.events.append(event)

    def on_run_end(self, sim, result):
        self.calls.append('on_run_end')
        self.result = result

    def close(self):
        self.calls.append('close')


@pytest.fixture
def still_agent():
    return ConstAgent(delta_xy=(0., 0.), deposit=0.)


def test_observer_call_order(small_env, still_agent):
    spy = SpyObserver()
    sim = Simulation(small_env, still_agent, observers=[spy], seed=42)
    sim.run(5)
    assert spy.calls == ['on_run_start'] + ['on_step'] * 5 + ['on_run_end', 'close']


def test_run_result_matches_events(small_env, still_agent):
    spy = SpyObserver()
    sim = Simulation(small_env, still_agent, observers=[spy], seed=42)
    result = sim.run(5)
    assert result.num_steps == 5
    assert not result.terminated
    assert result.total_reward == pytest.approx(sum(e.reward for e in spy.events))
    assert spy.result is result


def test_metrics_observer_history(small_env, still_agent):
    metrics = MetricsObserver()
    sim = Simulation(small_env, still_agent, observers=[metrics], seed=42)
    result = sim.run(4)
    assert len(metrics.history) == 4
    assert metrics.total_reward == pytest.approx(result.total_reward)
    assert all('num_agents' in record for record in metrics.history)


def test_early_stop_on_termination(small_field, still_agent):
    expensive_cost = lambda action: np.full(action.shape[1:], 10.0)  # noqa: E731
    env = Env(small_field, Dynamics(init_agent_ratio=0.2, agents_die=True,
                                    op_action_cost=expensive_cost))
    spy = SpyObserver()
    sim = Simulation(env, still_agent, observers=[spy], seed=42)
    result = sim.run(50)
    assert result.terminated
    assert result.num_steps < 50
    assert spy.calls[-2:] == ['on_run_end', 'close']


def test_seeded_runs_are_reproducible(small_field):
    from die.registry import AgentId, make_agent

    def run_once() -> float:
        env = Env(small_field, Dynamics(init_agent_ratio=0.2))
        agent = make_agent(AgentId.physarum, small_field, seed=5)
        return Simulation(env, agent, seed=123).run(10).total_reward

    assert run_once() == pytest.approx(run_once())


def test_render_frames_memoized_per_step(small_env, still_agent):
    call_count = 0

    def counting_frames():
        nonlocal call_count
        call_count += 1
        return [np.zeros((4, 4, 3))]

    class Consumer(SimObserver):
        def on_step(self, sim, event):
            sim.render_frames()
            sim.render_frames()  # second consumer of the same step

    sim = Simulation(small_env, still_agent, observers=[Consumer()],
                     seed=42, frame_source=counting_frames)
    sim.run(3)
    assert call_count == 3


def test_default_frame_source_produces_frames(small_env, still_agent):
    sim = Simulation(small_env, still_agent, seed=42)
    sim.run(1)
    frames = sim.render_frames()
    assert len(frames) > 0
    assert all(isinstance(frame, np.ndarray) for frame in frames)
