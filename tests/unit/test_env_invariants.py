"""Behavioral invariants of the environment physics."""

import numpy as np
import pytest

from die.agents.static import BrownianAgent, ConstAgent
from die.env import BoundaryCondition, Dynamics, Env, zero_cost


def total_channel(env: Env, channel: str) -> float:
    return float(env.medium.sel(channel=channel).sum())


def alive_coords(env: Env) -> np.ndarray:
    alive = env.agents.sel(channel='alive').values > 0
    return env.agents.sel(channel=['x', 'y']).values[:, alive]


@pytest.fixture
def still_agent() -> ConstAgent:
    return ConstAgent(delta_xy=(0., 0.), deposit=0.)


def run_steps(env: Env, agent, num_steps: int):
    obs = env.observe()
    results = []
    for _ in range(num_steps):
        action = agent.forward(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        results.append((reward, terminated, info))
    return results


class TestFoodConservation:
    def test_food_depletes(self, small_field, seeded, still_agent):
        env = Env(small_field, Dynamics(init_agent_ratio=0.2, food_infinite=False,
                                        op_action_cost=zero_cost))
        food_before = total_channel(env, 'env_food')
        run_steps(env, still_agent, 10)
        food_after = total_channel(env, 'env_food')
        assert food_after < food_before

    def test_food_monotonically_depletes(self, small_field, seeded, still_agent):
        env = Env(small_field, Dynamics(init_agent_ratio=0.2, food_infinite=False,
                                        op_action_cost=zero_cost))
        totals = [total_channel(env, 'env_food')]
        for _ in range(5):
            run_steps(env, still_agent, 1)
            totals.append(total_channel(env, 'env_food'))
        assert all(later < earlier for earlier, later in zip(totals, totals[1:]))

    def test_food_infinite_is_conserved(self, small_field, seeded, still_agent):
        env = Env(small_field, Dynamics(init_agent_ratio=0.2, food_infinite=True,
                                        op_action_cost=zero_cost))
        food_before = total_channel(env, 'env_food')
        run_steps(env, still_agent, 10)
        food_after = total_channel(env, 'env_food')
        assert food_after == pytest.approx(food_before)


class TestBoundary:
    def test_wrap_keeps_coords_in_bounds(self, small_field, seeded):
        env = Env(small_field, Dynamics(init_agent_ratio=0.2,
                                        boundary=BoundaryCondition.wrap))
        agent = ConstAgent(delta_xy=(0.2, 0.2))
        run_steps(env, agent, 10)
        coords = alive_coords(env)
        assert np.all(coords >= 0.)
        assert np.all(coords < 1.)

    def test_wrap_is_modular(self, small_field, seeded):
        env = Env(small_field, Dynamics(init_agent_ratio=0.2,
                                        boundary=BoundaryCondition.wrap))
        start = alive_coords(env).copy()
        agent = ConstAgent(delta_xy=(0.2, 0.2))
        run_steps(env, agent, 1)
        expected = (start + 0.2) % 1.
        np.testing.assert_allclose(alive_coords(env), expected, atol=1e-9)

    def test_limit_clips_at_bounds(self, small_field, seeded):
        env = Env(small_field, Dynamics(init_agent_ratio=0.2,
                                        boundary=BoundaryCondition.limit))
        agent = ConstAgent(delta_xy=(0.2, 0.2))
        run_steps(env, agent, 10)  # 10 * 0.2 = 2.0 >> 1.0: every agent hits the corner
        coords = alive_coords(env)
        np.testing.assert_allclose(coords, 1.)


class TestLifecycle:
    def test_starvation_kills_agents(self, small_field, seeded, still_agent):
        expensive_cost = lambda action: np.full(action.shape[1:], 10.0)  # noqa: E731
        env = Env(small_field, Dynamics(init_agent_ratio=0.2, agents_die=True,
                                        op_action_cost=expensive_cost))
        results = run_steps(env, still_agent, 3)
        _, terminated, info = results[-1]
        assert info['num_agents'] == 0
        assert terminated

    def test_agents_survive_without_death_pressure(self, small_field, seeded, still_agent):
        env = Env(small_field, Dynamics(init_agent_ratio=0.2, agents_die=False))
        num_before = int((env.agents.sel(channel='alive').values > 0).sum())
        results = run_steps(env, still_agent, 5)
        _, terminated, info = results[-1]
        assert info['num_agents'] == num_before
        assert not terminated


class TestReward:
    def test_reward_equals_energy_gain(self, small_field, seeded, still_agent):
        # NB: 'limit' boundary -- under 'wrap', agents sitting exactly on the
        # x=1.0/y=1.0 seam relocate to 0.0 on the first step (1.0 % 1.0 == 0.0),
        # which changes the set of feeding cells mid-step.
        env = Env(small_field, Dynamics(init_agent_ratio=0.2, op_action_cost=zero_cost,
                                        boundary=BoundaryCondition.limit))
        env_food = env.medium.sel(channel='env_food')
        agent_mask = env.medium.sel(channel='agents').values > 0
        expected_gain = float((env.dynamics.rate_feed * env_food * agent_mask).sum())

        (reward, _, _), = run_steps(env, still_agent, 1)
        assert reward == pytest.approx(expected_gain)

    def test_action_cost_reduces_reward(self, small_field, seeded):
        agent = BrownianAgent(move_scale=0.01, deposit_scale=0.1)
        env_free = Env(small_field, Dynamics(init_agent_ratio=0.2, op_action_cost=zero_cost))
        env_costly = Env(small_field, Dynamics(init_agent_ratio=0.2))

        env_free.reset(seed=7)
        np.random.seed(7)
        (reward_free, _, _), = run_steps(env_free, agent, 1)
        env_costly.reset(seed=7)
        np.random.seed(7)
        (reward_costly, _, _), = run_steps(env_costly, agent, 1)
        assert reward_costly < reward_free


class TestChemical:
    def test_chem_decays_without_deposit(self, small_field, seeded, still_agent):
        env = Env(small_field, Dynamics(init_agent_ratio=0.2, op_action_cost=zero_cost))
        env.medium.loc[dict(channel='chem1')] = 1.0
        chem_before = total_channel(env, 'chem1')
        run_steps(env, still_agent, 1)
        chem_after = total_channel(env, 'chem1')
        assert chem_after < chem_before

    def test_deposit_adds_chem(self, small_field, seeded):
        env = Env(small_field, Dynamics(init_agent_ratio=0.2, rate_decay_chem=0.))
        agent = ConstAgent(delta_xy=(0., 0.), deposit=1.0)
        chem_before = total_channel(env, 'chem1')
        run_steps(env, agent, 1)
        chem_after = total_channel(env, 'chem1')
        assert chem_after > chem_before


class TestResetSeeding:
    def test_reset_same_seed_is_reproducible(self, small_field):
        env = Env(small_field, Dynamics(init_agent_ratio=0.2))
        env.reset(seed=7)
        medium_first = env.medium.copy(deep=True)
        agents_first = env.agents.copy(deep=True)

        # Perturb the state, then reset with the same seed (cache-restore path)
        run_steps(env, BrownianAgent(), 3)
        env.reset(seed=7)

        np.testing.assert_array_equal(env.medium.values, medium_first.values)
        np.testing.assert_array_equal(env.agents.values, agents_first.values)

    def test_reset_same_seed_across_instances(self, small_field):
        env1 = Env(small_field, Dynamics(init_agent_ratio=0.2))
        env2 = Env(small_field, Dynamics(init_agent_ratio=0.2))
        env1.reset(seed=11)
        env2.reset(seed=11)
        np.testing.assert_array_equal(env1.medium.values, env2.medium.values)
        np.testing.assert_array_equal(env1.agents.values, env2.agents.values)

    def test_reset_different_seeds_differ(self, small_field):
        env = Env(small_field, Dynamics(init_agent_ratio=0.2))
        env.reset(seed=1)
        medium_one = env.medium.copy(deep=True)
        env.reset(seed=2)
        assert not np.array_equal(env.medium.values, medium_one.values)
