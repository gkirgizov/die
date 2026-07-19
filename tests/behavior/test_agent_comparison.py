"""High-level behavioral comparisons between agent strategies.

These encode the project's qualitative claims (README) as seeded,
threshold-calibrated assertions:
- Physarum (stigmergic trail-following) forages food more effectively
  than Brownian random motion.
- The gradient agent climbs chemical gradients.

Thresholds were calibrated empirically (seeds 7/42/123 gave physarum/brownian
food-consumption ratios of ~1.27-1.29 at this field size); the asserted floor
is deliberately conservative.
"""

import numpy as np

from die import AgentId, DynamicsId, Simulation, make_agent, make_env
from die.agents.gradient import GradientAgent
from die.env import Dynamics, Env

FIELD = 32
STEPS = 150
SEED = 42


def food_consumed(agent_id: AgentId, seed: int = SEED) -> float:
    """Total env food eaten over a seeded run (foraging effectiveness)."""
    env = make_env(DynamicsId.st_perlin, FIELD, agent_ratio=0.15)
    agent = make_agent(agent_id, FIELD, seed=seed)
    sim = Simulation(env, agent, seed=seed)
    sim.reset()
    food_before = float(env.medium.sel(channel='env_food').sum())
    sim.run(STEPS, reset=False)
    food_after = float(env.medium.sel(channel='env_food').sum())
    return food_before - food_after


class TestPhysarumVsBrownian:
    def test_physarum_forages_more_than_brownian(self):
        brownian = food_consumed(AgentId.brownian)
        physarum = food_consumed(AgentId.physarum)
        assert brownian > 0
        assert physarum > 1.15 * brownian

    def test_result_holds_across_seeds(self):
        ratios = []
        for seed in (7, 123):
            brownian = food_consumed(AgentId.brownian, seed)
            physarum = food_consumed(AgentId.physarum, seed)
            ratios.append(physarum / brownian)
        assert min(ratios) > 1.1


class TestGradientFollowing:
    def test_gradient_agent_climbs_chem_gradient(self, seeded):
        env = Env((FIELD, FIELD), Dynamics(init_agent_ratio=0.2))
        # Impose a linear chemical ramp increasing along +x
        ramp = np.linspace(0., 1., FIELD)[:, None] * np.ones((FIELD, FIELD))
        env.medium.loc[dict(channel='chem1')] = ramp

        agent = GradientAgent(max_agents=FIELD * FIELD, inertia=0.,
                              noise_scale=0., sense_offset=0.,
                              normalized_grad=True, seed=1)
        action = agent.forward(env.observe())

        alive = env.agents.sel(channel='alive').values > 0
        dx = action.sel(channel='dx').values[alive]
        # essentially all alive agents should move toward higher chem (+x)
        assert (dx > 0).mean() > 0.9
        assert dx.mean() > 0
