"""Minimal quick-start: run an agent in the environment with a live view.

This is the README example. For the full-featured entry point use the CLI:
    die run --agent physarum
"""

from die import AgentId, DynamicsId, Simulation, TqdmObserver, make_agent, make_env
from die.view import InteractiveViewObserver


def run_minimal(agent_id: AgentId = AgentId.physarum,
                field_size: int = 256,
                agent_ratio: float = 0.1,
                steps: int = 300,
                seed: int | None = None):
    env = make_env(DynamicsId.st_perlin, field_size, agent_ratio=agent_ratio)
    agent = make_agent(agent_id, field_size, seed=seed)

    sim = Simulation(env, agent, seed=seed,
                     observers=[TqdmObserver(), InteractiveViewObserver()])
    result = sim.run(steps)
    print(f'Total reward: {result.total_reward:.3f}')


if __name__ == '__main__':
    run_minimal(AgentId.brownian, steps=300)
    run_minimal(AgentId.physarum, steps=300)
