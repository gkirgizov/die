import logging

import numpy as np
from tqdm import trange

from core.agent.static import BrownianAgent
from core.agent.gradient import PhysarumAgent
from core.agent.base import Agent
from core.env import Env, Dynamics
from core.plotting import InteractivePlotter
from core.utils import setup_logging


def run_minimal(agent: Agent, agent_ratio=0.1, field_size=(256, 256), iters=1000):
    # Setup the environment
    dynamics = Dynamics(init_agent_ratio=agent_ratio)
    env = Env(field_size, dynamics)
    plotter = InteractivePlotter.get(env, agent)

    total_reward = 0
    obs = env._get_current_obs
    for i in (pbar := trange(iters)):
        # Step: action & observation
        action = agent.forward(obs)
        obs, reward, _, _, stats = env.step(action)
        total_reward += reward
        # Visualisation & logging
        pbar.set_postfix(total_reward=np.round(total_reward, 3))
        plotter.draw()


if __name__ == '__main__':
    setup_logging(logging.WARNING)

    random_agent = BrownianAgent(move_scale=0.01)
    run_minimal(random_agent, agent_ratio=0.05, iters=200)

    physarum_agent = PhysarumAgent(max_agents=256*256,
                                   scale=0.006,
                                   turn_angle=30,
                                   sense_offset=0.04)
    run_minimal(physarum_agent, agent_ratio=0.15, iters=200)
