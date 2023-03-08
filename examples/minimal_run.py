import logging

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm, trange

from core.agent import Agent, RandomAgent, PhysarumAgent
from core.env import Env, Dynamics
from core.utils import setup_logging


def run_minimal(agent: Agent, agent_ratio=0.1, field_size=(256, 256), iters=1000):
    # Setup the environment
    dynamics = Dynamics(init_agent_ratio=agent_ratio)
    env = Env(field_size, dynamics)

    total_reward = 0
    obs = env._get_current_obs
    for i in (pbar := trange(iters)):
        # Step: action & observation
        action = agent.forward(obs)
        obs, reward, _, _, stats = env.step(action)
        total_reward += reward
        # Visualisation & logging
        pbar.set_postfix(total_reward=np.round(total_reward, 3))
        env.render(show=True)


if __name__ == '__main__':
    setup_logging(logging.WARNING)

    random_agent = RandomAgent(move_scale=0.01)
    run_minimal(random_agent, agent_ratio=0.05, iters=300)

    physarum_agent = PhysarumAgent(max_agents=256*256,
                                   scale=0.01,
                                   turn_angle=30,
                                   sense_offset=0.03)
    run_minimal(physarum_agent, agent_ratio=0.15, iters=300)
