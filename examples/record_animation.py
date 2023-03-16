from pathlib import Path

from core.agent.static import BrownianAgent
from core.agent.gradient import PhysarumAgent
from core.agent.base import Agent
from core.env import Env, Dynamics
from core.utils import setup_logging


def record(agent: Agent,
           field_size=(256, 256),
           iters=150,
           save_dir='samples',
           **dynamics_params):
    filename = f'{agent.__class__.__name__}.gif'
    filepath = Path(save_dir) / filename
    print('Starting animation...')

    env = Env(field_size, Dynamics(**dynamics_params))
    env.render_animation(agent.forward, str(filepath), num_frames=iters)
    print('Saved animation to file:', filepath)


if __name__ == '__main__':
    setup_logging()
    field_size = (196, 196)
    max_agents = field_size[0] * field_size[1]

    random_agent = BrownianAgent(move_scale=0.01)
    record(random_agent, field_size, init_agent_ratio=0.05)

    physarum_agent = PhysarumAgent(max_agents=max_agents,
                                   scale=0.007,
                                   turn_angle=30,
                                   sense_offset=0.04)
    record(physarum_agent, field_size, init_agent_ratio=0.15)
