from pathlib import Path

from core.agent import RandomAgent, PhysarumAgent, Agent
from core.env import Env, Dynamics
from core.utils import setup_logging


def record(agent: Agent,
           field_size=(256, 256),
           iters=300,
           save_dir='samples',
           **dynamics_params):
    filepath = Path(save_dir) / agent.__class__.__name__
    print('Starting animation...')

    env = Env(field_size, Dynamics(**dynamics_params))
    env.render_animation(agent.forward, str(filepath), num_frames=iters)
    print('Saved animation to file:', filepath)


if __name__ == '__main__':
    setup_logging()
    field_size = (256, 256)
    max_agents = field_size[0] * field_size[1]

    random_agent = RandomAgent(move_scale=0.01)
    record(random_agent, field_size, init_agent_ratio=0.05)

    physarum_agent = PhysarumAgent(max_agents=max_agents,
                                   scale=0.01,
                                   turn_angle=30,
                                   sense_offset=0.03)
    record(physarum_agent, field_size, init_agent_ratio=0.15)
