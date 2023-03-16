
from matplotlib import pyplot as plt
from tqdm import tqdm

from core.agent.static import ConstAgent, BrownianAgent
from core.agent.gradient import GradientAgent, PhysarumAgent
from core.agent.base import Agent
from core.base_types import ActType
from core.data_init import WaveSequence
from core.env import Env, Dynamics
from core.utils import setup_logging


def _manual_step(env: Env, action: ActType):
    """Use this manual step instead of Env.step
    for debugging and customizing update cycle."""

    # env._agent_move_async(action)
    env._agent_move(action)
    env._agent_deposit_and_layout(action)

    env._agent_feed(action)
    env._agent_lifecycle()

    env._medium_resource_dynamics()
    env._medium_diffuse_decay()

    return env._get_current_obs, 0, False, False, {}


def run_agent(env: Env, agent: Agent, iters=1000, show_each=1):

    total_reward = 0
    obs = env._get_current_obs

    for i in tqdm(range(1, iters+1)):
        action = agent.forward(obs)
        obs, reward, _, _, stats = env.step(action)
        total_reward += reward

        if show_each > 0 and i % show_each == 0:
            print(f'drawing progress at iteration {i}: '
                  f'total_reward={total_reward}')
            print(stats)
            env.render()
            plt.show()


def try_const_agent(**kwargs):
    return ConstAgent(delta_xy=(-0.01, 0.005), deposit=0.1)


def try_random_agent(**kwargs):
    return BrownianAgent(move_scale=0.01, deposit_scale=0.1)


def try_gradient_agent(num_agents, **kwargs):
    return GradientAgent(num_agents,
                         sense_offset=0.03,
                         inertia=0.95,
                         scale=0.01,
                         deposit=4.5,
                         noise_scale=0.025,
                         normalized_grad=True)


def try_physarum_agent(num_agents, **kwargs):
    return PhysarumAgent(num_agents,
                         turn_angle=35,
                         sense_angle=120,
                         sense_offset=0.03,
                         turn_tolerance=0.05,

                         inertia=0.,
                         scale=0.0075,
                         deposit=4.5,
                         noise_scale=0.0,
                         normalized_grad=True)


def run_experiment(field_size=156,
                   agent_id='rand',
                   dynamics_id='st-perlin',
                   iters=1000,
                   agent_ratio=0.15,
                   ):
    setup_logging()

    max_agents = field_size * field_size
    field_size = (field_size, field_size)

    agents = {
        'const': try_const_agent(),
        'rand': try_random_agent(),
        'grad': try_gradient_agent(max_agents),
        'physarum': try_physarum_agent(max_agents),
    }

    wave_flow = \
        WaveSequence(field_size, dt=0.01).get_flow_operator(scale=0.5, decay=0.5)
    dynamics_choice = {
        'st-perlin': Dynamics(init_agent_ratio=agent_ratio, food_infinite=False),
        'dyn-pred': Dynamics(init_agent_ratio=agent_ratio, food_infinite=False, op_food_flow=wave_flow),
    }

    # Setup environment
    env = Env(field_size, dynamics_choice[dynamics_id])
    # Setup agent
    agent = agents[agent_id]
    # Run the agent-env loop
    run_agent(env, agent, iters=iters)


if __name__ == '__main__':
    run_experiment(field_size=156,
                   agent_id='physarum',
                   dynamics_id='st-perlin',
                   agent_ratio=0.1,
                   )
