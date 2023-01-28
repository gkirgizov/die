import logging

import matplotlib.pyplot
from matplotlib import pyplot as plt
from tqdm import tqdm

from core.agent import ConstAgent, Agent, RandomAgent, GradientAgent, PhysarumAgent
from core.base_types import ActType
from core.data_init import WaveSequence
from core.env import Env, Dynamics


def try_agent_action(agent: Agent,
                     field_size=(256, 256),
                     iters=1000,
                     show_each=1,
                     ):
    food_flow = \
        WaveSequence(field_size, dt=0.01).get_flow_operator(scale=0.8, decay=1)
    env = Env(field_size, Dynamics(init_agent_ratio=0.15,
                                   # op_food_flow=food_flow,
                                   # food_infinite=False,
                                   food_infinite=True,
                                   apply_sense_mask=False,
                                   diffuse_sigma=.4, rate_decay_chem=0.05,
                                   ))

    def manual_step(action: ActType):
        # env._agent_move_async(action)
        env._agent_move(action)
        env._agent_act_on_medium(action)
        env._agent_feed(action)
        env._agent_lifecycle()

        env._medium_resource_dynamics()
        env._medium_diffuse_decay()

        return env._get_current_obs, 0, False, False, {}

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
    agent = ConstAgent(delta_xy=(-0.01, 0.005), deposit=0.1)
    return agent


def try_random_agent(**kwargs):
    agent = RandomAgent(move_scale=0.01, deposit_scale=0.1)
    return agent


def try_gradient_agent(num_agents, **kwargs):
    agent = GradientAgent(num_agents,
                          inertia=0.98, scale=0.025, deposit=5,
                          kind='gaussian_noise', noise_scale=0.,
                          normalized_grad=True)
    return agent


def try_physarum_agent(num_agents, **kwargs):
    agent = PhysarumAgent(num_agents,
                          turn_angle=35,
                          sense_angle=90,
                          sense_offset=0.01,
                          turn_tolerance=0.05,
                          inertia=0.,
                          scale=0.01,
                          deposit=0.5,
                          noise_scale=0.0,
                          normalized_grad=True)
    return agent


if __name__ == '__main__':
    # setup logging
    logging.basicConfig(level=logging.INFO)
    # disable matplotlib warnings, mabye put that into EnvDrawer
    matplotlib.pyplot.set_loglevel('error')

    # field_size = (512, 512)
    # field_size = (256, 256)
    field_size = (156, 156)
    # field_size = (94, 94)
    # field_size = (32, 32)
    # field_size = (16, 16)
    num_agents = field_size[0] * field_size[1]

    # agent = try_const_agent()
    # agent = try_random_agent()
    # agent = try_gradient_agent(num_agents)
    agent = try_physarum_agent(num_agents)

    try_agent_action(agent, field_size, iters=1000)

