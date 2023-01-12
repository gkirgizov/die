import logging

from matplotlib import pyplot as plt
from tqdm import tqdm

from core.agent import ConstAgent, Agent, RandomAgent, GradientAgent
from core.base_types import ActType
from core.env import Env, Dynamics


def try_agent_action(agent: Agent,
                     field_size=(256, 256),
                     iters=1000,
                     show_each=1,
                     ):
    env = Env(field_size, Dynamics(init_agent_ratio=0.8,
                                   food_infinite=True))

    def manual_step(action: ActType):
        # env._agent_move_async(action)
        env._agent_move(action)
        env._agents_to_medium()

        env._agent_feed(action)
        env._agent_act_on_medium(action)
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
    try_agent_action(agent, **kwargs)


def try_random_agent(**kwargs):
    agent = RandomAgent(move_scale=0.01, deposit_scale=0.1)
    try_agent_action(agent, **kwargs)


def try_gradient_agent(**kwargs):
    agent = GradientAgent(inertia=0.95, scale=1.0, deposit=0.005,
                          kind='gaussian_noise', noise_scale=0.015)
    try_agent_action(agent, **kwargs)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    field_size = (256, 256)
    # field_size = (128, 128)
    # field_size = (32, 32)
    # try_const_agent(field_size=field_size)
    # try_random_agent(field_size=field_size)
    try_gradient_agent(field_size=field_size)

