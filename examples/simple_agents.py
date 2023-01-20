import logging

from matplotlib import pyplot as plt
from tqdm import tqdm

from core.agent import ConstAgent, Agent, RandomAgent, GradientAgent
from core.base_types import ActType
from core.data_init import WaveSequence
from core.env import Env, Dynamics


def try_agent_action(agent: Agent,
                     field_size=(256, 256),
                     iters=1000,
                     show_each=1,
                     ):
    food_flow = \
        WaveSequence(field_size, dt=0.01).get_flow_operator(scale=0.5, decay=1)
    env = Env(field_size, Dynamics(init_agent_ratio=0.35,
                                   op_food_flow=food_flow,
                                   food_infinite=False,
                                   # food_infinite=True,
                                   ))

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


def try_gradient_agent(field_size, **kwargs):
    agent = GradientAgent(field_size,
                          inertia=0.98, scale=0.1, deposit=5,
                          kind='gaussian_noise', noise_scale=0.001,
                          normalized_grad=True)
    try_agent_action(agent, field_size=field_size, **kwargs)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # field_size = (512, 512)
    # field_size = (256, 256)
    field_size = (196, 196)
    # field_size = (94, 94)
    # field_size = (32, 32)
    # try_const_agent(field_size=field_size)
    # try_random_agent(field_size=field_size)
    try_gradient_agent(field_size=field_size)

