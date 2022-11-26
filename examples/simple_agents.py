from matplotlib import pyplot as plt
from tqdm import tqdm

from core.agent import ConstAgent, Agent, RandomAgent
from core.env import Env


def try_agent_action(agent: Agent,
                     field_size=(256, 256),
                     iters=200,
                     show_each=-1,
                     ):
    env = Env(field_size)

    for i in tqdm(range(1, iters+1)):
        obs = env._get_current_obs
        action = agent.forward(obs)

        # env._agent_move_async(action)
        env._agent_move(action)
        env._agent_act_on_medium(action)

        if show_each > 0 and i % show_each == 0:
            print(f'drawing progress at iteration {i}')
            env.plot()
            plt.show()


def try_const_agent(**kwargs):
    agent = ConstAgent(delta_xy=(0.001, 0.0025), deposit=0.1)
    try_agent_action(agent, show_each=10, **kwargs)


def try_random_agent(**kwargs):
    agent = RandomAgent(move_scale=0.01, deposit_scale=0.25)
    try_agent_action(agent, show_each=20, **kwargs)


if __name__ == '__main__':
    try_const_agent()
    # try_random_agent(field_size=(512, 512))
