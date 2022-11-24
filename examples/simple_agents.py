from matplotlib import pyplot as plt

from core.agent import ConstAgent, Agent, RandomAgent
from core.env import Env


def try_agent_action(agent: Agent,
                     field_size=(256, 256),
                     iters=100,
                     show_each=-1,
                     ):
    env = Env(field_size)

    for i in range(1, iters+1):
        obs = env._get_current_obs
        action = agent.forward(obs)

        env._agent_move_async(action)
        env._agent_act_on_medium(action)

        if show_each > 0 and i % show_each == 0:
            print(f'iteration {i}')
            # env.plot()
            # plt.show()


def try_const_agent(**kwargs):
    agent = ConstAgent(delta_xy=(0.05, 0.025), deposit=0.01)
    try_agent_action(agent, show_each=2, **kwargs)


def try_random_agent(**kwargs):
    agent = RandomAgent(move_scale=0.05, deposit_scale=0.10)
    try_agent_action(agent, show_each=2, **kwargs)


if __name__ == '__main__':
    # try_const_agent()
    try_random_agent(field_size=(128, 128))
