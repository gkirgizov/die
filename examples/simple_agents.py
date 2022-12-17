from matplotlib import pyplot as plt
from tqdm import tqdm

from core.agent import ConstAgent, Agent, RandomAgent, GradientAgent
from core.env import Env


def try_agent_action(agent: Agent,
                     field_size=(256, 256),
                     iters=1000,
                     show_each=-1,
                     ):
    env = Env(field_size)

    for i in tqdm(range(1, iters+1)):
        obs = env._get_current_obs
        action = agent.forward(obs)

        # env._agent_move_async(action)
        env._agent_move(action)
        # env._agent_act_on_medium(action)
        # env._medium_diffuse_decay()

        if show_each > 0 and i % show_each == 0:
            num_agents = int(env._get_agent_mask.sum())
            print(f'drawing progress at iteration {i}: '
                  f'num_agents={num_agents}')
            env.plot()
            plt.show()


def try_const_agent(**kwargs):
    agent = ConstAgent(delta_xy=(-0.01, 0.005), deposit=0.1)
    try_agent_action(agent, show_each=20, **kwargs)


def try_random_agent(**kwargs):
    agent = RandomAgent(move_scale=0.01, deposit_scale=0.25)
    try_agent_action(agent, show_each=20, **kwargs)


def try_gradient_agent(**kwargs):
    agent = GradientAgent(scale=0.1, deposit=0.1, kind='gaussian_noise')
    try_agent_action(agent, show_each=25, **kwargs)


if __name__ == '__main__':
    # field_size = (256, 256)
    field_size = (16, 16)
    try_const_agent(field_size=field_size)
    # try_random_agent(field_size=field_size)
    # try_gradient_agent(field_size=field_size)

