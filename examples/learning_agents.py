import logging

import numpy as np
from evotorch.algorithms import CMAES, PGPE
from evotorch.logging import StdOutLogger
from evotorch.neuroevolution import NEProblem
from evotorch.neuroevolution.net import count_parameters
from matplotlib import pyplot as plt
from tqdm import tqdm, trange

from core.agent.evo import NeuralAutomataAgent, ConvolutionModel
from core.data_init import WaveSequence
from core.env import Env, Dynamics
from core.plotting import InteractivePlotter
from core.utils import setup_logging


def run_agent(env: Env,
              agent: NeuralAutomataAgent,
              epochs: int = 100,
              epoch_iters: int = 50,
              ):
    plotter = InteractivePlotter.get(env, agent)

    def run_epoch(agent: NeuralAutomataAgent) -> float:
        obs = env._get_current_obs
        epoch_reward = 0.

        for i in (pbar := trange(epoch_iters)):
            action = agent.forward(obs)
            obs, reward, _, _, stats = env.step(action)
            epoch_reward += reward

            pbar.set_postfix(epoch_reward=np.round(epoch_reward, 3), **stats)
            plotter.draw()
        return epoch_reward

    print(f'Network has {count_parameters(agent.model)} parameters')

    # btw, setting device is described here:
    # https://docs.evotorch.ai/v0.4.1/examples/notebooks/Brax_Experiments_with_PGPE/
    problem = NEProblem(
        'max',
        network=agent,
        network_eval_func=run_epoch,
        initial_bounds=[-0.5, 0.5],
        # device='cuda:0',
        # num_actors='max',
        num_actors=1,
    )

    searcher = CMAES(
        problem,
        stdev_init=0.01,
        popsize=10,
        separable=True,
    )

    # radius_init = 1.5  # (approximate) radius of initial hypersphere that we will sample from
    # max_speed = radius_init / 15.  # Rule-of-thumb from the paper
    # center_learning_rate = max_speed / 2.
    # searcher = PGPE(
    #     problem,
    #     popsize=10,  # For now we use a static population size
    #     radius_init=radius_init,  # The searcher can be initialised directely with an initial radius, rather than stdev
    #     center_learning_rate=center_learning_rate,
    #     stdev_learning_rate=0.1,  # stdev learning rate of 0.1 was used across all experiments
    #     optimizer="clipup",  # Using the ClipUp optimiser
    #     optimizer_config={
    #         'max_speed': max_speed,  # with the defined max speed
    #         'momentum': 0.9,  # and momentum fixed to 0.9
    #     }
    # )

    StdOutLogger(searcher)
    searcher.run(epochs)


def run_experiment(field_size=156,
                   epochs: int = 100,
                   epoch_iters: int = 50,
                   dynamics_id='st-perlin',
                   agent_ratio=0.15,
                   ):
    setup_logging()
    max_agents = field_size * field_size
    field_size = (field_size, field_size)

    wave_flow = \
        WaveSequence(field_size, dt=0.01).get_flow_operator(scale=0.5, decay=0.5)
    dynamics_choice = {
        'st-perlin': Dynamics(init_agent_ratio=agent_ratio, food_infinite=False),
        'dyn-pred': Dynamics(init_agent_ratio=agent_ratio, food_infinite=False, op_food_flow=wave_flow),
    }

    # Setup environment
    env = Env(field_size, dynamics_choice[dynamics_id])
    # Setup agent
    agent = NeuralAutomataAgent(initial_obs=env._get_current_obs,
                                kernel_sizes=[3, 3, 3],
                                p_agent_dropout=0.25,
                                scale=0.01)
    # Run the agent-env loop
    run_agent(env, agent, epochs, epoch_iters)


if __name__ == '__main__':
    setup_logging(logging.ERROR)

    run_experiment(field_size=128,
                   epochs=100,
                   epoch_iters=5,
                   dynamics_id='dyn-pred',
                   agent_ratio=0.05,
                   )
