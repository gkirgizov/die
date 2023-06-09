import logging
import os

import mlflow
import numpy as np
from IPython.utils.tz import utcnow
from evotorch.algorithms import CMAES, PGPE
from evotorch.logging import StdOutLogger, PandasLogger, MlflowLogger
from evotorch.neuroevolution import NEProblem
from evotorch.neuroevolution.net import count_parameters
from tqdm import trange

from core.agent.evo import NeuralAutomataAgent
from core.data_init import WaveSequence
from core.env import Env, Dynamics
from core.plotting import InteractivePlotter
from core.utils import setup_logging, make_net


def run_agent(env: Env,
              agent: NeuralAutomataAgent,
              epochs: int = 100,
              epoch_iters: int = 50,
              ) -> NeuralAutomataAgent:

    def run_epoch(agent: NeuralAutomataAgent,
                  iters=epoch_iters,
                  plotter=None) -> float:
        obs = env._get_current_obs
        epoch_reward = 0.

        for i in (pbar := trange(iters)):
            action = agent.forward(obs)
            obs, reward, _, _, stats = env.step(action)
            epoch_reward += reward

            pbar.set_postfix(epoch_reward=np.round(epoch_reward, 3), **stats)
            if plotter:
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

    # searcher = CMAES(
    #     problem,
    #     stdev_init=0.1,
    #     popsize=10,
    #     separable=True,
    # )

    radius_init = 1.5  # (approximate) radius of initial hypersphere that we will sample from
    max_speed = radius_init / 15.  # Rule-of-thumb from the paper
    center_learning_rate = max_speed / 2.
    searcher = PGPE(
        problem,
        popsize=10,  # For now we use a static population size
        radius_init=radius_init,  # The searcher can be initialised directely with an initial radius, rather than stdev
        center_learning_rate=center_learning_rate,
        stdev_learning_rate=0.1,  # stdev learning rate of 0.1 was used across all experiments
        optimizer="clipup",  # Using the ClipUp optimiser
        optimizer_config={
            'max_speed': max_speed,  # with the defined max speed
            'momentum': 0.9,  # and momentum fixed to 0.9
        }
    )

    # Create the MLFlow client and 'run' object for logging into
    client = mlflow.tracking.MlflowClient()
    run = mlflow.start_run()
    _ = MlflowLogger(searcher, client=client, run=run)

    searcher.run(epochs)

    # Get the best solution
    solution_agent = make_net(problem, solution=searcher.status["pop_best"])
    assert isinstance(solution_agent, NeuralAutomataAgent)

    # Save the agent
    models_dir = 'saved_models'
    agent_id = f'{agent.__class__.__name__.lower()}'
    solver_id = f'{searcher.__class__.__name__.lower()}_epochs{epochs}x{epoch_iters}'
    timestamp = utcnow().strftime('%Y%m%d-%H%M%S')
    agent_dir = f'{models_dir}/{agent_id}_{solver_id}'
    os.makedirs(agent_dir, exist_ok=True)
    agent_file = f'{agent_dir}/{timestamp}.pt'

    print(f'Saving the agent to: {agent_file}...')
    solution_agent.save(agent_file)

    # Try running the best solution
    input('Press Anything to run the best model...')
    plotter = InteractivePlotter.get(env, solution_agent)
    env.reset()
    reward = run_epoch(solution_agent, iters=epoch_iters*100, plotter=plotter)
    print(f'Final reward of the best solution: {reward}')

    return solution_agent


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
        'st-perlin': Dynamics(init_agent_ratio=agent_ratio, food_infinite=True),
        'st-perlin-wide': Dynamics(init_agent_ratio=agent_ratio, food_infinite=True,
                                   rate_decay_chem=0.025, diffuse_sigma=.8),
        'dyn-pred': Dynamics(init_agent_ratio=agent_ratio, food_infinite=False, op_food_flow=wave_flow),
    }

    # Setup environment
    env = Env(field_size, dynamics_choice[dynamics_id])
    # Setup agent
    agent = NeuralAutomataAgent(kernel_sizes=[3, 3,],
                                # p_agent_dropout=0.25,
                                scale=0.01,
                                deposit=2.0,
                                )
    # Run the agent-env loop
    run_agent(env, agent, epochs, epoch_iters)


if __name__ == '__main__':
    setup_logging(logging.ERROR)

    run_experiment(field_size=96,
                   epochs=1000,
                   epoch_iters=30,
                   # dynamics_id='dyn-pred',
                   dynamics_id='st-perlin-wide',
                   agent_ratio=0.10,
                   )
