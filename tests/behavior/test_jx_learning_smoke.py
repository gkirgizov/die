"""End-to-end learning smoke tests for the JAX/evosax pipeline.

Mirrors ``test_learning_smoke.py`` (the evotorch twin) so both backends are
held to the same contract. Marked slow: these run the full trainer, including
MLflow tracking and headless rollout recording.
"""

import numpy as np
import pytest

pytest.importorskip('jax')
pytest.importorskip('evosax')

from die.env import Dynamics, Env  # noqa: E402
from die.jx.agent import JaxNCAAgent  # noqa: E402
from die.jx.env import init_state  # noqa: E402
from die.jx.model import init_params  # noqa: E402
from die.jx.rollout import make_episode_return  # noqa: E402
from die.jx.state import EnvConfig  # noqa: E402
from die.sim import Simulation  # noqa: E402
from die.train import HistoryLogger, MlflowTrainLogger, TrainConfig  # noqa: E402
from die.train.evosax_backend import EvosaxBackend  # noqa: E402
from die.train.jax_trainer import JaxEvolutionTrainer  # noqa: E402

FIELD = (24, 24)


def tiny_setup() -> tuple[Env, JaxNCAAgent, EnvConfig]:
    """A 24x24 st-perlin-wide-like world, small enough to evolve in seconds."""
    dynamics = dict(init_agent_ratio=0.15, food_infinite=True,
                    rate_decay_chem=0.025, diffuse_sigma=0.8)
    env = Env(FIELD, Dynamics(**dynamics))
    jx_config = EnvConfig(field_size=FIELD, max_agents=FIELD[0] * FIELD[1], **dynamics)
    agent = JaxNCAAgent.create(FIELD, seed=0, kernel_sizes=(3,), scale=0.01, deposit=2.0)
    return env, agent, jx_config


@pytest.mark.slow
def test_fitness_improves_and_checkpoint_roundtrips(tmp_path):
    env, agent, jx_config = tiny_setup()
    history = HistoryLogger()
    config = TrainConfig(generations=6, episode_steps=20, popsize=8, seed=7,
                         eval_seed_mode='fixed', checkpoint_every=0,
                         record_every=0, out_dir=tmp_path)
    trainer = JaxEvolutionTrainer(env, agent, EvosaxBackend('pgpe'), config,
                                  jx_config=jx_config, loggers=[history])
    best = trainer.train()

    stats = history.stats
    assert len(stats) == 6
    # best-ever is monotone by construction; it must top the initial
    # population mean, i.e. search made progress over a random start
    assert stats[-1].best_ever_fitness >= stats[0].best_fitness
    assert stats[-1].best_ever_fitness > stats[0].mean_fitness

    checkpoint = tmp_path / 'checkpoints' / 'best.msgpack'
    assert checkpoint.exists()
    loaded = JaxNCAAgent.load(checkpoint)
    assert loaded.init_params == best.init_params

    # the reloaded agent must act identically on the same seeded env
    reward_best = Simulation(env, best, seed=7).run(10).total_reward
    reward_loaded = Simulation(env, loaded, seed=7).run(10).total_reward
    assert reward_loaded == pytest.approx(reward_best)


@pytest.mark.slow
def test_trained_agent_beats_random_parameters():
    """The evolved kernel must actually forage better than untrained ones."""
    env, agent, jx_config = tiny_setup()
    episode_steps, seed = 20, 7
    config = TrainConfig(generations=8, episode_steps=episode_steps, popsize=8,
                         seed=seed, eval_seed_mode='fixed', checkpoint_every=0,
                         record_every=0)
    trainer = JaxEvolutionTrainer(env, agent, EvosaxBackend('pgpe'), config,
                                  jx_config=jx_config)
    best = trainer.train()

    import jax

    episode_return = jax.jit(make_episode_return(
        agent.model, jx_config, episode_steps, agent.action_coefs))
    initial_state = init_state(seed, jx_config)

    trained = float(episode_return(best.params, initial_state))
    random_returns = [
        float(episode_return(init_params(agent.model, jx_config, jax.random.key(i)),
                             initial_state))
        for i in range(5)
    ]
    assert trained > float(np.median(random_returns))


@pytest.mark.slow
def test_mlflow_tracking_and_artifacts(tmp_path):
    mlflow = pytest.importorskip('mlflow')

    env, agent, jx_config = tiny_setup()
    tracking_uri = f'sqlite:///{tmp_path / "mlflow.db"}'
    config = TrainConfig(generations=2, episode_steps=10, popsize=4, seed=3,
                         eval_seed_mode='fixed', checkpoint_every=0,
                         record_every=2, record_steps=5, out_dir=tmp_path)
    trainer = JaxEvolutionTrainer(
        env, agent, EvosaxBackend('pgpe'), config, jx_config=jx_config,
        loggers=[MlflowTrainLogger(experiment='die-jx-test', tracking_uri=tracking_uri,
                                   artifact_location=str(tmp_path / 'mlartifacts'))],
        run_context={'dynamics': 'test', 'backend': 'evosax'})
    trainer.train()

    # rollout animation was recorded headlessly during training
    assert (tmp_path / 'rollouts' / 'gen0002.gif').exists()

    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name('die-jx-test')
    assert experiment is not None
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 1
    run = runs[0]

    assert run.data.params['train.popsize'] == '4'
    assert run.data.params['run.backend'] == 'evosax'
    assert run.data.params['searcher.backend'] == 'evosax'
    # the JAX env config is tracked too, so runs are reproducible
    assert run.data.params['jx.diffuse_sigma'] == '0.8'

    fitness_history = client.get_metric_history(run.info.run_id, 'fitness/best')
    assert len(fitness_history) == 2

    artifact_dirs = {artifact.path for artifact in client.list_artifacts(run.info.run_id)}
    assert 'checkpoints' in artifact_dirs
    assert 'rollouts' in artifact_dirs
