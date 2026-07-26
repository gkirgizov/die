"""End-to-end learning smoke tests: tiny evolution runs must work and improve.

Marked slow: these exercise the full evotorch + trainer + MLflow pipeline.
"""

import pytest

pytest.importorskip('evotorch')

from die.agents.neural import NeuralAutomataAgent  # noqa: E402
from die.env import Dynamics, Env  # noqa: E402
from die.train import EvolutionTrainer, HistoryLogger, MlflowTrainLogger, TrainConfig  # noqa: E402
from die.train.evotorch_backend import EvotorchBackend  # noqa: E402


def tiny_setup() -> tuple[Env, NeuralAutomataAgent]:
    env = Env((24, 24), Dynamics(init_agent_ratio=0.15, food_infinite=True,
                                 rate_decay_chem=0.025, diffuse_sigma=0.8))
    agent = NeuralAutomataAgent(kernel_sizes=[3], scale=0.01, deposit=2.0)
    return env, agent


@pytest.mark.slow
def test_fitness_improves_and_checkpoint_roundtrips(tmp_path):
    env, agent = tiny_setup()
    history = HistoryLogger()
    config = TrainConfig(generations=4, episode_steps=20, popsize=4, seed=7,
                         eval_seed_mode='fixed', checkpoint_every=0,
                         record_every=0, out_dir=tmp_path)
    trainer = EvolutionTrainer(env, agent, EvotorchBackend('pgpe'),
                               config, loggers=[history])
    best = trainer.train()

    stats = history.stats
    assert len(stats) == 4
    # best-ever is monotone by construction; it must top the initial
    # population mean, i.e. search made progress over a random start
    assert stats[-1].best_ever_fitness >= stats[0].best_fitness
    assert stats[-1].best_ever_fitness > stats[0].mean_fitness

    checkpoint = tmp_path / 'checkpoints' / 'best.pt'
    assert checkpoint.exists()
    loaded = NeuralAutomataAgent.load(checkpoint)
    assert loaded.init_params == best.init_params

    # the reloaded agent must act identically on the same seeded env
    from die.sim import Simulation
    reward_best = Simulation(env, best, seed=7).run(10).total_reward
    reward_loaded = Simulation(env, loaded, seed=7).run(10).total_reward
    assert reward_loaded == pytest.approx(reward_best)


@pytest.mark.slow
def test_mlflow_tracking_and_artifacts(tmp_path):
    mlflow = pytest.importorskip('mlflow')

    env, agent = tiny_setup()
    tracking_uri = f'sqlite:///{tmp_path / "mlflow.db"}'
    config = TrainConfig(generations=2, episode_steps=10, popsize=4, seed=3,
                         eval_seed_mode='fixed', checkpoint_every=0,
                         record_every=2, record_steps=5, out_dir=tmp_path)
    trainer = EvolutionTrainer(
        env, agent, EvotorchBackend('pgpe'), config,
        loggers=[MlflowTrainLogger(experiment='die-test', tracking_uri=tracking_uri,
                                   artifact_location=str(tmp_path / 'mlartifacts'))],
        run_context={'dynamics': 'test'})
    trainer.train()

    # rollout animation was recorded headlessly during training
    assert (tmp_path / 'rollouts' / 'gen0002.gif').exists()

    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name('die-test')
    assert experiment is not None
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 1
    run = runs[0]

    assert run.data.params['train.popsize'] == '4'
    assert run.data.params['run.dynamics'] == 'test'
    assert 'searcher.algorithm' in run.data.params

    fitness_history = client.get_metric_history(run.info.run_id, 'fitness/best')
    assert len(fitness_history) == 2

    artifact_dirs = {artifact.path for artifact in client.list_artifacts(run.info.run_id)}
    assert 'checkpoints' in artifact_dirs
    assert 'rollouts' in artifact_dirs
