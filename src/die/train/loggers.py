"""Training progress loggers: stdout and MLflow experiment tracking."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from die.train.searcher import GenStats

if TYPE_CHECKING:
    from die.agents.neural import TorchAgent
    from die.train.trainer import EvolutionTrainer

log = logging.getLogger('die.train')


class TrainLogger:
    """No-op base class for training loggers: override what you need."""

    def on_train_start(self, trainer: EvolutionTrainer) -> None:
        pass

    def on_generation(self, trainer: EvolutionTrainer, stats: GenStats) -> None:
        pass

    def on_checkpoint(self, trainer: EvolutionTrainer, generation: int, path: Path) -> None:
        pass

    def on_rollout(self, trainer: EvolutionTrainer, generation: int, path: Path) -> None:
        pass

    def on_train_end(self, trainer: EvolutionTrainer,
                     best_agent: TorchAgent, checkpoint: Path) -> None:
        pass

    def close(self) -> None:
        """Release resources; idempotent, called in a ``finally`` block."""


class HistoryLogger(TrainLogger):
    """Collects per-generation stats in memory (used by tests and notebooks)."""

    def __init__(self):
        self.stats: list[GenStats] = []

    def on_generation(self, trainer: EvolutionTrainer, stats: GenStats) -> None:
        self.stats.append(stats)


class StdoutTrainLogger(TrainLogger):
    """Progress bar over generations with fitness stats."""

    def __init__(self):
        self._bar = None

    def on_train_start(self, trainer: EvolutionTrainer) -> None:
        from tqdm import tqdm
        self._bar = tqdm(total=trainer.config.generations, desc='generations')

    def on_generation(self, trainer: EvolutionTrainer, stats: GenStats) -> None:
        if self._bar is None:
            return
        self._bar.update(1)
        self._bar.set_postfix(best=round(stats.best_fitness, 3),
                              mean=round(stats.mean_fitness, 3),
                              best_ever=round(stats.best_ever_fitness, 3))

    def on_checkpoint(self, trainer, generation: int, path: Path) -> None:
        log.info('checkpoint saved: %s', path)

    def on_train_end(self, trainer, best_agent, checkpoint: Path) -> None:
        self.close()
        print(f'Training finished; best agent saved to: {checkpoint}')

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None


class MlflowTrainLogger(TrainLogger):
    """Full-fidelity MLflow tracking of an evolution run.

    Logs the complete run configuration as params, per-generation fitness
    metrics, checkpoints and rollout animations as artifacts, and the final
    best model. Requires the ``train`` extra (mlflow).
    """

    def __init__(self,
                 experiment: str = 'die',
                 tracking_uri: str | None = None,
                 artifact_location: str | None = None,
                 run_name: str | None = None):
        self._experiment = experiment
        self._tracking_uri = tracking_uri
        self._artifact_location = artifact_location
        self._run_name = run_name
        self._run_active = False

    def on_train_start(self, trainer: EvolutionTrainer) -> None:
        import mlflow

        if self._tracking_uri:
            mlflow.set_tracking_uri(self._tracking_uri)
        # NB: mlflow >= 3.14 rejects the legacy file:// store; the expected
        # local backend is sqlite. Artifact location is set explicitly on
        # experiment creation so artifacts land next to the tracking db
        # instead of a cwd-relative default.
        client = mlflow.MlflowClient()
        if client.get_experiment_by_name(self._experiment) is None and self._artifact_location:
            client.create_experiment(self._experiment,
                                     artifact_location=self._artifact_location)
        mlflow.set_experiment(self._experiment)
        mlflow.start_run(run_name=self._run_name)
        self._run_active = True
        mlflow.log_params(trainer.describe())

    def on_generation(self, trainer: EvolutionTrainer, stats: GenStats) -> None:
        import mlflow

        metrics = {
            'fitness/best': stats.best_fitness,
            'fitness/mean': stats.mean_fitness,
            'fitness/best_ever': stats.best_ever_fitness,
            **{f'fitness/{key}': value for key, value in stats.extra.items()},
        }
        mlflow.log_metrics(metrics, step=stats.generation)

    def on_checkpoint(self, trainer, generation: int, path: Path) -> None:
        import mlflow
        mlflow.log_artifact(str(path), artifact_path='checkpoints')

    def on_rollout(self, trainer, generation: int, path: Path) -> None:
        import mlflow
        mlflow.log_artifact(str(path), artifact_path='rollouts')

    def on_train_end(self, trainer, best_agent, checkpoint: Path) -> None:
        import mlflow
        mlflow.log_artifact(str(checkpoint), artifact_path='checkpoints')
        self.close()

    def close(self) -> None:
        if self._run_active:
            import mlflow
            mlflow.end_run()
            self._run_active = False
