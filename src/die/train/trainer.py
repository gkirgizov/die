"""Evolutionary training of neural agents, built on the Simulation runner."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from die.agents.neural import TorchAgent
from die.env import Env
from die.sim import Simulation
from die.train.searcher import GenStats, SearcherBackend

if TYPE_CHECKING:
    from die.train.loggers import TrainLogger


@dataclass
class TrainConfig:
    generations: int = 100
    episode_steps: int = 50
    popsize: int = 10
    seed: int | None = None
    #: 'fixed': every evaluation runs on the same env init (fastest, exploitable);
    #: 'per_generation': env init seed varies per generation -- fair within
    #: a generation, varied across them.
    eval_seed_mode: Literal['fixed', 'per_generation'] = 'per_generation'
    checkpoint_every: int = 25   # 0 disables periodic checkpoints (final one always saved)
    record_every: int = 25       # 0 disables periodic rollout animations
    record_steps: int = 100
    out_dir: Path = field(default_factory=lambda: Path('runs'))


class EvolutionTrainer:
    """Runs an evolutionary search over agent parameters.

    Fitness of a candidate is the total reward of a (seeded) episode run
    through the same :class:`~die.sim.Simulation` machinery used everywhere
    else. Progress is reported to :class:`~die.train.loggers.TrainLogger` s.
    """

    def __init__(self,
                 env: Env,
                 agent: TorchAgent,
                 backend: SearcherBackend,
                 config: TrainConfig,
                 loggers: Sequence[TrainLogger] = (),
                 run_context: dict | None = None):
        self.env = env
        self.agent = agent
        self.backend = backend
        self.config = config
        self.loggers = list(loggers)
        #: Extra run parameters (e.g. dynamics id, field size) for loggers.
        self.run_context = run_context or {}
        self.history: list[GenStats] = []
        self._current_eval_seed: int | None = config.seed

    def describe(self) -> dict:
        """Flat description of the run for experiment tracking."""
        config = {f'train.{key}': str(value)
                  for key, value in asdict(self.config).items()}
        agent_params = {f'agent.{key}': str(value)
                        for key, value in self.agent.init_params.items()}
        backend_params = {f'searcher.{key}': str(value)
                          for key, value in getattr(self.backend, 'params', {}).items()}
        context = {f'run.{key}': str(value) for key, value in self.run_context.items()}
        return {**config, **agent_params, **backend_params, **context}

    def evaluate(self, agent: TorchAgent) -> float:
        """Fitness: total episode reward on the shared env, headless."""
        sim = Simulation(self.env, agent, seed=self._current_eval_seed)
        result = sim.run(self.config.episode_steps)
        return result.total_reward

    def _eval_seed_for_generation(self, generation: int) -> int | None:
        if self.config.seed is None:
            return None
        if self.config.eval_seed_mode == 'fixed':
            return self.config.seed
        return self.config.seed + 10007 * generation

    def train(self) -> TorchAgent:
        config = self.config
        self.backend.setup(self.agent, self.evaluate,
                           popsize=config.popsize, seed=config.seed)
        self.history = []

        for logger in self.loggers:
            logger.on_train_start(self)
        try:
            for generation in range(config.generations):
                self._current_eval_seed = self._eval_seed_for_generation(generation)
                stats = self.backend.run_generation()
                self.history.append(stats)
                for logger in self.loggers:
                    logger.on_generation(self, stats)

                if config.checkpoint_every and (generation + 1) % config.checkpoint_every == 0:
                    path = self._save_checkpoint(f'gen{generation + 1:04d}')
                    for logger in self.loggers:
                        logger.on_checkpoint(self, generation, path)
                if config.record_every and (generation + 1) % config.record_every == 0:
                    path = self._record_rollout(generation + 1)
                    for logger in self.loggers:
                        logger.on_rollout(self, generation, path)

            best = self.backend.best_agent()
            final_path = self._save_checkpoint('best')
            for logger in self.loggers:
                logger.on_train_end(self, best, final_path)
            return best
        finally:
            for logger in self.loggers:
                logger.close()

    def _save_checkpoint(self, tag: str) -> Path:
        path = self.config.out_dir / 'checkpoints' / f'{tag}.pt'
        path.parent.mkdir(parents=True, exist_ok=True)
        self.backend.best_agent().save(path)
        return path

    def _record_rollout(self, generation: int) -> Path:
        """Records an animation of the current best agent, headless."""
        from die.view.record import AnimationRecorder

        path = self.config.out_dir / 'rollouts' / f'gen{generation:04d}.gif'
        recorder = AnimationRecorder(path, fps=10)
        rollout_seed = self.config.seed if self.config.seed is not None else 0
        sim = Simulation(self.env, self.backend.best_agent(),
                         observers=[recorder], seed=rollout_seed)
        sim.run(self.config.record_steps)
        return path
