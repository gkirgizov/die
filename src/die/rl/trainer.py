"""Host-side harness around the pure PPO ``train`` function.

The division of labor mirrors :class:`~die.train.jax_trainer.JaxEvolutionTrainer`:
the device does the learning in one jitted call, and this class does everything
that must touch the host -- building the world banks, reporting through the
:class:`~die.train.loggers.TrainLogger` seam, saving the checkpoint, and
recording a replay animation through the ordinary xarray ``Simulation``.

One structural difference from the ES trainer is deliberate: the whole training
run executes as a single ``lax.scan`` before any logging happens, so loggers
receive the full per-iteration history *after the fact* rather than live. That
is the price of keeping ``train`` a pure function of ``(key, hypers)`` -- the
property the meta-learning phase depends on -- and it is the right trade while
runs are minutes long.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import jax
import numpy as np

from die.env import Env
from die.jx.interface import make_init_bank
from die.rl.agent import RLAgent
from die.rl.config import HyperParams, TrainStatic
from die.rl.ppo import make_evaluate, make_train
from die.train.loggers import IterStats

if TYPE_CHECKING:
    from die.train.loggers import TrainLogger


@dataclass
class RLRunConfig:
    """Host-side run bookkeeping (the jittable parts live in TrainStatic)."""

    seed: int | None = None
    out_dir: Path = field(default_factory=lambda: Path('runs'))
    record_steps: int = 100     # 0 disables the final rollout animation
    eval_steps: int = 50        # held-out evaluation horizon
    #: Seeds for the held-out evaluation bank; disjoint from training seeds by
    #: construction so evaluation measures generalization, not memorization.
    eval_bank_offset: int = 10_000
    #: Total units of progress, read by StdoutTrainLogger to size its bar. Set
    #: from ``TrainStatic.num_iterations`` by the trainer; the loggers speak
    #: 'generations' because ES got there first.
    generations: int = 0


class RLTrainer:
    """Trains an :class:`RLAgent` with PPO and reports through TrainLogger."""

    def __init__(self,
                 env: Env,
                 static: TrainStatic,
                 hypers: HyperParams,
                 config: RLRunConfig,
                 loggers: Sequence[TrainLogger] = (),
                 run_context: dict | None = None):
        self.env = env
        self.static = static
        self.hypers = hypers
        self.config = config
        self.loggers = list(loggers)
        self.run_context = run_context or {}
        self.config.generations = static.num_iterations

    def describe(self) -> dict:
        """Flat run description for experiment tracking."""
        import dataclasses

        static = {f'rl.{f.name}': str(getattr(self.static, f.name))
                  for f in dataclasses.fields(self.static)}
        hypers = {}
        for name, value in vars(self.hypers).items():
            if hasattr(value, '__dataclass_fields__') or hasattr(value, '__dict__'):
                hypers.update({f'rl.{name}.{key}': str(val)
                               for key, val in vars(value).items()})
            else:
                hypers[f'rl.{name}'] = str(value)
        run = {f'run.{key}': str(value) for key, value in self.run_context.items()}
        from die.train.jax_trainer import jax_device_summary
        return {**static, **hypers, **run,
                'rl.seed': str(self.config.seed),
                'jx.device': jax_device_summary()}

    def train(self) -> RLAgent:
        static, hypers, config = self.static, self.hypers, self.config
        seed = config.seed if config.seed is not None else 0

        train_bank = make_init_bank(range(seed, seed + static.bank_size), hypers.env)
        eval_seeds = range(config.eval_bank_offset + seed,
                           config.eval_bank_offset + seed + static.bank_size)
        eval_bank = make_init_bank(eval_seeds, hypers.env)

        train_fn = jax.jit(make_train(static, train_bank))
        evaluate = jax.jit(make_evaluate(static, eval_bank), static_argnums=(4, 5))

        for logger in self.loggers:
            logger.on_train_start(self)
        try:
            runner, metrics = train_fn(jax.random.key(seed), hypers)
            jax.block_until_ready(metrics)

            history = jax.tree.map(np.asarray, metrics)
            for i in range(static.num_iterations):
                stats = IterStats(iteration=i, metrics={
                    f'rl/{name}': float(getattr(history, name)[i])
                    for name in ('raw_return', 'total_loss', 'policy_loss',
                                 'value_loss', 'entropy', 'approx_kl',
                                 'clip_fraction', 'mean_alive')})
                for logger in self.loggers:
                    logger.on_iteration(self, stats)

            held_out = float(evaluate(runner.params, runner.normalizer,
                                      jax.random.key(seed + 1), hypers,
                                      config.eval_steps, static.num_envs))
            final = IterStats(iteration=static.num_iterations,
                              metrics={'rl/eval_return': held_out})
            for logger in self.loggers:
                logger.on_iteration(self, final)

            agent = RLAgent(params=runner.params,
                            normalizer=runner.normalizer,
                            features=static.features,
                            num_blocks=static.num_blocks,
                            kernel_size=static.kernel_size,
                            hidden=static.hidden,
                            action_scale=float(hypers.action_scale),
                            deposit_scale=float(hypers.deposit_scale),
                            normalize_obs=static.normalize_obs)

            checkpoint = config.out_dir / 'checkpoints' / f'final{agent.checkpoint_suffix}'
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            agent.save(checkpoint)
            for logger in self.loggers:
                logger.on_checkpoint(self, static.num_iterations, checkpoint)

            if config.record_steps:
                rollout_path = self._record_rollout(agent)
                for logger in self.loggers:
                    logger.on_rollout(self, static.num_iterations, rollout_path)

            for logger in self.loggers:
                logger.on_train_end(self, agent, checkpoint)
            return agent
        finally:
            for logger in self.loggers:
                logger.close()

    def _record_rollout(self, agent: RLAgent) -> Path:
        """Animates the trained policy in the xarray env, headless."""
        from die.sim import Simulation
        from die.view.record import AnimationRecorder

        path = self.config.out_dir / 'rollouts' / 'final.gif'
        recorder = AnimationRecorder(path, fps=10)
        seed = self.config.seed if self.config.seed is not None else 0
        Simulation(self.env, agent, observers=[recorder],
                   seed=seed).run(self.config.record_steps)
        return path
