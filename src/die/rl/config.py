"""The static/traced split for training, mirroring the one in ``EnvParams``.

Two objects, and which one a setting belongs to is the whole point:

:class:`TrainStatic` holds shapes, counts and architecture. Changing any of them
recompiles, and none of them can vary along a ``vmap`` axis.

:class:`HyperParams` holds every traced scalar -- optimization, action scaling,
reward shaping, and the traced half of the env's own physics. Changing these
costs nothing, so a whole population of configurations trains in one call. That
is the property the meta-learning phase is built on, and it has to be designed
in from the start: rejax's per-field split is retrofit-hostile, and purejaxrl,
which bakes its config in as Python values, cannot do it at all.

The rule, restated: traced is a continuous scalar that only multiplies or adds.
Static is anything setting an array shape or a loop trip count.
"""

from __future__ import annotations

import dataclasses
from typing import Literal

from flax import struct

from die.jx.state import EnvParams
from die.rl.reward import RewardParams

#: How the PPO objective factorizes over the agent population.
#:
#: ``per_agent`` treats every alive agent's transition as its own sample -- the
#: point of exposing per-agent rewards at all, and worth ~10^5x more signal per
#: unit of simulation.
#:
#: ``joint`` sums log-probabilities across a world's agents and uses one
#: advantage per world step. That is the same objective evolution optimizes, so
#: it is the honest RL-vs-ES comparison; it is also the only shape a
#: single-agent library like rejax can consume.
Factorization = Literal['per_agent', 'joint']


@struct.dataclass
class HyperParams:
    """Everything traced. Sweep by stacking these and ``vmap``-ing ``train``."""

    # --- optimization --------------------------------------------------------
    #: Deliberately well above the usual PPO default: the reward here is dense
    #: and per-agent-per-step, and an iteration sees ~10^5 agent transitions, so
    #: gradient estimates are far less noisy than the single-agent regime those
    #: defaults come from. 3e-4 does not move the policy at all on the 16x16
    #: world (return -2.7 -> -2.9, approx_kl ~3e-5).
    #:
    #: **This default is very likely wrong and is kept only for continuity.**
    #: The 1e-2 it encodes came from a single-seed sweep, which docs/JOURNAL.md
    #: retracts: across three seeds 1e-3 beats it for `per_agent` too, 4.39 +/-
    #: 1.41 against 3.21 +/- 2.70. Prefer 1e-3, which is what
    #: die.bench.canonical and the rejax cross-check both train at. Changing the
    #: default is deliberately left alone -- it would silently move every
    #: existing calibration that was measured against it.
    #:
    #: A rate belongs to the (algorithm, objective) pair, not to the env; see
    #: the tuned per-configuration values in tests/behavior.
    learning_rate: float = 1e-2
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    gamma: float = 0.99
    gae_lambda: float = 0.95

    # --- action space --------------------------------------------------------
    #: Multiplies the policy's [-1, 1] output into world units. Kept here rather
    #: than in EnvParams so the policy's support and the env's action space are
    #: defined in one place -- mismatching them is a documented way to make an
    #: algorithm silently unable to learn.
    action_scale: float = 0.1
    deposit_scale: float = 1.0

    # --- objective and physics ----------------------------------------------
    reward: RewardParams = RewardParams()
    env: EnvParams = EnvParams(field_size=(64, 64), max_agents=64 * 64)


@dataclasses.dataclass(frozen=True)
class TrainStatic:
    """Shapes, counts and architecture. Each distinct value costs a compilation."""

    # --- rollout shape -------------------------------------------------------
    #: Independent worlds. Prefer scaling this over `num_steps`: agents within a
    #: world are coupled through the medium, so their transitions are correlated
    #: and the effective sample size is far below the nominal count. Scaling
    #: parallel envs (rather than rollout length) is also what arXiv:2506.03404
    #: measures as the better axis for PPO specifically.
    num_envs: int = 8
    num_steps: int = 64
    num_iterations: int = 100

    # --- update shape --------------------------------------------------------
    #: A minibatch element is one (world, timestep) field snapshot; the loss then
    #: reduces over the alive agents inside it. With hundreds of agents per
    #: snapshot the effective batch is enormous, so few epochs are right --
    #: standard small-batch PPO settings overfit each rollout here.
    num_epochs: int = 2
    num_minibatches: int = 4

    # --- architecture --------------------------------------------------------
    features: int = 32
    num_blocks: int = 2
    kernel_size: int = 3
    hidden: int = 64

    # --- behaviour -----------------------------------------------------------
    factorization: Factorization = 'per_agent'
    #: Unbounded channels make this close to mandatory; see die.rl.obs.
    normalize_obs: bool = True
    anneal_lr: bool = True
    #: Distinct initial worlds to draw episodes from. Perlin init is host-side,
    #: so these are built once up front and indexed into.
    bank_size: int = 8

    def __post_init__(self):
        if self.factorization not in ('per_agent', 'joint'):
            raise ValueError(f'Unknown factorization: {self.factorization!r}')
        batch = self.num_envs * self.num_steps
        if batch % self.num_minibatches:
            raise ValueError(
                f'num_envs * num_steps ({batch}) must divide evenly into '
                f'num_minibatches ({self.num_minibatches})')

    @property
    def batch_size(self) -> int:
        """World-snapshots per rollout, before splitting into minibatches."""
        return self.num_envs * self.num_steps

    @property
    def minibatch_size(self) -> int:
        return self.batch_size // self.num_minibatches
