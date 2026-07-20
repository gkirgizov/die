"""Running observation normalization.

Necessary rather than optional here: **the medium's channels are unbounded**.
Chemical concentration accumulates past 1 (AGENTS.md flags the same fact for
the renderer, which clips before display), and agent energy stock has no
ceiling either. Feeding raw values to a network is how RL runs diverge, and the
failure looks like a bad learning rate rather than like a scaling problem.

This is also the cheapest of the plasticity interventions surveyed in
arXiv:2411.04832 -- together with LayerNorm in the trunk it covers most of what
that literature recommends before anything exotic.

The statistics live in a pytree and update functionally, so they ride inside the
training state and stay ``vmap``-able along with everything else.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import struct

from die.jx.interface import Obs

#: Row order of :attr:`die.jx.interface.Obs.agent`.
OBS_AGENT_FOOD, OBS_AGENT_ALIVE = 0, 1

#: Guards the first update, where variance is still undefined.
_EPS = 1e-4

#: Axis holding features/channels in each observation component.
_MEDIUM_AXIS, _AGENT_AXIS = -3, -2


def _broadcast_shape(x: jax.Array, feature_axis: int) -> tuple[int, ...]:
    """Shape that lines a ``[F]`` statistic up against ``x``'s feature axis."""
    trailing = x.ndim - (feature_axis % x.ndim) - 1
    return (-1,) + (1,) * trailing


@struct.dataclass
class RunningStats:
    """Streaming mean/variance over a feature axis (Chan's parallel update)."""

    mean: jax.Array
    var: jax.Array
    count: jax.Array

    @classmethod
    def create(cls, num_features: int) -> RunningStats:
        return cls(mean=jnp.zeros(num_features),
                   var=jnp.ones(num_features),
                   count=jnp.full((), _EPS))

    def merge(self, mean: jax.Array, var: jax.Array, count: jax.Array) -> RunningStats:
        """Folds one batch's moments in, without revisiting past samples."""
        delta = mean - self.mean
        total = self.count + count
        m2 = (self.var * self.count
              + var * count
              + jnp.square(delta) * self.count * count / total)
        return RunningStats(mean=self.mean + delta * count / total,
                            var=m2 / total,
                            count=total)

    def normalize(self, x: jax.Array, feature_axis: int, clip: float = 10.) -> jax.Array:
        """Standardizes along ``feature_axis``, clipping outliers.

        Clipping earns its place early in training, when few samples have been
        seen and the variance estimate is still poor enough to blow values up.
        """
        shape = _broadcast_shape(x, feature_axis)
        mean, var = self.mean.reshape(shape), self.var.reshape(shape)
        return jnp.clip((x - mean) / jnp.sqrt(var + _EPS), -clip, clip)


def _moments(x: jax.Array, feature_axis: int) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Mean, variance and sample count, reducing every axis but the features."""
    flat = jnp.moveaxis(x, feature_axis, 0)
    flat = flat.reshape(flat.shape[0], -1)
    return flat.mean(axis=-1), flat.var(axis=-1), jnp.asarray(flat.shape[-1], jnp.float32)


def _masked_moments(x: jax.Array,
                    mask: jax.Array,
                    feature_axis: int) -> tuple[jax.Array, jax.Array, jax.Array]:
    """As :func:`_moments`, but counting only samples the mask selects.

    Dead agent slots are parked at (0, 0) carrying zeros. Averaging over them
    would make the statistics track the *death rate* rather than the quantity
    being normalized -- the same phantom-participation trap AGENTS.md warns
    about for the physics.
    """
    flat = jnp.moveaxis(x, feature_axis, 0)
    flat = flat.reshape(flat.shape[0], -1)
    weights = jnp.broadcast_to(mask.reshape(-1), flat.shape)

    count = jnp.maximum(weights.sum(axis=-1), 1.)
    mean = (weights * flat).sum(axis=-1) / count
    var = (weights * jnp.square(flat - mean[:, None])).sum(axis=-1) / count
    return mean, var, count[0]


@struct.dataclass
class ObsNormalizer:
    """Per-channel statistics for the field and for per-agent proprioception."""

    medium: RunningStats
    agent: RunningStats

    @classmethod
    def create(cls, num_medium_channels: int = 3, num_agent_features: int = 2) -> ObsNormalizer:
        return cls(medium=RunningStats.create(num_medium_channels),
                   agent=RunningStats.create(num_agent_features))

    def update(self, obs: Obs) -> ObsNormalizer:
        """Folds in an observation, batched over any leading axes.

        Agent statistics are gathered over living slots only, using the alive
        row carried in the observation itself.
        """
        alive = obs.agent[..., OBS_AGENT_ALIVE, :]
        return ObsNormalizer(
            medium=self.medium.merge(*_moments(obs.medium, _MEDIUM_AXIS)),
            agent=self.agent.merge(*_masked_moments(obs.agent, alive, _AGENT_AXIS)))

    def apply(self, obs: Obs, clip: float = 10.) -> Obs:
        """Standardizes an observation.

        Two things pass through untouched. The alive row is an indicator, not a
        feature: among living agents it is identically 1, so its within-mask
        variance is exactly zero and normalizing it would divide by ``_EPS`` and
        emit clipped garbage. Cell indices are integer array positions, not
        measurements -- normalizing them would make them useless for gathering.
        """
        agent = self.agent.normalize(obs.agent, _AGENT_AXIS, clip)
        agent = agent.at[..., OBS_AGENT_ALIVE, :].set(obs.agent[..., OBS_AGENT_ALIVE, :])
        return obs.replace(medium=self.medium.normalize(obs.medium, _MEDIUM_AXIS, clip),
                           agent=agent)
