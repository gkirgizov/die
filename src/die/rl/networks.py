"""Policy and value networks: a convolutional field operator with local heads.

The architecture is forced by what the project is about. Information is supposed
to move between locations *only through the medium* -- that is the thesis under
test -- so the trunk is convolutional, and its receptive field is the agent's
sensory horizon rather than an incidental hyperparameter. Global attention would
hand agents a direct non-local channel and make the question vacuous; it belongs
in an ablation, not in the default.

Within that constraint the trunk follows SimBa (arXiv:2410.09754): pre-norm
residual blocks with a 4x expansion, which the 2025-26 literature converges on
as simultaneously the architecture answer and the plasticity answer. The
adaptation from ``Dense`` to ``Conv`` is *ours* -- no published result applies
SimBa to convolutional or NCA policies -- so it is validated against the plain
:class:`~die.jx.model.ConvNCA` baseline rather than assumed.

Modules here take a **single world**. Batching is ``jax.vmap`` in the training
loop, which XLA lowers back into batched convolutions, so nothing is lost.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import linen as nn

from die.jx.interface import Obs
from die.rl.obs import OBS_AGENT_FOOD

#: (dx, dy, deposit1) -- mirrors die.types.DataChannels.actions.
NUM_ACTIONS = 3


class ResidualConvBlock(nn.Module):
    """Pre-norm residual block, SimBa's shape with convolutions.

    ``LayerNorm -> Conv(4F, kxk) -> ReLU -> Conv(F, 1x1) -> + residual``. Only
    the expanding convolution mixes spatially; the projection is pointwise, so
    the block grows the receptive field by ``kernel_size - 1`` and costs far
    less than two spatial convolutions would. This is the transformer-FFN shape
    SimBa is built on, which is why the projection is 1x1.
    """

    features: int
    kernel_size: int = 3
    expansion: int = 4

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        residual = x
        x = nn.LayerNorm()(x)
        x = nn.Conv(self.features * self.expansion,
                    (self.kernel_size, self.kernel_size),
                    padding='CIRCULAR',
                    kernel_init=nn.initializers.he_normal())(x)
        x = nn.relu(x)
        x = nn.Conv(self.features, (1, 1),
                    kernel_init=nn.initializers.he_normal())(x)
        return residual + x


class ConvTrunk(nn.Module):
    """Maps the medium to a feature field: ``[C, W, H] -> [W, H, features]``.

    Circular padding throughout, matching the env's wrap boundary -- a policy
    that saw a hard edge the physics does not have would learn artefacts there.
    """

    features: int = 32
    num_blocks: int = 2
    kernel_size: int = 3

    @nn.compact
    def __call__(self, medium: jax.Array) -> jax.Array:
        # flax convolutions are channels-last; world arrays are channels-first.
        x = jnp.transpose(medium, (1, 2, 0))
        x = nn.Conv(self.features, (self.kernel_size, self.kernel_size),
                    padding='CIRCULAR',
                    kernel_init=nn.initializers.he_normal())(x)
        for _ in range(self.num_blocks):
            x = ResidualConvBlock(self.features, self.kernel_size)(x)
        return nn.LayerNorm()(x)

    @property
    def receptive_field(self) -> int:
        """How far an agent can see, in cells -- a scientific quantity here.

        The stem and each block's spatial convolution each widen it by
        ``kernel_size - 1``; the 1x1 projections do not.
        """
        return 1 + (self.num_blocks + 1) * (self.kernel_size - 1)


class ActorCritic(nn.Module):
    """One shared policy, evaluated at every agent's own cell.

    The trunk runs once per world step and each agent gathers the feature vector
    at its position, so this stays a local policy despite being computed
    globally -- a convolution *is* a shared policy over local receptive fields.
    The heads then add proprioception, which the field cannot supply: an agent's
    own energy is exactly the quantity its survival depends on, and it is not
    written anywhere in the medium.
    """

    features: int = 32
    num_blocks: int = 2
    kernel_size: int = 3
    hidden: int = 64
    init_log_std: float = -0.5

    @nn.compact
    def __call__(self, obs: Obs) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Returns ``(mean, log_std, value)`` for one world.

        ``mean`` and ``log_std`` are both ``[A, 3]`` and ``value`` is ``[A]``.
        The spread is a single state-independent parameter, but it is broadcast
        to the agent axis before returning so that every output carries the same
        leading dimensions -- otherwise ``vmap`` over worlds gives it a world
        axis and no agent axis, and it stops broadcasting against ``mean``.
        """
        field = ConvTrunk(self.features, self.num_blocks, self.kernel_size)(obs.medium)
        gathered = field[obs.cells[0], obs.cells[1]]                    # [A, features]
        proprioception = obs.agent[OBS_AGENT_FOOD][:, None]             # [A, 1]
        x = jnp.concatenate([gathered, proprioception], axis=-1)

        actor = nn.relu(nn.Dense(
            self.hidden, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2.)))(x))
        # Small final gain keeps the initial policy near its mean, the standard
        # guard against destructive first updates.
        mean = nn.Dense(NUM_ACTIONS,
                        kernel_init=nn.initializers.orthogonal(0.01))(actor)
        # State-independent spread, as in most continuous-control PPO.
        log_std = self.param('log_std',
                             nn.initializers.constant(self.init_log_std),
                             (NUM_ACTIONS,))

        critic = nn.relu(nn.Dense(
            self.hidden, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2.)))(x))
        value = nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.))(critic)[..., 0]

        return mean, jnp.broadcast_to(log_std, mean.shape), value

    @property
    def receptive_field(self) -> int:
        return ConvTrunk(self.features, self.num_blocks, self.kernel_size).receptive_field


# --- Squashed-gaussian action distribution -----------------------------------
#
# Actions are bounded: the env expects [-1, 1] and applies its own scaling. A
# tanh-squashed gaussian respects that exactly, which matters more than it
# looks -- Raffin (ICLR 2026 blogpost) shows that mismatched action bounds
# silently decide which algorithms can work at all in massively-parallel sim.
#
# Everything below takes and returns *pre-squash* actions. Rollouts store those
# rather than the squashed ones so that recomputing a log-probability under an
# updated policy never needs `atanh`, which diverges at the bounds exactly where
# a converging policy spends its time.


def _normal_log_prob(raw: jax.Array, mean: jax.Array, log_std: jax.Array) -> jax.Array:
    variance = jnp.exp(2. * log_std)
    return -0.5 * (jnp.square(raw - mean) / variance + 2. * log_std + jnp.log(2. * jnp.pi))


def _log_one_minus_tanh_sq(raw: jax.Array) -> jax.Array:
    """``log(1 - tanh(x)^2)`` without the cancellation of the naive form."""
    return 2. * (jnp.log(2.) - raw - jax.nn.softplus(-2. * raw))


def sample_raw(key: jax.Array, mean: jax.Array, log_std: jax.Array) -> jax.Array:
    """Draws a pre-squash action."""
    return mean + jnp.exp(log_std) * jax.random.normal(key, mean.shape)


def squash(raw: jax.Array) -> jax.Array:
    """Maps a pre-squash action into the env's ``[-1, 1]`` action space."""
    return jnp.tanh(raw)


def log_prob(raw: jax.Array, mean: jax.Array, log_std: jax.Array) -> jax.Array:
    """Log-density of the *squashed* action, summed over action dimensions."""
    base = _normal_log_prob(raw, mean, log_std).sum(axis=-1)
    return base - _log_one_minus_tanh_sq(raw).sum(axis=-1)


def entropy(log_std: jax.Array) -> jax.Array:
    """Entropy of the pre-squash gaussian.

    The squashed distribution has no closed-form entropy. This is an upper bound
    on it and is used purely as an exploration regularizer, so the bias is
    constant in the parameters that matter and does not distort the objective.
    """
    return (log_std + 0.5 * jnp.log(2. * jnp.pi * jnp.e)).sum(axis=-1)


def to_world_action(squashed: jax.Array,
                    action_scale: jax.Array,
                    deposit_scale: jax.Array) -> jax.Array:
    """``[A, 3]`` policy output -> ``[3, A]`` world action the env consumes."""
    coefficients = jnp.stack([action_scale, action_scale, deposit_scale])
    return squashed.T * coefficients[:, None]
