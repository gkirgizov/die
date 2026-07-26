"""Flax twin of :class:`die.agents.neural.ConvolutionModel`.

Kept deliberately faithful to the torch original: a stack of bias-free
convolutions with circular padding and a *single* tanh at the very end. There
are no inter-layer activations in the reference model -- do not "improve" this
without recalibrating the behavioral tests.

Dropout is omitted: ``p_agent_dropout`` defaults to 0 everywhere, including the
registry presets, and leaving it out keeps the episode scan RNG-free.
"""

from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp
from flax import linen as nn

from die.jx.env import agent_cells
from die.jx.state import EnvConfig, EnvState


class ConvNCA(nn.Module):
    """Maps the medium to an action field: ``[C_obs, W, H] -> [C_act, W, H]``."""

    kernel_sizes: Sequence[int] = (3, 3)
    obs_channels: int = 3
    act_channels: int = 3

    @nn.compact
    def __call__(self, medium: jax.Array) -> jax.Array:
        # flax convolutions are channels-last; the world arrays are channels-first.
        x = jnp.transpose(medium, (1, 2, 0))
        last = len(self.kernel_sizes) - 1
        for i, kernel_size in enumerate(self.kernel_sizes):
            features = self.act_channels if i == last else self.obs_channels
            x = nn.Conv(features=features,
                        kernel_size=(kernel_size, kernel_size),
                        padding='CIRCULAR',
                        use_bias=False)(x)
        # NB: the single output activation normalizes actions into [-1, 1].
        return jnp.transpose(jnp.tanh(x), (2, 0, 1))


def compute_action(model: ConvNCA,
                   params,
                   state: EnvState,
                   config: EnvConfig,
                   action_coefs: jax.Array) -> jax.Array:
    """Per-agent actions ``[3, max_agents]`` from the current world state.

    Gathers the action field at every agent slot -- including dead ones, which
    the env masks out -- mirroring ``field_by_agents(only_alive=False)``.
    """
    action_field = model.apply(params, state.medium)
    ix, iy = agent_cells(state.agents, config)
    per_agent = action_field[:, ix, iy]
    return per_agent * action_coefs[:, None]


def init_params(model: ConvNCA, config: EnvConfig, key: jax.Array):
    """Initializes model parameters against a dummy medium of the right shape."""
    dummy = jnp.zeros((model.obs_channels, config.width, config.height), jnp.float32)
    return model.init(key, dummy)


def params_from_torch(torch_model, kernel_sizes: Sequence[int]):
    """Converts a torch :class:`ConvolutionModel`'s weights into a flax pytree.

    Used by the parity tests to drive both implementations from identical
    weights. Torch conv kernels are ``(out, in, kh, kw)``; flax wants
    ``(kh, kw, in, out)``.
    """
    import numpy as np

    conv_weights = [module.weight.detach().numpy()
                    for module in torch_model.kernels
                    if hasattr(module, 'weight')]
    assert len(conv_weights) == len(kernel_sizes)
    layers = {f'Conv_{i}': {'kernel': jnp.asarray(np.transpose(w, (2, 3, 1, 0)))}
              for i, w in enumerate(conv_weights)}
    return {'params': layers}
