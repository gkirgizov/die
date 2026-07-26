"""Construction of JAX env configs from the shared registry ids.

Mirrors :func:`die.registry.make_dynamics` so both engines are driven by the
same ``DynamicsId`` presets with the same tuned values.
"""

from __future__ import annotations

from die.jx.state import EnvConfig
from die.registry import DynamicsId, FieldSize, normalize_field_size


def make_jx_config(dynamics_id: DynamicsId | str,
                   field_size: int | FieldSize,
                   agent_ratio: float = 0.1,
                   **overrides) -> EnvConfig:
    """Builds the JAX twin of a registered dynamics preset."""
    field_size = normalize_field_size(field_size)
    params: dict = dict(field_size=field_size,
                        max_agents=field_size[0] * field_size[1],
                        init_agent_ratio=agent_ratio)

    match DynamicsId(dynamics_id):
        case DynamicsId.st_perlin:
            params.update(food_infinite=False)
        case DynamicsId.st_perlin_wide:
            params.update(food_infinite=True, rate_decay_chem=0.025, diffuse_sigma=0.8)
        case DynamicsId.dyn_pred:
            params.update(food_infinite=False, food_flow='wave',
                          wave_scale=0.5, wave_decay=0.5)
        case _:
            raise ValueError(f'Unknown dynamics id: {dynamics_id}')
    return EnvConfig(**{**params, **overrides})
