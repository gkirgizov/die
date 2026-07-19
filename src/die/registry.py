"""Registries of agents and environment dynamics.

Single source of the string ids and tuned default parameters shared
by the CLI, examples and tests.
"""

from enum import StrEnum
from os import PathLike
from pathlib import Path

from die.agents.base import Agent
from die.agents.gradient import GradientAgent, PhysarumAgent
from die.agents.neural import NeuralAutomataAgent
from die.agents.static import BrownianAgent, ConstAgent
from die.data_init import WaveSequence
from die.env import Dynamics, Env

FieldSize = tuple[int, int]


class AgentId(StrEnum):
    const = 'const'
    brownian = 'brownian'
    gradient = 'gradient'
    physarum = 'physarum'
    neural = 'neural'


class DynamicsId(StrEnum):
    st_perlin = 'st-perlin'
    st_perlin_wide = 'st-perlin-wide'
    dyn_pred = 'dyn-pred'


def normalize_field_size(field_size: int | FieldSize) -> FieldSize:
    if isinstance(field_size, int):
        return (field_size, field_size)
    return tuple(field_size)


def make_agent(agent_id: AgentId | str,
               field_size: int | FieldSize,
               seed: int | None = None,
               **overrides) -> Agent:
    """Constructs an agent by id with the tuned default parameters."""
    field_size = normalize_field_size(field_size)
    max_agents = field_size[0] * field_size[1]

    match AgentId(agent_id):
        case AgentId.const:
            params: dict = dict(delta_xy=(-0.01, 0.005), deposit=0.1)
            return ConstAgent(**{**params, **overrides})
        case AgentId.brownian:
            params = dict(move_scale=0.01, deposit_scale=0.1)
            return BrownianAgent(**{**params, **overrides})
        case AgentId.gradient:
            params = dict(max_agents=max_agents,
                          sense_offset=0.03,
                          inertia=0.95,
                          scale=0.01,
                          deposit=4.5,
                          noise_scale=0.025,
                          normalized_grad=True,
                          seed=seed)
            return GradientAgent(**{**params, **overrides})
        case AgentId.physarum:
            params = dict(max_agents=max_agents,
                          turn_angle=35,
                          sense_angle=120,
                          sense_offset=0.03,
                          turn_tolerance=0.05,
                          inertia=0.,
                          scale=0.0075,
                          deposit=4.5,
                          noise_scale=0.0,
                          normalized_grad=True,
                          seed=seed)
            return PhysarumAgent(**{**params, **overrides})
        case AgentId.neural:
            params = dict(kernel_sizes=[3, 3], scale=0.01, deposit=2.0)
            return NeuralAutomataAgent(**{**params, **overrides})
    raise ValueError(f'Unknown agent id: {agent_id}')


def make_dynamics(dynamics_id: DynamicsId | str,
                  field_size: int | FieldSize,
                  agent_ratio: float = 0.1,
                  **overrides) -> Dynamics:
    """Constructs environment dynamics by id."""
    field_size = normalize_field_size(field_size)

    match DynamicsId(dynamics_id):
        case DynamicsId.st_perlin:
            params: dict = dict(init_agent_ratio=agent_ratio,
                                food_infinite=False)
        case DynamicsId.st_perlin_wide:
            params = dict(init_agent_ratio=agent_ratio,
                          food_infinite=True,
                          rate_decay_chem=0.025,
                          diffuse_sigma=0.8)
        case DynamicsId.dyn_pred:
            wave_flow = WaveSequence(field_size, dt=0.01) \
                .get_flow_operator(scale=0.5, decay=0.5)
            params = dict(init_agent_ratio=agent_ratio,
                          food_infinite=False,
                          op_food_flow=wave_flow)
        case _:
            raise ValueError(f'Unknown dynamics id: {dynamics_id}')
    return Dynamics(**{**params, **overrides})


def make_env(dynamics_id: DynamicsId | str,
             field_size: int | FieldSize,
             agent_ratio: float = 0.1,
             **overrides) -> Env:
    field_size = normalize_field_size(field_size)
    dynamics = make_dynamics(dynamics_id, field_size, agent_ratio, **overrides)
    return Env(field_size, dynamics)


def load_agent(path: str | PathLike) -> Agent:
    """Loads a trained agent from a checkpoint, dispatching on its format.

    Two trainers write checkpoints: the torch/evotorch path (``.pt``) and the
    JAX/evosax path (``.msgpack``). Suffix decides, with a content sniff as a
    fallback for renamed files -- torch checkpoints are zip archives.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == '.msgpack':
        return _load_jax_agent(path)
    if suffix in ('.pt', '.pth'):
        return NeuralAutomataAgent.load(path)

    with open(path, 'rb') as handle:
        magic = handle.read(4)
    if magic[:2] == b'PK':
        return NeuralAutomataAgent.load(path)
    return _load_jax_agent(path)


def _load_jax_agent(path: Path) -> Agent:
    try:
        from die.jx.agent import JaxNCAAgent
    except ImportError as exc:
        raise ImportError(
            f'Loading {path.name} needs the JAX stack: uv sync --extra train') from exc
    return JaxNCAAgent.load(path)
