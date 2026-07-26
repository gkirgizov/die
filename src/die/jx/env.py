"""Pure-JAX environment step, mirroring :meth:`die.env.Env.step`.

Physics parity with the xarray env is the design constraint here; see
``tests/unit/test_jx_env.py`` for the pinned cross-implementation assertions.
Two deliberate divergences are documented inline: scatter-add deposits and
non-terminating episodes.

The reward this module returns is the same scalar it always was. The per-agent
vector behind it -- which is what reinforcement learning needs, and which
``gained.sum()`` collapses away -- is available from :func:`step_with_aux`.
Keeping the scalar signature intact is deliberate: a ``[max_agents]`` reward
would *broadcast* rather than raise through ``Simulation``'s accumulator and
``rollout``'s ``.sum()``, so the two shapes must not share a return type.
"""

from __future__ import annotations

import random

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from die.data_init import DataInitializer
from die.jx.state import (
    ACT_DEPOSIT,
    ACT_DX,
    ACT_DY,
    AGENT_ALIVE,
    AGENT_FOOD,
    AGENT_X,
    AGENT_Y,
    MEDIUM_AGENTS,
    MEDIUM_CHEM,
    MEDIUM_FOOD,
    EnvParams,
    EnvState,
    resolve_diffuse_radius,
)
from die.types import DataChannels


@struct.dataclass
class StepAux:
    """Per-agent quantities that the scalar reward collapses away.

    :param gained: ``[max_agents]`` energy change this step, already masked by
        the pre-step alive state (dead slots contribute exactly 0).
    :param alive: ``[max_agents]`` alive mask *before* the lifecycle stage --
        the set of agents that actually acted, and therefore the set whose
        transitions are real. This is the mask RL losses must reduce over.
    :param died: ``[max_agents]`` 1.0 where an agent was alive at the start of
        the step and is dead at the end. This is the per-agent terminal signal;
        value bootstrapping must be cut here.
    """

    gained: jax.Array
    alive: jax.Array
    died: jax.Array


def init_state(seed: int | None, config: EnvParams) -> EnvState:
    """Builds the initial world state on the host, then moves it to device.

    Initialization is intentionally *not* jitted: it reuses the existing
    :class:`~die.data_init.DataInitializer` pipeline verbatim, so a given seed
    produces the same world as :meth:`die.env.Env.reset` (up to the float32
    cast). Perlin noise is the expensive part and is why the trainer caches
    these states per seed.
    """
    if seed is not None:
        # Field init draws from the global numpy RNG and (perlin) from stdlib
        # `random` -- the same three-headed seeding as the xarray env.
        random.seed(seed)
        np.random.seed(seed)

    medium = DataInitializer(config.field_size, DataChannels.medium) \
        .with_const('env_food', 0.5) \
        .with_food_perlin(threshold=config.food_threshold,
                          octaves=config.food_octaves,
                          scale=config.food_scale) \
        .with_agents(ratio=config.init_agent_ratio) \
        .build(name='medium')
    agents = DataInitializer.agents_from_medium(medium, max_agents=config.max_agents)

    return EnvState(
        medium=jnp.asarray(medium.values, dtype=jnp.float32),
        agents=jnp.asarray(agents.values, dtype=jnp.float32),
        step=jnp.asarray(0, dtype=jnp.int32),
    )


def gaussian_kernel1d(sigma, radius: int) -> jax.Array:
    """1D gaussian kernel matching ``scipy.ndimage.gaussian_filter``.

    skimage's ``gaussian`` delegates to scipy with ``truncate=4.0``; replicating
    the radius rule and normalization exactly is what keeps chem1 within ~1e-6
    of the reference implementation.

    ``radius`` is passed in rather than derived from sigma because it sets the
    kernel's *shape*: deriving it here would make sigma shape-affecting, and so
    impossible to ``vmap`` over. :attr:`~die.jx.state.EnvParams.blur_radius`
    applies scipy's rule when a concrete sigma allows it.

    The weights are computed in float32 (rather than the float64-then-cast the
    host implementation used) so that a traced sigma works. Measured cost: the
    blur moves from 1.2e-7 to 1.8e-7 away from skimage, against a 1e-6 parity
    tolerance.
    """
    x = jnp.arange(-radius, radius + 1, dtype=jnp.float32)
    kernel = jnp.exp(-(x ** 2) / (2.0 * sigma ** 2))
    return kernel / kernel.sum()


def gaussian_blur_wrap(field: jax.Array, sigma, radius: int | None = None) -> jax.Array:
    """Separable gaussian blur with periodic ('wrap') boundary.

    ``dm_pix.gaussian_blur`` only offers SAME/VALID padding, so the periodic
    case the env needs is hand-rolled: pad with ``mode='wrap'``, then two
    valid-mode 1D convolutions.

    Leaving ``radius`` implicit applies scipy's rule to a concrete sigma; a
    traced sigma requires passing it.
    """
    radius = resolve_diffuse_radius(sigma, radius)
    if radius == 0:
        return field
    kernel = gaussian_kernel1d(sigma, radius)

    padded = jnp.pad(field, radius, mode='wrap')
    # Convolve along each axis in turn (separability).
    blurred = jax.vmap(lambda row: jnp.convolve(row, kernel, mode='valid'))(padded)
    blurred = jax.vmap(lambda col: jnp.convolve(col, kernel, mode='valid'),
                       in_axes=1, out_axes=1)(blurred)
    return blurred


def _wave_field(t: jax.Array, config: EnvParams) -> jax.Array:
    """Closed-form port of :meth:`die.data_init.WaveSequence.__getitem__`.

    Evaluated inside the scan rather than precomputed: it is pure trigonometry,
    so this is exact and costs no device memory.
    """
    xs = jnp.linspace(0.0, 1.0, config.width)
    ys = jnp.linspace(0.0, 1.0, config.height)
    # NB: dim order is reversed in the xarray meshgrid this mirrors.
    grid_y, grid_x = jnp.meshgrid(ys, xs, indexing='xy')
    x = (grid_x - 0.5) * 2
    y = (grid_y - 0.5) * 2

    pi = jnp.pi
    r = jnp.sqrt(x ** 2 + y ** 2)
    rwave = r + jnp.cos(pi * x) + jnp.sin(0.4 * pi * y)
    z_waves = jnp.cos(pi * (rwave + t))

    sx, sy = 3, 3
    z_islands = jnp.sin(pi * x * sx + t) + jnp.cos(pi * y * sy + t)

    mix = 0.25
    return (1 - mix) * z_waves + mix * z_islands


def cells_at(coords: jax.Array, config: EnvParams) -> tuple[jax.Array, jax.Array]:
    """Maps real-valued ``[2, N]`` coords to array indices (xarray 'nearest').

    Split out from :func:`agent_cells` because the scripted agents sense at an
    *offset* from where they stand, so they need the same rounding rule applied
    to coordinates that are not the agents' own -- and a second copy of that
    rule is exactly how the two engines would drift apart. Out-of-range coords
    clip rather than wrap, matching what ``sel(method='nearest')`` does when an
    offset pushes a lookup past the edge of the grid.
    """
    ix = jnp.round(coords[0] * (config.width - 1)).astype(jnp.int32)
    iy = jnp.round(coords[1] * (config.height - 1)).astype(jnp.int32)
    ix = jnp.clip(ix, 0, config.width - 1)
    iy = jnp.clip(iy, 0, config.height - 1)
    return ix, iy


def agent_cells(agents: jax.Array, config: EnvParams) -> tuple[jax.Array, jax.Array]:
    """Maps real-valued agent coords to array indices (xarray 'nearest')."""
    return cells_at(agents[AGENT_X:AGENT_Y + 1], config)


def regrow_food(food: jax.Array, config: EnvParams) -> jax.Array:
    """Logistic regrowth -- the twin of :meth:`die.env.Env._regrow`.

    ``food += rate * food * (1 - food / capacity)``: renewable resource,
    fastest at half capacity and self-limiting there. Deterministic, and that
    is why it belongs here rather than in the noise layer -- the physics core
    stays RNG-free and bit-comparable across both engines, which is the
    invariant that lets any of these results be cross-checked at all.

    It provably cannot revive a cell at exactly zero, since the term is
    proportional to what is already present. That is a property of logistic
    growth rather than a gap, and it is precisely why reappearance is a
    separate, stochastic mechanism (:func:`die.jx.interface.respawn_food`).

    Computed unconditionally rather than skipped when the rate is zero.
    ``regrow_rate`` is a *traced* leaf -- a continuous scalar that only ever
    multiplies, exactly what :class:`~die.jx.state.EnvParams` says belongs on
    that side so it can be swept under ``vmap`` -- and a Python ``if`` on a
    tracer raises ``TracerBoolConversionError`` the moment params arrive as a
    jit argument rather than a closed-over constant. At rate 0 the added term
    is identically zero, so the guard bought a field of elementwise ops next to
    a gaussian blur and cost jit-compatibility.
    """
    return food + config.regrow_rate * food * (1. - food / config.food_capacity)


def step_with_aux(state: EnvState,
                  action: jax.Array,
                  config: EnvParams) -> tuple[EnvState, jax.Array, StepAux]:
    """One world step, also reporting the per-agent quantities.

    ``action`` is ``[3, max_agents]`` (dx, dy, deposit1). Substage order matches
    the xarray env exactly -- each leaves the world valid for the next.
    """
    medium, agents = state.medium, state.agents
    alive = agents[AGENT_ALIVE]

    # --- Motor stage: move, honouring the boundary condition -----------------
    coords = agents[AGENT_X:AGENT_Y + 1] + action[ACT_DX:ACT_DY + 1]
    if config.boundary == 'wrap':
        coords = coords % 1.0          # also handles negative overflow
    elif config.boundary == 'limit':
        coords = jnp.clip(coords, 0.0, 1.0)
    else:
        raise ValueError(f'Unknown boundary condition: {config.boundary}')
    agents = agents.at[AGENT_X:AGENT_Y + 1].set(coords)

    # --- Deposit chemical & lay agents out onto the field --------------------
    ix, iy = agent_cells(agents, config)
    deposit = action[ACT_DEPOSIT] * alive
    # NB: divergence from the xarray env -- numpy fancy-index `+=` keeps only
    # the last write when agents collide on a cell, whereas this accumulates.
    # Scatter-add is the physically intended semantics (see the feeding-
    # collision bug in docs/NOTES.md).
    medium = medium.at[MEDIUM_CHEM, ix, iy].add(deposit)
    agents_layer = jnp.zeros_like(medium[MEDIUM_AGENTS]).at[ix, iy].max(alive)
    medium = medium.at[MEDIUM_AGENTS].set(agents_layer)

    # --- Lifecycle stage: feed from the medium, burn the cost of acting ------
    agent_mask = (agents_layer > 0).astype(medium.dtype)
    consumed_field = config.rate_feed * medium[MEDIUM_FOOD] * agent_mask
    # Gathered for every slot (dead ones are masked out below), mirroring
    # `field_by_agents(only_alive=False)`: co-located agents each read the
    # cell value while the field is deducted only once.
    consumed = consumed_field[ix, iy]
    if not config.food_infinite:
        medium = medium.at[MEDIUM_FOOD].add(-consumed_field)

    w_deposit, w_dist = config.cost_weights
    dist = jnp.sqrt(action[ACT_DX] ** 2 + action[ACT_DY] ** 2)
    burned = w_deposit * jnp.abs(action[ACT_DEPOSIT]) + w_dist * dist
    gained = (consumed - burned) * alive
    agents = agents.at[AGENT_FOOD].add(gained)

    if config.agents_die:
        have_food = agents[AGENT_FOOD] > 1e-4
        agents = agents * have_food     # zeroes all channels of dead slots

    # Alive *after* the lifecycle stage, so `died` marks this step's terminals.
    died = alive * (1.0 - agents[AGENT_ALIVE])

    # --- Medium dynamics stage ----------------------------------------------
    if config.food_flow == 'wave':
        num_ts = len(np.arange(*config.wave_t_bounds, config.wave_dt))
        t = (state.step % num_ts) * config.wave_dt
        flow = _wave_field(t, config)
        # Clipped at zero, mirroring WaveSequence.get_flow_operator: the wave
        # term is a signed cosine mixture, and negative *food* inverts every
        # quantity downstream -- feeding drains the agent and the env's own
        # deduction adds resource back. See that docstring for the measurement.
        medium = medium.at[MEDIUM_FOOD].set(jnp.maximum(
            config.wave_scale * flow + (1 - config.wave_decay) * medium[MEDIUM_FOOD],
            0.))
    elif config.food_flow != 'none':
        raise ValueError(f'Unknown food flow: {config.food_flow}')

    medium = medium.at[MEDIUM_FOOD].set(regrow_food(medium[MEDIUM_FOOD], config))

    diffused = gaussian_blur_wrap(medium[MEDIUM_CHEM],
                                  config.diffuse_sigma, config.blur_radius)
    medium = medium.at[MEDIUM_CHEM].set(diffused * (1.0 - config.rate_decay_chem))

    # NB: divergence from Simulation -- an extinct population is not terminated
    # early, it simply yields zero reward forever. Fitness-equivalent, and it
    # keeps the episode shape static for `lax.scan`.
    reward = gained.sum()
    aux = StepAux(gained=gained, alive=alive, died=died)
    return EnvState(medium=medium, agents=agents, step=state.step + 1), reward, aux


def step(state: EnvState, action: jax.Array, config: EnvParams) -> tuple[EnvState, jax.Array]:
    """One world step. ``action`` is ``[3, max_agents]`` (dx, dy, deposit1).

    Returns the *scalar* episode reward contribution, as it always has. Use
    :func:`step_with_aux` when the per-agent breakdown is needed.
    """
    next_state, reward, _ = step_with_aux(state, action, config)
    return next_state, reward
