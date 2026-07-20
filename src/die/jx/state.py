"""World state and static configuration of the JAX env.

The split matters for jit: :class:`EnvState` is a pytree that flows *through*
traced code, while :class:`EnvParams` splits its own fields into two kinds.
Fields marked ``pytree_node=False`` land in the treedef -- they are static, are
baked into the compiled function, and changing one triggers a recompile. Every
other field is a traced leaf, so it can be ``vmap``-ed over: that is what makes
a population of environment configurations evaluable in a single call, and it is
the prerequisite for meta-learning over configurations.

The rule for deciding which side a field belongs on: static is anything that
determines an array *shape* or a Python control-flow branch; traced is a
continuous scalar that only ever multiplies or adds inside the computation.
Getting this wrong is silent -- a shape-affecting field left traced fails later,
inside ``jit``, with an opaque tracer error.
"""

from __future__ import annotations

import jax
from flax import struct

# Channel orders mirror die.types.DataChannels -- kept as plain indices here
# because JAX arrays carry no channel labels.
MEDIUM_AGENTS, MEDIUM_FOOD, MEDIUM_CHEM = 0, 1, 2
AGENT_X, AGENT_Y, AGENT_ALIVE, AGENT_FOOD = 0, 1, 2, 3
ACT_DX, ACT_DY, ACT_DEPOSIT = 0, 1, 2

#: Kernel truncation used by scipy/skimage's gaussian filter, which the blur
#: replicates exactly. See :func:`die.jx.env.gaussian_kernel1d`.
DEFAULT_TRUNCATE = 4.0


def diffuse_radius_for(sigma: float, truncate: float = DEFAULT_TRUNCATE) -> int:
    """The kernel radius ``scipy.ndimage.gaussian_filter`` would pick for sigma."""
    return int(truncate * sigma + 0.5)


def resolve_diffuse_radius(sigma, radius: int | None = None) -> int:
    """Static kernel radius, deriving it from sigma only when that is possible.

    The radius sets an array shape, so it can never be traced. When it is left
    implicit this applies scipy's rule -- which needs a concrete sigma, and so
    is exactly the case that breaks under ``vmap``. The error says so, because
    the failure would otherwise surface much later as an opaque tracer error.
    """
    if radius is not None:
        return radius
    try:
        return diffuse_radius_for(float(sigma))
    except (TypeError, jax.errors.ConcretizationTypeError) as exc:
        raise TypeError(
            f'Cannot derive a diffusion kernel radius from a traced sigma '
            f'({sigma!r}): the radius sets an array shape and must stay static. '
            f'Pass it explicitly -- EnvParams(diffuse_radius=...), or '
            f'diffuse_radius_for(sigma) evaluated outside the trace.') from exc


@struct.dataclass
class EnvState:
    """Fixed-shape world state.

    :param medium: ``[3, width, height]`` -- channels (agents, env_food, chem1).
    :param agents: ``[4, max_agents]`` -- channels (x, y, alive, agent_food).
        Dead slots are parked at (0, 0) with ``alive=0``; every per-agent
        computation must mask by ``alive`` (same invariant as the xarray env).
    :param step: step counter, drives time-dependent food flow.
    """

    medium: jax.Array
    agents: jax.Array
    step: jax.Array


class EnvParams(struct.PyTreeNode):
    """Dynamics parameters -- the JAX twin of :class:`die.env.Dynamics`.

    Operators that are callables in ``Dynamics`` become string tags here so the
    static half stays hashable and jit-cacheable.

    Static because they set a shape or pick a branch:
    ``field_size``/``max_agents`` (array shapes), ``boundary``/``food_flow``
    (Python branches), ``food_infinite``/``agents_die`` (Python branches),
    ``init_agent_ratio`` (host-side init only), ``diffuse_radius`` (kernel
    length), and ``wave_dt``/``wave_t_bounds`` -- those last two feed a
    host-side ``np.arange`` whose *length* sets the flow period, so they cannot
    be tracers.

    Traced, and therefore sweepable: every remaining scalar.
    """

    # --- static: shapes and branches -----------------------------------------
    field_size: tuple[int, int] = struct.field(pytree_node=False)
    max_agents: int = struct.field(pytree_node=False)
    boundary: str = struct.field(pytree_node=False, default='wrap')  # 'wrap' | 'limit'
    #: 'none' -> identity flow; 'wave' -> WaveSequence flow (dyn-pred)
    food_flow: str = struct.field(pytree_node=False, default='none')
    food_infinite: bool = struct.field(pytree_node=False, default=False)
    agents_die: bool = struct.field(pytree_node=False, default=False)
    init_agent_ratio: float = struct.field(pytree_node=False, default=0.1)
    #: Food-field initialization, twins of the same fields on
    #: :class:`die.env.Dynamics`. Static despite being continuous scalars:
    #: they are consumed only by the host-side :func:`die.jx.env.init_state`,
    #: never inside traced code, so sweeping them means building a different
    #: init bank rather than vmapping. Same reasoning as ``init_agent_ratio``.
    food_scale: float = struct.field(pytree_node=False, default=1.0)
    food_octaves: int = struct.field(pytree_node=False, default=8)
    food_threshold: float = struct.field(pytree_node=False, default=1.0)
    #: Steps before an episode is truncated (``None`` never truncates). Unlike
    #: every other field this has no twin in :class:`die.env.Dynamics` -- the
    #: xarray engine leaves episode length to ``Simulation.run``. It lives here
    #: because the RL interface auto-resets, and so has to know.
    max_steps: int | None = struct.field(pytree_node=False, default=None)
    #: Enables per-step medium noise (see ``food_noise`` / ``comm_noise``).
    #: A static gate rather than "nonzero scale means on", so that a run without
    #: noise pays nothing: the perturbation would otherwise draw two full
    #: ``[W, H]`` normal fields per world step just to multiply them by zero.
    #: Also has no xarray twin -- noise is applied by the RL interface layer, so
    #: the physics core stays RNG-free and bit-comparable across both engines.
    stochastic_medium: bool = struct.field(pytree_node=False, default=False)
    wave_dt: float = struct.field(pytree_node=False, default=0.01)
    wave_t_bounds: tuple[float, float] = struct.field(pytree_node=False, default=(0.0, 10.0))
    #: Half-length of the diffusion kernel. ``None`` derives it from
    #: ``diffuse_sigma``, which is only possible while sigma is a concrete
    #: float -- see :meth:`blur_radius`.
    diffuse_radius: int | None = struct.field(pytree_node=False, default=None)

    # --- traced: continuous scalars ------------------------------------------
    rate_feed: float = 0.1
    rate_decay_chem: float = 0.1
    diffuse_sigma: float = 0.5
    #: weights of linear_action_cost: (deposit, distance)
    cost_weights: tuple[float, float] = (0.02, 0.01)
    wave_scale: float = 0.5
    wave_decay: float = 0.5
    #: Relative standard deviation of *multiplicative* per-step noise on the
    #: food channel -- a resource whose level fluctuates rather than sitting
    #: still, so a policy cannot memorize one field. Multiplicative because food
    #: is reward and the perturbation must not create it; see
    #: :func:`die.jx.interface.perturb_medium`. Requires ``stochastic_medium``.
    food_noise: float = 0.0
    #: Standard deviation of additive per-step noise on the chemical channel.
    #: This is the signal-to-noise ratio of the *communication medium*: chem is
    #: the only route by which information reaches a distant agent, so raising
    #: this degrades cooperation specifically, rather than perception in general.
    #: Requires ``stochastic_medium``.
    comm_noise: float = 0.0

    def __post_init__(self):
        # Pin the kernel radius at construction, while sigma is still a concrete
        # host float. Because the field is static it then rides through on the
        # treedef, so the radius survives jit and vmap even once sigma has become
        # a tracer -- which is the normal case whenever params are passed *as an
        # argument* to a jitted function rather than closed over.
        if self.diffuse_radius is None:
            try:
                radius = diffuse_radius_for(float(self.diffuse_sigma))
            except (TypeError, jax.errors.ConcretizationTypeError):
                return      # already traced: a radius must have been pinned upstream
            object.__setattr__(self, 'diffuse_radius', radius)

    @property
    def width(self) -> int:
        return self.field_size[0]

    @property
    def height(self) -> int:
        return self.field_size[1]

    @property
    def blur_radius(self) -> int:
        """Static half-length of the diffusion kernel.

        Normally pinned by ``__post_init__`` using scipy's rule, which is what
        keeps the blur bit-comparable with the xarray engine.

        Because the radius is a shape it cannot vary along a ``vmap`` axis, so
        sweeping ``diffuse_sigma`` varies the *weights* within a fixed
        truncation window. Sweeping upward from a narrowly-constructed params
        therefore truncates silently: construct with the widest sigma of the
        intended range, or pass ``diffuse_radius`` explicitly.
        """
        return resolve_diffuse_radius(self.diffuse_sigma, self.diffuse_radius)


#: Historical name. The type was a plain frozen dataclass of dynamics settings
#: before the static/traced split; it is the same set of fields, so presets in
#: ``die.jx.factory`` and every existing call site keep working unchanged.
EnvConfig = EnvParams
