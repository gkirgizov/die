"""Jittable JAX ports of the scripted (non-learned) agents.

These exist so the hand-written baselines can be benchmarked on the JAX env in
the same jitted, ``scan``-ed rollout as the learned ones, instead of being
stuck on the xarray engine behind a Python loop. Parity with the originals in
:mod:`die.agents.static` and :mod:`die.agents.gradient` is the design
constraint; ``tests/unit/test_jx_scripted.py`` pins it.

The port forces one structural change. The xarray agents are *stateful objects*
-- :class:`~die.agents.gradient.GradientAgent` carries the momentum term and
the heading it senses ahead along, mutating both inside ``forward`` -- and
hidden mutable state cannot cross a ``jit`` boundary. So the carry becomes an
explicit pytree threaded through :meth:`ScriptedPolicy.act`, which is what makes
these ``scan``-able. Randomness moves the same way: the numpy agents own a
``default_rng``, these take a key per step.

Two behavioral notes, both deliberate:

- **Actions are masked by ``alive`` here, unconditionally.** The xarray family
  is inconsistent about this -- :class:`~die.agents.static.BrownianAgent` masks
  (via ``DataInitializer.action_for``) while ``ConstAgent`` and
  ``GradientAgent`` do not, so their dead slots receive a live action and drift
  away from the (0, 0) park. It changes no physics (the env masks deposit,
  feeding and reward by ``alive`` itself, and dead slots never come back), but
  AGENTS.md's rule is to mask every per-agent computation, and an unmasked
  action makes ``alive`` mean two different things in two engines.
- **Only the *distribution* of ``BrownianPolicy`` matches its original**, since
  numpy's and JAX's uniform streams are unrelated. The bounds and moments are
  the contract; the samples are not.
"""

from __future__ import annotations

from typing import Protocol

import jax
import jax.numpy as jnp
from flax import struct

from die.jx.env import cells_at
from die.jx.interface import Obs
from die.jx.state import MEDIUM_CHEM, MEDIUM_FOOD, EnvParams

#: Row order of :attr:`die.jx.interface.Obs.agent` -- (agent_food, alive).
OBS_AGENT_FOOD, OBS_AGENT_ALIVE = 0, 1


# --- JAX twins of the trig helpers in die.utils ------------------------------
# Reimplemented rather than imported because the originals route through complex
# numpy scalars, which promotes float32 tracers and costs a device round-trip.

def polar2xy(r, theta):
    return r * jnp.cos(theta), r * jnp.sin(theta)


def _angle(x, y):
    """``np.angle(x + 1j*y)``, which is **not** ``arctan2(y, x)`` at signed zeros.

    :func:`die.utils.xy2polar` takes the angle of a complex number it builds as
    ``x + np.multiply(1j, y)``, and that construction is not sign-of-zero
    neutral: the complex multiply contributes ``copysign(0, y)`` to the real
    part, so ``(-0.0, +0.0)`` collapses to ``(+0.0, +0.0)`` and reports angle 0
    where IEEE ``arctan2`` reports pi.

    Faithfully reproduced rather than cleaned up, because it is *reachable and
    behavior-changing here*: ``grad_clip`` zeroes weak gradients by multiplying
    by a boolean, which preserves the sign bits of the direction that was
    discarded. Physarum then reads those signed zeros -- ``undetermined_grad``
    fires on angle 0 but not on angle pi, so a weak gradient that happened to
    point left is treated as a *decided* direction while one pointing right is
    not. On a 16x16 world that is a few percent of agents per step, and it
    drives both the turn and the deposit mask. It is an artifact of the complex
    detour, not a designed rule; fixing it means changing both engines together.
    """
    return jnp.arctan2(0.0 + y, x + jnp.copysign(0.0, y))


def xy2polar(x, y):
    # hypot, not sqrt(x^2 + y^2): matches numpy's `abs(complex)` and does not
    # overflow when the gradient field is left unnormalized.
    return jnp.hypot(x, y), _angle(x, y)


def get_radians(coords: jax.Array) -> jax.Array:
    return _angle(coords[0], coords[1])


def renormalize_radians(rads):
    """Renormalizes radians into (-pi, pi], as :func:`die.utils.renormalize_radians`."""
    return jnp.mod(rads - jnp.pi, -2 * jnp.pi) + jnp.pi


def discretize(value, step):
    return jnp.floor_divide(value, step) * step


def gather_by_agents(field: jax.Array,
                     coords: jax.Array,
                     params: EnvParams,
                     offset=0.) -> jax.Array:
    """Samples ``field`` at each agent's cell -- :meth:`AgentIndexer.field_by_agents`.

    ``field`` is ``[width, height]`` or ``[C, width, height]``; the result is
    ``[max_agents]`` or ``[C, max_agents]``. ``offset`` is added to the coords
    *before* discretization, exactly as the xarray version does, so an agent can
    sense a cell it is not standing on.
    """
    ix, iy = cells_at(coords + offset, params)
    return field[..., ix, iy]


@struct.dataclass
class EmptyState:
    """Carry of a memoryless policy. A pytree with no leaves, so it scans."""


@struct.dataclass
class GradientState:
    """What a gradient-following agent remembers between steps.

    :param grad: ``[2, max_agents]`` the post-momentum gradient, which the next
        step blends into by ``inertia``. Initialized to noise rather than zero
        so the population starts with scattered headings.
    :param direction: ``[max_agents]`` heading in radians. Read one step *later*
        than it is written -- it aims the sense offset, so an agent looks ahead
        of where it was already going.
    """

    grad: jax.Array
    direction: jax.Array


#: Carry of any policy in this module.
ScriptedState = EmptyState | GradientState


class ScriptedPolicy(Protocol):
    """Functional, explicit-state form of the :class:`die.agents.base.Agent` loop."""

    def init(self, key: jax.Array, params: EnvParams) -> ScriptedState:
        """Draws the initial carry."""
        ...

    def act(self,
            key: jax.Array,
            state: ScriptedState,
            obs: Obs,
            params: EnvParams) -> tuple[jax.Array, ScriptedState]:
        """Returns ``([3, max_agents]`` action, next carry).

        The action is ``(dx, dy, deposit)`` in **world units** -- already
        multiplied by the policy's own scale and deposit coefficients, which is
        what the xarray agents' ``forward`` returns and what
        :func:`die.jx.env.step` expects.
        """
        ...


class NullPolicy(struct.PyTreeNode):
    """Does nothing at all.

    The do-nothing baseline, promoted from the ad-hoc ``jnp.zeros`` that the
    test files kept rebuilding. AGENTS.md records why it is worth a name: it has
    already caught two configurations in which stillness out-earns foraging, and
    in those a learner can look successful while having learned nothing.
    """

    def init(self, key: jax.Array, params: EnvParams) -> EmptyState:
        return EmptyState()

    def act(self, key, state: EmptyState, obs: Obs,
            params: EnvParams) -> tuple[jax.Array, EmptyState]:
        return jnp.zeros((3, params.max_agents), dtype=jnp.float32), state


class ConstPolicy(struct.PyTreeNode):
    """Every agent applies the same fixed action -- :class:`die.agents.static.ConstAgent`."""

    delta_xy: tuple[float, float] = (-0.01, 0.005)
    deposit: float = 0.1

    def init(self, key: jax.Array, params: EnvParams) -> EmptyState:
        return EmptyState()

    def act(self, key, state: EmptyState, obs: Obs,
            params: EnvParams) -> tuple[jax.Array, EmptyState]:
        alive = obs.agent[OBS_AGENT_ALIVE]
        dx, dy = self.delta_xy
        action = jnp.stack([jnp.full_like(alive, dx),
                            jnp.full_like(alive, dy),
                            jnp.full_like(alive, self.deposit)])
        return action * alive, state


class BrownianPolicy(struct.PyTreeNode):
    """Uniform random walk -- :class:`die.agents.static.BrownianAgent`.

    Matches the original's *support*: ``[-move_scale, move_scale]`` for each
    coordinate and ``[0, deposit_scale]`` for the deposit, mirroring
    ``DataInitializer.get_random(size, a, b) = (b - a) * U + a``. The original
    additionally rounds the unit sample to 3 decimals before rescaling, which
    quantizes the support onto a 1e-3 lattice; that is an artifact of the field
    initializer it borrows and is not reproduced here.
    """

    move_scale: float = 0.01
    deposit_scale: float = 0.1

    def init(self, key: jax.Array, params: EnvParams) -> EmptyState:
        return EmptyState()

    def act(self, key, state: EmptyState, obs: Obs,
            params: EnvParams) -> tuple[jax.Array, EmptyState]:
        alive = obs.agent[OBS_AGENT_ALIVE]
        move_key, deposit_key = jax.random.split(key)
        move = jax.random.uniform(move_key, (2, params.max_agents),
                                  minval=-self.move_scale, maxval=self.move_scale)
        deposit = jax.random.uniform(deposit_key, (1, params.max_agents),
                                     minval=0., maxval=self.deposit_scale)
        return jnp.concatenate([move, deposit]) * alive, state


class GradientPolicy(struct.PyTreeNode):
    """Follows the chemical gradient with momentum -- :class:`GradientAgent`.

    Defaults are the tuned ones :func:`die.registry.make_agent` uses for
    ``AgentId.gradient``, so constructing this bare gives the same agent the
    xarray benchmark runs.

    ``normalized_grad`` and ``grad_clip`` are static because they pick Python
    branches (and ``grad_clip``'s ``None`` disables a whole term); every other
    knob is a traced leaf and can be swept under ``vmap``, same discipline as
    :class:`~die.jx.state.EnvParams`.
    """

    scale: float = 0.01
    deposit: float = 4.5
    inertia: float = 0.95
    sense_offset: float = 0.03
    #: Measured as a fraction of the 0.4-sigma exploration noise, not of ``scale``.
    noise_scale: float = 0.025
    normalized_grad: bool = struct.field(pytree_node=False, default=True)
    grad_clip: float | None = struct.field(pytree_node=False, default=1e-5)

    def _noise(self, key: jax.Array, params: EnvParams) -> jax.Array:
        return 0.4 * jax.random.normal(key, (2, params.max_agents))

    def init(self, key: jax.Array, params: EnvParams) -> GradientState:
        grad = self._noise(key, params)
        return GradientState(grad=grad, direction=get_radians(grad))

    def gradient_field(self, chem: jax.Array) -> jax.Array:
        """``[2, width, height]`` chemical gradient, normalized and clipped.

        ``jnp.gradient`` reproduces numpy's central differences with one-sided
        stencils at the edges, which is what the original stacks. Note the
        clipping mask is built from the **raw** norm even after normalization has
        already divided it out -- so it removes cells whose gradient was
        numerically meaningless, not cells whose normalized gradient is small
        (there are none). Order matters and is preserved.
        """
        grad = jnp.stack(jnp.gradient(chem))
        norm = jnp.sqrt(jnp.sum(grad ** 2, axis=0))
        if self.normalized_grad:
            # `np.true_divide` + `nan_to_num` in the original: a zero gradient
            # divides 0/0 to nan and lands back on 0. Guard the denominator so
            # the nan never exists, which also keeps this differentiable.
            safe = jnp.where(norm > 0., norm, 1.)
            grad = jnp.where(norm > 0., grad / safe, 0.)
        if self.grad_clip is not None:
            grad = grad * (norm >= self.grad_clip)
        return grad

    def process_gradient(self, key: jax.Array, grad: jax.Array, state: GradientState
                         ) -> tuple[jax.Array, jax.Array]:
        """Hook for subclasses; also returns the deposit mask (all-on here)."""
        return grad, jnp.ones_like(grad[0], dtype=bool)

    def act(self, key, state: GradientState, obs: Obs,
            params: EnvParams) -> tuple[jax.Array, GradientState]:
        """One step. ``key`` splits into ``(turn_key, noise_key)``, in that order.

        The split order is part of the contract rather than an implementation
        detail: the parity tests reproduce ``turn_key`` to feed the numpy agent
        the same random turns, which is the only way a stochastic branch can be
        compared step for step.
        """
        turn_key, noise_key = jax.random.split(key)
        alive = obs.agent[OBS_AGENT_ALIVE]

        # Sense the gradient one sense_offset ahead of the current heading.
        grad_field = self.gradient_field(obs.medium[MEDIUM_CHEM])
        offset = jnp.stack(polar2xy(self.sense_offset, state.direction))
        grad = gather_by_agents(grad_field, obs.coords, params, offset)

        grad, deposit_mask = self.process_gradient(turn_key, grad, state)
        grad = ((1. - self.inertia) * grad + self.inertia * state.grad
                + self.noise_scale * self._noise(noise_key, params))

        # Deposit in proportion to the food actually underfoot -- no offset here,
        # an agent reports what it found rather than what it is heading towards.
        food = gather_by_agents(obs.medium[MEDIUM_FOOD], obs.coords, params)
        # clip(0.1, 1.0) of a boolean mask: 1.0 where set, 0.1 where not, so a
        # silenced agent still signals faintly. An all-true mask clips to 1.0,
        # which is why the base agent needs no separate deposit path.
        mask = jnp.clip(deposit_mask.astype(grad.dtype), 0.1, 1.0)
        deposit = self.deposit * food * mask

        action = jnp.concatenate([grad * self.scale, deposit[None]]) * alive
        return action, GradientState(grad=grad, direction=get_radians(grad))


@struct.dataclass
class TurnDecision:
    """Outcome of Physarum's discrete turn choice.

    :param turn: ``[max_agents]`` signed turn in radians.
    :param deposit_mask: agents whose turn was *decided* -- they earn a full
        deposit; the rest are attenuated to 0.1.
    :param undetermined: agents that fell through to a random turn. Exposed
        because it is the only stochastic branch in the policy, so it is where
        parity against the numpy agent has to be established explicitly.
    """

    turn: jax.Array
    deposit_mask: jax.Array
    undetermined: jax.Array


class PhysarumPolicy(GradientPolicy):
    """Physarum-style discrete turns -- :class:`die.agents.gradient.PhysarumAgent`.

    Where :class:`GradientPolicy` steers continuously, this one may only turn by
    ``turn_angle`` per step and only sees gradients within ``sense_angle`` of
    where it is already pointed. Defaults are the tuned ones from
    :func:`die.registry.make_agent`.
    """

    scale: float = 0.0075
    deposit: float = 4.5
    inertia: float = 0.0
    sense_offset: float = 0.03
    noise_scale: float = 0.0
    turn_angle: float = 35.
    sense_angle: float = 120.
    #: Tolerance for calling a turn definite, relative to ``turn_angle``.
    turn_tolerance: float = 0.05

    def init(self, key: jax.Array, params: EnvParams) -> GradientState:
        grad = self._noise(key, params)
        # Headings start snapped to the turn lattice: every subsequent heading is
        # a whole number of turns away, so an unsnapped start would put the
        # population permanently off-grid.
        return GradientState(grad=grad,
                             direction=discretize(get_radians(grad),
                                                  jnp.radians(self.turn_angle)))

    def choose_turn(self, key: jax.Array, drads: jax.Array,
                    state: GradientState) -> TurnDecision:
        """Picks a turn from the delta between desired and actual heading.

        Three ways to end up undetermined, and they are not the same failure:
        the gradient carries no direction at all (it was clipped to zero, so
        ``drads`` is exactly 0), the desired heading is already within tolerance
        (nothing to correct), or it lies outside the sensing cone (seen, but not
        reachable in one turn). The first two silence the agent's deposit --
        it has no discovery to report -- while the third does not: an agent
        turning towards something off to the side still found something.
        """
        turn_radians = jnp.radians(self.turn_angle)
        sense_radians = jnp.radians(self.sense_angle)
        atol = turn_radians * self.turn_tolerance

        dir_delta = renormalize_radians(state.direction - drads)
        # np.isclose(a, b, rtol, atol) is |a - b| <= atol + rtol * |b|, and the
        # original passes a=0, so the reference side is the *measured* value.
        undetermined_grad = jnp.abs(drads) <= 1e-8 + 1e-5 * jnp.abs(drads)
        undetermined_turn = jnp.abs(dir_delta) <= atol + 1e-2 * jnp.abs(dir_delta)
        unseen_grad = jnp.abs(dir_delta) > sense_radians
        undetermined = undetermined_grad | undetermined_turn | unseen_grad

        random_turn = (jax.random.randint(key, drads.shape, 0, 2) - 0.5) * 2
        decided = dir_delta * ~undetermined
        turn = jnp.where(decided > atol, -1.,                 # right, clockwise
                         jnp.where(decided < -atol, 1.,       # left, ccw
                                   random_turn))
        return TurnDecision(turn=turn * turn_radians,
                            deposit_mask=~(undetermined_grad | undetermined_turn),
                            undetermined=undetermined)

    def process_gradient(self, key: jax.Array, grad: jax.Array, state: GradientState
                         ) -> tuple[jax.Array, jax.Array]:
        """Replaces the sensed gradient with a unit step along the new heading."""
        magnitude, drads = xy2polar(grad[0], grad[1])
        decision = self.choose_turn(key, drads, state)
        direction = renormalize_radians(state.direction + decision.turn)
        radius = 1. if self.normalized_grad else magnitude
        return jnp.stack(polar2xy(radius, direction)), decision.deposit_mask
