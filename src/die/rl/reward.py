"""Turning per-agent energy change into a learning signal.

This is the design decision the project actually cares about, so it is a
parameter rather than a hardcoded choice.

The env reports raw energy change per agent. Rewarding an agent with exactly
that makes depositing pheromone *pure cost to self* -- the benefit lands on
whoever passes later -- so a selfish learner converges on never communicating,
and the central research question ("do agents have enough pressure to learn
communication?", docs/NOTES.md) is answered by the reward function instead of by
the experiment. Rewarding every agent with the population mean restores the
incentive but credits each agent for everyone else's foraging, which is the
high-variance structural credit assignment problem.

``share`` interpolates between the two. At ``share=1`` the summed reward equals
the total population gain -- precisely the quantity evolution optimizes -- which
is what makes an apples-to-apples RL-vs-ES comparison possible.

This lives in ``die.rl`` and not in :class:`~die.jx.state.EnvParams` on purpose:
``EnvParams`` mirrors :class:`die.env.Dynamics` under the "two engines, one
physics" invariant, and reward shaping has no twin there. It is an objective,
not physics.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import struct


@struct.dataclass
class RewardParams:
    """All traced, so the selfish/collective axis is sweepable.

    :param share: 0 rewards an agent for its own gain only; 1 rewards every
        agent with the population mean. Values between mix the two.
    :param survival_bonus: added per step to each agent that survived it.
        **Not optional in a world with death.** With ``agents_die=True`` and no
        bonus, an agent whose expected per-step gain is negative maximizes
        return by dying immediately -- starving is a perfectly good way to stop
        accumulating negative reward. The bonus is what makes "not-dying" pay.
    :param death_penalty: subtracted from an agent on the step it dies.
    """

    share: float = 1.0
    survival_bonus: float = 0.0
    death_penalty: float = 0.0


def shape_reward(raw: jax.Array,
                 alive: jax.Array,
                 died: jax.Array,
                 params: RewardParams) -> jax.Array:
    """``[max_agents]`` shaped reward for one world's step.

    Every argument is per-agent and pre-step-masked: ``alive`` is the set of
    agents that actually acted. Averaging happens over that set, not over all
    slots -- dead slots carry zeros and would otherwise pull the collective term
    toward zero in proportion to the death rate.

    Vmap this over parallel worlds; the collective term must stay within a
    world, since agents in different worlds share nothing.
    """
    num_alive = jnp.maximum(alive.sum(), 1.)
    collective = (raw * alive).sum() / num_alive

    shaped = (1. - params.share) * raw + params.share * collective
    survived = alive * (1. - died)
    shaped = shaped + params.survival_bonus * survived - params.death_penalty * died
    return shaped * alive
