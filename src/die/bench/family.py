"""A seeded family of environments: the dataset agents are scored on.

Shaped as a **one-axis-at-a-time sweep** around a single baseline rather than a
grid. A full cross product of five axes would be hundreds of configurations
whose differences could not be attributed to anything; varying one axis from a
fixed baseline means every scenario has an unambiguous reading -- "this is the
baseline, but scarcer" -- and the baseline itself appears once as the shared
control that every axis passes through.

Each scenario is instantiated at several **world seeds**, and the distinction
between the two kinds of variation matters:

- A *seed* changes the food field while the physics stays fixed. Averaging over
  seeds is what makes a scenario's number a property of the scenario rather
  than of one lucky perlin draw.
- An *axis value* changes the physics. That is the thing being measured.

Seeds here are disjoint from :data:`die.bench.canonical.TRAIN_SEEDS`, so every
number this family produces is out-of-sample for the trained agents.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from die.bench.canonical import canonical_params
from die.jx.state import EnvParams

#: World seeds each scenario is instantiated at. Disjoint from the training
#: seeds *and* from canonical.EVAL_SEEDS, so the family is a third, independent
#: sample of worlds.
FAMILY_SEEDS = tuple(range(200, 206))


@dataclass(frozen=True)
class Scenario:
    """One point of the family: a named environment config.

    :param axis: which knob this scenario varies from the baseline. Scenarios
        sharing an axis form a response curve; ``'baseline'`` is the control.
    :param value: the varied knob's value, used as the x-coordinate when the
        axis is plotted.
    """

    name: str
    axis: str
    value: float
    overrides: dict[str, Any] = field(default_factory=dict)

    def params(self) -> EnvParams:
        return canonical_params(**self.overrides)


def _noise(food: float = 0.0, comm: float = 0.0) -> dict[str, Any]:
    """Noise overrides, with the static gate that actually switches noise on.

    ``stochastic_medium`` is a separate static flag from the two magnitudes, so
    setting only ``food_noise`` silently does nothing -- the perturbation is
    skipped wholesale when the gate is off (see ``jx.interface.perturb_medium``).
    """
    return dict(stochastic_medium=True, food_noise=food, comm_noise=comm)


def build_family() -> list[Scenario]:
    """The full dataset, scarce through rich through noisy.

    Bounds worth knowing, both documented in ``jx.interface.perturb_medium``:
    ``food_noise`` stops being interpretable above ~0.5, because clipping the
    multiplicative draw at zero starts adding food (+0.4% at 0.5, +8.3% at 1.0)
    and the "harder" world becomes measurably a richer one. So this family
    stops at 0.5.
    """
    scenarios = [Scenario('baseline', 'baseline', 0.0, {})]

    # --- abundance: the same patches, more or less food in them --------------
    for scale in (0.25, 0.5, 2.0, 4.0):
        scenarios.append(Scenario(f'food-scale-{scale}', 'food_scale', scale,
                                  dict(food_scale=scale)))

    # --- patchiness: few large basins vs fine speckle ------------------------
    # Gradient followers should care about this far more than about abundance:
    # it changes how far a useful chemical gradient reaches.
    #
    # Checked for an abundance confound, since changing perlin's frequency
    # could in principle change how much field survives the mask. Measured over
    # the six family seeds, total food is 16.7-18.5 across octaves 2..32 while
    # the per-seed spread is up to +/-3.0 -- so the axis is abundance-neutral
    # within seed noise, and a response along it is about structure. Do not read
    # a single seed here: one draw showed 22.1 vs 16.2, which is seed, not axis.
    for octaves in (2, 4, 16, 32):
        scenarios.append(Scenario(f'food-octaves-{octaves}', 'food_octaves', octaves,
                                  dict(food_octaves=octaves)))

    # --- resource noise: a field that will not hold still --------------------
    for sigma in (0.1, 0.25, 0.5):
        scenarios.append(Scenario(f'food-noise-{sigma}', 'food_noise', sigma,
                                  _noise(food=sigma)))

    # --- communication noise: corrupting the only long-range channel ---------
    # Separated from food noise because it degrades cooperation specifically
    # rather than perception in general.
    for sigma in (0.02, 0.05, 0.1):
        scenarios.append(Scenario(f'comm-noise-{sigma}', 'comm_noise', sigma,
                                  _noise(comm=sigma)))

    # --- crowding: how many agents share the same field ----------------------
    for ratio in (0.05, 0.1, 0.4):
        scenarios.append(Scenario(f'agent-ratio-{ratio}', 'init_agent_ratio', ratio,
                                  dict(init_agent_ratio=ratio)))

    # --- renewable resource: food that grows back ----------------------------
    # Rates bracket the *balanced* point, where renewal offsets consumption.
    # Measured under physarum over 40 steps: the non-renewable baseline runs
    # 16.15 -> 10.18 total food, and 0.02 lands within 1% of where it started.
    # So 0.01 is a world still running down, 0.02 sustainable, 0.04 recovering.
    for rate in (0.01, 0.02, 0.04):
        scenarios.append(Scenario(f'regrow-{rate}', 'regrow_rate', rate,
                                  dict(regrow_rate=rate, food_capacity=0.5)))

    # --- reappearing resource: food returning where it was exhausted ---------
    # The only axis on which a depleted cell can come back. Bracketed the same
    # way: 0.005 is the balanced rate.
    for rate in (0.002, 0.005, 0.01):
        scenarios.append(Scenario(f'respawn-{rate}', 'respawn_rate', rate,
                                  dict(stochastic_respawn=True, respawn_rate=rate,
                                       respawn_amount=0.15)))

    # --- non-stationary resource: food that moves under the agents -----------
    # Swept rather than left at one point: this was previously a single
    # scenario and produced by far the largest effects in the family, which is
    # the worst possible thing to under-sample. wave_decay sets how fast
    # standing resource is replaced by the incoming wave.
    for decay in (0.25, 0.5, 0.75):
        scenarios.append(Scenario(f'wave-decay-{decay}', 'wave_decay', decay,
                                  dict(food_flow='wave', wave_scale=0.5,
                                       wave_decay=decay)))

    return scenarios


def dynamic_params(**overrides):
    """The training config for the dynamic-world checkpoints: a *renewable* world.

    Regrowth and respawn at their balanced rates, and deliberately **no wave**.

    An earlier version combined all three, on the reasoning that a policy shown
    only one mechanism would have a blind spot. Measured, that was wrong, and
    wrong in an instructive way: the three are not comparable in reward scale.
    A wave world pays a do-nothing baseline of ~70 while a renewable one pays
    ~3.5, so a combined objective is ~95% wave, and the resulting checkpoints
    were wave specialists -- better than anything else on wave scenarios
    (+152 over baseline) and *worse than the static-trained agents* on the
    renewable ones they were supposed to be good at (-4.0 against +2.1).

    The two are different questions, so they get different worlds. This one is
    "food depletes and comes back"; the wave is "food moves", and it stays a
    test-time axis, where the static-trained agents can be asked whether they
    generalize to it without anyone having trained on it.
    """
    from die.bench.canonical import canonical_params
    base = dict(regrow_rate=0.02, food_capacity=0.5,
                stochastic_respawn=True, respawn_rate=0.005, respawn_amount=0.15)
    return canonical_params(**{**base, **overrides})


def family_axes() -> list[str]:
    """Distinct axes present in the family, baseline excluded."""
    seen = [s.axis for s in build_family() if s.axis != 'baseline']
    return list(dict.fromkeys(seen))
