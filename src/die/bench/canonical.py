"""The one config that training and evaluation both import.

Deliberately *not* a registered ``DynamicsId``. The presets in
:mod:`die.registry` are world flavours; this is a benchmark definition, and it
pins settings a flavour has no business pinning (seed ranges, episode length).
``st-perlin-wide`` is its nearest preset and differs where it matters:
``food_infinite=True``, which AGENTS.md records as degenerate -- an unlimited
resource makes doing nothing nearly as good as foraging.

The numbers come from ``tests/behavior/test_rl_rejax_crosscheck.py`` so results
here stay comparable with the figures already in ``docs/JOURNAL.md``: a
do-nothing baseline of 2.68-2.76 and trained policies at 4.5-6.2.
"""

from __future__ import annotations

from die.jx.state import EnvParams

FIELD: tuple[int, int] = (16, 16)
MAX_AGENTS: int = FIELD[0] * FIELD[1]

#: Worlds the policies train on, and the disjoint ones they are scored on.
#: Disjoint because the interesting question is generalization: every agent is
#: evaluated on food fields it has never seen.
TRAIN_SEEDS = tuple(range(4))
EVAL_SEEDS = tuple(range(100, 104))

#: Rollout shape. ``EVAL_STEPS`` is the episode length every reported return is
#: summed over, so it is part of the benchmark's definition, not a tuning knob.
EVAL_STEPS: int = 40
NUM_ENVS: int = 8

#: PPO at the rate ``docs/JOURNAL.md`` settles on after retracting 1e-2 as a
#: single-seed artifact (4.39 +/- 1.41 vs 3.21 +/- 2.70 across three seeds).
#: Note this disagrees with ``HyperParams.learning_rate``'s own docstring, which
#: still argues for the retracted value.
LEARNING_RATE: float = 1e-3
ITERATIONS: int = 120
NUM_STEPS: int = 32

#: Seeds each trainable agent is trained at. The journal is emphatic that
#: single-seed results on this benchmark have been retracted twice, at effect
#: sizes comparable to the seed spread.
TRAIN_REPEATS = (0, 1, 2)


def canonical_params(**overrides) -> EnvParams:
    """The benchmark world: depleting food, nothing terminates mid-episode.

    ``agents_die=False`` and ``max_steps=None`` keep every episode the same
    length, so a return is a sum over a fixed horizon and differences between
    agents cannot come from one of them ending early.
    """
    base = dict(field_size=FIELD,
                max_agents=MAX_AGENTS,
                init_agent_ratio=0.2,
                agents_die=False,
                food_infinite=False,
                rate_decay_chem=0.025,
                diffuse_sigma=0.8)
    return EnvParams(**{**base, **overrides})
