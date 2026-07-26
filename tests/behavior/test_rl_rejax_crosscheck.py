"""Two PPO implementations, one benchmark.

The vendored PPO in ``die/rl/ppo.py`` is our own reading of the algorithm, so it
needs a witness. These run rejax's PPO -- shared code: none -- over the same
environment and compare.

The benchmark is deliberately the *shared* one, which means the vendored path
runs handicapped too. rejax can only consume the ``joint`` factorization (one
scalar reward per world step), so that is what it uses. ``max_steps`` stays
``None`` and ``agents_die`` stays ``False`` so nothing terminates mid-rollout:
rejax collapses truncation into termination, and on this env that difference is
worth about 40 points of return (docs/JOURNAL.md).

**Each configuration runs at its own tuned learning rate.** Comparing two
algorithms at a third one's optimum measures nothing, and getting this wrong
once already inverted a conclusion here -- rejax at the vendored path's 1e-2
looks catastrophic and unstable, while at 1e-3 it is the strongest and tightest
result on the branch.

Thresholds come from 3-seed sweeps but the tests run one pinned seed, to fit CI.
They are regression guards, not evidence; the seed spreads that justify them are
in the journal. Recalibrate deliberately if the physics changes.
"""

import numpy as np
import pytest

jax = pytest.importorskip('jax')
pytest.importorskip('rejax')
jnp = jax.numpy

from die.jx.interface import make_init_bank, reset, step  # noqa: E402
from die.jx.state import EnvParams  # noqa: E402
from die.rl.config import HyperParams, TrainStatic  # noqa: E402
from die.rl.ppo import make_evaluate, make_train  # noqa: E402
from die.rl.rejax_backend import RejaxParams, make_rejax_ppo, train_rejax  # noqa: E402

FIELD = (16, 16)
MAX_AGENTS = FIELD[0] * FIELD[1]
TRAIN_SEEDS, EVAL_SEEDS = range(4), range(100, 104)
ITERATIONS, NUM_ENVS, NUM_STEPS, EVAL_STEPS = 120, 8, 32, 40

#: Tuned per (algorithm, objective) pair -- see the module docstring.
LR_VENDORED_PER_AGENT = 1e-3
LR_VENDORED_JOINT = 3e-3
LR_REJAX = 1e-3


def env_params() -> EnvParams:
    """The shared benchmark: depleting food, nothing ever terminates."""
    return EnvParams(field_size=FIELD, max_agents=MAX_AGENTS, init_agent_ratio=0.2,
                     agents_die=False, food_infinite=False,
                     rate_decay_chem=0.025, diffuse_sigma=0.8)


def do_nothing_return(params: EnvParams) -> float:
    """The bar that matters: not 'better than random', but 'better than inaction'."""
    bank = make_init_bank(EVAL_SEEDS, params)
    _, state = jax.vmap(reset, in_axes=(0, None, None))(
        jax.random.split(jax.random.key(0), NUM_ENVS), bank, params)
    total = 0.
    for i in range(EVAL_STEPS):
        timestep, state = jax.vmap(step, in_axes=(0, 0, 0, None, None))(
            jax.random.split(jax.random.key(i), NUM_ENVS), state,
            jnp.zeros((NUM_ENVS, 3, MAX_AGENTS), jnp.float32), bank, params)
        total += float((timestep.reward * timestep.alive).sum(axis=-1).mean())
    return total


def vendored_return(factorization: str, learning_rate: float, seed: int = 0) -> float:
    params = env_params()
    static = TrainStatic(num_envs=NUM_ENVS, num_steps=NUM_STEPS, num_iterations=ITERATIONS,
                         num_epochs=2, num_minibatches=4, features=16, num_blocks=1,
                         hidden=32, bank_size=4, factorization=factorization)
    hypers = HyperParams(env=params, learning_rate=learning_rate)
    runner, _ = jax.jit(make_train(static, make_init_bank(TRAIN_SEEDS, params)))(
        jax.random.key(seed), hypers)
    evaluate = jax.jit(make_evaluate(static, make_init_bank(EVAL_SEEDS, params)),
                       static_argnums=(4, 5))
    return float(evaluate(runner.params, runner.normalizer, jax.random.key(99),
                          hypers, EVAL_STEPS, NUM_ENVS))


def rejax_return(conv: bool, learning_rate: float = LR_REJAX, seed: int = 0) -> float:
    """Held-out return, from rejax's own evaluation callback."""
    params = env_params()
    total = ITERATIONS * NUM_ENVS * NUM_STEPS
    algorithm = make_rejax_ppo(
        make_init_bank(TRAIN_SEEDS, params),
        RejaxParams(env=params, max_steps_in_episode=EVAL_STEPS),
        conv=conv, eval_bank=make_init_bank(EVAL_SEEDS, params),
        total_timesteps=total, num_envs=NUM_ENVS, num_steps=NUM_STEPS,
        num_epochs=2, num_minibatches=4, eval_freq=total, learning_rate=learning_rate)
    _, (_, returns) = train_rejax(algorithm, jax.random.key(seed))
    return float(np.asarray(returns).mean(axis=-1)[-1])


@pytest.mark.slow
def test_the_benchmark_is_not_degenerate():
    """Guards the premise of every claim below, as the ES-side smoke tests do.

    Two configurations have already been caught making stillness optimal here:
    ``food_infinite=True`` and ``init_agent_ratio=1.0`` (agents fill every cell,
    so there is nowhere to forage to). Both scored do-nothing far above any
    trained policy. See docs/JOURNAL.md.
    """
    assert do_nothing_return(env_params()) == pytest.approx(2.68, abs=0.5)


@pytest.mark.slow
def test_an_independent_ppo_implementation_clears_the_baseline():
    """The cross-check's core claim: rejax's loss, our env, no shared code.

    3-seed spread behind the threshold: 6.22 +- 1.35, worst seed 4.55.
    """
    assert rejax_return(conv=True) > 1.5 * do_nothing_return(env_params())


@pytest.mark.slow
def test_the_convolutional_prior_is_what_makes_the_env_trainable():
    """rejax's own MLP, on the same loss and rate, is not worse -- it is unusable.

    The only difference between the two runs is the network, and this is the
    largest and tightest effect on the branch (-32.51 +- 0.68 across seeds).
    Note the two also differ in output-layer init, so this measures 'the conv
    trunk as configured' rather than convolution in isolation.
    """
    conv, mlp = rejax_return(conv=True), rejax_return(conv=False)
    assert mlp < -20.
    assert conv > mlp + 25.


@pytest.mark.slow
def test_both_implementations_agree_that_the_task_is_learnable():
    """Tuned, all three configurations clear inaction -- and land close together.

    The gap that is *not* here is the interesting one: ``per_agent`` (4.68 +-
    1.22) and ``joint`` (4.56 +- 0.82) are indistinguishable at this scale,
    though the per-agent objective sees ~10^5x more reward signal. Pinned loosely
    on purpose -- the claim is 'comparable', and a regression that made them
    diverge is what should fail here.
    """
    baseline = do_nothing_return(env_params())
    per_agent = vendored_return('per_agent', LR_VENDORED_PER_AGENT)
    joint = vendored_return('joint', LR_VENDORED_JOINT)

    assert per_agent > baseline and joint > baseline
    assert abs(per_agent - joint) < 0.6 * max(per_agent, joint)
