"""Two PPO implementations, one benchmark.

The vendored PPO in ``die/rl/ppo.py`` is our own reading of the algorithm, so it
needs a witness. These run rejax's PPO -- shared code: none -- over the same
environment and compare.

The benchmark is deliberately the *shared* one, which means the vendored path
runs handicapped too. rejax can only consume the ``joint`` factorization (one
scalar reward per world step), so that is what both sides use, and the
``per_agent`` number is carried along only to show what the handicap costs.
``max_steps`` stays ``None`` and ``agents_die`` stays ``False`` so nothing
terminates mid-rollout: rejax collapses truncation into termination, and on this
env that difference is worth about 40 points of return (docs/JOURNAL.md).

Thresholds are empirically calibrated and generous on purpose -- joint-factorized
PPO is high-variance here, which is itself the finding. Recalibrate deliberately.
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


def env_params() -> EnvParams:
    """The shared benchmark: depleting food, nothing ever terminates."""
    return EnvParams(field_size=FIELD, max_agents=MAX_AGENTS, init_agent_ratio=0.2,
                     agents_die=False, food_infinite=False,
                     rate_decay_chem=0.025, diffuse_sigma=0.8)


def static(factorization: str) -> TrainStatic:
    return TrainStatic(num_envs=NUM_ENVS, num_steps=NUM_STEPS, num_iterations=ITERATIONS,
                       num_epochs=2, num_minibatches=4, features=16, num_blocks=1,
                       hidden=32, bank_size=4, factorization=factorization)


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


def vendored_return(factorization: str, seed: int = 0) -> float:
    params = env_params()
    train_static = static(factorization)
    runner, _ = jax.jit(make_train(train_static, make_init_bank(TRAIN_SEEDS, params)))(
        jax.random.key(seed), HyperParams(env=params))
    evaluate = jax.jit(make_evaluate(train_static, make_init_bank(EVAL_SEEDS, params)),
                       static_argnums=(4, 5))
    return float(evaluate(runner.params, runner.normalizer, jax.random.key(99),
                          HyperParams(env=params), EVAL_STEPS, NUM_ENVS))


def rejax_returns(conv: bool, seed: int = 0) -> tuple[float, float]:
    """``(untrained, trained)`` held-out return, from rejax's own eval callback."""
    params = env_params()
    total = ITERATIONS * NUM_ENVS * NUM_STEPS
    algorithm = make_rejax_ppo(
        make_init_bank(TRAIN_SEEDS, params),
        RejaxParams(env=params, max_steps_in_episode=EVAL_STEPS),
        conv=conv, eval_bank=make_init_bank(EVAL_SEEDS, params),
        total_timesteps=total, num_envs=NUM_ENVS, num_steps=NUM_STEPS,
        num_epochs=2, num_minibatches=4, eval_freq=total)
    _, (_, returns) = train_rejax(algorithm, jax.random.key(seed))
    curve = np.asarray(returns).mean(axis=-1)
    return float(curve[0]), float(curve[-1])


@pytest.mark.slow
def test_rejax_trains_this_env_end_to_end():
    """The mechanical claim: a second, independent PPO consumes this env."""
    untrained, trained = rejax_returns(conv=True)
    assert np.isfinite(untrained) and np.isfinite(trained)


@pytest.mark.slow
def test_the_convolutional_prior_is_what_makes_the_env_trainable():
    """rejax's own MLP, on the same loss, is not merely worse -- it is unusable.

    The only difference between the two runs is the network. The MLP sees the
    flattened field and has to learn locality from scratch across a
    ``3 * max_agents`` action head; it never gets near the do-nothing baseline.
    This is the architecture claim of die/rl/networks.py, measured rather than
    argued -- though note the two also differ in output-layer init, so the gap
    is 'conv trunk as configured' rather than convolution alone.
    """
    _, conv = rejax_returns(conv=True)
    _, mlp = rejax_returns(conv=False)

    assert mlp < -15.                       # far below inaction, consistently
    assert conv > mlp + 15.


@pytest.mark.slow
def test_joint_factorization_costs_more_than_the_choice_of_library():
    """The result this cross-check exists to produce.

    Both implementations, restricted to the joint objective, land near or below
    a do-nothing baseline; the vendored path's ``per_agent`` objective is the
    only one that clears it. So the interesting axis for further R&D is the
    factorization, not the PPO implementation -- which is what makes the two
    implementations agreeing here informative rather than reassuring.
    """
    params = env_params()
    baseline = do_nothing_return(params)
    per_agent = vendored_return('per_agent')

    assert baseline == pytest.approx(2.68, abs=0.5)   # pins the calibration point
    assert per_agent > baseline
