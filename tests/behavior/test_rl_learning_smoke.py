"""End-to-end learning claims for the PPO pipeline.

Mirrors ``test_jx_learning_smoke.py`` so both learners are held to a comparable
contract, with one addition the ES tests do not have: a **do-nothing baseline**.
Beating an untrained network only shows the optimizer moved; beating stillness
shows the task was worth solving. Under ``food_infinite=True`` stillness scores
11.2 against a wanderer's 7.0 -- foraging is strictly worse than inaction, and a
learner can look successful while learning nothing (docs/JOURNAL.md, 2026-07-20).
Thresholds are empirically calibrated; recalibrate deliberately if physics changes.
"""

import numpy as np
import pytest

jax = pytest.importorskip('jax')
jnp = jax.numpy

from die.env import Dynamics, Env  # noqa: E402
from die.jx.interface import make_init_bank, reset, step  # noqa: E402
from die.jx.state import EnvParams  # noqa: E402
from die.rl.agent import RLAgent  # noqa: E402
from die.rl.config import HyperParams, TrainStatic  # noqa: E402
from die.rl.ppo import make_evaluate, make_train  # noqa: E402
from die.sim import Simulation  # noqa: E402
from die.train import HistoryLogger  # noqa: E402

FIELD = (16, 16)
MAX_AGENTS = FIELD[0] * FIELD[1]
TRAIN_SEEDS, EVAL_SEEDS = range(4), range(100, 104)


def env_params(**overrides) -> EnvParams:
    """A depleting-food world: foraging must beat sitting still."""
    params = dict(field_size=FIELD, max_agents=MAX_AGENTS, init_agent_ratio=0.2,
                  food_infinite=False, rate_decay_chem=0.025, diffuse_sigma=0.8)
    return EnvParams(**{**params, **overrides})


def tiny_static(**overrides) -> TrainStatic:
    static = dict(num_envs=8, num_steps=32, num_iterations=120, num_epochs=2,
                  num_minibatches=4, features=16, num_blocks=1, hidden=32, bank_size=4)
    return TrainStatic(**{**static, **overrides})


def fixed_policy_return(params: EnvParams, action_fn, steps=40, envs=8) -> float:
    """Scores a hand-written policy on the held-out worlds."""
    bank = make_init_bank(EVAL_SEEDS, params)
    _, state = jax.vmap(reset, in_axes=(0, None, None))(
        jax.random.split(jax.random.key(0), envs), bank, params)
    total = 0.
    for i in range(steps):
        timestep, state = jax.vmap(step, in_axes=(0, 0, 0, None, None))(
            jax.random.split(jax.random.key(i), envs), state, action_fn(envs), bank, params)
        total += float((timestep.reward * timestep.alive).sum(axis=-1).mean())
    return total


def do_nothing(envs: int) -> jax.Array:
    return jnp.zeros((envs, 3, MAX_AGENTS), jnp.float32)


def train_policy(params: EnvParams, static: TrainStatic, seed: int = 0):
    bank = make_init_bank(TRAIN_SEEDS, params)
    runner, metrics = jax.jit(make_train(static, bank))(
        jax.random.key(seed), HyperParams(env=params))
    return runner, metrics


@pytest.mark.slow
def test_the_task_is_not_degenerate():
    """Guards the premise of every other claim here.

    With infinite food the optimal policy is to sit on a food cell and eat
    forever at zero action cost, so 'the agent learned' becomes unfalsifiable.
    """
    depleting = fixed_policy_return(env_params(food_infinite=False), do_nothing)
    infinite = fixed_policy_return(env_params(food_infinite=True), do_nothing)

    assert infinite > depleting * 3     # inaction pays hugely when food never runs out
    assert depleting < 4.0              # ...and barely pays when it does


@pytest.mark.slow
def test_ppo_beats_doing_nothing():
    """The claim worth making: the policy forages, rather than merely surviving."""
    params, static = env_params(), tiny_static()
    runner, _ = train_policy(params, static)

    evaluate = jax.jit(make_evaluate(static, make_init_bank(EVAL_SEEDS, params)),
                       static_argnums=(4, 5))
    trained = float(evaluate(runner.params, runner.normalizer, jax.random.key(99),
                             HyperParams(env=params), 40, 8))
    baseline = fixed_policy_return(params, do_nothing)

    assert baseline == pytest.approx(2.76, abs=0.5)   # pins the calibration point
    assert trained > 1.5 * baseline


@pytest.mark.slow
def test_training_improves_the_return():
    params, static = env_params(), tiny_static()
    _, metrics = train_policy(params, static)
    returns = np.asarray(metrics.raw_return)

    assert np.all(np.isfinite(returns))
    assert returns[-10:].mean() > returns[:10].mean()


@pytest.mark.slow
def test_noise_training_does_not_hurt_clean_performance():
    """Medium noise acts as a regularizer here, not a robustness tradeoff.

    Calibrated loosely: the finding is that noise-training is *not worse* on
    clean worlds (measured +38% at 200 iterations), which is the surprising
    direction and the one worth pinning against regression.
    """
    static = tiny_static()
    clean = env_params()
    noisy = env_params(stochastic_medium=True, food_noise=0.25, comm_noise=0.05)

    evaluate = jax.jit(make_evaluate(static, make_init_bank(EVAL_SEEDS, clean)),
                       static_argnums=(4, 5))
    scores = {}
    for name, params in (('clean', clean), ('noisy', noisy)):
        runner, _ = train_policy(params, static)
        scores[name] = float(evaluate(runner.params, runner.normalizer,
                                      jax.random.key(99), HyperParams(env=clean), 40, 8))

    assert scores['noisy'] > 0.8 * scores['clean']


@pytest.mark.slow
def test_checkpoint_replays_in_the_xarray_env(tmp_path):
    """A trained policy must survive the trip back to the reference engine."""
    params, static = env_params(), tiny_static(num_iterations=10)
    runner, _ = train_policy(params, static)

    agent = RLAgent(params=runner.params, normalizer=runner.normalizer,
                    features=static.features, num_blocks=static.num_blocks,
                    kernel_size=static.kernel_size, hidden=static.hidden)
    path = tmp_path / 'policy.msgpack'
    agent.save(path)
    loaded = RLAgent.load(path)

    env = Env(FIELD, Dynamics(init_agent_ratio=0.2, food_infinite=False))
    first = Simulation(env, agent, seed=7).run(10).total_reward
    second = Simulation(env, loaded, seed=7).run(10).total_reward
    assert second == pytest.approx(first)
    assert np.isfinite(first)


@pytest.mark.slow
def test_trainer_reports_through_the_logger_seam(tmp_path):
    from die.rl.trainer import RLRunConfig, RLTrainer

    params = env_params()
    static = tiny_static(num_iterations=6)
    env = Env(FIELD, Dynamics(init_agent_ratio=0.2, food_infinite=False))
    history = HistoryLogger()

    trainer = RLTrainer(env=env, static=static, hypers=HyperParams(env=params),
                        config=RLRunConfig(seed=1, out_dir=tmp_path, record_steps=5),
                        loggers=[history])
    agent = trainer.train()

    # one IterStats per iteration, plus the final held-out evaluation
    assert len(history.iterations) == static.num_iterations + 1
    assert 'rl/raw_return' in history.iterations[0].metrics
    assert 'rl/eval_return' in history.iterations[-1].metrics
    # RL metrics must not be filed under the ES 'fitness/' prefix
    assert not any(key.startswith('fitness/') for key in history.iterations[0].metrics)

    assert (tmp_path / 'checkpoints' / 'final.msgpack').exists()
    assert (tmp_path / 'rollouts' / 'final.gif').exists()
    assert isinstance(agent, RLAgent)
