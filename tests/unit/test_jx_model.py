"""The Flax NCA model and the batched rollout."""

import numpy as np
import pytest
import torch as th

jax = pytest.importorskip('jax')
jnp = jax.numpy

from die.agents.neural import NeuralAutomataAgent  # noqa: E402
from die.env import BoundaryCondition, Dynamics, Env  # noqa: E402
from die.jx.env import init_state  # noqa: E402
from die.jx.model import ConvNCA, compute_action, init_params, params_from_torch  # noqa: E402
from die.jx.rollout import make_episode_return, make_evaluate_population, stack_params  # noqa: E402
from die.jx.state import EnvConfig  # noqa: E402
from die.sim import Simulation  # noqa: E402

FIELD = (16, 16)
KERNELS = [3, 3]
SCALE, DEPOSIT = 0.01, 2.0


@pytest.fixture
def config() -> EnvConfig:
    return EnvConfig(field_size=FIELD, max_agents=FIELD[0] * FIELD[1],
                     init_agent_ratio=0.2, boundary='limit')


@pytest.fixture
def action_coefs() -> jax.Array:
    return jnp.asarray([SCALE, SCALE, DEPOSIT], dtype=jnp.float32)


@pytest.fixture
def model() -> ConvNCA:
    return ConvNCA(kernel_sizes=tuple(KERNELS))


class TestConvNCA:
    def test_output_shape_and_range(self, model, config):
        params = init_params(model, config, jax.random.key(0))
        state = init_state(7, config)
        out = model.apply(params, state.medium)
        assert out.shape == (3, *FIELD)
        assert np.all(np.abs(np.asarray(out)) <= 1.0)  # single trailing tanh

    def test_circular_padding_is_shift_equivariant(self, model, config):
        params = init_params(model, config, jax.random.key(0))
        state = init_state(7, config)
        shifted = jnp.roll(state.medium, 3, axis=1)
        out_then_shift = jnp.roll(model.apply(params, state.medium), 3, axis=1)
        shift_then_out = model.apply(params, shifted)
        np.testing.assert_allclose(np.asarray(shift_then_out),
                                   np.asarray(out_then_shift), atol=1e-5)

    def test_action_shape_and_scaling(self, model, config, action_coefs):
        params = init_params(model, config, jax.random.key(0))
        state = init_state(7, config)
        action = compute_action(model, params, state, config, action_coefs)
        assert action.shape == (3, config.max_agents)
        # |tanh| <= 1, so each channel is bounded by its coefficient
        bounds = np.asarray(action_coefs)[:, None]
        assert np.all(np.abs(np.asarray(action)) <= bounds + 1e-6)


class TestTorchParity:
    """Identical weights must produce identical action fields."""

    def test_matches_torch_convolution_model(self, model, config):
        th.manual_seed(0)
        torch_agent = NeuralAutomataAgent(kernel_sizes=KERNELS, scale=SCALE, deposit=DEPOSIT)
        params = params_from_torch(torch_agent.model, KERNELS)
        state = init_state(7, config)

        medium = np.asarray(state.medium)
        torch_out = torch_agent.model.forward(
            th.as_tensor(medium.copy(), dtype=th.float32)[None]).detach().numpy().squeeze()
        jax_out = np.asarray(model.apply(params, state.medium))
        np.testing.assert_allclose(jax_out, torch_out, atol=1e-6)

    def test_episode_return_tracks_simulation(self, model, config, action_coefs):
        """End-to-end: same weights, same seed, both engines.

        Tolerance is loose on purpose -- once the neural agent moves agents
        around they collide, and the two implementations resolve collided
        deposits differently (see TestDepositCollisions in test_jx_env.py).
        """
        steps = 20
        th.manual_seed(0)
        torch_agent = NeuralAutomataAgent(kernel_sizes=KERNELS, scale=SCALE, deposit=DEPOSIT)
        params = params_from_torch(torch_agent.model, KERNELS)

        episode_return = jax.jit(make_episode_return(model, config, steps, action_coefs))
        jax_return = float(episode_return(params, init_state(7, config)))

        env = Env(FIELD, Dynamics(init_agent_ratio=0.2, boundary=BoundaryCondition.limit))
        result = Simulation(env, torch_agent, seed=7).run(steps)

        assert jax_return == pytest.approx(result.total_reward, rel=0.05)


class TestBatchedEvaluation:
    def test_evaluates_whole_population(self, model, config, action_coefs):
        popsize = 4
        keys = jax.random.split(jax.random.key(0), popsize)
        population = stack_params([init_params(model, config, key) for key in keys])

        evaluate = make_evaluate_population(model, config, 10, action_coefs)
        fitnesses = np.asarray(evaluate(population, init_state(7, config)))

        assert fitnesses.shape == (popsize,)
        assert np.all(np.isfinite(fitnesses))

    def test_batched_matches_individual(self, model, config, action_coefs):
        keys = jax.random.split(jax.random.key(1), 3)
        params_list = [init_params(model, config, key) for key in keys]
        state = init_state(7, config)

        episode_return = jax.jit(make_episode_return(model, config, 10, action_coefs))
        individually = [float(episode_return(params, state)) for params in params_list]

        evaluate = make_evaluate_population(model, config, 10, action_coefs)
        batched = np.asarray(evaluate(stack_params(params_list), state))

        np.testing.assert_allclose(batched, individually, rtol=1e-5)

    def test_reuses_compilation_across_states(self, model, config, action_coefs):
        """Varying the initial world per generation must not trigger recompiles."""
        population = stack_params([init_params(model, config, jax.random.key(i))
                                   for i in range(2)])
        evaluate = make_evaluate_population(model, config, 5, action_coefs)

        evaluate(population, init_state(7, config))
        compilations = evaluate._cache_size()
        evaluate(population, init_state(11, config))
        assert evaluate._cache_size() == compilations
