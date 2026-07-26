"""PPO over the population-batched env, as one jittable pure function.

``make_train(static, bank)`` returns ``train(key, hypers)``: no Python state, no
callbacks, no side effects. That shape is the point. It makes a population of
configurations a single ``jax.vmap`` -- which is what the meta-learning phase
needs, and what purejaxrl's design (config baked in as Python values, so only
the seed can be mapped) cannot provide.

Vendored rather than taken from a library because the novel axis here is exactly
the one every JAX RL library abstracts over: reward and terminals are
``[max_agents]`` arrays, and the loss reduces over an alive mask. rejax takes the
env as a static field and assumes the gymnax 5-tuple; adapting it would mean
subclassing the part that matters. PPO is a few hundred lines, and the masking
is where the correctness risk lives, so it stays visible.

**Masking is the thing to get right.** Dead slots are parked at (0, 0) carrying
zeros, and they outnumber live agents roughly nine to one at default settings.
They must be excluded from the loss, from advantage normalization, and from the
statistics used to normalize observations -- independently, in each place.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import optax
from flax import struct
from jax import lax

from die.jx.interface import Obs, get_obs, reset, step
from die.jx.state import EnvState
from die.rl.config import HyperParams, TrainStatic
from die.rl.networks import (
    ActorCritic,
    entropy,
    log_prob,
    sample_raw,
    squash,
    to_world_action,
)
from die.rl.obs import ObsNormalizer
from die.rl.reward import shape_reward

_EPS = 1e-8


class Transition(NamedTuple):
    """One collected step, across parallel worlds. Leading axes are ``[T, E]``.

    ``obs`` is stored **already normalized**, with the statistics current at
    collection time. Re-normalizing during the update epochs with newer
    statistics would put the numerator and denominator of the PPO ratio on
    different scales -- a silent bug that looks like instability.
    """

    obs: Obs
    raw_action: jax.Array       # pre-squash, so recomputation never needs atanh
    log_prob: jax.Array
    value: jax.Array
    reward: jax.Array           # shaped
    raw_reward: jax.Array       # unshaped env gain, for ES-comparable metrics
    done: jax.Array
    alive: jax.Array


@struct.dataclass
class RunnerState:
    params: Any
    opt_state: Any
    env_state: EnvState
    normalizer: ObsNormalizer
    key: jax.Array


@struct.dataclass
class Metrics:
    """Per-iteration diagnostics, stacked over iterations by the outer scan."""

    total_loss: jax.Array
    policy_loss: jax.Array
    value_loss: jax.Array
    entropy: jax.Array
    approx_kl: jax.Array
    clip_fraction: jax.Array
    #: Total unshaped env gain per world per rollout -- the quantity evolution
    #: maximizes, so this is what an RL-vs-ES comparison should read.
    raw_return: jax.Array
    mean_alive: jax.Array


def masked_mean(x: jax.Array, mask: jax.Array) -> jax.Array:
    return (x * mask).sum() / jnp.maximum(mask.sum(), 1.)


def masked_normalize(x: jax.Array, mask: jax.Array) -> jax.Array:
    """Standardizes over masked entries only, leaving the rest untouched.

    Including dead slots here would drag the mean toward zero in proportion to
    the death rate, so advantages would silently rescale as the population
    thins -- a moving target for the policy loss.
    """
    mean = masked_mean(x, mask)
    variance = masked_mean(jnp.square(x - mean), mask)
    return (x - mean) / jnp.sqrt(variance + _EPS)


def compute_gae(reward: jax.Array,
                value: jax.Array,
                done: jax.Array,
                last_value: jax.Array,
                gamma: jax.Array,
                gae_lambda: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Generalized advantage estimation, backwards over the time axis.

    ``done`` cuts both the bootstrap and the recursion. In the default
    configuration the only terminal is death (individual) or extinction
    (whole-world), and both genuinely have zero continuation value, so this is
    exact. Setting ``EnvParams.max_steps`` introduces time-limit truncation,
    which this treats as termination -- biased, in the usual way, and the reason
    the RL default leaves ``max_steps`` unset and lets the rollout boundary
    handle bootstrapping instead.
    """

    def backwards(carry, transition):
        advantage, next_value = carry
        step_reward, step_value, step_done = transition
        continues = 1. - step_done
        delta = step_reward + gamma * next_value * continues - step_value
        advantage = delta + gamma * gae_lambda * continues * advantage
        return (advantage, step_value), advantage

    _, advantages = lax.scan(
        backwards,
        (jnp.zeros_like(last_value), last_value),
        (reward, value, done),
        reverse=True)
    return advantages, advantages + value


def _to_world_level(transitions: Transition) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Collapses the agent axis for the ``joint`` factorization.

    One advantage per world step, over the summed reward and summed value of
    the agents that acted -- which is the objective evolution optimizes.
    """
    reward = (transitions.reward * transitions.alive).sum(axis=-1)
    value = (transitions.value * transitions.alive).sum(axis=-1)
    done = (transitions.done * transitions.alive).sum(axis=-1) / jnp.maximum(
        transitions.alive.sum(axis=-1), 1.)
    return reward, value, jnp.round(done)


def _build_network(static: TrainStatic) -> ActorCritic:
    return ActorCritic(features=static.features,
                       num_blocks=static.num_blocks,
                       kernel_size=static.kernel_size,
                       hidden=static.hidden)


def make_evaluate(static: TrainStatic, bank: EnvState):
    """Builds ``evaluate(params, normalizer, key, hypers, num_steps) -> return``.

    Deterministic: the policy acts on its mean rather than sampling, so repeated
    evaluations of the same parameters on the same worlds agree exactly. Point
    it at a bank the training run never saw to measure generalization rather
    than on-policy drift.

    Returns total *unshaped* env gain per world, averaged over worlds -- the
    same quantity evolution reports as fitness, so the two are comparable.
    """
    network = _build_network(static)

    def evaluate(params, normalizer: ObsNormalizer, key: jax.Array,
                 hypers: HyperParams, num_steps: int, num_envs: int = 8) -> jax.Array:
        env_params = hypers.env
        key, reset_key = jax.random.split(key)
        _, env_state = jax.vmap(reset, in_axes=(0, None, None))(
            jax.random.split(reset_key, num_envs), bank, env_params)

        def one_step(carry, step_key):
            env_state = carry
            obs = jax.vmap(get_obs, in_axes=(0, None))(env_state, env_params)
            if static.normalize_obs:
                obs = normalizer.apply(obs)
            mean, _, _ = jax.vmap(lambda single: network.apply(params, single))(obs)
            action = jax.vmap(to_world_action, in_axes=(0, None, None))(
                squash(mean), hypers.action_scale, hypers.deposit_scale)
            timestep, env_state = jax.vmap(step, in_axes=(0, 0, 0, None, None))(
                jax.random.split(step_key, num_envs), env_state, action, bank, env_params)
            return env_state, (timestep.reward * timestep.alive).sum(axis=-1)

        _, rewards = lax.scan(one_step, env_state, jax.random.split(key, num_steps))
        return rewards.sum(axis=0).mean()

    return evaluate


def make_train(static: TrainStatic, bank: EnvState):
    """Builds ``train(key, hypers) -> (RunnerState, Metrics)``.

    ``static`` and ``bank`` are captured; ``hypers`` flows through as traced
    values, so mapping over a stack of them costs one compilation.
    """
    network = _build_network(static)

    def apply_network(params, obs: Obs):
        """Evaluates the shared policy across parallel worlds."""
        return jax.vmap(lambda single: network.apply(params, single))(obs)

    def train(key: jax.Array, hypers: HyperParams):
        env_params = hypers.env

        # --- initialization --------------------------------------------------
        key, init_key, reset_key = jax.random.split(key, 3)
        reset_keys = jax.random.split(reset_key, static.num_envs)
        obs, env_state = jax.vmap(reset, in_axes=(0, None, None))(
            reset_keys, bank, env_params)

        template = jax.tree.map(lambda leaf: leaf[0], obs)
        params = network.init(init_key, template)

        gradient_tx = optax.chain(
            optax.clip_by_global_norm(hypers.max_grad_norm),
            optax.scale_by_adam())
        opt_state = gradient_tx.init(params)

        normalizer = ObsNormalizer.create(
            num_medium_channels=template.medium.shape[0],
            num_agent_features=template.agent.shape[0])

        # --- rollout collection ----------------------------------------------
        def collect(runner: RunnerState, _):
            key, action_key, step_key = jax.random.split(runner.key, 3)

            raw_obs = jax.vmap(get_obs, in_axes=(0, None))(runner.env_state, env_params)
            if static.normalize_obs:
                new_normalizer = runner.normalizer.update(raw_obs)
                obs = new_normalizer.apply(raw_obs)
            else:
                new_normalizer, obs = runner.normalizer, raw_obs

            mean, log_std, value = apply_network(runner.params, obs)
            raw_action = sample_raw(action_key, mean, log_std)
            action_log_prob = log_prob(raw_action, mean, log_std)

            world_action = jax.vmap(to_world_action, in_axes=(0, None, None))(
                squash(raw_action), hypers.action_scale, hypers.deposit_scale)
            step_keys = jax.random.split(step_key, static.num_envs)
            timestep, next_env_state = jax.vmap(step, in_axes=(0, 0, 0, None, None))(
                step_keys, runner.env_state, world_action, bank, env_params)

            reward = jax.vmap(shape_reward, in_axes=(0, 0, 0, None))(
                timestep.reward, timestep.alive, timestep.terminated, hypers.reward)
            # A world reset ends every trajectory in it, not just the dead ones.
            done = jnp.maximum(timestep.terminated,
                               timestep.truncated[:, None].astype(jnp.float32))

            transition = Transition(obs=obs,
                                    raw_action=raw_action,
                                    log_prob=action_log_prob,
                                    value=value,
                                    reward=reward,
                                    raw_reward=timestep.reward,
                                    done=done,
                                    alive=timestep.alive)
            runner = RunnerState(params=runner.params,
                                 opt_state=runner.opt_state,
                                 env_state=next_env_state,
                                 normalizer=new_normalizer,
                                 key=key)
            return runner, transition

        # --- loss ------------------------------------------------------------
        def loss_fn(params, batch: Transition, advantages, targets, entropy_coef):
            mean, log_std, value = apply_network(params, batch.obs)
            new_log_prob = log_prob(batch.raw_action, mean, log_std)
            alive = batch.alive

            if static.factorization == 'joint':
                # Sum log-probabilities within a world step: one sample per
                # world, exactly the objective evolution optimizes.
                new_log_prob = (new_log_prob * alive).sum(axis=-1)
                old_log_prob = (batch.log_prob * alive).sum(axis=-1)
                value = (value * alive).sum(axis=-1)
                old_value = (batch.value * alive).sum(axis=-1)
                mask = jnp.ones_like(new_log_prob)
            else:
                old_log_prob = batch.log_prob
                old_value = batch.value
                mask = alive

            ratio = jnp.exp(new_log_prob - old_log_prob)
            normalized = masked_normalize(advantages, mask)
            unclipped = ratio * normalized
            clipped = jnp.clip(ratio, 1. - hypers.clip_eps, 1. + hypers.clip_eps) * normalized
            policy_loss = -masked_mean(jnp.minimum(unclipped, clipped), mask)

            value_clipped = old_value + jnp.clip(
                value - old_value, -hypers.clip_eps, hypers.clip_eps)
            value_loss = 0.5 * masked_mean(
                jnp.maximum(jnp.square(value - targets),
                            jnp.square(value_clipped - targets)), mask)

            policy_entropy = entropy(log_std).mean()
            total = policy_loss + hypers.value_coef * value_loss - entropy_coef * policy_entropy

            diagnostics = (policy_loss, value_loss, policy_entropy,
                           masked_mean((ratio - 1.) - jnp.log(ratio + _EPS), mask),
                           masked_mean((jnp.abs(ratio - 1.) > hypers.clip_eps).astype(
                               jnp.float32), mask))
            return total, diagnostics

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

        # --- one iteration: collect, estimate advantages, update -------------
        def iteration(runner: RunnerState, iteration_index):
            runner, transitions = lax.scan(collect, runner, None, length=static.num_steps)

            last_obs = jax.vmap(get_obs, in_axes=(0, None))(runner.env_state, env_params)
            if static.normalize_obs:
                last_obs = runner.normalizer.apply(last_obs)
            *_, last_value = apply_network(runner.params, last_obs)

            if static.factorization == 'joint':
                reward, value, done = _to_world_level(transitions)
                last_value = (last_value * transitions.alive[-1]).sum(axis=-1)
            else:
                reward, value, done = (transitions.reward, transitions.value,
                                       transitions.done)

            advantages, targets = compute_gae(
                reward, value, done, last_value, hypers.gamma, hypers.gae_lambda)

            # Linear anneal, applied to the traced learning rate directly rather
            # than through an optax schedule, so the rate stays vmap-able.
            progress = iteration_index / static.num_iterations
            decay = 1. - progress if static.anneal_lr else 1.
            entropy_coef = hypers.entropy_coef

            def epoch(carry, _):
                params, opt_state, key = carry
                key, permutation_key = jax.random.split(key)

                flat = jax.tree.map(
                    lambda leaf: leaf.reshape((static.batch_size,) + leaf.shape[2:]),
                    (transitions, advantages, targets))
                order = jax.random.permutation(permutation_key, static.batch_size)
                shuffled = jax.tree.map(lambda leaf: leaf[order], flat)
                minibatches = jax.tree.map(
                    lambda leaf: leaf.reshape(
                        (static.num_minibatches, -1) + leaf.shape[1:]),
                    shuffled)

                def minibatch(carry, batch):
                    params, opt_state = carry
                    batch_transitions, batch_advantages, batch_targets = batch
                    (total, diagnostics), grads = grad_fn(
                        params, batch_transitions, batch_advantages, batch_targets,
                        entropy_coef)
                    updates, opt_state = gradient_tx.update(grads, opt_state, params)
                    updates = jax.tree.map(
                        lambda u: -hypers.learning_rate * decay * u, updates)
                    return (params := optax.apply_updates(params, updates),
                            opt_state), (total, *diagnostics)

                (params, opt_state), stats = lax.scan(
                    minibatch, (params, opt_state), minibatches)
                return (params, opt_state, key), stats

            key, epoch_key = jax.random.split(runner.key)
            (params, opt_state, _), stats = lax.scan(
                epoch, (runner.params, runner.opt_state, epoch_key),
                None, length=static.num_epochs)

            runner = RunnerState(params=params,
                                 opt_state=opt_state,
                                 env_state=runner.env_state,
                                 normalizer=runner.normalizer,
                                 key=key)
            total, policy_loss, value_loss, policy_entropy, approx_kl, clip_fraction = (
                jnp.mean(stat) for stat in stats)
            metrics = Metrics(
                total_loss=total,
                policy_loss=policy_loss,
                value_loss=value_loss,
                entropy=policy_entropy,
                approx_kl=approx_kl,
                clip_fraction=clip_fraction,
                raw_return=(transitions.raw_reward * transitions.alive).sum(
                    axis=(0, 2)).mean(),
                mean_alive=transitions.alive.sum(axis=-1).mean())
            return runner, metrics

        runner = RunnerState(params=params, opt_state=opt_state, env_state=env_state,
                             normalizer=normalizer, key=key)
        return lax.scan(iteration, runner, jnp.arange(static.num_iterations))

    return train
