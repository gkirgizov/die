"""The same environment, trained by somebody else's PPO.

:mod:`die.rl.ppo` is vendored, which means its correctness rests on our own
reading of the algorithm. This module runs the identical environment through
`rejax <https://github.com/keraJLi/rejax>`_'s PPO so the two can be compared on
a shared benchmark. Where they agree, the vendored loss is corroborated by an
implementation that shares none of its code; where they disagree, one of them is
wrong and the disagreement says where to look.

**It is a cross-check, not a drop-in alternative.** rejax speaks single-agent
gymnax, and adapting this environment to that shape costs three things that the
vendored path was built specifically to avoid:

1. **Joint factorization only.** gymnax hands back one scalar reward per step, so
   log-probabilities are summed over the whole population and one advantage
   covers a whole world-step. That is the objective evolution optimizes, and it
   discards the per-agent signal that motivates the RL path at all. Measured,
   this is most of the difference: on the 16x16 benchmark the vendored PPO
   scores 5.9 per-agent against 0.9 joint, with a do-nothing baseline of 2.7.
2. **No alive-masking.** rejax's loss cannot know that most action dimensions
   belong to dead slots, so at ``init_agent_ratio=0.2`` roughly four fifths of
   the action vector is noise in the log-probability. :func:`die.rl.ppo`'s
   ``joint`` path masks it; this one cannot. The cost is measured directly by
   the ``masked``/``unmasked`` comparison in the cross-check test rather than
   estimated.
3. **Truncation collapses into termination.** gymnax returns a single ``done``,
   and rejax zeroes the bootstrap on it. That distinction is the reason
   :mod:`die.jx.interface` returns ``terminated`` and ``truncated`` separately,
   and getting it wrong is not a subtlety here: setting ``max_steps=64`` on the
   benchmark below turns a policy that reaches +5.9 into one that reaches -36.
   So this path runs with ``max_steps=None`` and ``agents_die=False``, where
   nothing ever terminates and both implementations bootstrap only at the
   rollout boundary. ``max_steps_in_episode`` survives purely as rejax's
   evaluation horizon.

The architecture is held fixed across the comparison on purpose. rejax's stock
policy is an MLP over the flattened field, which would confound "different PPO"
with "different network" -- so :class:`ConvPPO` swaps in the same convolutional
trunk the vendored path uses, leaving rejax's *update rule* as the only variable.
:class:`~rejax.PPO` itself, with its MLP, remains available and measures
something else worth knowing: what the convolutional prior is actually worth.
"""

from __future__ import annotations

from functools import partial

import distrax
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import struct
from gymnax.environments import spaces
from gymnax.environments.environment import Environment
from rejax import PPO
from rejax.evaluate import evaluate

from die.jx.interface import Obs, get_obs, reset, step_env
from die.jx.state import EnvParams, EnvState
from die.rl.networks import NUM_ACTIONS, ConvTrunk
from die.rl.obs import OBS_AGENT_FOOD


@struct.dataclass
class RejaxParams:
    """gymnax-shaped parameters: our env params plus what rejax insists on.

    Kept separate from :class:`~die.jx.state.EnvParams` so the gymnax-isms stay
    confined to this module. ``env`` is a traced pytree and rides through
    ``vmap`` untouched; ``max_steps_in_episode`` is read in Python by rejax when
    it builds its evaluation callback, so it has to be a concrete int.
    """

    env: EnvParams
    #: Multiply the policy's [-1, 1] output into world units, as in HyperParams.
    action_scale: float = 0.1
    deposit_scale: float = 1.0
    #: rejax's evaluation horizon only. Because ``EnvParams.max_steps`` is None
    #: on this path, the environment itself never truncates -- see the module
    #: docstring on why that matters.
    max_steps_in_episode: int = struct.field(pytree_node=False, default=40)


def obs_size(params: EnvParams) -> int:
    width, height = params.field_size
    return 3 * width * height + 4 * params.max_agents


def action_size(params: EnvParams) -> int:
    return NUM_ACTIONS * params.max_agents


def flatten_obs(obs: Obs) -> jax.Array:
    """Packs the structured observation into the flat vector gymnax promises.

    ``[medium | agent | cells]``, in that order. ``cells`` are grid indices and
    are carried as float here purely because the observation must be one array;
    :func:`unflatten_obs` casts them back before they are used to gather.
    """
    return jnp.concatenate([obs.medium.ravel(),
                            obs.agent.ravel(),
                            obs.cells.ravel().astype(jnp.float32)])


def unflatten_obs(flat: jax.Array, params: EnvParams) -> Obs:
    """Inverse of :func:`flatten_obs`, for a single (unbatched) observation."""
    width, height = params.field_size
    agents = params.max_agents
    medium_size = 3 * width * height
    medium = flat[:medium_size].reshape(3, width, height)
    agent = flat[medium_size:medium_size + 2 * agents].reshape(2, agents)
    cells = flat[medium_size + 2 * agents:].reshape(2, agents)
    # Indices survive the float round-trip exactly: they are small integers and
    # float32 is exact well past any plausible field size.
    return Obs(medium=medium, agent=agent, cells=cells.astype(jnp.int32))


class DiePopulationEnv(Environment):
    """The DIE core wearing single-agent gymnax semantics.

    The population is presented as one agent with a ``[3 * max_agents]`` action
    vector and a scalar reward. That scalar is the **unshaped** total energy
    gain over alive agents -- the same quantity :func:`die.rl.ppo.make_evaluate`
    reports and the same one evolution maximizes -- so returns from this path,
    the vendored path and ``die train`` are all directly comparable.

    Auto-reset is left to :meth:`gymnax.environments.environment.Environment.step`
    rather than to :func:`die.jx.interface.step`, so the library's own episode
    semantics are what is being tested.

    Note that :meth:`rejax.algos.Algorithm.create` deep-copies its config, so
    the bank held here is duplicated once per configured run. It is a handful of
    field snapshots, but it is the reason ``algorithm.env.bank`` is never the
    object handed to this constructor.
    """

    def __init__(self, bank: EnvState, params: RejaxParams):
        self.bank = bank
        self._params = params

    @property
    def default_params(self) -> RejaxParams:
        return self._params

    @property
    def name(self) -> str:
        return 'DiePopulation-v0'

    def reset_env(self, key: jax.Array, params: RejaxParams):
        obs, state = reset(key, self.bank, params.env)
        return flatten_obs(obs), state

    def step_env(self, key: jax.Array, state: EnvState,
                 action: jax.Array, params: RejaxParams):
        world_action = action.reshape(NUM_ACTIONS, params.env.max_agents)
        scales = jnp.stack([params.action_scale,
                            params.action_scale,
                            params.deposit_scale])
        timestep, next_state = step_env(
            key, state, world_action * scales[:, None], params.env)

        reward = (timestep.reward * timestep.alive).sum()
        return (flatten_obs(timestep.obs), next_state, reward,
                timestep.truncated, {})

    def get_obs(self, state: EnvState, params: RejaxParams | None = None, key=None):
        params = params or self._params
        return flatten_obs(get_obs(state, params.env))

    def observation_space(self, params: RejaxParams) -> spaces.Box:
        # Chem is unbounded (AGENTS.md), so the box genuinely is.
        return spaces.Box(-jnp.inf, jnp.inf, (obs_size(params.env),), jnp.float32)

    def action_space(self, params: RejaxParams) -> spaces.Box:
        # [-1, 1] exactly: the env applies its own scaling, so the policy's
        # support and the action space match by construction. Mismatching them
        # is a documented way to make an algorithm silently unable to learn.
        return spaces.Box(-1., 1., (action_size(params.env),), jnp.float32)


class _ConvBody(nn.Module):
    """Shared trunk-and-gather, identical to :class:`die.rl.networks.ActorCritic`.

    Reproduced here rather than reused because rejax wants the actor and the
    critic as two separate modules with two separate parameter trees, which is
    also how its own MLP policy and ``VNetwork`` are arranged.
    """

    env_params: EnvParams
    features: int = 32
    num_blocks: int = 1
    kernel_size: int = 3
    hidden: int = 32

    @nn.compact
    def __call__(self, flat_obs: jax.Array) -> tuple[jax.Array, jax.Array]:
        """``[obs_dim] -> ([A, hidden] features, [A] alive mask)``."""
        obs = unflatten_obs(flat_obs, self.env_params)
        field = ConvTrunk(self.features, self.num_blocks, self.kernel_size)(obs.medium)
        gathered = field[obs.cells[0], obs.cells[1]]
        proprioception = obs.agent[OBS_AGENT_FOOD][:, None]
        x = jnp.concatenate([gathered, proprioception], axis=-1)
        x = nn.relu(nn.Dense(
            self.hidden, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2.)))(x))
        return x, obs.agent[1]


class ConvActor(nn.Module):
    """Convolutional policy in rejax's actor protocol.

    A diagonal Gaussian rather than the vendored path's tanh-squashed one,
    matching :class:`rejax.networks.GaussianPolicy`: rejax clips the sampled
    action to the action space instead of squashing it. That difference is left
    in deliberately -- it is part of what makes this an independent check, and
    the action space is ``[-1, 1]`` either way.
    """

    env_params: EnvParams
    features: int = 32
    num_blocks: int = 1
    kernel_size: int = 3
    hidden: int = 32
    init_log_std: float = -0.5

    def setup(self):
        self.body = _ConvBody(self.env_params, self.features, self.num_blocks,
                              self.kernel_size, self.hidden)
        self.mean_head = nn.Dense(NUM_ACTIONS,
                                  kernel_init=nn.initializers.orthogonal(0.01))
        self.log_std = self.param('log_std',
                                  nn.initializers.constant(self.init_log_std),
                                  (NUM_ACTIONS,))

    def _mean(self, flat_obs: jax.Array) -> jax.Array:
        """``[B, obs_dim] -> [B, 3 * A]``, laid out to match ``flatten``'s order."""
        def one(single):
            x, _ = self.body(single)
            # [A, 3] -> [3, A] -> flat, so the env's reshape recovers (dx, dy, deposit).
            return self.mean_head(x).T.ravel()

        return jax.vmap(one)(flat_obs)

    def _action_dist(self, flat_obs: jax.Array) -> distrax.Distribution:
        mean = self._mean(flat_obs)
        agents = self.env_params.max_agents
        scale = jnp.broadcast_to(jnp.exp(self.log_std)[:, None],
                                 (NUM_ACTIONS, agents)).ravel()
        return distrax.MultivariateNormalDiag(loc=mean, scale_diag=scale)

    def __call__(self, obs, rng):
        dist = self._action_dist(obs)
        action = dist.sample(seed=rng)
        return action, dist.log_prob(action), dist.entropy()

    def act(self, obs, rng):
        """Deterministic, unlike rejax's stock ``act``.

        Evaluation has to match :func:`die.rl.ppo.make_evaluate`, which acts on
        the distribution mean, or the two paths' reported returns differ by an
        exploration term rather than by anything about the algorithms.
        """
        return jnp.clip(self._mean(obs), -1., 1.)

    def action_log_prob(self, obs, rng):
        dist = self._action_dist(obs)
        action = dist.sample(seed=rng)
        return action, dist.log_prob(action)

    def log_prob_entropy(self, obs, action):
        dist = self._action_dist(obs)
        return dist.log_prob(action), dist.entropy()


class ConvCritic(nn.Module):
    """Joint value: the summed value of the agents that are actually alive.

    Masking here is legitimate even though the loss cannot mask -- this is the
    network's own output, not a reduction over rejax's batch, so it costs
    nothing and mirrors ``die.rl.ppo._to_world_level``.
    """

    env_params: EnvParams
    features: int = 32
    num_blocks: int = 1
    kernel_size: int = 3
    hidden: int = 32

    @nn.compact
    def __call__(self, flat_obs: jax.Array) -> jax.Array:
        body = _ConvBody(self.env_params, self.features, self.num_blocks,
                         self.kernel_size, self.hidden)
        head = nn.Dense(1, kernel_init=nn.initializers.orthogonal(1.))

        def one(single):
            x, alive = body(single)
            return (head(x)[..., 0] * alive).sum()

        return jax.vmap(one)(flat_obs)


class ConvPPO(PPO):
    """rejax's PPO with this project's convolutional policy substituted in.

    Only :meth:`create_agent` changes: the rollout, GAE, clipping, minibatching
    and optimizer are rejax's, untouched, which is the whole point.
    """

    @classmethod
    def create_agent(cls, config, env, env_params):
        kwargs = config.pop('agent_kwargs', {})
        kwargs.setdefault('features', 16)
        kwargs.setdefault('num_blocks', 1)
        kwargs.setdefault('kernel_size', 3)
        kwargs.setdefault('hidden', 32)
        return {'actor': ConvActor(env_params.env, **kwargs),
                'critic': ConvCritic(env_params.env, **kwargs)}


def make_rejax_ppo(bank: EnvState,
                   params: RejaxParams,
                   *,
                   conv: bool = True,
                   eval_bank: EnvState | None = None,
                   **config) -> PPO:
    """Configures a rejax PPO run against the DIE core.

    :param conv: use this project's convolutional trunk (the algorithm
        cross-check). ``False`` uses rejax's stock MLP over the flattened field,
        which measures what the convolutional prior buys instead.
    :param eval_bank: worlds to evaluate on. rejax evaluates on the environment
        it trains on, which measures on-policy drift rather than generalization;
        pointing it at a disjoint bank makes its reported return mean the same
        thing :func:`die.rl.ppo.make_evaluate` reports, which is the only reason
        the two are comparable at all.
    """
    env = DiePopulationEnv(bank, params)
    algorithm = ConvPPO if conv else PPO
    config.setdefault('learning_rate', 1e-2)   # see HyperParams.learning_rate
    config.setdefault('max_grad_norm', 0.5)
    ppo = algorithm.create(env=env, env_params=params, **config)

    if eval_bank is None:
        return ppo

    held_out = DiePopulationEnv(eval_bank, params)

    def eval_callback(algo, train_state, key):
        return evaluate(algo.make_act(train_state), key, held_out, params,
                        num_seeds=8, max_steps_in_episode=params.max_steps_in_episode)

    return ppo.replace(eval_callback=eval_callback)


@partial(jax.jit, static_argnames=('algorithm',))
def train_rejax(algorithm: PPO, key: jax.Array):
    """``(train_state, evaluation)``, where evaluation is ``(lengths, returns)``."""
    return algorithm.train(key)
