"""Reinforcement learning on the JAX env core.

Where ``die.jx`` is the environment -- physics, state, the gymnax-shaped
interface -- this package is the learner: observation normalization, policy
networks, and the training loop itself.

The framing that shapes everything here: there is **one** policy, a
convolutional field operator, instantiated across a population of locations.
Not N interacting agents with their own parameters, so none of the usual
multi-agent machinery applies; but not plain batching either, because the
agents are coupled through the medium they share. What survives that
distinction is credit assignment -- whether an agent is rewarded for its own
foraging or the population's -- which is a parameter here rather than an
assumption.

Nothing in ``die.jx`` imports from this package; the arrow points one way.
"""
