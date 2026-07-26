"""Pure-JAX core of the environment, used by the evolutionary training path.

This package mirrors the physics of :mod:`die.env` as fixed-shape, side-effect
free array functions so that a whole population of candidate agents can be
evaluated with a single ``vmap``-ed, ``scan``-ed, jitted call.

It deliberately *coexists* with the xarray env rather than replacing it: the
xarray env remains the reference implementation and the substrate for
interactive simulation, scripted agents (physarum, gradient) and the view
layer. Cross-implementation parity is pinned by tests.

Nothing outside this package (plus the evosax backend and jax trainer) imports
it, so the core package stays importable without jax installed.
"""

from die.jx.state import EnvConfig, EnvState

__all__ = ['EnvConfig', 'EnvState']
