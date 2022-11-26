
"""
Plan
- [x] some basic data init ENVs per channels for tests YET minimal
- [x] test data init: plot channels
- [x] make buffered update more efficient

core
- [x] agent move async
- [x] fix MockConstAgent
- [x] test agent moving

- [ ] some boundary conditions
- [ ] get some avg stats (like a reward)
- [ ] test agent lifecycle
- [ ] test medium diffusion
- [ ] test medium food dynamics
- [ ] agent action rescaling & saturation -- on which side? Env OR Agent?
      well, must be Env to make hacks "unaccessible" in principle
- [ ] ...
- [ ] test agent feeding & life cycle
- [ ] test deposit with communication

plotting
- [ ] plotting NB: all tests are isolated visual cases, really
      maybe with some statistical tests *over the image*
- [ ] aspect ratio
- [ ] subplot with agent stats
- [ ] advanced dynamic vis (see https://docs.xarray.dev/en/stable/user-guide/plotting.html)

"""
