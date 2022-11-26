
"""
Plan
- [x] some basic data init ENVs per channels for tests YET minimal
- [x] test data init: plot channels

Necessary core
- [x] agent move async
- [x] fix MockConstAgent
- [x] test agent moving
- [x] make buffered update more efficient
- [ ] agent moving separate coords separately!

- [x] some boundary conditions
- [ ] get some avg stats (like a reward)
- [ ] agent lifecycle
- [x] medium diffusion & decay
- [ ] medium food dynamics -- const | random add
- [ ] agent action rescaling & saturation -- on which side? Env OR Agent?
      well, must be Env to make hacks "unaccessible" in principle
- [ ] ...
- [ ] test agent feeding & life wrap
- [ ] test deposit with communication

Important core enhancements:
- [ ] always update position with continous data channels in `agents` array
      i.e. in coordinates store only approximations; but don't lose info on precision.

Aux: plotting
- plotting NB: all tests are isolated visual cases, really
  maybe with some statistical tests *over the image*
- [ ] aspect ratio
- [ ] subplot with agent stats
- [ ] advanced dynamic vis (see https://docs.xarray.dev/en/stable/user-guide/plotting.html)

Well packaged
- [ ] make tests out of visual examples

"""
