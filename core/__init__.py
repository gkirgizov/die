
"""
Plan
- [ ] Stage 0.1: const moving
- [ ] Stage 0.2: classic physarum? without feeding/dying; just moving
- [ ] Stage 0.3: basic intelligance: some obviously working learning agent (e.g. evolutionary)
-- can write an article here; publish as a new Env for ALife research
- [ ] Stage 0.4: RL
- [ ] Stage 0.5: FEP

Necessary core
- [x] some basic data init ENVs per channels for tests YET minimal
- [x] test data init: plot channels

- [x] agent move async
- [x] fix MockConstAgent
- [x] test agent moving
- [x] make buffered update more efficient
- [x] 0.1: agent moving separate coords separately!
    - [x] add agent info array with purely informational unordered unindexed array
    - [x] make & test DataInitializer for new agents, subst usages of its methods
        - [x] agents from medium MAPPING
        - [x] get_current_obs
        - [x] modify simple agents
        - [x] check postprocess action in Agent
        - [x] test mapping & DataInit
    - [x] return agents map to field? make a separate method for mapping them
        - [x] consider alive agents only
        - [x] --think if needed doing this in-the-loop ?
        - [x] rewrite other methods:
        - [x] agents lifecycle & deposit
        - [x] agents feed & consume
    Bugs
    - [x] sort out why agents are ...lost?...
    - [ ] BUG with agents feeding somehow unconstrained growth from where? from collisions?
          what can it be? possibly it's data collisions in 'feeding' step.
          try rewriting it step-by-step

- [x] plotting agents array with food etc.
- [ ] plotting LIVE
- [ ] medium food dynamics -- const | random add
- [x] agent action cost -- not obvious & important

- [x] some boundary conditions
- [x] medium diffusion & decay
- [ ] 0.2: come up with Physarum kernel
      some summing-up kernel that determines direction?
      like *chemical-weighted sum of coordinates*? then we get direction vector.
      ah, I see! that's like *a specific case of general convolving agent* with const-linear weigth mask!
      - [ ] understand why gradient agent doesn't return the characteristic pattern of physarum?
            ?add some inertia coefficient
            ?or possibly add requirement for *continuity* of the *derivative* (1st or 2nd order)
      - [ ] reproduce physarum environment with basics (no food, just inertia)
      - [ ] add some *native* collision resolution -- natural stochasticity of agents

      - [ ] setup clear Reward & performance characteristics (allmost done)
      - [ ] setup baseline Dynamics hyperparams
            criteria is some baseline performance of Random agent

-- does that count as a paper on generalisation of original Physarum paper? making it continuous?
   how about Lenia?
   sounds like not quite; small incremental enhancement.

- [ ] get some avg stats (like a reward)
- [ ] ...
- [ ] test deposit with communication

Important core enhancements:
- [ ] always update position with continous data channels in `agents` array
      i.e. in coordinates store only approximations; but don't lose info on precision.
- [ ] **implement separate 2-way mapper for such indexers**

Aux: plotting
- plotting NB: all tests are isolated visual cases, really
  maybe with some statistical tests *over the image*
- [ ] aspect ratio
- [ ] subplot with agent stats
- [ ] subplot with agent sense neighbourhoods?
- [ ] advanced dynamic vis (see https://docs.xarray.dev/en/stable/user-guide/plotting.html)

Well packaged
- [ ] make tests out of visual examples
- [ ] implement gym.Env API fully
tests
- [ ] make test for double-way mapping agents --> medium and back



"""
