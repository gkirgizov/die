
"""
Plan
- [x] Stage 0.1: const moving
- [x] Stage 0.2: classic physarum? without feeding/dying; just moving
- [ ] Stage 0.3: basic intelligence: some obviously working learning agent (e.g. evolutionary)
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
    - [x] only alive selection -- is it working?
    - [x] BUG with agents feeding somehow unconstrained growth from where? from collisions?
          what can it be? possibly it's data collisions in 'feeding' step.
          try rewriting it step-by-step

- [x] plotting agents array with food etc.
- [x] plotting LIVE

- [x] agent action cost -- not obvious & important
- [x] some boundary conditions
- [x] medium diffusion & decay
- [x] rewrite Agents to work on 'agents' array with FieldMapping
        because now i can't store direction of agents fixed by agent id
        --> seems like it's simpler to just select gradient field and act on agents array.
        that is: change Action format from Field to Agent -> it's far more logical
        possibly this will simplify Env logic
        - [x] agent_move
        seems like will need single mapping from Action array to Medium by Agent coords
        - [x] agent_act_on_medium will be like agents_to_medium
    - [x] add medium RANDOMIZATION of agents <-- will be resolved by previous point
          because randomization in agents is still liable to *aliasing*
- [ ] 0.2: come up with Physarum kernel
      some summing-up kernel that determines direction?
      like *chemical-weighted sum of coordinates*? then we get direction vector.
      ah, I see! that's like *a specific case of general convolving agent* with const-linear weigth mask!
      - [x] understand why gradient agent doesn't return the characteristic pattern of physarum?
      - [x] add some inertia coefficient
          (a) inertia on gradient directly ~= inertia on direction
          (b) or possibly add requirement for *continuity* of the *derivative* (1st or 2nd order)
      - [x] try adding noise
            -- it doesn't change a lot; but be careful with a scale.
            Clusters still form, but bigger
      - [x] try direction-only (normalized) gradient
      - [x] experiment 1 on medium food dynamics -- const | random add
          give agents an aim for movement
          --> obviously with gradient they try repeating the moving pattern.
      - [x] understand how coagulation works, how it depends on:
          - number of agents
          - agent speed
          - env character
          --> see results below on living thing 1
      - [x] reproduce physarum environment with basics (no food, just inertia)
            differences:
            - sense neighbourhooud is immediate not further
            Need make agents:
            - always moving (independent scale)
            - always turning by specific amount (gradient inertia doesn't work)
      - [x] add some *native* collision resolution -- natural stochasticity of agents

      - [ ] Exp check that order of operations is logical:
            - Deposit happens after medium diffuse & agent sensing & grad compute
              so that they don't sense their own trails so much.
            - Actually, Move must happen after Deposit => sense will be more pure
      - [ ] make agent deposit only on successfull turn
      - [x] create gradient + offset agent
            separate Blind zone, Discretization, Sense offset from basic GradAgent
            why? allow same gradient agent but with additional conditions
      Fixes
      - [ ] align sense mask on Medium with Agent's sense mask
      - [ ] fix "Jittering" agents
        h1: conflicting write to 'agents' channel with duplicate index

      Experiment Log. Modifications (a-d):

      Experiment 1 results on gradient agent:
      -> gradient scale + noise don't work for physarum because
      (a) noise forgets direction inertia (so cells don't tend to follow their routes)
      (b) gradient scale is still gradient-dependent, so coagulation is still there

      So, overall gradient strategy is like conservative exploitation strategy.
      It depends on ratio of environment decay/diffuse and agent deposit.
      In general agents "trail" to their own deposits, coagulating fast.

      Experiment 2 results:
      (c) discrete turn:
          Agents can begin to "spin" but still return to their own gradient.
      (d) blind zone of gradient:
          Agents are less prone to spinning, but they coagulate noisely, still well.
      (e) sense offset:
          Really helps. Agents bein to move "ahead".
          *Maybe* it is a quazi-prediction of where the food will be available?
          - [ ] EXP NOTE maybe learning agents will come up with something like that?

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
- [x] aspect ratio
- [x] subplot with agent stats
- [ ] subplot with agent sense neighbourhoods?
- [ ] advanced dynamic vis (see https://docs.xarray.dev/en/stable/user-guide/plotting.html)

Well packaged
- [ ] make tests out of visual examples
- [ ] implement gym.Env API fully
tests
- [ ] make test for double-way mapping agents --> medium and back

----------

Living thing 1 (Gradient coagulating agent):
- scale controls characteristic size of "cells"
- inertia controls their "responsiveness" and how fast they change/move/merge
- they are stable and not-moving in absence of nearby agents
- sometimes can get almost pure stable rings of characteristic size (inertia was 0.9)
- at inertia 0.2 get more dynamic form change: got several metamorphoses
  from metastable smooth "triangle" circles to several embedded circles.

Env params:
- higher diffuse => better communication; less diffuse => slower communication & convergence

"""
