# Distributed Intelligence Environment — *or just DIE*

**DIE** is an Artificial Life project aimed at reproducing emergence of distributed intelligence under environmental pressures using learning cellular automata models.

----

It implements nature-like **[Gym](https://github.com/Farama-Foundation/Gymnasium) environment** with essential pressures for foraging, feeding, *not-dying*, together with **distributed agents** for solving it. Agents are a kind of learning cellular automata, hence they're *cooperative* (share the policy), observation and action spaces are *continuous*, and computations are *vectorized* for efficiency.

Here how 2 agents look like running in the environment: simulating *brownian motion* and *slime mold* behavior.

![Brownian motion agent](img/BrownianAgent.gif "Brownian motion agent animation")

Left pane shows environment with food (green), agents (red) and their pheromone (blue). Right pane shows only movement traces of the agents. You can see that in time agents gradually consume the food.

![Physarum agent](img/PhysarumAgent.gif "Physarum agent animation")

This is a Physarum (slime mold) agent that communicates found food by releasing the pheromone and moves towards zones with more pheromone. Thanks to this information sharing between particles Physarum agent is much more efficient than Brownian agent in finding and consuming food.

### Quick Start

Minimal example for running an agent in the environment with default settings:
```python
def run_minimal(agent: Agent, agent_ratio=0.1, field_size=(256, 256), iters=300):
  # Setup the environment
  dynamics = Dynamics(init_agent_ratio=agent_ratio)
  env = Env(field_size, dynamics)
  plotter = InteractivePlotter.get(env, agent)

  total_reward = 0
  obs = env._get_current_obs
  for i in (pbar := tqdm.trange(iters)):
    # Step: action & observation
    action = agent.forward(obs)
    obs, reward, _, _, stats = env.step(action)
    total_reward += reward
    # Visualisation & logging
    pbar.set_postfix(total_reward=np.round(total_reward, 3))
    plotter.draw()
```

To reproduce the GIF-s above you can run it with following agents:
```python
run_minimal(BrownianAgent(move_scale=0.01))

run_minimal(PhysarumAgent(max_agents=256*256,
                          scale=0.007,
                          turn_angle=30,
                          sense_offset=0.04))
```

More example, including this one, can be found in `examples` directory.

### Features

- **Artificial Life system** with environmental pressures and agents' needs.
- **Natural rewards** for effective feeding and staying alive.
- **Gym environment** in the format of multi-channel 2D data arrays. Agents can sense (read) certain channels and can act (write to) other channels.
- **Agent implementations** for Physarum, Gradient, Constant, and Brownian motion behaviors.
- **Controlled dynamics** of the environment.
- **Dynamic visualisation** of environments (see GIFs above).
- (TBD) *Learning agents*.
- (TBD) *Flexible environment builder* with arbitrary data channels.
- (TBD) *Easily reproducible experiments* thanks to saving & loading of experiment setup (all parameters of environments & agents)

The environmental pressure is designed in such a way so that agents must learn to collectively predict environmental dynamics to be able to thrive. Otherwise their resources deplete, and they die.

### Background

The project embodies several sources of inspiration:
- **Embodied Intelligence** assumption.
  It states that flexible and general intelligence arises only in agents with certain needs embedded in environments with certain pressures.
- **Distributed intelligence** assumption. 
  Every intelligence as a distributed intelligence, be it a collection of cells, neural circuit, ant colony, or human community. Centralised top-down control is only a by-product of bottom-up cooperation.
- **Cellular Automata.**
  It is a working model demonstrating how complex coordinated behavior arises from local interactions.
- **Free Energy Principle & Active Inference.**
  It refers to a theory describing how intelligent behavior arises from the prior needs (aims) and predictive capabilities of agents.


### Milestones

1. [x] Basic environment with Brownian motion agent.
2. [x] Physarum agent.
3. [ ] Neural Cellular Automata agent based on evolutionary approach.
4. [ ] Neural Cellular Automata agent based on Active Inference. 

### Literature & Sources

- Mordvintsev, A. et al. (2020) Growing Neural Cellular Automata. Distill. [Link to semantic scholar.](https://api.semanticscholar.org/CorpusID:213719058)
- Parr, T., Pezzulo, G., & Friston, K.J. (2022). Active Inference. [Link to semantic scholar.](https://api.semanticscholar.org/CorpusID:247833519)
- M. Levin work, for example: Levin, M. (2012). Morphogenetic fields in embryogenesis, regeneration, and cancer: Non-local control of complex patterning. Bio Systems, 109 3, 243-61. [Link to semantic scholar.](https://api.semanticscholar.org/CorpusID:767009)
- Jones, Jeff Dale. (2010) Characteristics of Pattern Formation and Evolution in Approximations of Physarum Transport Networks. Artificial Life 16: 127-153. *[Link to semantic scholar.](https://api.semanticscholar.org/CorpusID:7511776Physarum)*
- Salimans, T., Ho, J., Chen, X., & Sutskever, I. (2017). Evolution Strategies as a Scalable Alternative to Reinforcement Learning. ArXiv, abs/1703.03864. [Link to semantic scholar.](https://api.semanticscholar.org/CorpusID:11410889)
