# Distributed Intelligence Environment -- *or just DIE*

**DIE** is an Artificial Life project aimed at reproducing emergence of distributed intelligence under environmental pressures.

It implements nature-like **[Gym](https://github.com/Farama-Foundation/Gymnasium) environment** with essential pressures for foraging, feeding, *not-dying*, together with **distributed agents** for solving it.

TODO: GIF of random and physarum and gradient agents

### Quick Start

TODO: two agents from GIF

### Features

- **Artificial Life system** with environmental pressures and agents' needs.
- **Natural rewards** for effective feeding and staying alive.
- **Gym environment** in the format of multi-channel 2D data arrays. Agents can sense (read) certain channels and can act (write to) other channels.
- **Agent implementations** for Physarum, Gradient, Constant, and Random behaviors.
- **Controlled dynamics** of the environment.
- **Dynamic visualisation** of environments (see GIFs above).
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
- **Free Energy Principle.**
  It refers to a theory describing how intelligent behavior arises from the prior needs (aims) and predictive capabilities of agents.


### Milestones

1. [x] Create basic environment with Random agents.
2. [x] Reproduce Physarum agent behavior.
3. [ ] Implement Neural Cellular Automata based on evolutionary approach.
4. [ ] Implement Neural Cellular Automata based on Free Energy Principle models. 

### Literature & Sources

- Physarum
- Neural CA
- FEP
- Michael Levin ideas
- OpenAI evolution

*PS: Possibly the philosophical setpoint for this project is to create intelligence that would be able to learn how to die with dignity.*