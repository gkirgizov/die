# Distributed Intelligence Environment — *or just DIE*

**DIE** is an Artificial Life project aimed at reproducing emergence of distributed intelligence under environmental pressures using learning cellular automata models.

----

It implements nature-like **[Gym](https://github.com/Farama-Foundation/Gymnasium) environment** with essential pressures for foraging, feeding, *not-dying*, together with **distributed agents** for solving it. Agents are a kind of learning cellular automata, hence they're *cooperative* (share the policy), observation and action spaces are *continuous*, and computations are *vectorized* for efficiency.

Here how 2 agents look like running in the environment: simulating *brownian motion* and *slime mold* behavior.

![Brownian motion agent](img/BrownianAgent.gif "Brownian motion agent animation")

Left pane shows environment with food (green), agents (red) and their pheromone (blue). Right pane shows only movement traces of the agents. You can see that in time agents gradually consume the food.

![Physarum agent](img/PhysarumAgent.gif "Physarum agent animation")

This is a Physarum (slime mold) agent that communicates found food by releasing the pheromone and moves towards zones with more pheromone. Thanks to this information sharing between particles Physarum agent is much more efficient than Brownian agent in finding and consuming food.

## Installation

Requires Python ≥ 3.12. Using [uv](https://docs.astral.sh/uv/):

```bash
uv sync --all-extras        # everything, incl. the training stack
uv sync                     # core only (no evotorch/mlflow)
```

## Quick Start

The `die` CLI is the main entry point:

```bash
die run --agent physarum                    # live view of a Physarum agent
die run --agent brownian --headless --seed 42
die record --agent physarum --out samples/  # save a GIF, no display needed
die train --generations 100                 # evolve a Neural CA agent
die replay runs/latest/checkpoints/best.pt  # watch a trained agent
die --help                                  # all commands & options
```

The same, minimal, from Python (see `examples/minimal_run.py`):

```python
from die import AgentId, DynamicsId, Simulation, TqdmObserver, make_agent, make_env
from die.view import InteractiveViewObserver

env = make_env(DynamicsId.st_perlin, field_size=256, agent_ratio=0.1)
agent = make_agent(AgentId.physarum, field_size=256)
sim = Simulation(env, agent, seed=42,
                 observers=[TqdmObserver(), InteractiveViewObserver()])
result = sim.run(300)
print(result.total_reward)
```

The `Simulation` runner owns the agent–env loop; presentation is pluggable
via observers (`InteractiveViewObserver` for a live matplotlib view,
`AnimationRecorder` for headless GIF/MP4 recording, `MetricsObserver` for
in-memory stats).

## Training & Observability

`die train` evolves a **Neural Cellular Automata** agent with evolution
strategies ([evotorch](https://evotorch.ai) PGPE/CMA-ES behind a swappable
`SearcherBackend` seam):

```bash
die train --dynamics st-perlin-wide --generations 100 --popsize 10 \
          --field-size 96 --seed 1 --out runs/exp1
```

Every run is tracked with **MLflow** (local SQLite backend by default):
full config as params, per-generation fitness metrics, periodic model
checkpoints and rollout animations of the current best agent as artifacts.

```bash
mlflow ui --backend-store-uri sqlite:///runs/exp1/mlflow.db
```

## Features

- **Artificial Life system** with environmental pressures and agents' needs.
- **Natural rewards** for effective feeding and staying alive.
- **Gym environment** in the format of multi-channel 2D data arrays. Agents can sense (read) certain channels and can act (write to) other channels.
- **Agent implementations** for Physarum, Gradient, Constant, and Brownian motion behaviors.
- **Learning agents**: evolutionary Neural Cellular Automata (`die train`).
- **Controlled dynamics** of the environment.
- **Dynamic visualisation** of environments (see GIFs above), recorded with `die record`.
- **Reproducible experiments**: seeded runs, agent serialization, MLflow tracking.

The environmental pressure is designed in such a way so that agents must learn to collectively predict environmental dynamics to be able to thrive. Otherwise their resources deplete, and they die.

## Development

```bash
uv sync --all-extras --dev
uv run pytest                # fast suite: unit + behavioral invariants
uv run pytest -m slow        # learning smoke tests (full training pipeline)
uv run ruff check .
```

Tests include high-level behavioral assertions — e.g. Physarum must
out-forage Brownian motion by a calibrated margin, fitness must improve
over a short evolution run — see `tests/behavior/`.

Research notes and the modernization roadmap (JAX port, evosax) live in
[docs/NOTES.md](docs/NOTES.md).

## Background

The project embodies several sources of inspiration:
- **Embodied Intelligence** assumption.
  It states that flexible and general intelligence arises only in agents with certain needs embedded in environments with certain pressures.
- **Distributed intelligence** assumption. 
  Every intelligence as a distributed intelligence, be it a collection of cells, neural circuit, ant colony, or human community. Centralised top-down control is only a by-product of bottom-up cooperation.
- **Cellular Automata.**
  It is a working model demonstrating how complex coordinated behavior arises from local interactions.
- **Free Energy Principle & Active Inference.**
  It refers to a theory describing how intelligent behavior arises from the prior needs (aims) and predictive capabilities of agents.

## Milestones

1. [x] Basic environment with Brownian motion agent.
2. [x] Physarum agent.
3. [x] Neural Cellular Automata agent based on evolutionary approach.
4. [ ] Neural Cellular Automata agent based on Active Inference. 

## Literature & Sources

- Mordvintsev, A. et al. (2020) Growing Neural Cellular Automata. Distill. [Link to semantic scholar.](https://api.semanticscholar.org/CorpusID:213719058)
- Parr, T., Pezzulo, G., & Friston, K.J. (2022). Active Inference. [Link to semantic scholar.](https://api.semanticscholar.org/CorpusID:247833519)
- M. Levin work, for example: Levin, M. (2012). Morphogenetic fields in embryogenesis, regeneration, and cancer: Non-local control of complex patterning. Bio Systems, 109 3, 243-61. [Link to semantic scholar.](https://api.semanticscholar.org/CorpusID:767009)
- Jones, Jeff Dale. (2010) Characteristics of Pattern Formation and Evolution in Approximations of Physarum Transport Networks. Artificial Life 16: 127-153. *[Link to semantic scholar.](https://api.semanticscholar.org/CorpusID:7511776Physarum)*
- Salimans, T., Ho, J., Chen, X., & Sutskever, I. (2017). Evolution Strategies as a Scalable Alternative to Reinforcement Learning. ArXiv, abs/1703.03864. [Link to semantic scholar.](https://api.semanticscholar.org/CorpusID:11410889)
