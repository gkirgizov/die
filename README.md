# Distributed Intelligence Environment — *or just DIE*

**DIE** is an Artificial Life project aimed at reproducing emergence of distributed intelligence under environmental pressures using learning cellular automata models.

The name is also the objective function: whatever fails to cooperate — dies.

----

It implements nature-like **[Gym](https://github.com/Farama-Foundation/Gymnasium) environment** with essential pressures for foraging, feeding, *not-dying*, together with **distributed agents** for solving it. Agents are a kind of learning cellular automata, hence they're *cooperative* (share the policy), observation and action spaces are *continuous*, and computations are *vectorized* for efficiency.

Here how 2 agents look like running in the environment: simulating *brownian motion* and *slime mold* behavior.

![Brownian motion agent](img/BrownianAgent.gif "Brownian motion agent animation")

Left pane shows environment with food (green), agents (red) and their pheromone (blue). Right pane shows only movement traces of the agents. You can see that in time agents gradually consume the food.

![Physarum agent](img/PhysarumAgent.gif "Physarum agent animation")

This is a Physarum (slime mold) agent that communicates found food by releasing the pheromone and moves towards zones with more pheromone. Thanks to this information sharing between particles Physarum agent is much more efficient than Brownian agent in finding and consuming food.

## Quick Start

Requires Python ≥ 3.12 and [uv](https://docs.astral.sh/uv/):

```bash
uv sync --all-extras

die run --agent physarum                    # watch a slime mold emerge, live
die record --agent brownian --out samples/  # save a GIF, no display needed
die train --generations 100                 # evolve a Neural CA agent from scratch
die replay runs/latest/checkpoints/best.pt  # watch what evolution came up with
```

The same from Python — the whole loop is three objects (see `examples/minimal_run.py`):

```python
from die import AgentId, DynamicsId, Simulation, make_agent, make_env
from die.view import InteractiveViewObserver

env = make_env(DynamicsId.st_perlin, field_size=256)
agent = make_agent(AgentId.physarum, field_size=256)
result = Simulation(env, agent, seed=42,
                    observers=[InteractiveViewObserver()]).run(300)
```

## Learning

`die train` evolves a **Neural Cellular Automata** agent with evolution strategies — no gradients, just variation and selection: kernels that feed their agents get to stay. Fitting, for a project called DIE.

Search runs on **JAX + evosax** (PGPE with ClipUp) by default: the environment has a second, pure-functional implementation in `die/jx/`, so an entire population is evaluated in one jitted, vectorized call — ~30x faster per generation than the torch path on CPU. The original **evotorch** backend is still there behind the same `SearcherBackend` seam:

```bash
die train --backend evosax     # JAX (default), checkpoints to .msgpack
die train --backend evotorch   # torch reference path, checkpoints to .pt
```

Either way, `die replay <checkpoint>` brings the trained agent back into the interactive environment.

Every run is tracked with **MLflow**: full config, per-generation fitness, model checkpoints, and rollout animations of the current best agent.

```bash
mlflow ui --backend-store-uri sqlite:///runs/latest/mlflow.db
```

## Features

- **Artificial Life system** with environmental pressures and agents' needs.
- **Natural rewards** for effective feeding and staying alive.
- **Gym environment** in the format of multi-channel 2D data arrays. Agents can sense (read) certain channels and can act (write to) other channels.
- **Agent implementations** for Physarum, Gradient, Constant, and Brownian motion behaviors.
- **Learning agents**: evolutionary Neural Cellular Automata, evolved on a vectorized JAX core.
- **Controlled dynamics** of the environment.
- **Dynamic visualisation** of environments (see GIFs above), recorded with `die record`.
- **Reproducible experiments**: seeded runs, agent serialization, MLflow tracking.

The environmental pressure is designed in such a way so that agents must learn to collectively predict environmental dynamics to be able to thrive. Otherwise their resources deplete, and they die.

## Development

```bash
uv sync --extra train --extra train-evotorch --extra video --dev
uv run pytest                # fast: unit + behavioral invariants
uv run pytest -m slow        # the full training pipeline
uv run ruff check .
```

On an NVIDIA machine, add `--extra train-cuda` for GPU-accelerated training
(linux x86_64; use `jax[cuda13]` if your driver is CUDA 13). `die train` prints
the JAX device it resolved, so it is obvious when a run silently fell back to CPU.
Apple GPUs are not supported — [docs/NOTES.md](docs/NOTES.md) records why.

Behavioral claims are tested, not just told: Physarum must out-forage Brownian motion by a calibrated margin, and a short evolution run must actually improve fitness (`tests/behavior/`). The architecture map lives in [AGENTS.md](AGENTS.md); research notes and the roadmap in [docs/NOTES.md](docs/NOTES.md).

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
