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

### Reinforcement learning

`die rl` trains a policy with **PPO** on the same JAX core. The difference is
how much of the environment's signal is used: evolution scores a whole episode
with one number, while RL consumes *every agent's per-step energy change* —
roughly 10⁵ times more signal per unit of simulation. That is what makes a
network of ~88k parameters trainable here, against the 162-parameter kernel
evolution can search.

```bash
die rl --iterations 200                        # PPO on the vectorized core
die rl --reward-share 0                        # selfish agents: does communication survive?
die rl --food-noise 0.25 --comm-noise 0.05     # a world that doesn't sit still
die replay runs/latest/checkpoints/final.msgpack
```

A second, independent PPO implementation lives behind the same environment:
`die/rl/rejax_backend.py` runs it through [rejax](https://github.com/keraJLi/rejax),
so the vendored loss has a witness that shares none of its code.

```bash
uv sync --extra train-rejax   # distrax pulls a TFP nightly; kept out of `train`
uv run pytest tests/behavior/test_rl_rejax_crosscheck.py -m slow
```

It is a cross-check rather than an alternative — rejax speaks single-agent
gymnax, so it can only consume the joint objective. What it found is recorded in
[docs/JOURNAL.md](docs/JOURNAL.md), and it was not what the design predicted:
the convolutional prior dominates everything (an MLP on the same loss never
approaches the baseline), while the per-agent objective — the whole reason for
vendoring a PPO — is currently indistinguishable from the joint one.

It is one policy instantiated across a population of locations, not a
multi-agent system: a convolution *is* a shared policy over local receptive
fields. `--reward-share` sets whether an agent is rewarded for its own foraging
or the population's — the knob that decides whether depositing pheromone can
ever pay off. Findings and calibrations live in [docs/JOURNAL.md](docs/JOURNAL.md).

Every run is tracked with **MLflow**: full config, per-generation fitness, model checkpoints, and rollout animations of the current best agent.

```bash
mlflow ui --backend-store-uri sqlite:///runs/latest/mlflow.db
```

## Benchmarking agents

Every agent — scripted, evolved, and RL-trained — is scored on one common
footing: **total unshaped energy gain over a fixed horizon**, the quantity all
three paths in this repo already agree on.

```bash
uv run python benchmarks/train_bench.py --out models    # trains what needs training
uv run python benchmarks/bench_agents.py --out results  # scores everyone
uv run python benchmarks/plot_bench.py --results results
```

Agents are compared across a **family of generated environments** — one axis
varied at a time from a fixed baseline, each at six world seeds:

| axis | what it varies | values |
|---|---|---|
| `food_scale` | abundance (same patches, more or less food) | 0.25 – 4.0 |
| `food_octaves` | patchiness (broad basins vs fine speckle) | 2 – 32 |
| `food_noise` | resource that will not hold still | 0.1 – 0.5 |
| `comm_noise` | corruption of the chemical channel | 0.02 – 0.1 |
| `init_agent_ratio` | crowding | 0.05 – 0.4 |
| `regrow_rate` | resource that **grows back** | 0.01 – 0.04 |
| `respawn_rate` | resource that **reappears where it was exhausted** | 0.002 – 0.01 |
| `wave_decay` | resource that **moves** under the agents | 0.25 – 0.75 |

The last three are the dynamic axes, and they are three mechanisms rather than
one because each does something the others cannot. The wave *moves* resource
without creating any. Logistic regrowth renews it in place but provably cannot
revive a cell foraged to exactly zero, since the term is proportional to what is
already there. Only stochastic respawn creates resource where there is none.

Renewal rates are calibrated against measured consumption, not chosen: under
physarum over 40 steps the non-renewable baseline runs 16.15 → 10.18 total food,
and the middle rate on each axis lands within 1% of where it started. An
uncalibrated rate would just make the world richer, which is the failure mode
already recorded for food noise.

Learned agents are trained twice — on the static world and on a **renewable**
one (regrowth + respawn) — so the table separates "cannot handle a changing
resource" from "was never shown one".

The family's seeds are disjoint from both the training and evaluation seeds, so
every number is out-of-sample: the benchmark measures **generalization to unseen
worlds**, not recall of a memorized one.

Two things worth knowing before reading results:

- **The bar is doing nothing, not random.** Agents pay to move, so `null` is a
  genuinely strong baseline — it has already caught configurations where
  stillness out-earns foraging and a learner looked successful having learned
  nothing.
- **Never read a single seed.** Held-out return on this benchmark varies by more
  than the effects being measured; two conclusions have been retracted for this
  (see [docs/JOURNAL.md](docs/JOURNAL.md)). Trained agents are therefore stored
  and reported at three training seeds each, and every figure shows the spread.

### Visualising performance

`plot_bench.py` writes two figures into `results/`:

- **`heatmap.png`** — agent × scenario, coloured by *advantage over doing
  nothing on that same scenario*. Diverging, centred at zero: blue beats
  inaction, red loses to it, gray ties. Reading raw return instead would hide
  the sign — a return of +2 looks respectable and is actually a loss.
- **`axis_curves.png`** — one panel per environment axis, showing how each agent
  degrades along it, with a band for the spread across world seeds.

Both read `results/benchmark.csv` (one tidy row per agent × scenario × seed), so
figures can be reworked without re-running the sweep.

For a *qualitative* look at any single policy, replay a checkpoint into the live
view or a GIF — this runs the xarray engine, so it works for every agent type:

```bash
die replay models/jx_nca_seed0.msgpack --field-size 16 --agent-ratio 0.2 --steps 400
die replay models/rl_ppo_seed1.msgpack --field-size 16 --agent-ratio 0.2 \
    --record out/ppo.gif --headless
```

Pass `--agent-ratio 0.2` to match the benchmark's crowding; `die replay`
otherwise defaults to 0.1 and a different dynamics preset, so a rollout left at
the defaults is not the world the reported numbers came from.

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
