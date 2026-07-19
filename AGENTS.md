# AGENTS.md — a working map of DIE

Orientation for anyone (human or agent) doing R&D here. It deliberately stays
at the level of architecture, invariants and conventions — the things that
save time and don't go stale. For research direction and roadmap, see
[docs/NOTES.md](docs/NOTES.md).

## What this is

An Artificial Life research project: cooperative cellular-automata agents
forage in a 2D multi-channel medium under survival pressure; neural agents
are evolved, not backpropagated. It is research code — correctness of the
physics and reproducibility of experiments matter more than product polish.

## Architecture in one breath

`types → env → agents → sim → view / train → cli`, one directory each under
`src/die/`. The dependency arrow only points left: the env knows nothing of
presentation, the sim layer knows nothing of matplotlib, the CLI contains no
logic.

- **World state is `xarray.DataArray`**: the medium is `(channel, x, y)` with
  float coords on [0, 1]; agents are a flat fixed-size `(channel, index)`
  array. Channel names live in `die/types.py::DataChannels`.
- **`env.py`** is gym-*shaped* (5-tuple `step`, seeded `reset`), not
  gym-compliant: no `spaces`, and observations are live views, not copies.
- **`agents/`**: scripted agents (numpy) and neural agents (small torch conv
  stacks). Every agent is reconstructable from its `init_params` property —
  that is the serialization contract (JSON for scripted, torch for neural).
- **`sim/`** is the machine level: `Simulation` owns the agent-env loop;
  everything else — live view, recording, metrics, training evaluation —
  plugs in as a `SimObserver`. If you need per-step behavior, write an
  observer; do not re-inline the loop.
- **`view/`** is the only layer that turns world arrays into pixels.
  Recording is imageio-only and must work headless; matplotlib appears only
  in the interactive path.
- **`train/`**: `EvolutionTrainer` evaluates candidates through the same
  `Simulation`, behind a generation-granular `SearcherBackend` seam. The
  concrete ES library (evotorch, pinned, semi-dormant upstream) is
  deliberately confined to one module so it can be swapped (JAX/evosax is
  the intended successor — see the roadmap).
- **`registry.py`** is the single source of agent/dynamics ids and tuned
  defaults. CLI, examples and tests all construct through it — new agents or
  dynamics presets get registered there, not duplicated.

## Invariants & gotchas (each of these has cost real debugging time)

- **Aliasing**: `env.observe()` and `StepEvent.obs` expose *live* env arrays.
  Observers and agents must never mutate them; copy if you need to.
- **Identity**: update `env.agents` / `env.medium` in place (`.loc[...]`,
  `.values[:] = ...`), never rebind them — `AgentIndexer` holds references
  and silently desyncs otherwise.
- **Dead agents are parked at (0, 0)** with `alive=0` in the fixed-size
  agents array. Any per-agent computation must mask by `alive`, or dead
  slots phantom-participate.
- **Seeding is three-headed**: field init draws from the global numpy RNG
  *and* stdlib `random` (perlin noise); torch has its own. Use
  `Simulation.reset(seed)` / `Env.reset(seed=...)`, which handle all of it;
  agents additionally carry their own `default_rng(seed)`.
- **The coordinate seam**: the grid includes both 0.0 and 1.0; the `wrap`
  boundary maps 1.0 → 0.0, so edge agents relocate on the first step even
  with zero movement. Use the `limit` boundary when a test assumes agents
  stay put.
- **Seeded resets are snapshot-cached** inside `Env` — perlin field
  generation is the expensive part, and the cache is what makes
  per-candidate training evaluation affordable. Don't bypass it in hot paths.
- **Rendered channels are unbounded** (concentrations grow past 1); clip to
  display range before showing or encoding frames.

## Workflow

- uv only: `uv sync --all-extras --dev`. Lint: `uv run ruff check .`.
- `uv run pytest` is the fast suite (unit + behavioral invariants);
  `-m slow` runs the full training pipeline (evolution + MLflow artifacts).
- **Tests encode the science**: behavioral claims (Physarum out-forages
  Brownian by a calibrated margin; a short evolution run improves fitness)
  are seeded assertions in `tests/behavior/` with empirically calibrated
  thresholds. If you change the physics and one fails, recalibrate
  deliberately — don't loosen thresholds to make CI pass.
- Everything must run headless (CI has no display). The interactive view is
  optional sugar with a backend fallback chain.
- Experiment tracking is MLflow on a local **sqlite** backend
  (`sqlite:///<out>/mlflow.db`) — mlflow ≥ 3.14 rejects the legacy `file://`
  store.
