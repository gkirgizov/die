# DIE — Research Journal

Dated findings and lessons that changed decisions, about learning agents and
the Env alike. Complements [NOTES.md](NOTES.md) (roadmap/design notebook) and
AGENTS.md (stable invariants): entries land here first, and graduate to
AGENTS.md only once they harden into conventions. Newest first within each
date. Claims carry the measurement that backs them.

## 2026-07-20 — Second PPO implementation (branch `rl-rejax`)

Everything below is on the 16×16 benchmark: `init_agent_ratio=0.2`,
`food_infinite=False`, `agents_die=False`, 120 iterations × 8 envs × 32 steps,
deterministic evaluation on held-out worlds, **do-nothing baseline 2.68**.

### Finding: setting `max_steps` destroys learning (+5.9 → −36)

The single largest effect measured on this branch, and it is not the learner —
it is the episode boundary. `compute_gae` treats every `done` as a termination,
so a time-limit truncation zeroes a bootstrap that should have been taken.

| `max_steps` | held-out return |
|---|---|
| `None` | **5.91** |
| 64 | **−36.17** |

The textbook description of truncation bias is "mild". Here it inverts the
result. This is what `jx/interface.py` returning `terminated` and `truncated`
*separately* is for, and it is why the RL path leaves `max_steps` unset and lets
the rollout boundary do the bootstrapping.

**Consequence for any gymnax-shaped library:** gymnax returns one `done`. So
rejax cannot be run on a truncating configuration here at all — the cross-check
is pinned to `max_steps=None`, `agents_die=False`, where nothing terminates.

### Finding: a fully-packed grid is degenerate, like infinite food was

rejax's loss cannot mask dead slots, so the tempting fix is
`init_agent_ratio=1.0` (every cell an agent, zero dead slots, deterministic
count, renderer untouched). It does not work — for a reason worth recording:

| config | do-nothing | best trained |
|---|---|---|
| ratio 0.2 | 2.68 | 5.91 |
| ratio 1.0 | **16.90** | −5.63 |

When agents occupy every cell there is nowhere to forage *to*. The spatial
search problem — the thing this environment is about — is removed by
construction, and stillness wins again. **Second instance of the same trap**
after `food_infinite=True`; the do-nothing baseline caught both.

So the cross-check runs on the sparse grid and rejax eats ~80% dead action
dimensions rather than have the benchmark bent to suit it.

### Finding: the convolutional prior, not the PPO implementation, is decisive

Same rejax loss, same env, only the network differs:

| network | untrained | trained |
|---|---|---|
| conv trunk (ours, substituted into rejax) | 3.4 | −20.5 ± 18.2 |
| rejax stock MLP over the flattened field | −25.3 | −32.8 ± 0.8 |

The MLP never approaches the baseline. Caveat kept explicit: the two also differ
in output-layer initialization (ours uses a small-gain orthogonal final layer),
and the MLP's untrained score of −25 shows that init alone already costs it —
so this measures "conv trunk as configured", not convolution in isolation.

### Finding: single-seed RL numbers on this benchmark are not trustworthy

Re-measuring across 3 seeds instead of 1:

| setup | untrained | trained (3 seeds) |
|---|---|---|
| vendored, `per_agent` | 2.88 | **3.76 ± 2.34** |
| vendored, `joint` | 2.88 | −2.05 ± 6.08 |
| rejax, conv (`joint` only) | 3.42 | −20.50 ± 18.17 |

**Partial retraction.** The 5.46/5.91 figures in the earlier entry and in PR #6
are seed-0 results; the seed-averaged effect is real but much smaller (+0.88
over untrained, ±2.34) and `test_ppo_beats_doing_nothing` passes on a pinned
seed rather than on a robust claim. Nothing about the *mechanism* changes —
`per_agent` is still the only configuration that does not degrade — but the
magnitude was overstated and any future comparison here needs seed averaging.

**Cross-check verdict:** the two implementations agree on the thing that
matters and it is not flattering to either. Restricted to the joint objective,
neither reliably beats inaction. The axis worth spending R&D on is the
factorization (per-agent credit), not the choice of PPO implementation — which
is exactly what a cross-check is for: it moved the suspicion off our vendored
loss and onto the objective.

### Note: what adapting to a single-agent library actually costs

Three deviations `jx/interface.py` documents as deliberate all get paid for in
`rl/rejax_backend.py`: joint-only rewards, no alive-masking in the loss, and
truncation collapsed into termination. Plus one operational surprise —
`rejax.Algorithm.create` deep-copies its config, so the init bank is duplicated
per configured run.

## 2026-07-20 — First RL results (branch `rl-ppo`)

### Finding: `food_infinite=True` makes foraging degenerate

Measured with fixed policies on held-out worlds (40 steps × 8 worlds):

| | do-nothing | random wander |
|---|---|---|
| `food_infinite=True` | **11.21** | 7.02 |
| `food_infinite=False` | 2.76 | **3.24** |

With infinite food the optimal policy is to sit on a food cell and eat forever
at zero action cost — stillness beats foraging, so any learner "succeeds" by
learning nothing. An early PPO result (3.5× over untrained, since retracted)
was inflated exactly this way: the config was inherited from the smoke tests.

**Lesson:** validate the *task* before validating the learner — a do-nothing
baseline is one rollout and would have caught this immediately.
**Open question:** `tests/behavior/test_jx_learning_smoke.py` and the
`st-perlin-wide` preset both use `food_infinite=True`; the ES behavioral
claims may partly rest on the same degenerate objective. Not touched — those
thresholds are calibrated — but worth a deliberate revisit.

### Finding: additive noise clipped at zero is a food dispenser

First noise design: `channel += σ·N(0,1)` then clip at 0. On an empty cell this
rectifies — negative draws become 0, positive draws become free resource — with
mean +0.02 food/step at σ=0.05. Every policy scored *higher* under "noise";
the entire robustness experiment ran inverted before this was caught.

Fix: **food noise is multiplicative** (`food *= 1 + σ·N`), mean-preserving and
unable to create resource on empty cells. The clip still bites at large σ:
`E[max(1+σZ,0)] = Φ(1/σ) + σφ(1/σ)` → +0.4% at σ=0.5, **+8.3% at σ=1.0**. So
`food_noise` is only interpretable below ~0.6; sweep results above that measure
enrichment, not difficulty. **Chem noise stays additive on purpose**: chem is
signal, not reward, and a corrupted channel must be able to fabricate false
positives, not merely attenuate real messages.

**Lesson:** in a survival environment, any perturbation touching a
reward-bearing channel must be checked for mean bias *first* — the bias does
not look like a bug, it looks like a result.

### Finding: training under medium noise is a regularizer, not a tradeoff

16×16 foraging (`food_infinite=False`), 200 iterations, held-out worlds,
2–3 seeds, do-nothing baseline 2.76:

| policy | eval clean | eval noisy (f=0.25, c=0.05) |
|---|---|---|
| clean-trained | 5.46 | 5.59 |
| noise-trained | **7.52** | **7.26** |

Noise-trained wins **everywhere, including clean worlds** (+38%). Under
comm-noise sweep 0→1.0 it degrades gracefully (7.52→6.38, −15%) while the
clean-trained policy slightly *improves* (5.46→6.06) — consistent with noise
breaking up the self-trail coagulation NOTES.md logs as the gradient agent's
characteristic failure mode. Robustness came free; the interesting axis now is
whether higher comm noise changes *what* is learned (deposit rates), not just
how much.

### Finding: PPO learns the foraging task; LR must be ~30× the usual default

PPO reaches 2.0× the do-nothing baseline (5.46 vs 2.76) and 1.7× a random
wanderer on worlds training never saw, deterministic evaluation. But at the
canonical PPO learning rate 3e-4 the policy **does not move at all**
(approx_kl ≈ 3e-5, return flat over 150 iterations, only the critic trains).
Measured sweep: 3e-4 flat, 1e-3 slow, **1e-2 best on every seed**, 3e-2 noisier
without being better. Why the regime differs: reward is dense and
per-agent-per-step, and one iteration carries ~10⁵ agent-transitions, so
gradient noise is far below the single-agent Atari/MuJoCo regime the default
comes from. `policy_loss ≈ 0` is *expected* under normalized advantages —
`approx_kl` and held-out return are the signals that matter.

### Lessons: Env/JAX structure (each cost a debugging round)

- **A per-agent reward vector escapes silently.** `Simulation`'s
  `total_reward += reward` and `rollout`'s `rewards.sum()` both *work* with a
  `[max_agents]` reward — broadcast, no exception, wrong numbers downstream.
  Hence `jx.env.step` keeps its scalar return byte-identical and per-agent
  data travels in a separate `StepAux` / `TimeStep` channel.
- **Pin shape-derived values at construction, not at use.** The blur radius
  derives from `diffuse_sigma`; deriving it inside a jitted function fails the
  moment params are passed *as an argument* (sigma is then a tracer — the
  ordinary case, not the vmap-exotic one). `EnvParams.__post_init__` resolves
  it while sigma is still concrete; being static it rides on the treedef.
  Consequence: sweeping sigma varies kernel *weights* inside a truncation
  window fixed at construction — construct with the widest sigma of the sweep
  range (pinned by `test_sweeping_sigma_keeps_the_constructing_window`).
- **The static/traced split is the meta-learning enabler and is
  retrofit-hostile.** One flax struct, `pytree_node=False` for anything
  shape- or branch-determining; 4 distinct traced configs → `_cache_size()==1`,
  one static change → 2. The traced-fields test asserts *names*, not counts,
  so growing the config is a deliberate act.
- **Dead slots poison statistics, not just physics.** Alive-masking the obs
  normalizer moves tracked food-mean from 0.082 (death-rate tracker) to 0.445
  (actual quantity). Same trap again in GAE, advantage normalization, and loss
  reduction — each masks independently.
- **Noise lives in the RL interface layer, not `jx/env.py`.** Physics stays
  RNG-free and bit-comparable across both engines ("two engines, one physics").
  Cost, accepted and documented: noise has no xarray twin, so noise-trained
  policies replay in a clean world.
- **State-independent `log_std` breaks under `vmap` over worlds** unless
  broadcast to the per-agent shape inside the module — outside, it acquires a
  world axis but no agent axis and stops broadcasting against `mean`.

### Lessons: architecture

- The conv trunk's **receptive field is a scientific quantity** (the agent's
  sensory horizon): exposed as a property, pinned by a locality test proving a
  perturbation beyond it *cannot* alter the output — the "information moves
  only through the medium" thesis as an architectural fact, and the reason
  attention is an ablation rather than the default.
- SimBa recipe adapted Dense→Conv (pre-norm residual, 4× expansion, spatial
  conv only in the expansion, 1×1 projection). **Our adaptation, no published
  precedent** — to be validated against plain `ConvNCA`, not assumed.
- RL trunk is ~88k params vs the evolved kernel's 162. That gap is the
  argument for gradients: ES is capped by its optimizer, not by the problem.

### Method notes (kept honest)

- Evaluation is deterministic (act on the mean) on a **held-out world bank**;
  training-return drift is not evidence of learning.
- On-policy return climbing while held-out stays flat = memorizing the bank.
- Retraction: the "3.5× over untrained" figure from earlier today was measured
  on the degenerate infinite-food task and is void.
