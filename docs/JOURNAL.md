# DIE — Research Journal

Dated findings and lessons that changed decisions, about learning agents and
the Env alike. Complements [NOTES.md](NOTES.md) (roadmap/design notebook) and
AGENTS.md (stable invariants): entries land here first, and graduate to
AGENTS.md only once they harden into conventions. Newest first within each
date. Claims carry the measurement that backs them.

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
