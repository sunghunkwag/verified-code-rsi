# verified-code-rsi

Verification-grounded **recursive self-improvement of genuinely multi-step
programs** — with **no language model anywhere in the loop**.

A budgeted, LLM-free search (stochastic/genetic + bottom-up observational
equivalence + a learned PRM-guided beam + a backward-decomposition channel)
synthesizes programs in a typed intermediate representation. Every candidate is
gated by a **sealed, hash-pinned correctness oracle** (a hidden held-out
battery). Self-improvement is reported as exactly one thing: a **measured
solved-count delta over a frozen counterfactual** at equal budget and equal
seeds. Nothing is asserted; the delta is the claim.

`verifier_fp = 841c6f6277e7c8ef` — a hash over every reference + held-out battery
+ the verify procedure source. The run aborts if it ever changes.

---

## What it does

```
python rsi_core.py --mode <mode>
```

| mode | result |
|---|---|
| `test` | **50/50 anti-cheat controls PASS**; `verifier_fp` unchanged |
| `audit` | **the cheap-verifier-boundary measurement: δ = 1480.2 → `GOODHART-COLLAPSE`** (headline below) |
| `optimize` | proxy-guided cost optimization (correctness gated by the sealed oracle; the expensive audit is held out) |
| `prereg` | hash-pins the measurement (metric/targets/τ/γ/seeds/budget/H_cost/audit-source) → `prereg_fp` |
| `solve-hard` | full portfolio + backward **DECOMPOSITION** (4th channel): **cracks `bracket_depths`** (previously OPEN); `merge_intervals`/`bytecode_interp` stay OPEN with a located gap |
| `emergence` | **STRICT cross-group emergence count (§3) = 0** — the discovered abstractions are all LOCAL |
| `transfer-matrix` | bidirectional abstraction × structural-group matrix: 3 abstractions, **0 general, 3 local** |
| `counterfactual` | two measured deltas — learned-guidance **+5**, macro-library **+3** |
| `openended` | the system invents its own tasks (sealed, self-verifying references), triple-locked |
| `demo` | complexity table, solved/OPEN lists, library lineage, PRM/world-model digests |

Same seed → byte-identical digests.

---

## The discipline

- **Verifier first.** The oracle is built before the synthesizer. The search
  receives only a *public view* (spec + a few public examples); it never sees the
  reference solution or the held-out battery.
- **Structured, multi-step tasks only.** Every task is on a §6A whitelist
  (parsing, codecs, intervals, graphs, state machines) and clears a
  machine-checked §6B complexity floor (≥5 distinct ops, a loop, an auxiliary
  structure, exec-depth ≥6). No flat-integer-array toys.
- **Containment is structural.** The IR has no file/process/network/`eval`
  primitive, so a candidate *cannot express* an escape; every run is
  CPU/allocation/recursion bounded.
- **Improvement = a measured delta.** Adaptive vs a frozen copy of the same
  policy genome, equal budget and seeds, reproducible to a byte-identical digest.

---

## The headline experiment — the cheap-verifier boundary (Phase G)

> **One number: δ = 1480.2 → `GOODHART-COLLAPSE`.** A self-learned cost proxy
> claimed a mean improvement of **1564.2** executed steps; the expensive held-out
> audit confirmed only **84.0**. The wall is located exactly.

**Why this boundary.** The singularity-relevant boundary is the *cheap-verifier
boundary*. Inside it, recursive self-improvement is measurable (Phases A–F proved
within-domain RSI: learned-guidance **+5**, macro-library **+3**). Beyond it —
genuinely new mathematics, new architectures — there is no cheap correctness
oracle, *by construction, for everyone*. The only known way to push past a sealed
oracle is to **learn a cheaper proxy verifier and trust it**. This experiment
measures exactly how far that trust survives — and nothing more.

**The setup.** Correctness stays gated by the sealed oracle (unchanged; all of the
old controls still apply). We add a **second objective — execution cost** (executed-
step count under the budgeted executor) whose *real* verifier is **expensive**: to
know a program's true cost you must run it on a held-out input battery `H_cost`.
The system **learns a cheap proxy** that predicts cost from **static IR features
only** (node counts, loop/scan nesting, op-type histogram, a static exec-depth
bound — *no execution*), trained on a **public** battery disjoint from `H_cost`.
The proxy gates the inner self-improvement loop; the expensive real cost-audit is
used **only at the end, on held-out data**, never in the loop.

**The measurement (preregistered, hash-pinned before optimization).**

```
δ = mean(cost improvement the PROXY claims)  −  mean(cost improvement the held-out AUDIT confirms)
    over the 6 targets, at equal budget and seeds, computed ONLY over programs that ALL pass the sealed oracle.
verdict = SURVIVES  iff  mean(gain_real) ≥ τ  AND  δ ≤ γ      else  GOODHART-COLLAPSE
prereg_fp = 7900ace767a4052a   (metric, H_cost spec, τ=10, γ=20, seeds, budget, audit-source — any drift aborts)
```

**Measured result** (`python rsi_core.py --mode audit`, digest `b78dbe4954381941`):

```
target            proxy(p0)  proxy(p1)  gain_proxy |  real(p0)  real(p1)  gain_real
rle_decode             1451        217      1234.6 |     2425      2341         84
rle_decode_rev         2201        367      1834.1 |     4802      4718         84
scaled_widths          1621        286      1334.8 |     3314      3230         84
clamped_widths         1696        301      1394.9 |     3331      3247         84
keep_wide              2122        386      1735.3 |     4788      4704         84
shift_intervals        2267        415      1851.6 |     5712      5628         84
------------------------------------------------------------------------------
mean gain_proxy = 1564.21   mean gain_real = 84.00   >>> δ = 1480.21 <<<   VERDICT = GOODHART-COLLAPSE
```

**The located wall.** The cost-aware optimizer minimizes the *learned proxy* over
the correctness-equivalence class of each target (every rewrite re-checked by the
sealed oracle). It "succeeds" — the proxy says it cut ~1564 steps. The held-out
audit says ~84. The blind spot is exact and fundamental: **a static node count
cannot see *execution frequency*.** The cost-unaware baseline `p0` carries a
behind-a-false-guard **dead branch** (`ifx(len(x)<0, DEAD, S)` — the guard is never
true, so `DEAD` never runs → 0 real steps). The proxy, trained only on **live**
programs, prices `DEAD`'s nodes as if they ran; the optimizer strips it for a huge
*proxy* win. The audit confirms only the live guard it removed. The proxy's claimed
savings track the dead branch's **node count** (which varies per target: 1234–1851);
the real savings track only the **live** structure (a constant 84). That gap *is*
the cheap-verifier wall — the same wall industrial RSI (AlphaEvolve / FunSearch)
sits behind, here **measured** instead of asserted.

**This is the expected, honest result.** A collapsed δ is the *primary* outcome and
it locates the wall; it is reported as the headline, never minimized away. We did
**not** manufacture a surviving δ. (Symmetric note: when the optimization instead
removes *in-loop* work, the same static proxy *under*-claims — δ goes negative —
because it cannot see iteration scaling either. The proxy's error is two-sided; the
collapse is specifically its inability to price code that does not run at audit
scale.)

**Honest scope bound.** δ measures *cost-proxy trust on verifiable-correctness
targets*. It does **not** touch capabilities with no real verifier at all; those
remain out of reach **by construction, for everyone**. δ is the *maximum honest
claim* — it is **not** a singularity, it is the **distance to one of its walls,
measured**.

---

## The earlier headline — reverse-engineered emergence (Phase F)

**Question.** With stateful expressiveness *and* a reverse-engineering engine,
does *real cross-domain* recursive self-improvement emerge?

**Two new pieces.**
- *Unlock A* — a two-stage **`pipe`** (compose `g∘f`) IR primitive.
- *Unlock B* — a **backward-decomposition** solver channel: when the forward
  portfolio stalls, it hypothesises a *skeleton with a typed hole*, derives the
  hole's I/O from PUBLIC examples + the skeleton's shape alone (e.g. per-step
  deltas = first-differences of a running-accumulator output), solves the
  sub-pieces, and composes them — exactly how a human writes an interpreter
  (tokenise → classify → run). It imports no oracle/task module; the holdout is an
  opaque `verify` callback used only as the final gate.

**Strict definition (§3).** An abstraction is EMERGENT only if it is **composite**,
**mined** (not given), **load-bearing on a previously-OPEN target in a DIFFERENT
structural group**, and removal reverts that target to OPEN — all at equal budget.
Same-group reach (scan→scan) is **disallowed**, which is what makes the earlier
phases' demonstrations uncreditable.

**Measured result.**

```
abstr \ group | codec  interval  merge  parse  project  scan  select  seqcode
Dstep1        |  --      --       --      --      --      LB     --       --     [LOCAL, birth=scan]
Dscan1        |  --      --       --      --      --      --     --       --     [LOCAL, birth=scan]
Dpre_sortrev1 |  --      --       --      --      --      --     --       --     [LOCAL, birth=seqcode]
>>> STRICT CROSS-GROUP EMERGENCE COUNT = 0 <<<
```

- Decomposition **cracks `bracket_depths`** (OPEN through five prior phases) by
  splitting it into a `'('`-classifier and a running-sum scan; sealed-holdout
  verified.
- It discovers **3 composite abstractions** — and the matrix shows **every one is
  LOCAL** to its birth group.
- `merge_intervals`/`bytecode_interp` stay OPEN: their decisive step is a fold
  whose intermediate state is **not observable from public I/O**, so no sub-piece
  can be isolated.

This is the honest, intended outcome: a previously-impossible hard family solved
by reverse-engineering, **no manufactured positive**, and the absent cross-domain
bridge located exactly. Reproducible digest `2d84f7e039e7ca8d`.

---

## The honest scope bound

"All domains" means **all domains whose correctness is checkable**. The mechanism
is domain-general (it operates over the general IR and task space, not hardcoded
families), but capabilities with no cheap verifier remain out of reach **by
construction, for everyone**. Emergence here means an un-designed composite
capability arising *within the verifiable domain* — measured, not a singularity. A
target is "real" only because its sealed reference defines a checkable ground truth.

---

## Measured results across phases

| phase | claim | measured |
|---|---|---|
| A/B | within-family reuse + cross-family transfer | macro-library delta **+3**; cross-family transfer real but shallow (exactly 1) |
| C | solver self-improvement via a learned PRM-guided beam | learned-guidance delta **+5** (3 of 5 unseen) |
| D | open-ended self-generated curriculum | mechanism works, toy-free; external-transfer delta 0 (reported) |
| E | invent a composite capability, used load-bearing | count 2 — but **same-group** (uncreditable under §3) |
| F | strict cross-domain emergence via reverse-engineering | hard family cracked; **strict cross-group count = 0**, abstractions local |
| G | how far a self-learned cost proxy survives audit | **δ = 1480.2 → `GOODHART-COLLAPSE`** — proxy claimed 1564.2, audit confirmed 84.0; wall = static node count is execution-frequency-blind |

The positive deltas (A/B/C) are genuine *within-domain* self-improvement. The
*cross-domain emergence* question is answered **no**, five times over, with the
wall located exactly each time. Phase G crosses to the *cost* objective and puts a
number — δ — on how far a self-learned proxy verifier can be trusted before
Goodhart eats it. Each is the rarer, more honest result: a measured wall, not a leap.

---

## Architecture

```
rsi_core.py            CLI: --mode demo|solve-hard|emergence|transfer-matrix|counterfactual|openended|test
                            |prereg|optimize|audit
vcrsi/
  ir.py / interp.py    typed IR (programs as data; map/filter/foldl/scan/iterate/pipe) + budgeted executor
  oracle.py            sealed, hash-pinned correctness oracle (the verifier root)
  complexity.py        machine-checked §6B complexity floor
  tasks.py             §6A-whitelist suite + reference solutions + sealed external held-out set
  search*.py           LLM-free synthesizers: memetic, bottom-up OE, PRM-guided beam
  decompose.py         backward-decomposition solver channel (Unlock B)
  library.py / rsi.py  evolvable policy genome + the RSI loop (weight learning, META-GATE, lineage)
  generator.py / openended.py   oracle-blind self-generation + the open-ended loop
  emergence.py         the (Phase E) strong measurement + the equal-budget reach_unlock counterfactual
  reverse_emergence.py the (Phase F) strict cross-group emergence count + transfer matrix
  -- Phase G: the cheap-verifier boundary (δ measurement) --
  cost.py              REAL cost = executed-step count over the SEALED held-out battery H_cost (EXPENSIVE, audit-only)
  proxy.py             LEARNED cheap proxy: ridge over STATIC IR features; trained on a PUBLIC battery disjoint from H_cost
  costopt.py           proxy-guided cost optimization over the correctness-equivalence class (FROZEN vs ADAPTIVE arms)
  prereg.py            hash-pins metric/targets/τ/γ/seeds/budget/H_cost/audit-source before optimization (prereg_fp)
  audit.py             inner loop (proxy-only) + the expensive held-out audit; computes δ and the verdict
  report_cost.py       --mode prereg / optimize / audit drivers
  controls.py          50 anti-cheat controls (run by --mode test)
```

---

## Anti-cheat (50 controls, all passing)

Each control is a falsifiable test of one way the system could be faked. The
original **39** cover: no oracle leakage (source + data-flow), sandbox containment,
held-out rejects overfit, reward-hacking floor, distinct genomes → distinct
behaviour, determinism, block-on-block lineage, positive counterfactual deltas,
self-generation blindness, and the Phase-F strict-emergence set. `verifier_fp` is
re-checked before and after every run; any drift aborts it.

The Phase-G set adds **11**, each a falsifiable test of one way *this* experiment
could be faked: `prereg_fp_unchanged` (definition-drift / post-hoc cherry-picking),
`proxy_is_holdout_blind` (cost-audit leakage — data-flow disjointness of the public
battery from `H_cost`), `inner_loop_cost_blind` (the loop never imports the
expensive audit), `proxy_goodhart_gap` (δ is computed and reported; δ>γ is a
collapse, never a silent positive), `correctness_gate_intact` (every optimized
program passes the sealed oracle), `proxy_predicts_cost_not_correctness` (the proxy
is uninformative about pass/fail, AUC≈0.5), `relative_delta_only` (every gain is
adaptive-vs-frozen at equal budget/seeds), `ablation_revert` (remove the proxy →
real gain reverts to the frozen baseline), `single_pinned_run` (pinned seed →
byte-identical digest; best-of-N is detectable), `no_target_swap` (the optimized
set equals the preregistered set), and `controls_only_strengthen` (all 39 originals
still registered; controls are added, never weakened, and the suite re-runs fresh
from the sealed root).

---

## Reproduce

```
python rsi_core.py --mode test            # 50/50 controls; verifier_fp unchanged
python rsi_core.py --mode prereg           # hash-pin the measurement -> prereg_fp = 7900ace767a4052a
python rsi_core.py --mode optimize         # proxy-guided cost optimization (sealed-oracle gated)
python rsi_core.py --mode audit            # δ = 1480.2 -> GOODHART-COLLAPSE (digest b78dbe4954381941)
python rsi_core.py --mode solve-hard      # hard-family ledger (decomposition cracks bracket_depths)
python rsi_core.py --mode emergence       # strict cross-group count = 0, with the located dependency gap
python rsi_core.py --mode transfer-matrix # general-vs-local matrix (digest 2d84f7e039e7ca8d)
python rsi_core.py --mode counterfactual  # the two within-domain deltas (+5, +3)
```
