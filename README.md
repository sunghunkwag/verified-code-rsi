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
| `test` | **39/39 anti-cheat controls PASS**; `verifier_fp` unchanged |
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

## The headline experiment — reverse-engineered emergence (Phase F)

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

The positive deltas (A/B/C) are genuine *within-domain* self-improvement. The
*cross-domain emergence* question is answered **no**, five times over, with the
wall located exactly each time — the rarer and more honest result.

---

## Architecture

```
rsi_core.py            CLI: --mode demo|solve-hard|emergence|transfer-matrix|counterfactual|openended|test|...
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
  controls.py          39 anti-cheat controls (run by --mode test)
```

---

## Anti-cheat (39 controls, all passing)

Each control is a falsifiable test of one way the system could be faked. They
cover: no oracle leakage (source + data-flow), sandbox containment, held-out
rejects overfit, reward-hacking floor, distinct genomes → distinct behaviour,
determinism, block-on-block lineage, positive counterfactual deltas, and
self-generation blindness — plus the Phase-F set: `decomposition_no_leakage`,
`emergent_is_cross_group_and_was_open` (a same-group plant is **rejected**),
`groups_not_gerrymandered`, `emergent_is_composite_and_mined`, and
`hard_family_solutions_are_holdout_verified`. `verifier_fp` is re-checked before
and after every run; any drift aborts it.

---

## Reproduce

```
python rsi_core.py --mode test            # 39/39 controls; verifier_fp unchanged
python rsi_core.py --mode solve-hard      # hard-family ledger (decomposition cracks bracket_depths)
python rsi_core.py --mode emergence       # strict cross-group count = 0, with the located dependency gap
python rsi_core.py --mode transfer-matrix # general-vs-local matrix (digest 2d84f7e039e7ca8d)
python rsi_core.py --mode counterfactual  # the two within-domain deltas (+5, +3)
```
