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
| `test` | **51/51 anti-cheat controls PASS** (incl. `no_planted_strawman`); `verifier_fp` unchanged |
| `audit` | **the cheap-verifier-boundary measurement: δ = −803.5 → `PROXY-CONSERVATIVE`** (raw-seed baseline; headline below) |
| `optimize` | proxy-guided cost optimization over genuine programs (correctness gated by the sealed oracle; the expensive audit is held out) |
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

## The headline experiment — the cheap-verifier boundary, measured honestly (Phase H)

> **One number: δ = −803.5 → `PROXY-CONSERVATIVE`.** On a **raw-synthesizer**
> baseline (no planted structure), the learned cost proxy claimed a mean saving of
> **53.2** executed steps; the expensive held-out audit confirmed **856.7**. The
> proxy is **not** Goodharted in the dangerous (over-claim) direction — it
> *under*-claims. The cheap-verifier over-claim wall is **not** reached by this
> proxy on this family. *(This corrects Phase G — see "A correction" below.)*

**Why this boundary.** The singularity-relevant boundary is the *cheap-verifier
boundary*. Inside it, recursive self-improvement is measurable (Phases A–F proved
within-domain RSI: learned-guidance **+5**, macro-library **+3**). Beyond it —
genuinely new mathematics, new architectures — there is no cheap correctness
oracle, *by construction, for everyone*. The only known way to push past a sealed
oracle is to **learn a cheaper proxy verifier and trust it**. This experiment
measures how far that trust survives — and nothing more.

**The setup.** Correctness stays gated by the sealed oracle (unchanged; all old
controls still apply). We add a **second objective — execution cost** (executed-step
count) whose *real* verifier is **expensive**: to know a program's true cost you
must run it on a held-out battery `H_cost`. The system **learns a cheap proxy** that
predicts cost from **static IR features only** (node counts, loop/scan nesting,
op-type histogram, a static exec-depth bound — *no execution*), trained on a
**public** battery disjoint from `H_cost`. The proxy gates the inner loop; the
expensive audit is used **only at the end, on held-out data**.

**The honest baseline (Phase H's fix).** The baseline `p0` is the synthesizer's
**raw output**, byte-for-byte — control `no_planted_strawman` checks
`ast_digest(p0) == ast_digest(seed)` and `node_count(p0) == node_count(seed)` for
every target. The cost-aware arm `p1` is a **proxy-guided descent** over a *genuine*
oracle-gated rewrite neighborhood of real programs (hoist-to-child, map/map fusion,
algebraic identities) — **no move inserts a dead branch, an always-false guard, or
no-op padding**. Both `p0` and `p1` are raw search outputs.

```
δ = mean(gain_proxy) − mean(gain_real),   gain = proxy/real_cost(p0) − (p1),
    over the 6 targets, equal budget/seeds, ONLY over programs that pass the sealed oracle.
verdict:  δ > γ → GOODHART (over-claim);  δ < −γ → PROXY-CONSERVATIVE (under-claim);  |δ| ≤ γ → NO-GOODHART.
prereg_fp = 7900ace767a4052a   (metric, H_cost spec, τ=10, γ=20, seeds, budget, audit-source — unchanged)
```

**Measured result** (`python rsi_core.py --mode audit`, digest `cdc997cae0720130`):

```
target            proxy(p0) proxy(p1) gain_proxy | real(p0) real(p1) gain_real
rle_decode             215       215        0.0  |    2341     2341         0     (already minimal)
rle_decode_rev         370       229      140.9  |    4718     2383      2335     sub(snd,sdiv(snd,0)) -> snd
scaled_widths          283       283        0.0  |    3230     3230         0     (already minimal)
clamped_widths         306       306        0.0  |    3247     3247         0     (already minimal)
keep_wide              385       292       93.0  |    4704     3300      1404     imin(fst,snd) -> fst
shift_intervals        418       333       85.3  |    5628     4227      1401     imax(fst,add(snd,k)) -> add(snd,k)
------------------------------------------------------------------------------
mean gain_proxy = 53.19   mean gain_real = 856.67   >>> δ = −803.47 <<<   VERDICT = PROXY-CONSERVATIVE
```

**What this means.** On 3 of 6 targets the proxy-guided search found and removed
**genuine in-loop redundancy** the cost-blind synthesizer happened to ship (e.g.
`imax(fst, add(snd,k)) → add(snd,k)`, valid because `fst ≤ snd+k` on every interval
— discovered by trying the hoist and letting the sealed oracle confirm it). The
other 3 seeds were already minimal. **`gain_real` varies** `[0, 2335, 0, 0, 1404,
1401]` — the tell of a *real* baseline, not Phase G's identical-84 fabrication.
**Every target's real cost went down or stayed equal — no regression.** The proxy
*under*-claimed (δ < 0) because it is calibrated on cheap **small** public inputs
and the savings are **in-loop**, so their real cost grows with input size the static
proxy cannot see. A conservative proxy is *safe* to gate on; the dangerous
over-claim wall is **not reached** here.

**A correction — what Phase G got wrong.** Phase G reported `δ = 1480.2,
GOODHART-COLLAPSE` and called it "the cheap-verifier boundary, measured." **That was
not measured — it was constructed.** Its baseline was not the synthesizer's output;
it deliberately wrapped a correct seed in an always-false guard carrying a chunky
**dead branch** (`DEAD_COPIES` stacked copies of the seed). The proxy then "saved"
the node-count of structure that *never runs* (0 real steps), so the proxy↔real gap
was a property of the planted structure, not the optimization landscape. **Proof it
was a dial:** setting `DEAD_COPIES` 3→8 moved `δ` 1480→3294 while `gain_real` stayed
flat at 84. Phase H **deletes** that machinery, adds `no_planted_strawman` to forbid
it, and reports the natural δ above. The qualitative point Phase G *did* establish —
a static, execution-frequency-blind proxy is **gameable by construction** — remains
true; it is a worst-case *existence* note, **not** a measurement, and is no longer
called one.

**Honest scope bound.** δ measures *cost-proxy trust on verifiable-correctness
targets*. It does **not** touch capabilities with no real verifier at all; those
remain out of reach **by construction, for everyone**. δ is a measured gap, **not**
a singularity — the distance to one of its walls, and on this family that wall is
not crossed.

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
| G | *(superseded)* a self-learned cost proxy "Goodharted" | **δ = 1480.2 was CONSTRUCTED, not measured** — a planted dead branch with a `DEAD_COPIES` dial; retained only as a gameable-by-construction note |
| H | the **natural** proxy-vs-real gap on a raw-seed baseline | **δ = −803.5 → `PROXY-CONSERVATIVE`** — proxy claimed 53.2, audit confirmed 856.7 (varying per target); the over-claim wall is **not** reached |

The positive deltas (A/B/C) are genuine *within-domain* self-improvement. The
*cross-domain emergence* question is answered **no**, five times over. Phase G
crossed to the *cost* objective but **manufactured** its collapse; Phase H removes
the fabrication, forbids it with `no_planted_strawman`, and reports the natural δ —
which shows the static proxy is *not* badly Goodharted on these targets (it
under-claims). Each is the rarer, more honest result: a measured wall — or, here, a
measured *absence* of one — not a leap.

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
  -- Phase G/H: the cheap-verifier boundary (δ measurement) --
  cost.py              REAL cost = executed-step count over the SEALED held-out battery H_cost (EXPENSIVE, audit-only)
  proxy.py             LEARNED cheap proxy: ridge over STATIC IR features; trained on a PUBLIC battery disjoint from H_cost
  costopt.py           proxy-guided descent over a GENUINE rewrite neighborhood of real programs (baseline = raw seed; NO planted strawman)
  prereg.py            hash-pins metric/targets/τ/γ/seeds/budget/H_cost/audit-source before optimization (prereg_fp)
  audit.py             inner loop (proxy-only) + the expensive held-out audit; computes the natural δ and the gap-centric verdict
  report_cost.py       --mode prereg / optimize / audit drivers
  controls.py          51 anti-cheat controls (run by --mode test)
```

---

## Anti-cheat (51 controls, all passing)

Each control is a falsifiable test of one way the system could be faked. The
original **39** cover: no oracle leakage (source + data-flow), sandbox containment,
held-out rejects overfit, reward-hacking floor, distinct genomes → distinct
behaviour, determinism, block-on-block lineage, positive counterfactual deltas,
self-generation blindness, and the Phase-F strict-emergence set. `verifier_fp` is
re-checked before and after every run; any drift aborts it.

The cheap-verifier set adds **12**, each a falsifiable test of one way *this*
experiment could be faked: `prereg_fp_unchanged` (definition-drift / post-hoc
cherry-picking), `proxy_is_holdout_blind` (cost-audit leakage — public battery
disjoint from `H_cost`), `inner_loop_cost_blind` (the loop never imports the
expensive audit), `proxy_goodhart_gap` (δ is computed and reported; the gap-centric
verdict labels over-claim/under-claim/no-goodhart, never a silent positive),
`correctness_gate_intact` (every optimized program passes the sealed oracle),
`proxy_predicts_cost_not_correctness` (the proxy is uninformative about pass/fail,
AUC≈0.5, on size-matched programs), `relative_delta_only` (every gain is
adaptive-vs-frozen at equal budget/seeds), `ablation_revert` (remove the proxy →
the optimizer reverts to the raw seed), `single_pinned_run` (pinned seed →
byte-identical digest; best-of-N is detectable), `no_target_swap` (the optimized set
equals the preregistered set), `controls_only_strengthen` (all 39 originals still
registered; controls are added, never weakened), and — the Phase-H addition —
**`no_planted_strawman`**: the baseline `p0` is the synthesizer's raw output
(`ast_digest(p0) == ast_digest(seed)`, `node_count(p0) == node_count(seed)`), the
optimize path constructs no dead branch / always-false guard / no-op padding, and a
planted-baseline negative self-test confirms the check has teeth. This is the
control that closes the gap which let Phase G's fabricated baseline pass.

---

## Reproduce

```
python rsi_core.py --mode test            # 51/51 controls; verifier_fp unchanged
python rsi_core.py --mode prereg           # hash-pin the measurement -> prereg_fp = 7900ace767a4052a
python rsi_core.py --mode optimize         # proxy-guided descent over genuine programs (sealed-oracle gated)
python rsi_core.py --mode audit            # natural δ = -803.5 -> PROXY-CONSERVATIVE (digest cdc997cae0720130)
python rsi_core.py --mode solve-hard      # hard-family ledger (decomposition cracks bracket_depths)
python rsi_core.py --mode emergence       # strict cross-group count = 0, with the located dependency gap
python rsi_core.py --mode transfer-matrix # general-vs-local matrix (digest 2d84f7e039e7ca8d)
python rsi_core.py --mode counterfactual  # the two within-domain deltas (+5, +3)
```

The full recorded output of `--mode prereg`, `--mode audit`, and all **51/51**
controls of `--mode test` (every control's PASS line shown in full) is committed at
[`docs/PHASE_H_RESULTS.txt`](docs/PHASE_H_RESULTS.txt). The superseded Phase G run
(the constructed δ = 1480.2) remains at
[`docs/PHASE_G_RESULTS.txt`](docs/PHASE_G_RESULTS.txt) for the record, now labelled
as a gameable-by-construction note rather than a measurement.
