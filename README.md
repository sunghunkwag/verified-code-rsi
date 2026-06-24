# verified-code-rsi

Verification-grounded recursive self-improvement of **genuinely multi-step
programs**, with **no language model** anywhere in the loop.

A budgeted stochastic/genetic search synthesizes programs in a typed
intermediate representation. Every solution is gated by a **sealed, hash-pinned
correctness oracle** (a hidden held-out test battery). The search improves *its
own* ability to synthesize programs by evolving a policy genome (operator
weights + a library of mined subroutines), and that improvement is reported as
exactly one thing: a **measured solved-count delta over a frozen counterfactual**
at equal budget and equal seeds. Nothing is asserted; the delta is the claim.

This replaces a previous ~50,000-line monolith whose RSI machinery was real but
whose *domain* was a toy (integer-list arithmetic — `sum`/`count`/`max` — solved
by 2–5 instructions of a stack VM whose opcodes were already high-level
reductions). The toy domain is deleted. What survives is the one genuinely
verification-grounded discipline of that monolith's native-kernel section:
*build the verifier first, datafy the candidate, verify bit-exact, measure
against a frozen counterfactual.* This system is **2,944 lines** (5.8% of the
original).

---

## What the run actually produces (regenerable; see Audit Protocol)

From a fixed seeded run on this machine:

```
--mode test            : 10/10 anti-cheat controls PASS; verifier_fp unchanged
--mode counterfactual  : frozen arm solves 3, adaptive arm solves 6
                         MEASURED SELF-IMPROVEMENT DELTA = 6 - 3 = +3
--mode demo            : per-task complexity table; solved/OPEN lists;
                         a printed parent->child library lineage pair
```

The delta comes from tasks the frozen (default-policy) arm cannot reach at the
budget but the adaptive arm can, after learning operator weights and mining
reusable subroutines from its own solved programs. Four task families remain
**OPEN for both arms** and are reported as OPEN, never hidden (the honest wall;
see below).

`verifier_fp = e54fdb362f17675d` (hash over every reference + held-out battery +
the verify procedure source; the run aborts if it changes).

---

## Architecture (modules)

```
rsi_core.py        thin CLI entry point (--mode demo | counterfactual | test)
vcrsi/
  ir.py            typed IR: programs as data; library blocks; inlining; pp
  interp.py        resource-budgeted, side-effect-free executor (physical root)
  tasks.py         the §6A-whitelist task suite + reference solutions (in the IR)
  oracle.py        the sealed, hash-pinned correctness oracle (verifier root)
  complexity.py    the machine-checked complexity floor (§6B)
  search.py        the LLM-free synthesizer (sampling + memetic local search)
  library.py       the evolvable policy genome (weights + mined/encapsulated blocks)
  rsi.py           the RSI loop: weight learning, META-GATE, encapsulation, lineage
  counterfactual.py  adaptive vs frozen arms at equal budget/seeds
  controls.py      the anti-cheat controls (§4) as runnable checks
  report.py        --mode demo / counterfactual reports
```

---

## Verifier first, and where the irreducible wall sits (F_eff / F_theo)

The oracle was built **before** the generator. For each task the oracle generates
public training examples (the only thing the synthesizer ever sees) and a hidden
held-out battery at larger, unseen input sizes by running the task's reference
solution. A candidate is correct iff it reproduces the reference's output exactly
on the **full** held-out battery.

Everything above the following two-part root is improvable / evolvable (the IR,
the synthesizer, the operator weights, the subroutine library):

1. **The correctness oracle** — the sealed held-out battery + reference, hashed
   into `verifier_fp` and asserted unchanged on every run. **It cannot be
   released.** Release it and "improvement" collapses into the trivial discovery
   of *declaring victory* (reward hacking). This is proven constructively by the
   reward-hacking control: a speed-only search with the oracle removed is won by
   a wrong-but-fast program (the empty string, 6 interpreter steps), while the
   oracle-gated search's winner is correct.

2. **The physical executor** — the interpreter (`interp.py`) plus its CPU-step,
   allocation and recursion budgets. **It cannot be released.** Release it and
   "produces the right output" / "terminates" lose all meaning; there is nothing
   left to ground correctness in.

This is the F_eff/F_theo boundary stated honestly: the fixed layer does not
vanish, it sits at exactly this load-bearing root. The boundary is demonstrated,
not asserted — the reward-hacking control *shows* you cannot move it down.

The sandbox is structural, not bolted on: the IR has **no** file, process,
network or `eval`/`exec` primitive, so a candidate cannot express an escape; and
every run is bounded, so a runaway loop or memory bomb is cut off and scored as
failed rather than hanging the harness (verified by the sandbox-containment
control).

---

## The task suite (§6A whitelist) and the complexity floor (§6B)

Tasks are drawn only from the approved families (parsing/interpreting; encoding
round-trips; graph/structure; interval/sequence restructuring; small state
machines). Every input is structured (a string to parse, a list of pairs, …),
never a flat integer array reduced to a scalar. The banned toy reductions
(`sum`, `count`, `max`, …) are asserted absent.

"Complex" is **computed**, never claimed. `complexity.complexity_floor(task)`
derives, from the reference solution's AST plus an execution trace on the longest
public example, four metrics that must all hold: ≥5 distinct non-trivial
operations; a loop (or recursion) present; recursion or an auxiliary data
structure built; and a dynamic execution depth ≥6. The per-task table is printed
by `--mode demo`; a task that misses any threshold is rejected and cannot be
counted. Adopted programs must additionally clear an inlined op-count floor (so a
task cannot be "solved" by a below-floor program that hides work inside a
subroutine).

Per-task metrics from the run (`--mode demo`):

```
task                     fam   ops  loop rec/ds depth   ok
rle_decode                 2     5     Y      Y      7   OK
rle_decode_rev             2     6     Y      Y      9   OK
rle_decode_sorted          2     6     Y      Y      9   OK
rle_decode_twice           2     6     Y      Y     14   OK
rle_decode_palindrome      2     7     Y      Y     12   OK
rle_decode_rev_twice       2     7     Y      Y     16   OK
rle_rev_palindrome         2     8     Y      Y     12   OK
rle_rev_palindrome_twice   2     8     Y      Y     36   OK
caesar_encode              2     6     Y      Y      7   OK
caesar_decode              2     6     Y      Y      8   OK
rle_decode_shift1          2     9     Y      Y     29   OK
interleave_pairs           4     5     Y      Y      9   OK
merge_intervals            4    13     Y      Y      8   OK
bracket_depths             4     9     Y      Y      8   OK
bytecode_interp            1    10     Y      Y     10   OK
```
(All 15 tasks in the suite clear the floor; the counterfactual runs the 11-task
`DEFAULT_ORDER` curriculum.)

---

## The synthesizer (LLM-free)

Candidates are typed IR programs (typed values incl. lists/strings/pairs, the
combinators `map`/`filter`/`foldl`, conditionals, and callable library
subroutines). Synthesis is weighted typed sampling under the policy genome,
refined by a memetic local search (in-scope leaf swaps, fresh subtrees, and
"wrapping" edits that grow structure around a correct fragment). There is no LLM
call. Element typing is what makes accessor-heavy multi-step programs reachable:
the loop variable `it` carries its concrete type (a pair, a char, …), so the
generator reaches for `fst`/`snd` where the raw element would not type-check.

---

## Recursive self-improvement, operationalized

The policy genome is the only thing search behaviour depends on, so improving it
is the only way the system improves its own search. Two evolvable channels:

* **Weight learning** — operators appearing in solved programs are up-weighted,
  concentrating the same budget on productive structure.
* **Library** — subroutines are mined from solved programs and committed only if
  they pass an empirical **META-GATE**: an A/B test on the unsolved frontier
  (incumbent vs incumbent+candidate, equal budget and seeds) showing strictly
  more newly-solved tasks, no regression on solved tasks, and `verifier_fp`
  unchanged. An **encapsulation operator** (after SECTION 24 of the old monolith)
  rewrites adopted solutions to call existing blocks and freezes recurring
  block-containing patterns as new blocks — producing a **lineage** in which a
  block's body references an earlier block.

Lineage observed on a real run (the §4.8 control):
`B0` (atom, round 3) → `B5` (calls `B0`, round 4) → `B6` (calls `B5`, round 7);
every task's adopted solution uses a composed block, so all are load-bearing,
and each child is created and first used strictly later than its parent.

---

## Audit protocol (honest vs cheat signatures)

| command | honest signature | cheat signature |
|---|---|---|
| `python rsi_core.py --mode demo` | real solved/OPEN counts + the §6B metrics table + a printed parent→child lineage pair | no OPEN tasks ever; no lineage; metrics missing or below floor |
| `python rsi_core.py --mode counterfactual` (twice, same seed) | identical digests both times; **positive** adaptive−frozen delta | delta zero/negative but claimed positive; output differs between runs |
| `python rsi_core.py --mode test` | every §4/§9 control passes; `verifier_fp` unchanged | a control missing, skipped, or trivially true |
| auditor re-runs the held-out battery on each adopted program | recomputed solved-count == reported (control `recompute_solved_count`) | mismatch (reported count inflated) |
| `--mode test` reward-hacking control | oracle-removed search won by a WRONG-but-fast program | removing the oracle changes nothing (oracle decorative) |
| `grep -ri "self-invention\|open-ended\|agi\|unbounded\|emergent" .` | no hits over unproven behaviour | grandiose names over toy/unproven behaviour |
| `wc -l` + Phase-0 greps | a small fraction of 50k; toy symbols absent | tens of thousands of lines; toy machinery present |

Determinism: all randomness is seeded; an adoption log is a pure function of the
seed. `--mode counterfactual` prints both arms' adoption digests and is
byte-identical across two same-seed runs.

---

## The honest wall (OPEN tasks) and the escape hatch (§13)

These tasks remain **OPEN for both arms** at the run's budget and are reported as
OPEN:

* `caesar_encode` (family 2, substitution codec) — the per-character body
  `schr(add(sord(it), shift))` is a deceptive, gradient-poor target: a wrong
  shift produces zero common characters, so the fitness landscape is flat until
  the exact body is hit. Reachable only with much larger budget or a learned
  character-shift subroutine, which in turn requires a base-solvable task in that
  sub-family to mine it from (absent here).
* `merge_intervals`, `bracket_depths` (family 4) and `bytecode_interp` (family 1)
  — their reference solutions are ~13–30 IR nodes with stateful `foldl`
  accumulators; this is beyond what stochastic search + memetic local search
  reaches at the budgets used, and no single mined subroutine collapses them into
  range.

This is the anti-toy guarantee in action: rather than retreat to easier tasks,
the system reports the wall with evidence. The minimal IR enrichment that would
move it: a **typed bottom-up enumerator for loop/fold bodies with observational
equivalence** (so the stateful accumulator body of an interpreter or a merge is
found systematically rather than sampled), and a **curriculum that makes each
hard sub-pattern base-solvable in isolation** so it can be mined into a block and
reused. Both are compatible with this architecture; neither is implemented here.

What *is* demonstrated end-to-end and honestly measured: a verifier-first
framework; a family of genuinely multi-step programs (run-length codecs over
structured pair-lists, each clearing the §6B floor) synthesized LLM-free and
gated on held-out tests; a positive, reproducible adaptive-vs-frozen delta; and
the recursion (block-on-block lineage) and reward-hacking controls passing on a
real run.

---

# Phase B — cross-family transfer (the strong claim, measured)

Phase A proved **within-family** reuse (RLE-decode blocks lift harder RLE tasks).
Phase B tests the harder, more meaningful claim:

> **STRONG CLAIM:** a library block MINED while solving family A becomes
> *load-bearing* in an adopted, held-out solution of a structurally-different
> family B (load-bearing = removal makes the task OPEN), AND survives a Socratic
> counterexample search (no distinguishing input vs the sealed reference).

## The five ported mechanisms (each independently ablatable)

| module | mechanism | role |
|---|---|---|
| `search_oe.py` | **M1** bottom-up observational-equivalence solver | a 2nd deterministic solver; clean minimal blocks (dedup on public I/O, holdout sealed) |
| `transfer.py` | **M2** signature-based transfer trigger | decides *which* family-A blocks are candidates for a B-task (public shapes only) |
| `normalize.py` | **M3** verified normalizer | canonicalises a block to a family-agnostic form; accepted only if kernel-equivalent |
| `socratic.py` | **M4** Socratic gate (CEGIS) | rejects *spurious* transfer via a distinguishing-input search judged by the sealed reference |
| `archive.py` | **M5** MAP-Elites archive | quality-diversity, anti-collapse; mining draws across families |

The task suite (`tasks.py`) is extended to **7 structural families** (the
`family_diversity` control asserts ≥4 families, largest ≤40% — measured: 7
families, max 38%). The transfer experiment runs over 4 of them that host
solvable tasks: `seqcode` (string, map), `interval` (int-pair → int-pair, map),
`select` (int-pair filter), `project` (int-pair → int, map).

## Measured result (`--mode transfer`, rotate-B, all mechanisms on)

```
detector self-test (positive control): PASS -- planted A->B transfer:
   load_bearing=True, socratic_admit=True, cross_group=True, spurious_rejected=True

ROTATE-B MATRIX (all 5 mechanisms ON):
  held-out B   frozen  adaptive  lib_blocks  cross-family-transfers
  seqcode        2        1          1       0
  interval       0        0          4       0
  select         1        2          4       1
      project->select block B3 in drop_short: load_bearing=True socratic=True [COUNTS]
  project        1        1          3       0

TOTAL CROSS-FAMILY transfer_families (load-bearing AND Socratic) = 1
```

The matrix is reproducible (two same-seed runs produce the identical matrix).

Ablation (`--mode ablation`) — cross-family transfers per configuration:

```
  config    cross-family transfers
  all-on            1
  M1-off            0      <- the transfer DISAPPEARS without M1
  M2-off            1
  M3-off            1
  M4-off            1
  M5-off            0      <- the transfer DISAPPEARS without M5
```

## The result, stated bluntly: cross-family transfer is REAL but SHALLOW (exactly 1)

There is **exactly one** measured cross-family transfer, and it is genuine under
the full strong-claim definition: the block `B3` (the interval **width**
computation `sub(snd($0), fst($0))`) is mined from the `project` family while
`select` is held out (B-blind), appears in the adopted, held-out-passing solution
of `drop_short` (a `select` task), is **load-bearing** there (removing it and
re-synthesizing at equal budget/seed leaves the task OPEN), and **survives the
Socratic gate** (no distinguishing input over 120 fresh probes vs the sealed
reference). The detector self-test passes and the Socratic gate rejects a planted
spurious block, so this positive is credible, not an artefact.

It is genuinely cross-family: `project` is a map-projection (int-pair → int list)
and `select` is a filter-selection (keep int-pairs by predicate) — different
combinator, different output structure — and `width` plays a different role in
each (the value projected vs the value thresholded). This is library reuse of a
learned utility across structurally different algorithms.

But it is **shallow**, and the matrix says exactly where transfer can and cannot
happen:

1. **Cross-TYPE transfer is 0.** The IR is typed; `seqcode` blocks are
   string-typed (`srepeat`, `sconcat`) and cannot even be invoked in the int-pair
   families. No block ever transfers across the string/int boundary.

2. **Only a small shared utility transfers — no large algorithmic block does.**
   The one transfer is a 3-op width utility. A *large* block is load-bearing only
   if it captures a big chunk of B's solution, and two structurally-different
   families sharing a big chunk would make them re-skins (which the suite
   forbids). So transfer is confined to small utilities shared by same-typed
   families.

3. **The ablation localises the cause.** The transfer needs **M1** (the bottom-up
   OE solver, which yields the *clean minimal* width block — the stochastic
   search alone produces bloated, non-transferable blocks) **and M5** (the
   MAP-Elites archive, which keeps the `project` block available cross-family;
   collapse mode drops it). It is robust to disabling M2/M3/M4 — those are
   candidate-filters and validators, not enablers, so removing them cannot create
   or destroy a genuine transfer.

The honest conclusion: **within-family RSI works (Phase A, positive counterfactual
delta, above); cross-family transfer of a mined block is REAL but minimal — one
shared utility between two
type-compatible families, enabled specifically by clean-block synthesis (M1) and
quality-diversity spread (M5).** It does NOT extend across data types, and no
*large* algorithmic abstraction transfers, because the structure that genuinely
generalises across families (the `map`/`filter`/`fold` control skeleton) is
already a primitive, leaving only small first-order utilities for a mined block to
carry. The enrichment that would deepen this is *higher-order* lifting — blocks
parameterised by a function body, abstracting the control skeleton with a
family-specific hole — which the current first-order block representation cannot
express. That is named as the next step, not faked here.

## Phase B audit commands

| command | what it shows |
|---|---|
| `python rsi_core.py --mode transfer` | rotate-B matrix, B-blind mining, per-block load-bearing + Socratic proofs, detector self-test |
| `python rsi_core.py --mode ablation` | cross-family transfers per mechanism-ablation configuration |
| `python rsi_core.py --mode test` | all Phase-A + Phase-B controls (diversity, B-blind, normalizer, OE-leakage, archive-spread, Socratic-rejects-spurious, detector, ablation-runs) |
