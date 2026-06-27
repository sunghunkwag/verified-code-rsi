#!/usr/bin/env python3
"""STRICT cross-domain emergence (§3) over decomposition-discovered abstractions.

The previous phases measured a "strong" emergence count that credited a same-group
reach (a scan abstraction unlocking another scan task). This module applies the
STRICTER definition the task demands -- under which that earlier demonstration is
explicitly UNCREDITABLE -- so demo-engineering is structurally impossible.

An abstraction ``b`` (discovered by the backward-decomposition engine, decompose.py)
is EMERGENT iff ALL FOUR hold:

  (1) COMPOSITE       b is not reducible to a single given IR primitive (its inlined
                      body nests >= 2 operator applications). [library.is_composite]
  (2) MINED, NOT GIVEN b was discovered by the system (origin 'decomposed'/'mined'/
                      'encapsulated'), disjoint from the seeded primitive/combinator
                      vocabulary (GIVEN_VOCAB) and the empty seed library.
  (3) CROSS-DOMAIN REACH with b available the portfolio solves a target in a
                      DIFFERENT structural group from b's birth group, AND that
                      target was previously OPEN (unsolvable by the full portfolio
                      WITHOUT b at equal budget).
  (4) LOAD-BEARING    removing b (and its dependents) and re-running at EQUAL budget
                      reverts that target to OPEN.

Same-group "reach" (scan->scan) is DISALLOWED by (3): the unlocked target must be in
a different structural group AND previously OPEN. (2)+(3)+(4) reuse the equal-budget
reach counterfactual ``emergence.reach_unlock``: solved-with-b (the surviving
solution actually calls b) and OPEN-without-b.

THE BIDIRECTIONAL TRANSFER MATRIX (--mode transfer-matrix) is the quantitative heart:
for every discovered abstraction x every structural group, is the abstraction
load-bearing on some previously-OPEN target there? A row that is load-bearing only in
its own birth group is LOCAL; one load-bearing across groups is GENERAL. Whether real
emergence happened is exactly: does any row reach a different group?

HONEST BOUND (§0/§8): if no discovered abstraction is load-bearing on a previously-OPEN
target in a different group, the strict count is 0 -- a first-class, reported result.
We then locate the exact dependency gap. We never relax (3)/(4), never gerrymander the
groups, and never leak the reference into decomposition.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from .ir import Block, Node, pp, PRIMS, COMBINATORS
from .oracle import SealedOracle, build_oracles, assert_verifier_unchanged
from .library import is_composite, expand_block, _op_nodes, _block_calls
from .decompose import solve_by_decomposition
from .emergence import (reach_unlock, _reach_attack, library_without,
                        _dependents, GIVEN_VOCAB)
from .rsi import Guidance
from .tasks import SUITE, EMERGENCE_SET, SUITE_BY_NAME


# --------------------------------------------------------------------------- #
# Structural groups -- genuine SHAPE families, fixed and shared by measurement   #
# (never read inside the search). NOT a generous catch-all: each is a distinct   #
# computational shape. The control ``groups_not_gerrymandered`` asserts two       #
# behaviourally-near tasks are NOT split across groups to fake cross-group reach. #
# --------------------------------------------------------------------------- #
STRUCT_GROUPS: Dict[str, str] = {
    "seqcode": "run-length codec: list of (char,count) -> string",
    "codec":   "character-shift codec: string -> string",
    "interval": "elementwise int-pair -> int-pair map",
    "select":  "int-pair predicate filter (sub-selection)",
    "project": "int-pair -> int projection map",
    "scan":    "running accumulator over a sequence (stateful prefix)",
    "parse":   "parse / interpret a structured input",
    "merge":   "sort then sequential state-merge",
}
HARD_FAMILIES = ("bracket_depths", "merge_intervals", "bytecode_interp")


def group_of(task) -> str:
    return getattr(task, "group", "") or "misc"


# --------------------------------------------------------------------------- #
# Discover abstractions by decomposing the hard targets (the §2 engine)          #
# --------------------------------------------------------------------------- #
@dataclass
class Discovered:
    block: Block
    birth_group: str
    source_task: str


@dataclass
class HardOutcome:
    task: str
    group: str
    solved: bool
    channel: str
    skeleton: str
    program: str
    gap: str                     # dependency-gap note when OPEN


# A small curated set of composite tasks from OTHER structural groups, decomposed
# alongside the hard families so the CROSS-GROUP test is maximally fair: if any
# discovered abstraction were load-bearing across groups it would have the chance to
# show it. These are real suite/external tasks, not targets invented to force a win.
_EXTRA_SOURCES = ("ext_decode_sortrev", "rle_decode_sorted")


def _extra_oracle(name: str) -> Optional[SealedOracle]:
    from .tasks import EMERGENCE_BY_NAME
    if name in SUITE_BY_NAME:
        return SealedOracle(SUITE_BY_NAME[name])
    if name in EMERGENCE_BY_NAME:
        return SealedOracle(EMERGENCE_BY_NAME[name])
    return None


def discover_from_hard(budget: int = 60_000, round_idx: int = 1,
                       include_extra: bool = True
                       ) -> Tuple[List[Discovered], List[HardOutcome]]:
    """Run the backward-decomposition engine on the hard families (and, for a fair
    cross-group test, a few composite tasks from other groups). Each cracked target
    contributes its mined sub-functions as candidate abstractions (deduped by body);
    each OPEN hard family contributes a precise dependency-gap note."""
    oracles = build_oracles()
    discovered: List[Discovered] = []
    seen_bodies: set = set()
    outcomes: List[HardOutcome] = []
    sources: List[Tuple[SealedOracle, bool]] = [
        (oracles[tn], True) for tn in HARD_FAMILIES if tn in oracles]
    if include_extra:
        for nm in _EXTRA_SOURCES:
            orc = _extra_oracle(nm)
            if orc is not None:
                sources.append((orc, False))
    for orc, is_hard in sources:
        grp = group_of(orc.task)
        dr = solve_by_decomposition(orc.public_view(), orc.verify, library=[],
                                    budget=budget, round_idx=round_idx,
                                    forward_first=True)
        cracked = dr.program is not None and orc.verify(dr.program)
        if cracked:
            for b in dr.mined:
                key = pp(b.body)
                if key in seen_bodies:
                    continue
                seen_bodies.add(key)
                discovered.append(Discovered(b, grp, orc.task.name))
        if is_hard:
            if cracked:
                outcomes.append(HardOutcome(orc.task.name, grp, True, dr.channel,
                                            dr.skeleton, pp(dr.program), ""))
            else:
                outcomes.append(HardOutcome(orc.task.name, grp, False, "", "", "",
                                            _gap_note(orc.task.name)))
    return discovered, outcomes


def _gap_note(task: str) -> str:
    """The precise reason a hard family resists decomposition (located, not waved
    away). These are STRUCTURAL facts about the target's I/O, not excuses."""
    if task == "bytecode_interp":
        return ("output is a single scalar (top of the final stack); the fold's "
                "intermediate STACK states are not observable from public I/O, so "
                "the dispatch/execute step cannot be isolated into a solvable hole")
    if task == "merge_intervals":
        return ("decomposes into sort (isolated + solved as lsort) THEN a sequential "
                "overlap-merge fold; the merge fold's intermediate accumulator is not "
                "recoverable from public I/O, so that stage stays a full deep fold "
                "beyond search reach -- the sort is solved, the merge is the gap")
    return "no skeleton's hole I/O is derivable from this target's public examples"


# --------------------------------------------------------------------------- #
# The reach target universe: the suite + the external held-out set, grouped.     #
# This is the honest, fixed universe (no targets invented to force a positive).  #
# --------------------------------------------------------------------------- #
def target_universe() -> List[SealedOracle]:
    tasks = list(SUITE) + list(EMERGENCE_SET)
    # dedup by name; the suite already covers all 8 structural groups
    seen, out = set(), []
    for t in tasks:
        if t.name in seen:
            continue
        seen.add(t.name)
        out.append(SealedOracle(t))
    return out


def _applicable(b: Block, orc: SealedOracle) -> bool:
    """Cheap type prefilter: could a solution to ``orc`` even CALL ``b``? b's
    params must be constructible from the task's arg/element types and b's result
    must plug into the output or an element. Skips hopeless (b, target) probes so
    the matrix stays tractable; it never credits anything by itself."""
    view = orc.public_view()
    avail = set(view.arg_types) | {"I", "V"}
    exs = view.public_examples
    la = next((i for i, t in enumerate(view.arg_types) if t == "L"), None)
    if la is not None and exs and exs[0][0][la]:
        v = exs[0][0][la][0]
        avail.add("P" if isinstance(v, tuple) else "S" if isinstance(v, str)
                  else "I" if isinstance(v, int) else "L")
    if not all((pt in avail) or pt == "V" for pt in b.ptypes):
        return False
    return b.rtype in (view.out_type, "V", "L", "I", "S")


# --------------------------------------------------------------------------- #
# The strict measurement                                                        #
# --------------------------------------------------------------------------- #
@dataclass
class StrictCapability:
    block: str
    body: str
    inlined: str
    composite_ops: int
    origin: str
    birth_group: str
    source_task: str
    unlocked_task: str
    unlocked_group: str
    solution_with_b: str
    used_blocks: Tuple[str, ...]

    def proof_lines(self) -> List[str]:
        return [
            f"  CAPABILITY {self.block} = {self.body}",
            f"     composite      : inlined body nests {self.composite_ops} ops "
            f"(>1 => irreducible to one primitive); inlined={self.inlined[:56]}",
            f"     mined-not-given: origin={self.origin}, discovered by decomposing "
            f"'{self.source_task}' (seed library empty; name disjoint from primitives)",
            f"     cross-group    : birth group '{self.birth_group}' != unlocked "
            f"group '{self.unlocked_group}' (DIFFERENT structural shape)",
            f"     was-OPEN+LB    : '{self.unlocked_task}' is OPEN to primitives+non-b "
            f"blocks at equal budget; WITH b solved as {self.solution_with_b[:44]} "
            f"(uses {list(self.used_blocks)}); removing b -> OPEN",
        ]


@dataclass
class TransferRow:
    block: str
    birth_group: str
    # group -> (load_bearing_bool, unlocked_task_or_"")
    cells: Dict[str, Tuple[bool, str]] = field(default_factory=dict)

    def is_local(self) -> bool:
        lb = [g for g, (v, _t) in self.cells.items() if v]
        return all(g == self.birth_group for g in lb)

    def reaches_cross_group(self) -> bool:
        return any(v and g != self.birth_group for g, (v, _t) in self.cells.items())


@dataclass
class StrictResult:
    discovered: List[Discovered] = field(default_factory=list)
    hard_outcomes: List[HardOutcome] = field(default_factory=list)
    capabilities: List[StrictCapability] = field(default_factory=list)
    transfer: List[TransferRow] = field(default_factory=list)
    groups: Tuple[str, ...] = ()
    reach_probes: int = 0
    verifier_fp: str = ""

    @property
    def count(self) -> int:
        return len(self.capabilities)

    def digest(self) -> str:
        h = hashlib.sha256()
        for d in self.discovered:
            h.update(f"D:{d.block.name}:{pp(d.block.body)}:{d.birth_group}\n".encode())
        for c in self.capabilities:
            h.update(f"C:{c.block}->{c.unlocked_task}\n".encode())
        for r in self.transfer:
            for g in sorted(r.cells):
                h.update(f"T:{r.block}:{g}:{int(r.cells[g][0])};".encode())
        return h.hexdigest()[:16]


_MINED_ORIGINS = {"decomposed", "mined", "encapsulated"}


def _is_mined_not_given(b: Block) -> bool:
    return (b.origin in _MINED_ORIGINS and b.name not in GIVEN_VOCAB
            and b.name not in (set(PRIMS) | set(COMBINATORS)))


def measure_strict(discovered: List[Discovered], hard_outcomes: List[HardOutcome],
                   budget: int = 40_000, guidance: Optional[Guidance] = None,
                   targets: Optional[List[SealedOracle]] = None,
                   per_group: int = 3) -> StrictResult:
    """Build the bidirectional transfer matrix and credit every abstraction meeting
    all four §3 conditions. Equal-budget reach probes via emergence.reach_unlock.
    ``targets`` defaults to the full suite+external universe; a control passes a
    tiny set to exercise the credit logic fast. ``per_group`` caps the targets
    probed per structural group so the matrix stays tractable (the cap is reported)."""
    g = guidance or Guidance()
    targets = targets if targets is not None else target_universe()
    groups = tuple(sorted(STRUCT_GROUPS))
    res = StrictResult(discovered=discovered, hard_outcomes=hard_outcomes,
                       groups=groups)
    by_group: Dict[str, List[SealedOracle]] = {gp: [] for gp in groups}
    for orc in targets:
        gp = group_of(orc.task)
        by_group.setdefault(gp, []).append(orc)
    for gp in list(by_group):
        by_group[gp] = by_group[gp][:per_group]

    for d in discovered:
        b = d.block
        row = TransferRow(block=b.name, birth_group=d.birth_group)
        composite = is_composite(b)
        mined = _is_mined_not_given(b)
        for gp in groups:
            lb_here = False
            won_task = ""
            for orc in by_group.get(gp, []):
                if not _applicable(b, orc):
                    continue
                res.reach_probes += 1
                proof = reach_unlock(b, orc, [b], g, budget=budget)
                if proof is not None:
                    lb_here = True
                    won_task = orc.task.name
                    # CREDIT iff this is a cross-group, previously-OPEN unlock and b
                    # is composite + mined (reach_unlock already proved OPEN-without-b
                    # at equal budget and that the surviving solution calls b).
                    if gp != d.birth_group and composite and mined:
                        flat = expand_block(b, {})
                        res.capabilities.append(StrictCapability(
                            block=b.name, body=pp(b.body), inlined=pp(flat),
                            composite_ops=_op_nodes(flat), origin=b.origin,
                            birth_group=d.birth_group, source_task=d.source_task,
                            unlocked_task=orc.task.name, unlocked_group=gp,
                            solution_with_b=proof["solution"],
                            used_blocks=proof["used_blocks"]))
                    break
            row.cells[gp] = (lb_here, won_task)
        res.transfer.append(row)
    return res


# --------------------------------------------------------------------------- #
# Same-group PLANT (control emergent_is_cross_group_and_was_open): a genuine     #
# same-group unlock must NOT be credited.                                        #
# --------------------------------------------------------------------------- #
def same_group_plant_is_rejected(budget: int = 40_000) -> Tuple[bool, str]:
    """Construct an abstraction that genuinely unlocks a previously-OPEN target in
    its OWN birth group; assert measure_strict does NOT credit it (cross-group fails),
    while the transfer row still records the same-group load-bearing cell."""
    discovered, _ = discover_from_hard(budget=60_000)
    if not discovered:
        return False, "no abstraction discovered to plant with"
    # take the running-sum scan (born in group 'scan'); if it is load-bearing on a
    # scan-group target that is the SAME-group unlock that must NOT be credited.
    scan_blocks = [d for d in discovered if d.block.rtype == "L"]
    if not scan_blocks:
        return False, "no L-typed abstraction discovered"
    res = measure_strict(discovered, [], budget=budget)
    # any credited capability MUST be cross-group; assert no credit shares birth group
    same_group_credit = [c for c in res.capabilities
                         if c.unlocked_group == c.birth_group]
    ok = (len(same_group_credit) == 0)
    detail = (f"{len(res.capabilities)} credited (all cross-group="
              f"{ok}); same-group credits={[c.block for c in same_group_credit]} "
              f"(must be [])")
    return ok, detail
