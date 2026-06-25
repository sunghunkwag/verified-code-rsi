#!/usr/bin/env python3
"""The STRONG emergence measurement: did the system INVENT a composite capability
it was never given as a primitive, use it load-bearing, and does that invention
let it reach a task it could not reach before?

An abstraction ``b`` (a mined / encapsulated library block) is an EMERGENT
CAPABILITY iff ALL FOUR hold (§3):

  (1) COMPOSITE      b is NOT expressible as a single given IR primitive -- its
                     inlined body nests >= 2 operator applications (``is_composite``).
  (2) LOAD-BEARING   b is used in an adopted, holdout-passing solution, and removing
                     b reverts that task to OPEN at equal budget.
  (3) NOT GIVEN      b was MINED from the system's own solutions, never pre-seeded
                     (the seed library is empty; the credited set is disjoint from
                     the given primitives + the pre-seeded library).
  (4) ENABLES REACH  with b available the portfolio solves a task it does NOT solve
                     with primitives + non-b blocks at equal budget (ideally in a
                     HARDER, stateful family).

(2) and (4) are tested by the SAME counterfactual: attack a target task with the
full library and with the library MINUS b (and minus every block that transitively
calls b -- the honest "primitives + non-b blocks" set); b is credited only if the
target is solved WITH b and OPEN WITHOUT it, and the surviving solution actually
calls b (or a b-dependent). Everything is the sealed portfolio at the loop's attack
budget, so a credit is a real, holdout-verified, reproducible reach gain -- not a
claim. If no block satisfies all four, the count is 0 and that is reported as the
finding (§8); nothing is credited that does not pass every proof.

The HONEST BOUND (§0): invention is still bounded by the verifiable domain -- a
target is real only because its sealed reference defines checkable ground truth.
Emergence here = an UN-DESIGNED composite capability arising within that domain,
measured -- not a singularity.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .ir import Block, Node, pp, inline, PRIMS, COMBINATORS
from .interp import run
from .oracle import SealedOracle
from .complexity import adopted_program_ops, MIN_SOLUTION_OPS
from .library import (is_composite, input_coupled, score_abstraction,
                      expand_block, _op_nodes, _block_calls, stateful_policy)
from .search import synthesize
from .search_oe import oe_solve
from .openended import attack, solved_and_floor_ok, OpenEndedResult, _bm
from .rsi import Guidance

# The reach counterfactual budget -- IDENTICAL for the with-b and without-b arms,
# so a credit is never an artefact of unequal budget. The memetic search with a
# high block-call probability finds a block-using solution FAST when one exists
# (M1 abstraction-first); the bottom-up OE is a backup (it explodes with a large
# library, so it gets a small budget).
REACH_MEMETIC = 65_000
REACH_BLOCK_PROB = 0.5
REACH_OE = 35_000


def _reach_attack(orc: SealedOracle, library: List[Block]
                  ) -> Optional[Node]:
    """The deterministic reach probe: the memetic search (newest blocks reused
    first, M1) then a bounded bottom-up OE, both with the given library. Returns a
    holdout-verified, floor-clearing program or None. Equal budget for every call."""
    bm = _bm(library)
    pol = stateful_policy()
    pol.blocks = list(library)
    pol.block_prob = REACH_BLOCK_PROB if library else 0.0
    p, _ = synthesize(orc.public_view(), pol, REACH_MEMETIC, seed=7)
    if solved_and_floor_ok(p, orc, library):
        return p
    p = oe_solve(orc.public_view(), blocks=library, max_size=10,
                 eval_budget=REACH_OE)
    if solved_and_floor_ok(p, orc, library):
        return p
    return None

# The families the curated suite does NOT teach with a flat map/filter -- reaching
# one of these is the "harder family" bonus of §3(4).
HARDER_GROUPS = {"scan", "parse", "merge", "state"}
# the given vocabulary: a credited capability must be disjoint from this (§3(3)).
GIVEN_VOCAB = set(PRIMS) | set(COMBINATORS)


@dataclass
class InventedCapability:
    name: str
    body: str
    inlined_body: str
    composite_op_count: int          # operator nodes in the INLINED body (>=2)
    origin: str                      # 'mined' | 'encapsulated'
    calls: Tuple[str, ...]           # earlier blocks this one is built on
    m2: dict                         # the M2 multi-objective score components
    unlocked_task: str
    unlocked_family: str
    solution_with_b: str
    used_blocks: Tuple[str, ...]     # blocks load-bearing in the unlocking solution
    harder_family: bool

    def proof_lines(self) -> List[str]:
        return [
            f"  CAPABILITY {self.name} = {self.body}",
            f"     composite-proof : inlined body nests {self.composite_op_count} "
            f"operator nodes (>1 => irreducible to one primitive); "
            f"inlined = {self.inlined_body[:64]}",
            f"     not-given-proof : origin={self.origin}, built on {list(self.calls)} "
            f"(mined from the system's OWN solutions; seed library was empty)",
            f"     load-bearing    : used {list(self.used_blocks)} in the solution; "
            f"removing it reverts the task to OPEN at equal budget",
            f"     reach-unlock    : solved '{self.unlocked_task}' "
            f"[{self.unlocked_family}{' / HARDER' if self.harder_family else ''}] "
            f"as {self.solution_with_b[:54]}",
            f"     M2 score        : {self.m2['score']:.3f} (compression="
            f"{self.m2['compression']:.2f} transfer={self.m2['transfer']:.2f} "
            f"anti_cheat={self.m2['anti_cheat']:.2f})",
        ]


@dataclass
class EmergenceStrong:
    capabilities: List[InventedCapability] = field(default_factory=list)
    reach_attempted: int = 0
    library_size: int = 0
    composite_blocks: int = 0
    reach_target_names: List[str] = field(default_factory=list)
    hard_family_reach: Dict[str, bool] = field(default_factory=dict)

    @property
    def count(self) -> int:
        return len(self.capabilities)

    def digest(self) -> str:
        h = hashlib.sha256()
        for c in self.capabilities:
            h.update(f"{c.name}|{c.inlined_body}|{c.unlocked_task}\n".encode())
        for k in sorted(self.hard_family_reach):
            h.update(f"{k}={self.hard_family_reach[k]};".encode())
        return h.hexdigest()[:16]


# --------------------------------------------------------------------------- #
# Library surgery: remove a block AND everything that transitively calls it     #
# (the honest "primitives + non-b blocks" counterfactual set).                  #
# --------------------------------------------------------------------------- #
def _dependents(block_name: str, library: List[Block]) -> set:
    """Names of blocks that transitively call ``block_name`` (incl. itself)."""
    dep = {block_name}
    changed = True
    while changed:
        changed = False
        for b in library:
            if b.name in dep:
                continue
            if dep & set(b.calls()):
                dep.add(b.name)
                changed = True
    return dep


def library_without(library: List[Block], block_name: str) -> List[Block]:
    drop = _dependents(block_name, library)
    return [b for b in library if b.name not in drop]


# --------------------------------------------------------------------------- #
# Reach test for ONE block on ONE target oracle (the §3(2)+(4) counterfactual)  #
# --------------------------------------------------------------------------- #
def _solution_uses(prog: Node, dropped: set) -> bool:
    return bool(set(_block_calls(prog)) & dropped)


def reach_unlock(b: Block, target: SealedOracle, full_lib: List[Block],
                 guidance: Guidance) -> Optional[dict]:
    """Return a proof dict iff ``target`` is solved WITH the full library but OPEN
    with the library MINUS b (and its dependents), and the surviving solution
    actually calls b / a b-dependent. The with-b and without-b probes use the
    SAME deterministic channels at the SAME budget, so a credit is a real reach
    gain, not a budget artefact."""
    without = library_without(full_lib, b.name)
    dropped = _dependents(b.name, full_lib)
    prog_with = _reach_attack(target, full_lib)
    if prog_with is None:
        return None
    if not _solution_uses(prog_with, dropped):
        return None                                   # b not load-bearing here
    prog_without = _reach_attack(target, without)
    if prog_without is not None:
        return None                                   # solvable without b -> no unlock
    return {"solution": pp(prog_with),
            "used_blocks": tuple(sorted(set(_block_calls(prog_with)) & dropped))}


# --------------------------------------------------------------------------- #
# The whole measurement                                                         #
# --------------------------------------------------------------------------- #
def _dedup_targets(specs) -> list:
    seen, out = set(), []
    for sp in specs:
        key = (sp.arg_types, sp.out_type, pp(sp.reference))
        if key not in seen:
            seen.add(key)
            out.append(sp)
    return out


def measure_strong(res: OpenEndedResult, hard_suite: List[SealedOracle]
                   ) -> EmergenceStrong:
    """Credit every library block that satisfies §3(1)-(4). Target-driven so it is
    fast and honest: attack each reach target ONCE with the full library; if it is
    solved, the load-bearing block is among the ones its solution calls, so test
    removal only for THOSE composite blocks (equal-budget without-b probe). A block
    is credited the first time it is the difference between solved and OPEN."""
    out = EmergenceStrong()
    library = res.library
    out.library_size = len(library)
    bm = _bm(library)
    out.composite_blocks = sum(1 for b in library if is_composite(b, bm))
    seed_set = set(res.seed_blocks)                   # pre-seeded (empty)
    credited_names: set = set()

    # bound the reach probe set: the de-duped deep minted tasks (capped) + the hard
    # suite families. One credited capability is sufficient for the headline; the
    # cap keeps the measurement tractable and deterministic.
    minted_oracles = [SealedOracle(sp)
                      for sp in _dedup_targets(res.reach_targets)[:6]]
    out.reach_target_names = [o.task.name for o in minted_oracles]
    targets = minted_oracles + list(hard_suite)

    for tgt in targets:
        out.reach_attempted += 1
        prog_with = _reach_attack(tgt, library)
        if prog_with is None:
            continue                                  # unsolvable even with full lib
        used = set(_block_calls(prog_with))
        # the load-bearing block is among those the solution calls; test removal of
        # each composite, not-yet-credited one (cheap: usually 1-2 calls).
        for bname in sorted(used):
            if bname in credited_names or bname not in bm:
                continue
            b = bm[bname]
            if (b.name in seed_set or b.origin not in ("mined", "encapsulated")
                    or not is_composite(b, bm)):     # (1) composite + (3) not given
                continue
            without = library_without(library, bname)
            if not _solution_uses(prog_with, _dependents(bname, library)):
                continue
            if _reach_attack(tgt, without) is not None:
                continue                              # solvable without b -> no unlock
            # CREDIT: b is composite, mined, load-bearing, and reach-unlocking.
            flat = expand_block(b, bm)
            fam = tgt.task.group
            out.capabilities.append(InventedCapability(
                name=b.name, body=pp(b.body), inlined_body=pp(flat),
                composite_op_count=_op_nodes(flat), origin=b.origin,
                calls=tuple(b.calls()),
                m2=score_abstraction(b, res.adopted_pairs, bm),
                unlocked_task=tgt.task.name, unlocked_family=fam,
                solution_with_b=pp(prog_with),
                used_blocks=tuple(sorted(used & _dependents(bname, library))),
                harder_family=fam in HARDER_GROUPS))
            credited_names.add(bname)

    # frontier check: which hard suite families enter reach as the library grows?
    for orc in hard_suite:
        out.hard_family_reach[orc.task.name] = _reach_attack(orc, library) is not None
    return out
