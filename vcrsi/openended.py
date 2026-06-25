#!/usr/bin/env python3
"""The open-ended self-generated curriculum loop + the emergence measurement.

This is the only place the system's targets are NOT human-given. Each generation
the system synthesises its own reference programs (``generator.py``), keeps only
those passing the TRIPLE LOCK, attacks the survivors with its current portfolio,
and trains its guidance + library on what it solves -- with no human target inside
the loop (§3). We then measure whether this self-directed exploration produced
capability on an EXTERNAL, never-generated, human-authored held-out set (§4).

THE TRIPLE LOCK (all machine-checked; a task counts only if all three hold):
  L1  whitelist        §6A family, STRUCTURED input, NOT a flat-int reduction.
  L2  §6B floor        distinct_ops>=5, a loop, recursion-or-aux-structure,
                       exec_depth>=6 -- COMPUTED from the reference's AST + trace.
  L3  self-easiness     the CURRENT solver at LOW (probe) budget CANNOT solve it.
The L3 frontier ratchets because BOTH the probe (low budget) and the attack (high
budget) use the SAME improving guidance + library: as they improve, the probe
rejects more easy tasks (floor rises) while the attack cracks harder ones
(ceiling rises), so the newly-solved band climbs in §6B difficulty -- or it does
not, which we report honestly (§8).

THE HONEST BOUND (§0): self-generation does NOT escape the verifiable domain. A
generated task is real only because its sealed reference defines a checkable
ground truth; "emergence" here means capability arising from self-directed
exploration WITHIN that domain, measured against an external set -- not magic.
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .ir import Block, Node, pp
from .interp import run
from .oracle import SealedOracle
from .complexity import complexity_floor, adopted_program_ops, MIN_SOLUTION_OPS
from .generator import GenSpec, generate_spec, on_whitelist
from .search import synthesize
from .search_oe import oe_solve
from .prm_beam import prm_beam_synthesize
from .library import broad_policy, mine_blocks
from .archive import MapElites
from .rsi import Guidance

# Probe (L3) vs attack budgets. Both run the CURRENT solver; only the budget
# differs -- that gap, times an improving solver, is where the frontier lives.
PROBE_OE = 12_000
PROBE_MEMETIC = 9_000
PROBE_BEAM = (10, 18)          # (width, layers)
ATTACK_OE = 50_000
ATTACK_MEMETIC = 32_000
ATTACK_BEAM = (22, 30)
LIBRARY_CAP = 10


# --------------------------------------------------------------------------- #
# Solver portfolios (probe = low budget, attack = high budget; same machinery)  #
# --------------------------------------------------------------------------- #
def _policy_with(blocks: List[Block]):
    pol = broad_policy()
    pol.blocks = list(blocks)
    pol.block_prob = 0.22 if blocks else 0.0
    return pol


def probe_solves(orc: SealedOracle, guidance: Guidance,
                 blocks: List[Block]) -> bool:
    """L3 probe: can the CURRENT solver already do this at LOW budget?"""
    v = orc.public_view()
    p = oe_solve(v, blocks=blocks, max_size=10, eval_budget=PROBE_OE)
    if p is not None and orc.verify(p, _bm(blocks)):
        return True
    p, _ = synthesize(v, _policy_with(blocks), PROBE_MEMETIC, seed=5)
    if p is not None and orc.verify(p, _bm(blocks)):
        return True
    w, l = PROBE_BEAM
    p, _ = prm_beam_synthesize(v, guidance.prm, blocks, width=w, max_layers=l,
                               verify=lambda pr: orc.verify(pr, _bm(blocks)))
    return p is not None and orc.verify(p, _bm(blocks))


def attack(orc: SealedOracle, guidance: Guidance,
           blocks: List[Block]) -> Tuple[Optional[Node], str]:
    """Full portfolio at HIGH budget: OE, memetic, then the PRM-guided beam."""
    v = orc.public_view()
    bm = _bm(blocks)
    p = oe_solve(v, blocks=blocks, max_size=12, eval_budget=ATTACK_OE)
    if p is not None and orc.verify(p, bm):
        return p, "oe"
    p, _ = synthesize(v, _policy_with(blocks), ATTACK_MEMETIC, seed=7)
    if p is not None and orc.verify(p, bm):
        return p, "memetic"
    w, l = ATTACK_BEAM
    p, _ = prm_beam_synthesize(v, guidance.prm, blocks, width=w, max_layers=l,
                               verify=lambda pr: orc.verify(pr, bm))
    if p is not None and orc.verify(p, bm):
        return p, "beam"
    return None, "-"


def _bm(blocks: List[Block]) -> Dict[str, Block]:
    return {b.name: b for b in blocks}


def solved_and_floor_ok(prog: Optional[Node], orc: SealedOracle,
                        blocks: List[Block]) -> bool:
    """A win counts ONLY if the program passes the SEALED held-out battery AND
    clears the adopted-program floor (after inlining library calls)."""
    bm = _bm(blocks)
    return (prog is not None and orc.verify(prog, bm)
            and adopted_program_ops(prog, bm) >= MIN_SOLUTION_OPS)


# --------------------------------------------------------------------------- #
# The triple lock                                                              #
# --------------------------------------------------------------------------- #
def triple_lock(spec: GenSpec, blocks: List[Block], guidance: Guidance
                ) -> Tuple[bool, Optional[SealedOracle], str, dict]:
    """Return (passed_all_three, oracle, first_failed_lock, floor_metrics)."""
    if not on_whitelist(spec):                       # L1
        return False, None, "L1", {}
    orc = SealedOracle(spec)
    ok, m = complexity_floor(orc)                    # L2
    if not ok:
        return False, orc, "L2", m
    if probe_solves(orc, guidance, blocks):          # L3
        return False, orc, "L3", m
    return True, orc, "ok", m


# --------------------------------------------------------------------------- #
# Library admission (mine from a solved program; additive, deduped, capped)     #
# --------------------------------------------------------------------------- #
def _admit_blocks(library: List[Block], prog: Node, gen: int) -> int:
    have = {pp(b.body) for b in library}
    added = 0
    for c in mine_blocks(prog, library, gen, max_new=2):
        if len(library) >= LIBRARY_CAP:
            break
        if pp(c.body) not in have:
            have.add(pp(c.body))
            library.append(c)
            added += 1
    return added


# --------------------------------------------------------------------------- #
# Per-generation + whole-run result                                            #
# --------------------------------------------------------------------------- #
@dataclass
class GenStat:
    gen: int
    minted: int
    locked: int
    solved: int
    stalled: bool
    lock_fail: Dict[str, int]
    solved_groups: List[str]
    newly_metrics: List[Tuple[int, int]]   # (distinct_ops, exec_depth) of solves


@dataclass
class OpenEndedResult:
    per_gen: List[GenStat] = field(default_factory=list)
    guidance: Guidance = field(default_factory=Guidance)
    library: List[Block] = field(default_factory=list)
    archive: MapElites = field(default_factory=MapElites)
    total_attacks: int = 0
    total_solved: int = 0

    def frontier_trajectory(self) -> List[dict]:
        out = []
        for gs in self.per_gen:
            ms = gs.newly_metrics
            if ms:
                ops = sorted(m[0] for m in ms)
                dep = sorted(m[1] for m in ms)
                out.append({"gen": gs.gen, "solved": len(ms),
                            "ops_min": ops[0], "ops_med": ops[len(ops) // 2],
                            "depth_min": dep[0], "depth_med": dep[len(dep) // 2],
                            "groups": sorted(set(gs.solved_groups))})
            else:
                out.append({"gen": gs.gen, "solved": 0,
                            "stalled": gs.stalled})
        return out

    def digest(self) -> str:
        h = hashlib.sha256()
        for gs in self.per_gen:
            h.update(f"g{gs.gen}:{gs.minted},{gs.locked},{gs.solved};".encode())
        h.update(("|" + "|".join(self.guidance.wave_digests)).encode())
        for b in self.library:
            h.update(f"B:{pp(b.body)}\n".encode())
        return h.hexdigest()[:16]


# --------------------------------------------------------------------------- #
# The open-ended loop (§3): generate -> triple-lock -> attack -> learn          #
# --------------------------------------------------------------------------- #
def run_openended(generations: int = 6, batch: int = 8, seed: int = 0,
                  verbose: bool = False) -> OpenEndedResult:
    res = OpenEndedResult()
    g = res.guidance
    library = res.library
    solved_progs: List[Tuple[SealedOracle, Node]] = []

    for gen in range(generations):
        minted = locked = solved = 0
        lock_fail: Dict[str, int] = {"L1": 0, "L2": 0, "L3": 0}
        solved_groups: List[str] = []
        newly: List[Tuple[int, int]] = []
        registered: List[Tuple[GenSpec, SealedOracle, dict]] = []

        for i in range(batch):
            crng = random.Random(seed * 1009 + gen * 97 + i)
            sp = generate_spec(crng, gen, i, blocks=library)
            if sp is None:
                continue
            minted += 1
            ok, orc, reason, m = triple_lock(sp, library, g)
            if not ok:
                lock_fail[reason] = lock_fail.get(reason, 0) + 1
                continue
            locked += 1
            registered.append((sp, orc, m))

        if not registered:
            res.per_gen.append(GenStat(gen, minted, 0, 0, True, lock_fail, [], []))
            if verbose:
                print(f"  gen {gen}: minted {minted}, 0 passed the triple lock "
                      f"-> FRONTIER STALLED")
            continue

        for sp, orc, m in registered:
            res.total_attacks += 1
            prog, ch = attack(orc, g, library)
            if not solved_and_floor_ok(prog, orc, library):
                continue
            # a genuine win: solved AND sealed-holdout-verified
            solved += 1
            res.total_solved += 1
            solved_groups.append(sp.group)
            newly.append((m["distinct_ops"], m["max_exec_depth"]))
            g.train_on(prog, orc.public_view(), library)
            res.archive.consider_solution(sp.group, sp.out_type, prog, sp.name)
            _admit_blocks(library, prog, gen)
            solved_progs.append((orc, prog))
            if verbose:
                print(f"  gen {gen}: solved {sp.name} [{sp.group}/"
                      f"{sp.note.split(':')[-1]}] via {ch} "
                      f"(ops={m['distinct_ops']}, depth={m['max_exec_depth']})")
        g.record_digest()
        res.per_gen.append(GenStat(gen, minted, locked, solved, False,
                                   lock_fail, solved_groups, newly))
        if verbose:
            print(f"  gen {gen}: minted={minted} locked={locked} solved={solved}"
                  f" | lib={len(library)} | lockfail={lock_fail}")
    return res


# =========================================================================== #
# THE EMERGENCE MEASUREMENT (§4)                                               #
# --------------------------------------------------------------------------- #
# Open-ended arm: train guidance + library ONLY on the system's OWN generated   #
# tasks (run_openended), then evaluate FROZEN on an EXTERNAL, never-generated,   #
# human-authored held-out set. Baseline arm: identical budget/seeds but trained  #
# only on the FIXED suite (no self-generation). Emergence delta = (external      #
# solved by open-ended) - (external solved by baseline). The external set never  #
# enters generation or training in either arm (emergence_set_is_sealed).         #
# =========================================================================== #
def train_on_suite(n_attacks: int, seed: int) -> Tuple[Guidance, List[Block]]:
    """The baseline arm's training: the SAME machinery as the open-ended loop
    (attack -> train guidance -> mine library) but over the FIXED suite, with no
    self-generation. Performs exactly ``n_attacks`` attacks (cycling the suite)
    at the same per-task budget and seed scheme -> equal training budget."""
    from .tasks import SUITE
    g = Guidance()
    library: List[Block] = []
    order = list(SUITE)
    for a in range(n_attacks):
        task = order[a % len(order)]
        orc = SealedOracle(task)
        prog, _ch = attack(orc, g, library)
        if solved_and_floor_ok(prog, orc, library):
            g.train_on(prog, orc.public_view(), library)
            _admit_blocks(library, prog, a // max(1, len(order)))
        if (a + 1) % len(order) == 0:
            g.record_digest()
    g.record_digest()
    return g, library


def eval_on_external(externals: List[SealedOracle], guidance: Guidance,
                     library: List[Block], channel: str = "full") -> List[str]:
    """Count external-set tasks solved FROZEN (no training). ``channel='full'``
    uses the whole portfolio (the honest 'got better at unseen tasks' measure);
    ``channel='beam'`` uses ONLY the PRM-guided beam, isolating the learned
    guidance (the component the two arms differ in)."""
    solved: List[str] = []
    bm = _bm(library)
    for orc in externals:
        v = orc.public_view()
        if channel == "beam":
            w, l = ATTACK_BEAM
            p, _ = prm_beam_synthesize(v, guidance.prm, library, width=w,
                                       max_layers=l,
                                       verify=lambda pr: orc.verify(pr, bm))
            ok = solved_and_floor_ok(p, orc, library)
        else:
            p, _ch = attack(orc, guidance, library)
            ok = solved_and_floor_ok(p, orc, library)
        if ok:
            solved.append(orc.task.name)
    return sorted(solved)


@dataclass
class EmergenceResult:
    open_res: OpenEndedResult
    open_solved_full: List[str]
    base_solved_full: List[str]
    open_solved_beam: List[str]
    base_solved_beam: List[str]
    n_external: int
    attacks: int
    verifier_fp: str

    @property
    def delta_full(self) -> int:
        return len(self.open_solved_full) - len(self.base_solved_full)

    @property
    def delta_beam(self) -> int:
        return len(self.open_solved_beam) - len(self.base_solved_beam)

    def digest(self) -> str:
        h = hashlib.sha256()
        h.update(("F" + ",".join(self.open_solved_full)).encode())
        h.update(("|" + ",".join(self.base_solved_full)).encode())
        h.update(("B" + ",".join(self.open_solved_beam)).encode())
        h.update(("|" + ",".join(self.base_solved_beam)).encode())
        h.update(self.open_res.digest().encode())
        return h.hexdigest()[:16]


def run_emergence(generations: int = 5, batch: int = 7, seed: int = 0,
                  verbose: bool = False) -> EmergenceResult:
    from .tasks import EMERGENCE_SET
    from .oracle import assert_verifier_unchanged, build_oracles
    fp = assert_verifier_unchanged(build_oracles(), "emergence.start")

    open_res = run_openended(generations, batch, seed, verbose=verbose)
    # equal budget: the baseline performs the SAME number of attacks as the
    # open-ended arm actually performed (post-lock), same seeds.
    n_attacks = max(open_res.total_attacks, len(EMERGENCE_SET))
    base_g, base_lib = train_on_suite(n_attacks, seed)

    # FROZEN evaluation on the SEALED external set (never trained on).
    externals = [SealedOracle(t) for t in EMERGENCE_SET]
    return EmergenceResult(
        open_res=open_res,
        open_solved_full=eval_on_external(externals, open_res.guidance,
                                          open_res.library, "full"),
        base_solved_full=eval_on_external(externals, base_g, base_lib, "full"),
        open_solved_beam=eval_on_external(externals, open_res.guidance,
                                          open_res.library, "beam"),
        base_solved_beam=eval_on_external(externals, base_g, base_lib, "beam"),
        n_external=len(EMERGENCE_SET), attacks=n_attacks, verifier_fp=fp)
