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

from .ir import Block, Node, pp, inline
from .interp import run
from .oracle import SealedOracle
from .complexity import complexity_floor, adopted_program_ops, MIN_SOLUTION_OPS
from .generator import (GenSpec, generate_spec, on_whitelist, mint_curriculum,
                        _behav_sig)
from .search import synthesize, shrink
from .search_oe import oe_solve
from .prm_beam import prm_beam_synthesize
from .library import (broad_policy, stateful_policy, mine_blocks,
                      propose_encapsulations, re_encapsulate, score_abstraction,
                      is_composite, input_coupled, is_nontrivial_abstraction,
                      _block_calls)
from .archive import MapElites
from .rsi import Guidance

# Probe (L3) vs attack budgets. Both run the CURRENT solver; only the budget
# differs -- that gap, times an improving solver, is where the frontier lives.
# (The scan-enabled beam is expensive, so its budget is kept modest; OE+memetic
# carry the stateful solves and the strong measurement uses its own equal-budget
# OE+memetic reach probe -- see emergence._reach_attack.)
PROBE_OE = 8_000
PROBE_MEMETIC = 6_000
PROBE_BEAM = (6, 12)           # (width, layers)
ATTACK_OE = 40_000
ATTACK_MEMETIC = 46_000
ATTACK_BEAM = (16, 22)
LIBRARY_CAP = 12


# --------------------------------------------------------------------------- #
# Solver portfolios (probe = low budget, attack = high budget; same machinery)  #
# --------------------------------------------------------------------------- #
def _policy_with(blocks: List[Block]):
    pol = stateful_policy()
    pol.blocks = list(blocks)
    # a high block-call probability is the memetic counterpart of M1 abstraction-
    # first: the search reuses the newest learned atoms instead of re-deriving them,
    # which is what makes a deep scanify solvable as one block call (see emergence).
    pol.block_prob = 0.45 if blocks else 0.0
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
                               verify=lambda pr: orc.verify(pr, _bm(blocks)),
                               enable_scan=True)
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
                               verify=lambda pr: orc.verify(pr, bm),
                               enable_scan=True)
    if p is not None and orc.verify(p, bm):
        return p, "beam"
    return None, "-"


def _bm(blocks: List[Block]) -> Dict[str, Block]:
    return {b.name: b for b in blocks}


def _shrunk(prog: Node, orc: SealedOracle, library: List[Block]) -> Node:
    """Reduce a solved program to a smaller behaviourally-identical one (drops the
    harmless wrappers a stochastic search adds), so mined blocks are clean ATOMS
    and minting can find the map/scan subtree. Kept only if it still holdout-passes."""
    bm = _bm(library)
    s = shrink(prog, orc.public_view().public_examples, bm)
    return s if solved_and_floor_ok(s, orc, library) else prog


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
# Library admission -- M2 multi-objective abstraction score + anti-cheat guard   #
# --------------------------------------------------------------------------- #
M2_SCORE_MIN = 0.20      # admit only abstractions scoring at least this (M2)


def _admit_blocks(library: List[Block], prog: Node, gen: int,
                  adopted_pairs: List[Tuple[Node, str]]) -> List[Block]:
    """Mine candidate atoms from one solved program and admit each on the M2 score
    AND the input-coupled anti-cheat guard (a constant-pushing macro scores 0 on
    anti_cheat and is rejected). Returns the blocks newly admitted."""
    have = {pp(b.body) for b in library}
    bm = {b.name: b for b in library}
    admitted: List[Block] = []
    for c in mine_blocks(prog, library, gen, max_new=3):
        if len(library) >= LIBRARY_CAP:
            break
        if pp(c.body) in have:
            continue
        if input_coupled(c) <= 0.0:                       # anti-cheat: no constants
            continue
        if not is_nontrivial_abstraction(c):              # reject bare aliases
            continue
        sc = score_abstraction(c, adopted_pairs, bm)
        if sc["score"] < M2_SCORE_MIN:
            continue
        have.add(pp(c.body))
        library.append(c)
        bm[c.name] = c
        admitted.append(c)
    return admitted


# --------------------------------------------------------------------------- #
# Encapsulation (M1 depth-2 lineage): freeze a recurring block-CALLING pattern   #
# from adopted solutions as a NEW block whose body calls an EARLIER block. This   #
# is what turns a flat scan into a reusable stateful capability (the candidate    #
# emergent abstraction). Admitted on the M2 score + no-regression on adopted       #
# solutions + verifier_fp unchanged.                                              #
# --------------------------------------------------------------------------- #
def _encapsulate_oe(library: List[Block], adopted: List[Tuple[SealedOracle, Node]],
                    adopted_pairs: List[Tuple[Node, str]], gen: int) -> List[Block]:
    if not library or not adopted:
        return []
    bm = {b.name: b for b in library}
    # (1) rewrite adopted solutions to CALL existing blocks where their pattern
    #     occurs, so the library is genuinely exercised (load-bearing).
    rewritten: List[Tuple[SealedOracle, Node]] = []
    for orc, prog in adopted:
        re = re_encapsulate(prog, library)
        rewritten.append((orc, re if orc.verify(re, bm) else prog))
    progs = [p for _o, p in rewritten]
    # (2) propose NEW blocks whose body CALLS an existing block (depth-2 lineage).
    admitted: List[Block] = []
    for c in propose_encapsulations(progs, library, gen, max_new=4):
        if len(library) >= LIBRARY_CAP:
            break
        parents = [p for p in c.calls() if p in bm]
        if not parents or pp(c.body) in {pp(b.body) for b in library}:
            continue
        if (input_coupled(c) <= 0.0 or not is_composite(c, bm)
                or not is_nontrivial_abstraction(c)):     # reject bare aliases
            continue
        trial = library + [c]
        tbm = {b.name: b for b in trial}
        # no-regression: every adopted solution still verifies under the new block,
        # and the block is actually USED by at least one re-encapsulated solution.
        used = False
        ok = True
        for orc, prog in rewritten:
            re = re_encapsulate(prog, trial)
            if not orc.verify(re, tbm):
                ok = False
                break
            if c.name in _block_calls(re):
                used = True
        if not (ok and used):
            continue
        if score_abstraction(c, adopted_pairs, tbm)["score"] < M2_SCORE_MIN:
            continue
        library.append(c)
        bm[c.name] = c
        admitted.append(c)
    return admitted


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
    # data the STRONG emergence measurement (emergence.py) consumes:
    adopted_pairs: List[Tuple[Node, str]] = field(default_factory=list)
    reach_targets: List[GenSpec] = field(default_factory=list)   # deep minted tasks
    seed_blocks: List[str] = field(default_factory=list)         # pre-seeded (=[])
    encapsulated: List[str] = field(default_factory=list)        # depth-2 block names

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
    solved_refs: List[dict] = []          # block-free refs of solves (M3 sources)
    mint_registry: set = set()            # behavioural dedup of minted tasks

    for gen in range(generations):
        minted = locked = solved = 0
        lock_fail: Dict[str, int] = {"L1": 0, "L2": 0, "L3": 0}
        solved_groups: List[str] = []
        newly: List[Tuple[int, int]] = []
        registered: List[Tuple[GenSpec, SealedOracle, dict]] = []

        # --- the generation pool: base generated tasks + M3 non-shallow mints --- #
        pool: List[GenSpec] = []
        for i in range(batch):
            crng = random.Random(seed * 1009 + gen * 97 + i)
            sp = generate_spec(crng, gen, i, blocks=library)
            if sp is not None:
                pool.append(sp)
        # M3: compose the system's OWN verified solved references into structurally-
        # novel (accumulator-introducing) tasks. Split them by role:
        #   SKILL  (scanify/chain) -- shallow-enough to solve; the loop solves them
        #          to BUILD the stateful abstraction (mine + encapsulate). Not L3-
        #          gated and not counted as a frontier-novelty solve.
        #   REACH  (scan_twice)    -- the deep frontier: the inner scan appears twice
        #          so it is beyond flat reach until the encapsulated scan block
        #          exists. L3-gated; these are the strong measurement's reach targets.
        skill_mints: List[GenSpec] = []
        if solved_refs:
            for sp in mint_curriculum(solved_refs, mint_registry,
                                      n=max(4, batch), blocks=library):
                if "twice" in sp.note:
                    pool.append(sp)
                    res.reach_targets.append(sp)
                else:
                    skill_mints.append(sp)

        # SKILL PASS: solve the easy compositions to grow the library (the system
        # practises its own sub-skills before the composite frontier task).
        for sp in skill_mints[:max(2, batch // 2)]:
            orc = SealedOracle(sp)
            res.total_attacks += 1
            prog, _ch = attack(orc, g, library)
            if not solved_and_floor_ok(prog, orc, library):
                continue
            prog = _shrunk(prog, orc, library)
            g.train_on(prog, orc.public_view(), library)
            res.adopted_pairs.append((prog, sp.group))
            _admit_blocks(library, prog, gen, res.adopted_pairs)
            solved_progs.append((orc, prog))
            if verbose:
                print(f"  gen {gen}: SKILL-solved {sp.name} [{sp.note}] "
                      f"-> library grows (lib={len(library)})")

        for sp in pool:
            minted += 1
            ok, orc, reason, m = triple_lock(sp, library, g)
            if not ok:
                lock_fail[reason] = lock_fail.get(reason, 0) + 1
                continue
            locked += 1
            registered.append((sp, orc, m))

        # encapsulate after the skill pass too, so a scan block built from this
        # generation's skill solves is available before the frontier attacks.
        for b in _encapsulate_oe(library, solved_progs, res.adopted_pairs, gen):
            res.encapsulated.append(b.name)
            if verbose:
                print(f"  gen {gen}: ENCAPSULATED {b.name}={pp(b.body)[:54]} "
                      f"(calls {b.calls()})")

        if not registered:
            res.per_gen.append(GenStat(gen, minted, 0, 0, True, lock_fail, [], []))
            if verbose:
                print(f"  gen {gen}: minted {minted}, 0 passed the triple lock "
                      f"-> FRONTIER STALLED")
            continue

        gen_solved: List[Tuple[SealedOracle, Node]] = []
        for sp, orc, m in registered:
            res.total_attacks += 1
            prog, ch = attack(orc, g, library)
            if not solved_and_floor_ok(prog, orc, library):
                continue
            prog = _shrunk(prog, orc, library)        # clean atoms for mining
            # a genuine win: solved AND sealed-holdout-verified
            solved += 1
            res.total_solved += 1
            solved_groups.append(sp.group)
            newly.append((m["distinct_ops"], m["max_exec_depth"]))
            g.train_on(prog, orc.public_view(), library)
            res.archive.consider_solution(sp.group, sp.out_type, prog, sp.name)
            # record (program, family) for the M2 score + the strong measurement
            res.adopted_pairs.append((prog, sp.group))
            _admit_blocks(library, prog, gen, res.adopted_pairs)
            solved_progs.append((orc, prog))
            gen_solved.append((orc, prog))
            # store a BLOCK-FREE reference (inline calls) so M3 can compose it and
            # the sealed oracle can run it standalone next generation.
            bm = {b.name: b for b in library}
            solved_refs.append({"ref": inline(prog, bm), "arg_types": sp.arg_types,
                                "out_type": sp.out_type, "group": sp.group,
                                "gen_input": sp.gen_input})
            if verbose:
                print(f"  gen {gen}: solved {sp.name} [{sp.group}/"
                      f"{sp.note.split(':')[-1]}] via {ch} "
                      f"(ops={m['distinct_ops']}, depth={m['max_exec_depth']})")

        # --- encapsulation (M1 depth-2 lineage): freeze recurring block-calling
        #     patterns from this run's solutions as new stateful capabilities. --- #
        new_enc = _encapsulate_oe(library, solved_progs, res.adopted_pairs, gen)
        for b in new_enc:
            res.encapsulated.append(b.name)
            if verbose:
                print(f"  gen {gen}: ENCAPSULATED {b.name}={pp(b.body)[:54]} "
                      f"(calls {b.calls()})")
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
    pairs: List[Tuple[Node, str]] = []
    order = list(SUITE)
    for a in range(n_attacks):
        task = order[a % len(order)]
        orc = SealedOracle(task)
        prog, _ch = attack(orc, g, library)
        if solved_and_floor_ok(prog, orc, library):
            g.train_on(prog, orc.public_view(), library)
            pairs.append((prog, task.group))
            _admit_blocks(library, prog, a // max(1, len(order)), pairs)
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
                                       verify=lambda pr: orc.verify(pr, bm),
                                       enable_scan=True)
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
    strong: object = None              # emergence.EmergenceStrong (the headline)

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
        if self.strong is not None:
            h.update(("S" + self.strong.digest()).encode())
        return h.hexdigest()[:16]


def run_emergence(generations: int = 5, batch: int = 7, seed: int = 0,
                  verbose: bool = False) -> EmergenceResult:
    from .tasks import EMERGENCE_SET
    from .oracle import assert_verifier_unchanged, build_oracles
    fp = assert_verifier_unchanged(build_oracles(), "emergence.start")

    open_res = run_openended(generations, batch, seed, verbose=verbose)

    # (strong) THE HEADLINE: count un-designed composite capabilities that unlock
    # reach. Tested on the deep minted tasks + the hard suite families.
    from .emergence import measure_strong
    from .tasks import SUITE_BY_NAME
    hard = [SealedOracle(SUITE_BY_NAME[n]) for n in
            ("bracket_depths", "merge_intervals", "bytecode_interp")
            if n in SUITE_BY_NAME]
    strong = measure_strong(open_res, hard)

    # equal budget: the baseline performs the SAME number of attacks as the
    # open-ended arm actually performed (post-lock), same seeds.
    n_attacks = max(open_res.total_attacks, len(EMERGENCE_SET))
    base_g, base_lib = train_on_suite(n_attacks, seed)

    # (weak, retained) FROZEN evaluation on the SEALED external set.
    externals = [SealedOracle(t) for t in EMERGENCE_SET]
    return EmergenceResult(
        open_res=open_res,
        open_solved_full=eval_on_external(externals, open_res.guidance,
                                          open_res.library, "full"),
        base_solved_full=eval_on_external(externals, base_g, base_lib, "full"),
        open_solved_beam=eval_on_external(externals, open_res.guidance,
                                          open_res.library, "beam"),
        base_solved_beam=eval_on_external(externals, base_g, base_lib, "beam"),
        n_external=len(EMERGENCE_SET), attacks=n_attacks, verifier_fp=fp,
        strong=strong)
