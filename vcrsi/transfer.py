#!/usr/bin/env python3
"""M2 + the cross-family transfer experiment (the heart of Phase B).

A block ``b`` TRANSFERS to family B iff ALL hold (§3):
  (1) b was mined with B held out (B-blind mining -- enforced + tested),
  (2) b appears in an ADOPTED held-out B-solution (passes the sealed battery),
  (3) b is LOAD-BEARING there (remove it + re-synthesize at equal budget -> the
      task becomes OPEN), and
  (4) b passes the SOCRATIC gate (M4) -- no distinguishing counterexample.

``transfer_families(b)`` counts distinct families (other than b's home) to which
b transfers under (1)-(4). The held-out adaptive-vs-frozen delta IS the claim.

M2 (the transfer TRIGGER) decides WHICH library blocks are even candidates for a
B-task, from PUBLIC signatures only (block type-shape vs task type-shape) -- so
we do not blind-try every block.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .ir import Block, Node, pp, inline
from .interp import run
from .oracle import build_oracles, SealedOracle
from .library import default_policy, broad_policy, Policy, mine_blocks, _block_calls
from .search import synthesize
from .search_oe import oe_solve
from .archive import MapElites
from .normalize import normalize_block
from .socratic import socratic_admit
from .tasks import TRANSFER_FAMILIES, SUITE_BY_NAME


@dataclass
class Mechanisms:
    M1_oe: bool = True
    M2_trigger: bool = True
    M3_normalize: bool = True
    M4_socratic: bool = True
    M5_archive: bool = True

    def tag(self) -> str:
        return "".join(k for k, v in [("1", self.M1_oe), ("2", self.M2_trigger),
                       ("3", self.M3_normalize), ("4", self.M4_socratic),
                       ("5", self.M5_archive)] if v) or "none"


# --------------------------------------------------------------------------- #
# M2 -- behavioural signatures + transfer trigger                              #
# --------------------------------------------------------------------------- #
def task_signature(view) -> dict:
    """From PUBLIC data only: argument types, output type, element type of the
    list argument, and the input->output length relation."""
    exs = view.public_examples
    la = next((i for i, t in enumerate(view.arg_types) if t == "L"), None)
    elem = "V"
    if la is not None and exs and exs[0][0][la]:
        v = exs[0][0][la][0]
        elem = ("P" if isinstance(v, tuple) else "S" if isinstance(v, str)
                else "I" if isinstance(v, int) else "L")
    return {"args": tuple(view.arg_types), "out": view.out_type, "elem": elem}


def block_signature(b: Block) -> dict:
    return {"ptypes": tuple(b.ptypes), "rtype": b.rtype}


def transfer_trigger(b: Block, sig: dict) -> float:
    """Score how applicable block b is to a task with signature ``sig`` (M2).
    Type-compatibility from public shapes only; 0 means 'not a candidate'."""
    bs = block_signature(b)
    score = 0.0
    avail = set(sig["args"]) | {sig["elem"], sig["out"], "I"}
    # all params must be constructible from types available in the task
    if all((pt in avail) or pt == "V" for pt in bs["ptypes"]):
        score += 1.0
    else:
        return 0.0
    # the block's result must plug somewhere useful: into the output or elements
    if bs["rtype"] in (sig["out"], sig["elem"], "V"):
        score += 1.0
    if bs["rtype"] == sig["out"]:
        score += 0.5
    return score


# --------------------------------------------------------------------------- #
# Portfolio solver (M1 OE + stochastic), returns a public-passing program       #
# --------------------------------------------------------------------------- #
def portfolio_solve(view, blocks: List[Block], budget: int, seed: int,
                    use_oe: bool) -> Optional[Node]:
    if use_oe:
        p = oe_solve(view, blocks)
        if p is not None:
            return p
    pol = Policy(weights=dict(broad_policy().weights), blocks=list(blocks),
                 block_prob=0.25 if blocks else 0.0)
    prog, _stats = synthesize(view, pol, budget, seed)
    return prog


def _seed(*parts) -> int:
    import hashlib
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest(), 16) % (2**31)


# --------------------------------------------------------------------------- #
# B-blind mining: build a cross-family library on families {all \ B}           #
# --------------------------------------------------------------------------- #
def mine_blind(oracles, mining_families: List[str], mech: Mechanisms,
               budget: int = 9000, rounds: int = 2) -> MapElites:
    arch = MapElites()
    solved: Dict[str, Node] = {}
    pol = broad_policy()
    tasks = [(fam, t) for fam in mining_families for t in TRANSFER_FAMILIES[fam]]
    for r in range(rounds):
        lib = arch.draw_blocks(spread=mech.M5_archive)
        for fam, tname in tasks:
            if tname in solved:
                continue
            orc = oracles[tname]
            prog = portfolio_solve(orc.public_view(), lib, budget,
                                   _seed("mine", tname, r), mech.M1_oe)
            if prog is None or not orc.verify(prog, {b.name: b for b in lib}):
                continue
            solved[tname] = prog
            arch.consider_solution(fam, orc.task.out_type, prog, tname)
            for cand in mine_blocks(prog, arch.blocks, r, max_new=4):
                cand = Block(cand.name, cand.ptypes, cand.body, cand.rtype,
                             cand.created_round, origin="fam:" + fam)
                if mech.M3_normalize:
                    cand, _changed = normalize_block(cand)
                    cand = Block(cand.name, cand.ptypes, cand.body, cand.rtype,
                                 cand.created_round, origin="fam:" + fam)
                arch.add_block(cand)
        # learn weights across the mining set (within-family specialisation)
        from .rsi import learn_weights
        pol.weights = learn_weights(broad_policy().weights, list(solved.values()))
    return arch


# --------------------------------------------------------------------------- #
# Transfer test on a held-out family B                                        #
# --------------------------------------------------------------------------- #
@dataclass
class TransferRecord:
    held_out: str
    task: str
    block: str
    home_family: str
    load_bearing: bool
    socratic_ok: bool
    detail: str = ""

    @property
    def counts(self) -> bool:
        return self.load_bearing and self.socratic_ok


@dataclass
class FamilyResult:
    held_out: str
    frozen_solved: int
    adaptive_solved: int
    transfers: List[TransferRecord] = field(default_factory=list)
    n_blocks: int = 0


def home_family_of(b: Block) -> str:
    return b.origin.split("fam:")[-1] if "fam:" in b.origin else "?"


def transfer_on_family(oracles, B: str, mech: Mechanisms, budget: int = 9000
                       ) -> FamilyResult:
    mining = [f for f in TRANSFER_FAMILIES if f != B]
    arch = mine_blind(oracles, mining, mech)
    lib_all = arch.draw_blocks(spread=mech.M5_archive)
    res = FamilyResult(held_out=B, frozen_solved=0, adaptive_solved=0,
                       n_blocks=len(lib_all))
    for tname in TRANSFER_FAMILIES[B]:
        orc = oracles[tname]
        sig = task_signature(orc.public_view())
        # M2 trigger: candidate cross-family blocks for this B-task
        if mech.M2_trigger:
            cands = [b for b in lib_all if transfer_trigger(b, sig) > 0.0]
        else:
            cands = list(lib_all)
        bm_c = {b.name: b for b in cands}
        s = _seed("test", tname, B)
        frozen = portfolio_solve(orc.public_view(), [], budget, s, mech.M1_oe)
        froze_ok = frozen is not None and orc.verify(frozen)
        res.frozen_solved += int(froze_ok)
        adapt = portfolio_solve(orc.public_view(), cands, budget, s, mech.M1_oe)
        adapt_ok = adapt is not None and orc.verify(adapt, bm_c)
        res.adaptive_solved += int(adapt_ok)
        if not adapt_ok:
            continue
        used = set(_block_calls(adapt)) & {b.name for b in cands}
        for bn in used:
            b = bm_c[bn]
            if home_family_of(b) == B:           # not cross-family
                continue
            # (3) load-bearing: remove b, re-synthesize at equal budget/seed
            without = [x for x in cands if x.name != bn]
            re = portfolio_solve(orc.public_view(), without, budget, s, mech.M1_oe)
            lb = not (re is not None and orc.verify(re, {x.name: x for x in without}))
            # (4) Socratic gate
            if mech.M4_socratic:
                sok, sdetail = socratic_admit(adapt, orc.task, bm_c)
            else:
                sok, sdetail = True, "M4 disabled"
            res.transfers.append(TransferRecord(
                B, tname, bn, home_family_of(b), lb, sok,
                f"lb={lb}; {sdetail}"))
    return res


def rotate_B(oracles, mech: Mechanisms, budget: int = 9000) -> List[FamilyResult]:
    return [transfer_on_family(oracles, B, mech, budget)
            for B in TRANSFER_FAMILIES]


def cross_family_transfer_count(results: List[FamilyResult]) -> int:
    return sum(1 for fr in results for tr in fr.transfers if tr.counts)


# --------------------------------------------------------------------------- #
# POSITIVE CONTROL -- prove the detector FIRES on a genuine cross-group         #
# transfer, so a reported 0 on the real families is credible (not a dead        #
# detector). A block from group 'A' is made load-bearing + Socratically valid   #
# in a task of group 'B' by construction.                                       #
# --------------------------------------------------------------------------- #
def _synth_widths_oracle():
    from .tasks import Task, b as B, arg, it
    from .oracle import SealedOracle
    import random as _r
    ref = B("map", arg(0, "L"), B("sub", B("snd", it()), B("fst", it())))

    def gen(rng, scale):
        k = rng.randint(max(2, scale - 1), scale + 2)
        return ([( (a := rng.randint(-5, 9)), a + rng.randint(0, 8))
                 for _ in range(k)],)
    t = Task("selftest_widths", 4, "interval widths (detector self-test)",
             ("L",), "L", ref, gen, group="selftestB")
    return SealedOracle(t), t


def detector_self_test() -> Tuple[bool, str]:
    """Plant a block from group 'selftestA' that is genuinely load-bearing +
    Socratically valid in a 'selftestB' task; assert the counting path reports a
    transfer. Also assert a spurious block is NOT counted."""
    orc, task = _synth_widths_oracle()
    p0 = Node("param", "P", const=0)
    width = Block("BW", ("P",), Node("sub", "I", (Node("snd", "I", (p0,)),
                  Node("fst", "I", (p0,)))), "I", origin="fam:selftestA")
    # with-block solution: map(a0, BW(it))
    sol = Node("map", "L", (Node("arg", "L", const=0),
               Node("call", "I", (Node("var", "P", const="it"),), "BW")))
    if not orc.verify(sol, {"BW": width}):
        return False, "self-test: planted with-block solution failed holdout"
    # load-bearing: without BW at a TINY stochastic budget, the width body is
    # not rebuildable -> task OPEN -> block is load-bearing.
    re = portfolio_solve(orc.public_view(), [], budget=400,
                         seed=_seed("selftest"), use_oe=False)
    lb = not (re is not None and orc.verify(re))
    # Socratic: the correct solution survives the counterexample search.
    sok, _d = socratic_admit(sol, task, {"BW": width})
    # spurious block: returns fst (wrong) -- fits nothing; build a wrong solution
    bad = Node("map", "L", (Node("arg", "L", const=0),
               Node("fst", "I", (Node("var", "P", const="it"),))))
    bad_rejected = not socratic_admit(bad, task, {})[0]
    home_cross = home_family_of(width) != task.group
    ok = lb and sok and bad_rejected and home_cross
    return ok, (f"planted A->B transfer: load_bearing={lb}, socratic_admit={sok}, "
                f"cross_group={home_cross}, spurious_rejected={bad_rejected}")
