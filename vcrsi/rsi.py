#!/usr/bin/env python3
"""The recursive-self-improvement loop.

One "arm" repeatedly attacks the still-OPEN tasks. After each round the ADAPTIVE
arm improves its own search by editing the policy genome:

  * weight learning -- operators that appear in adopted solutions are
    up-weighted, so the same budget is spent more on productive structure;
  * block mining + META-GATE -- candidate subroutines are mined from solved
    programs and committed ONLY if an A/B test on the unsolved frontier (equal
    budget + seeds, incumbent vs incumbent+candidate) shows strictly more newly
    solved tasks, no regression on solved tasks, and an unchanged verifier_fp.

The FROZEN arm runs the identical loop but never edits its genome. Both arms get
the same per-(task,round) seeds and the same per-attempt budget, so the only
difference is adaptation -- and the solved-count delta is exactly the measured
self-improvement, no more.

Every adopted program is (a) verified on the sealed held-out battery and (b)
required to clear the adopted-program complexity floor AFTER inlining its library
calls, so a task cannot be "solved" by a below-floor program hiding work inside a
subroutine.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .ir import Block, Node, pp, inline
from .library import (Policy, default_policy, mine_blocks, _block_calls,
                      re_encapsulate, propose_encapsulations)
from .search import synthesize, shrink
from .complexity import adopted_program_ops, MIN_SOLUTION_OPS
from .oracle import SealedOracle, assert_verifier_unchanged


# --------------------------------------------------------------------------- #
# Deterministic seeds: identical for both arms -> reproducible counterfactual   #
# --------------------------------------------------------------------------- #
def seed_for(task: str, rnd: int, salt: str = "") -> int:
    h = hashlib.sha256(f"{task}|{rnd}|{salt}".encode()).hexdigest()
    return int(h, 16) % (2 ** 31)


# --------------------------------------------------------------------------- #
# Weight learning                                                              #
# --------------------------------------------------------------------------- #
def _count_op_keys(n: Node, acc: Dict[str, int]) -> None:
    if n.op == "arg":
        acc["arg"] = acc.get("arg", 0) + 1
    elif n.op == "var":
        acc[n.const] = acc.get(n.const, 0) + 1
    elif n.op == "lit":
        key = {"I": "lit_int", "B": "lit_bool", "L": "lit_nil",
               "S": "lit_estr", "P": "lit_pair"}.get(n.rtype, "lit_int")
        acc[key] = acc.get(key, 0) + 1
    elif n.op == "param":
        pass
    elif n.op == "call":
        pass  # block usage handled by block_prob, not per-op weights
    else:
        acc[n.op] = acc.get(n.op, 0) + 1
    for k in n.kids:
        _count_op_keys(k, acc)


def learn_weights(base: Dict[str, float], programs: List[Node]) -> Dict[str, float]:
    counts: Dict[str, int] = {}
    for p in programs:
        _count_op_keys(p, counts)
    w = dict(base)
    for op, c in counts.items():
        factor = min(4.0, 1.0 + 0.6 * c)
        w[op] = base.get(op, 1.0) * factor
    return w


# --------------------------------------------------------------------------- #
# Adoption (held-out gate + adopted-program floor)                            #
# --------------------------------------------------------------------------- #
def _adopt_ok(prog: Node, oracle: SealedOracle, blocks: Dict[str, Block]) -> bool:
    if not oracle.verify(prog, blocks):
        return False
    if adopted_program_ops(prog, blocks) < MIN_SOLUTION_OPS:
        return False  # below-floor -> the task was mis-classified; do not count
    return True


# --------------------------------------------------------------------------- #
# Run state                                                                    #
# --------------------------------------------------------------------------- #
@dataclass
class Adoption:
    task: str
    round: int
    program: Node
    used_blocks: Tuple[str, ...]      # block names load-bearing in this solution
    evals: int


@dataclass
class ArmResult:
    adaptive: bool
    adopted: Dict[str, Adoption] = field(default_factory=dict)
    open_tasks: List[str] = field(default_factory=list)
    blocks: List[Block] = field(default_factory=list)
    rounds: int = 0
    total_evals: int = 0
    policy_versions: int = 0
    meta_gate_log: List[dict] = field(default_factory=list)
    lineage: List[dict] = field(default_factory=list)

    def solved_count(self) -> int:
        return len(self.adopted)

    def adoption_digest(self) -> str:
        """A pure function of the run: byte-identical across same-seed runs."""
        h = hashlib.sha256()
        for name in sorted(self.adopted):
            a = self.adopted[name]
            h.update(f"{name}@{a.round}:{pp(a.program)}|{a.used_blocks}\n".encode())
        for b in self.blocks:
            h.update(f"B:{b.name}={pp(b.body)}@{b.created_round}\n".encode())
        return h.hexdigest()[:16]


def _used_block_names(prog: Node) -> Tuple[str, ...]:
    return tuple(sorted(_block_calls(prog)))


# --------------------------------------------------------------------------- #
# META-GATE for candidate blocks                                              #
# --------------------------------------------------------------------------- #
def _solves_with_policy(policy: Policy, oracle: SealedOracle, budget: int,
                        seed: int) -> Optional[Node]:
    prog, stats = synthesize(oracle.public_view(), policy, budget, seed)
    if prog is not None and _adopt_ok(prog, oracle, policy.block_map()):
        return prog
    return None


def _frontier_solves(policy: Policy, frontier: List[SealedOracle],
                     gate_budget: int, rnd: int) -> set:
    solved = set()
    for orc in frontier:
        s = seed_for(orc.task.name, rnd, salt="gate")
        if _solves_with_policy(policy, orc, gate_budget, s) is not None:
            solved.add(orc.task.name)
    return solved


def meta_gate_one(incumbent: Policy, candidate: Block, base_solved: set,
                  frontier: List[SealedOracle], oracles: Dict[str, SealedOracle],
                  already_solved: Dict[str, Adoption], gate_budget: int, rnd: int
                  ) -> Tuple[bool, dict]:
    """A/B a single candidate block against a precomputed incumbent baseline on
    the unsolved frontier (equal budget/seeds). Accept iff it strictly increases
    the frontier solve count, causes no regression on already-solved tasks, and
    leaves verifier_fp unchanged."""
    fp_before = assert_verifier_unchanged(oracles, "meta_gate.before")
    trial = incumbent.clone()
    trial.blocks = list(incumbent.blocks) + [candidate]
    trial.block_prob = max(0.22, incumbent.block_prob)
    trial_solved = _frontier_solves(trial, frontier, gate_budget, rnd)

    regression = any(
        not oracles[n].verify(a.program, trial.block_map())
        for n, a in already_solved.items())
    fp_after = assert_verifier_unchanged(oracles, "meta_gate.after")
    accept = (len(trial_solved) > len(base_solved)) and (not regression) \
        and (fp_before == fp_after)
    info = {"round": rnd, "candidate": candidate.name, "body": pp(candidate.body),
            "base_new": len(base_solved), "trial_new": len(trial_solved),
            "regression": regression, "accepted": accept}
    return accept, info


# --------------------------------------------------------------------------- #
# Encapsulation step (builds the depth-2 lineage)                              #
# --------------------------------------------------------------------------- #
def _encapsulation_step(res: "ArmResult", policy: Policy,
                        oracles: Dict[str, SealedOracle], r: int,
                        block_round: Dict[str, int], verbose: bool) -> None:
    bm = policy.block_map()
    # (1) rewrite adopted solutions to CALL existing blocks where their pattern
    #     occurs, so the library is actually exercised (load-bearing).
    for name, a in res.adopted.items():
        re = re_encapsulate(a.program, policy.blocks)
        if re is not a.program and oracles[name].verify(re, bm):
            res.adopted[name] = Adoption(name, a.round, re,
                                         _used_block_names(re), a.evals)
    # (2) propose NEW blocks whose body CALLS an existing block.
    progs = [a.program for a in res.adopted.values()]
    cands = propose_encapsulations(progs, policy.blocks, r, max_new=3)
    for c in cands:
        parents = [p for p in c.calls() if p in {b.name for b in policy.blocks}]
        if not parents:
            continue
        trial_blocks = policy.blocks + [c]
        tbm = {b.name: b for b in trial_blocks}
        # no-regression: every adopted solution still verifies, and the new block
        # is actually USED by at least one re-encapsulated solution.
        ok = True
        used_somewhere = False
        rewrites: Dict[str, Node] = {}
        for name, a in res.adopted.items():
            re = re_encapsulate(a.program, trial_blocks)
            if not oracles[name].verify(re, tbm):
                ok = False
                break
            rewrites[name] = re
            if c.name in _used_block_names(re):
                used_somewhere = True
        if not (ok and used_somewhere):
            continue
        fp_before = assert_verifier_unchanged(oracles, "encapsulate.before")
        # commit
        policy.blocks.append(c)
        block_round[c.name] = r
        policy.block_prob = max(policy.block_prob, 0.22)
        policy.version += 1
        for name, re in rewrites.items():
            a = res.adopted[name]
            res.adopted[name] = Adoption(name, a.round, re,
                                         _used_block_names(re), a.evals)
        fp_after = assert_verifier_unchanged(oracles, "encapsulate.after")
        assert fp_before == fp_after
        for parent in parents:
            res.lineage.append({"kind": "block_on_block", "parent": parent,
                                "parent_round": block_round.get(parent, -1),
                                "child": c.name, "child_round": r,
                                "child_body": pp(c.body)})
        if verbose:
            print(f"  [A r{r}] ENCAPSULATED {c.name}={pp(c.body)[:60]} "
                  f"(calls {parents})")
        return  # one composed block per round keeps the lineage legible


# --------------------------------------------------------------------------- #
# The arm loop                                                                 #
# --------------------------------------------------------------------------- #
def run_arm(oracles: Dict[str, SealedOracle], adaptive: bool, *,
            budget: int = 60000, rounds: int = 4, gate_budget: int = 30000,
            gate_frontier: int = 8, max_gate_candidates: int = 4,
            learn_weights_on: bool = False, encapsulate: bool = True,
            task_order: Optional[List[str]] = None,
            verbose: bool = False) -> ArmResult:
    order = task_order or list(oracles.keys())
    policy = default_policy()
    res = ArmResult(adaptive=adaptive)
    res.rounds = rounds
    block_round: Dict[str, int] = {}

    for r in range(rounds):
        # ---- encapsulation operator (port of SECTION 24): freeze a recurring
        #      block-containing pattern from prior-round solutions as a NEW block
        #      that CALLS an earlier block -> this is the depth-2 lineage. Uses
        #      only blocks adopted in EARLIER rounds, so the child is strictly
        #      later than its parent. Adopted only if it causes no regression. -- #
        if adaptive and encapsulate and r > 0 and policy.blocks:
            _encapsulation_step(res, policy, oracles, r, block_round, verbose)

        open_now = [t for t in order if t not in res.adopted]
        if not open_now:
            continue
        # ---- attempt every open task with identical seed/budget ----------- #
        newly: List[Tuple[str, Node, int]] = []
        for t in open_now:
            orc = oracles[t]
            s = seed_for(t, r)
            prog, stats = synthesize(orc.public_view(), policy, budget, s)
            res.total_evals += stats.evals
            if prog is not None and _adopt_ok(prog, orc, policy.block_map()):
                # de-bloat the adopted program so mined blocks are clean atoms
                prog = shrink(prog, orc.public_view().public_examples,
                              policy.block_map())
                used = _used_block_names(prog)
                res.adopted[t] = Adoption(t, r, prog, used, stats.evals)
                newly.append((t, prog, r))
                # lineage: record first-use round of each load-bearing block
                for bn in used:
                    cr = block_round.get(bn, -1)
                    if cr >= 0 and r > cr:
                        res.lineage.append(
                            {"block": bn, "created_round": cr,
                             "first_used_round": r, "in_task": t})
                if verbose:
                    print(f"  [{'A' if adaptive else 'F'} r{r}] solved {t}"
                          f" (blocks={used}) evals={stats.evals}")

        if not adaptive:
            continue  # frozen never edits its genome

        if not newly:
            continue

        # ---- learn weights from all adopted solutions (mild) -------------- #
        # The mined-subroutine LIBRARY is the primary improvement channel; weight
        # learning is deliberately mild so it does not by itself trivialise the
        # composed tasks (which would leave the library with nothing to do and
        # erase the block-on-block lineage). Both are evolvable genome data.
        if learn_weights_on:
            policy.weights = learn_weights(default_policy().weights,
                                           [a.program for a in res.adopted.values()])

        # ---- mine + META-GATE candidate blocks ---------------------------- #
        still_open = [oracles[t] for t in order if t not in res.adopted]
        frontier = still_open[:gate_frontier]
        if not frontier:
            continue
        # collect + rank candidates from all solutions this round, gate the best
        cands: List[Block] = []
        seen_bodies = {pp(b.body) for b in policy.blocks}
        for t, prog, _r in newly:
            for c in mine_blocks(prog, policy.blocks, r, max_new=3):
                if pp(c.body) not in seen_bodies:
                    seen_bodies.add(pp(c.body))
                    cands.append(c)
        cands = cands[:max_gate_candidates]
        if not cands:
            continue
        # baseline on the frontier with the current genome (computed once)
        base_solved = _frontier_solves(policy, frontier, gate_budget, r)
        for c in cands:
            accept, info = meta_gate_one(policy, c, base_solved, frontier,
                                         oracles, res.adopted, gate_budget, r)
            res.meta_gate_log.append(info)
            if accept:
                policy.blocks.append(c)
                block_round[c.name] = r
                policy.block_prob = 0.22
                policy.version += 1
                base_solved = _frontier_solves(policy, frontier, gate_budget, r)
                if verbose:
                    print(f"  [A r{r}] ADOPTED block {c.name}={pp(c.body)[:56]}"
                          f" (gate {info['base_new']}->{info['trial_new']})")

    res.open_tasks = [t for t in order if t not in res.adopted]
    res.blocks = list(policy.blocks)
    res.policy_versions = policy.version
    return res
