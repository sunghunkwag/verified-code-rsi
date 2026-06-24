#!/usr/bin/env python3
"""--mode demo / counterfactual drivers and their printed reports."""
from __future__ import annotations

from typing import List

from .ir import pp
from .oracle import build_oracles, assert_verifier_unchanged
from .complexity import complexity_floor
from .rsi import run_arm
from .counterfactual import run_counterfactual, DEFAULT_ORDER

# Locked, documented run parameters (the README's numbers regenerate from these).
BUDGET = 6000
ROUNDS = 7
GATE_BUDGET = 6000

# A dedicated curriculum that elicits the block-on-block lineage (the library is
# the sole adaptive channel here, so composed blocks are genuinely needed).
LINEAGE_ORDER = ["rle_decode", "rle_decode_rev", "rle_rev_palindrome",
                 "rle_rev_palindrome_twice"]


def _print_complexity_table(oracles) -> None:
    print("-" * 78)
    print("MACHINE-CHECKED COMPLEXITY FLOOR (§6B) -- per-task metrics")
    print("%-24s %3s %5s %5s %6s %5s %4s" %
          ("task", "fam", "ops", "loop", "rec/ds", "depth", "ok"))
    for name, orc in oracles.items():
        ok, m = complexity_floor(orc)
        print("%-24s %3d %5d %5s %6s %6d %4s" %
              (m["task"], m["family"], m["distinct_ops"],
               "Y" if m["has_loop"] else "n",
               "Y" if m["has_rec_or_struct"] else "n",
               m["max_exec_depth"], "OK" if ok else "FAIL"))


def _print_lineage(oracles) -> None:
    print("-" * 78)
    print("RECURSIVE SELF-IMPROVEMENT -- library lineage (block built on block)")
    res = run_arm(oracles, adaptive=True, budget=16000, rounds=8,
                  gate_budget=16000, gate_frontier=4, max_gate_candidates=5,
                  learn_weights_on=False, encapsulate=True,
                  task_order=LINEAGE_ORDER)
    created = {b.name: b.created_round for b in res.blocks}
    used = set()
    for a in res.adopted.values():
        used |= set(a.used_blocks)
    print("  library blocks (round created -> body):")
    for b in res.blocks:
        tag = ("calls " + ",".join(b.calls())) if b.calls() else "atom"
        lb = "load-bearing" if b.name in used else "unused"
        print(f"    {b.name}  r{b.created_round}  [{tag}, {lb}]  {pp(b.body)[:54]}")
    print("  task -> block actually used in its adopted solution:")
    for name in LINEAGE_ORDER:
        if name in res.adopted:
            print(f"    {name:24s} uses {res.adopted[name].used_blocks}")
    pairs = []
    for b in res.blocks:
        for parent in b.calls():
            if (parent in created and b.name in used and parent in used
                    and created[b.name] > created.get(parent, 99)):
                pairs.append((parent, created[parent], b.name, created[b.name]))
    if pairs:
        p, pr, c, cr = pairs[0]
        print(f"  >>> LINEAGE PAIR: parent {p} (round {pr}) <- child {c} "
              f"(round {cr}); both load-bearing, child strictly later. <<<")
    else:
        print("  >>> no block-on-block lineage on this run <<<")


def run_demo() -> int:
    oracles = build_oracles()
    fp = assert_verifier_unchanged(oracles, "demo")
    print("=" * 78)
    print("VERIFIED-CODE-RSI -- DEMO (adaptive run)")
    print(f"verifier_fp (hash-pinned correctness oracle) = {fp}")
    print("=" * 78)
    _print_complexity_table(oracles)

    order = [t for t in DEFAULT_ORDER if t in oracles]
    res = run_arm(oracles, adaptive=True, budget=BUDGET, rounds=ROUNDS,
                  gate_budget=GATE_BUDGET, learn_weights_on=True,
                  task_order=order)
    print("-" * 78)
    print(f"ADAPTIVE RUN over {len(order)} tasks "
          f"(budget={BUDGET}/attempt, rounds={ROUNDS}):")
    print(f"  SOLVED ({res.solved_count()}):")
    for name in order:
        if name in res.adopted:
            a = res.adopted[name]
            print(f"    {name:24s} @round {a.round}  blocks={a.used_blocks}")
            print(f"        {pp(a.program)[:70]}")
    print(f"  OPEN ({len(res.open_tasks)}) -- reported honestly, never hidden:")
    for name in res.open_tasks:
        print(f"    {name:24s} ({SUITE_NOTE(oracles, name)})")
    print(f"  blocks mined+gated this run: {[b.name for b in res.blocks]}")

    _print_lineage(oracles)
    assert_verifier_unchanged(oracles, "demo.end")
    print("=" * 78)
    print("Run `--mode counterfactual` for the measured adaptive-vs-frozen delta,")
    print("and `--mode test` for the anti-cheat controls.")
    return 0


def SUITE_NOTE(oracles, name) -> str:
    return oracles[name].task.note or oracles[name].task.spec[:40]


def run_counterfactual_mode() -> int:
    oracles = build_oracles()
    print("=" * 78)
    print("VERIFIED-CODE-RSI -- COUNTERFACTUAL (adaptive vs frozen)")
    print("equal per-attempt budget, equal per-(task,round) seeds; the ONLY")
    print("difference is that the adaptive arm improves its own search policy.")
    print("=" * 78)
    cf = run_counterfactual(oracles, budget=BUDGET, rounds=ROUNDS,
                            gate_budget=GATE_BUDGET)
    d = cf.to_dict()
    print(f"verifier_fp                : {d['verifier_fp']}")
    print(f"frozen  arm solved         : {d['frozen_solved']}  {d['frozen_tasks']}")
    print(f"adaptive arm solved        : {d['adaptive_solved']}  {d['adaptive_tasks']}")
    print(f"adaptive-only (the delta)  : {d['adaptive_only']}")
    print("-" * 78)
    print(f"MEASURED SELF-IMPROVEMENT DELTA = adaptive - frozen = "
          f"{d['adaptive_solved']} - {d['frozen_solved']} = {d['delta']}")
    print("-" * 78)
    print(f"library blocks adopted     : {d['blocks_adopted']}")
    print(f"frozen adoption digest     : {d['frozen_digest']}")
    print(f"adaptive adoption digest   : {d['adaptive_digest']}")
    print(f"frozen total evals         : {d['frozen_evals']}")
    print(f"adaptive total evals       : {d['adaptive_evals']}")
    print("=" * 78)
    print("Reproducible: re-run with the same seed -> byte-identical digests.")
    print("This delta IS the self-improvement claim; nothing else is asserted.")
    return 0
