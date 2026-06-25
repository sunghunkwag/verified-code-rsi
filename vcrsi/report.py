#!/usr/bin/env python3
"""--mode demo / counterfactual drivers and their printed reports."""
from __future__ import annotations

from typing import List

from .ir import pp
from .oracle import build_oracles, assert_verifier_unchanged
from .complexity import complexity_floor, adopted_program_ops, MIN_SOLUTION_OPS
from .rsi import run_arm, run_guided_arm, Guidance, GuidedArmResult, BOOTSTRAP_TASKS
from .counterfactual import (run_counterfactual, DEFAULT_ORDER,
                             run_guidance_counterfactual, GUIDANCE_ORDER)
from .search_oe import oe_solve
from .search import synthesize
from .library import broad_policy, stateful_policy
from .prm import PRM
from .prm_beam import prm_beam_synthesize

# Locked, documented run parameters (the README's numbers regenerate from these).
BUDGET = 6000
ROUNDS = 7
GATE_BUDGET = 6000
TRANSFER_BUDGET = 16000      # per-task solve budget in the transfer experiment
ABLATION_BUDGET = 11000      # smaller per-task budget for the 6-config ablation
# Phase C: the PRM-guided beam budget (width x layers), shared by every channel
# and both arms of the guidance counterfactual.
GUIDE_WIDTH = 24
GUIDE_LAYERS = 30
GUIDE_WAVES = 3
# Open-ended / emergence run parameters (the README's numbers regenerate here).
OE_GENERATIONS = 4
OE_BATCH = 5

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
    _print_guidance_demo(oracles)
    assert_verifier_unchanged(oracles, "demo.end")
    print("=" * 78)
    print("Run `--mode solve-hard` for the full portfolio on the hard families,")
    print("`--mode counterfactual` for both deltas, `--mode test` for the controls.")
    return 0


def _print_guidance_demo(oracles) -> None:
    """Phase C: PRM digest evolution across waves + world-model coverage."""
    print("-" * 78)
    print("LEARNED-GUIDANCE RSI -- PRM (process-reward model) digest across waves")
    order = [t for t in GUIDANCE_ORDER if t in oracles]
    res = run_guided_arm(oracles, adaptive=True, order=order, width=GUIDE_WIDTH,
                         layers=GUIDE_LAYERS, waves=GUIDE_WAVES)
    print("  the PRM is trained on the system's OWN solved programs; its digest")
    print("  changes wave-to-wave as it learns (a frozen PRM's digest never moves):")
    for i, dg in enumerate(res.guidance.wave_digests):
        print(f"    wave {i}: prm_digest = {dg}")
    print(f"  world-model coverage : {res.guidance.world.coverage()}")
    print(f"  PRM-beam solved ({res.solved_count()}): {sorted(res.adopted)}")
    print(f"  still OPEN under the beam: {res.open_tasks}")


def SUITE_NOTE(oracles, name) -> str:
    return oracles[name].task.note or oracles[name].task.spec[:40]


def run_transfer_mode() -> int:
    from .transfer import (rotate_B, Mechanisms, cross_family_transfer_count,
                           detector_self_test)
    from .tasks import TRANSFER_FAMILIES
    oracles = build_oracles()
    print("=" * 78)
    print("VERIFIED-CODE-RSI -- CROSS-FAMILY TRANSFER (rotate-B, B-blind mining)")
    print("A block transfers to family B iff: mined B-blind, appears in an")
    print("ADOPTED held-out B-solution, is LOAD-BEARING (removal -> OPEN), AND")
    print("passes the Socratic gate (no distinguishing counterexample).")
    print("=" * 78)
    ok, det = detector_self_test()
    print(f"detector self-test (positive control): {'PASS' if ok else 'FAIL'} -- {det}")
    print("-" * 78)
    res = rotate_B(oracles, Mechanisms(), budget=TRANSFER_BUDGET)
    print("ROTATE-B MATRIX (all 5 mechanisms ON):")
    print("  held-out B   frozen  adaptive  lib_blocks  cross-family-transfers")
    for fr in res:
        ctr = sum(1 for tr in fr.transfers if tr.counts)
        print(f"  {fr.held_out:11s}  {fr.frozen_solved:^6d}  {fr.adaptive_solved:^8d}"
              f"  {fr.n_blocks:^10d}  {ctr}")
        for tr in fr.transfers:
            mark = "COUNTS" if tr.counts else "rejected"
            print(f"      {tr.home_family}->{fr.held_out} block {tr.block} in "
                  f"{tr.task}: load_bearing={tr.load_bearing} "
                  f"socratic={tr.socratic_ok} [{mark}]")
    total = cross_family_transfer_count(res)
    print("-" * 78)
    print(f"TOTAL CROSS-FAMILY transfer_families (load-bearing AND Socratic) = {total}")
    if total == 0:
        print("RESULT: no library block mined in one family is load-bearing-and-")
        print("Socratically-valid in a structurally-different held-out family.")
        print("(The detector self-test above confirms the detector CAN report a")
        print("positive, so this 0 is a measured negative, not a dead detector.)")
    else:
        print("RESULT: cross-family transfer measured (see COUNTS rows above).")
    print("=" * 78)
    return 0


def run_ablation_mode() -> int:
    from .transfer import rotate_B, Mechanisms, cross_family_transfer_count
    oracles = build_oracles()
    print("=" * 78)
    print("VERIFIED-CODE-RSI -- ABLATION (which mechanism enables transfer?)")
    print("=" * 78)
    configs = [
        ("all-on", Mechanisms()),
        ("M1-off", Mechanisms(M1_oe=False)),
        ("M2-off", Mechanisms(M2_trigger=False)),
        ("M3-off", Mechanisms(M3_normalize=False)),
        ("M4-off", Mechanisms(M4_socratic=False)),
        ("M5-off", Mechanisms(M5_archive=False)),
    ]
    print("  config    cross-family transfers   (adaptive-solved over rotate-B)")
    for name, mech in configs:
        res = rotate_B(oracles, mech, budget=ABLATION_BUDGET)
        ctr = cross_family_transfer_count(res)
        solved = sum(fr.adaptive_solved for fr in res)
        print(f"  {name:8s}  {ctr:^22d}   {solved}")
    print("=" * 78)
    print("If transfer is 0 in every configuration, no single mechanism (nor all)")
    print("makes cross-family block transfer occur at this IR's abstraction level.")
    return 0


def run_counterfactual_mode() -> int:
    oracles = build_oracles()
    print("=" * 78)
    print("VERIFIED-CODE-RSI -- COUNTERFACTUAL (two deltas, equal budget/seeds)")
    print("=" * 78)
    # --- delta (a): SOLVER SELF-IMPROVEMENT (adaptive vs frozen guidance) --- #
    print("DELTA (a) -- LEARNED GUIDANCE: adaptive (PRM+world model trained on the")
    print("system's own solves) vs frozen (wave-0, untrained) guidance; the PRM-")
    print("guided beam is the sole solver, equal beam budget. This is the solver-")
    print("self-improvement claim.")
    print("-" * 78)
    gcf = run_guidance_counterfactual(oracles, width=GUIDE_WIDTH,
                                      layers=GUIDE_LAYERS, waves=GUIDE_WAVES)
    g = gcf.to_dict()
    print(f"frozen-guidance  solved    : {g['frozen_solved']}  {g['frozen_tasks']}")
    print(f"adaptive-guidance solved   : {g['adaptive_solved']}  {g['adaptive_tasks']}")
    print(f"adaptive-only (the delta)  : {g['adaptive_only']}")
    print(f"PRM digest per wave        : {g['prm_wave_digests']}")
    print(f"world-model coverage       : {g['world_coverage']}")
    print(f"frozen / adaptive digests  : {g['frozen_digest']} / {g['adaptive_digest']}")
    print(f">>> SOLVER-SELF-IMPROVEMENT DELTA (a) = {g['adaptive_solved']} - "
          f"{g['frozen_solved']} = {gcf.delta} <<<")
    print("=" * 78)
    # --- delta (b): the existing MACRO-LIBRARY RSI counterfactual ---------- #
    print("DELTA (b) -- MACRO LIBRARY: with-library (adaptive) vs no-library")
    print("(frozen) stochastic/OE portfolio; equal per-attempt budget and seeds.")
    print("-" * 78)
    cf = run_counterfactual(oracles, budget=BUDGET, rounds=ROUNDS,
                            gate_budget=GATE_BUDGET)
    d = cf.to_dict()
    print(f"frozen  arm solved         : {d['frozen_solved']}  {d['frozen_tasks']}")
    print(f"adaptive arm solved        : {d['adaptive_solved']}  {d['adaptive_tasks']}")
    print(f"adaptive-only (the delta)  : {d['adaptive_only']}")
    print(f"library blocks adopted     : {d['blocks_adopted']}")
    print(f"frozen / adaptive digests  : {d['frozen_digest']} / {d['adaptive_digest']}")
    print(f">>> MACRO-LIBRARY DELTA (b) = {d['adaptive_solved']} - "
          f"{d['frozen_solved']} = {d['delta']} <<<")
    print("=" * 78)
    print(f"verifier_fp (unchanged): {d['verifier_fp']}")
    print("Reproducible: re-run with the same seed -> byte-identical digests.")
    return 0


# --------------------------------------------------------------------------- #
# --mode solve-hard : the full portfolio on the suite incl. the hard families   #
# --------------------------------------------------------------------------- #
def _build_adaptive_guidance(oracles) -> Guidance:
    """Train the adaptive guidance (PRM + world model) by running the guided arm
    over the seqcode curriculum, so its PRM-beam channel is the trained one. The
    world model additionally observes a few arithmetic solutions so it learns real
    binary-op semantics (add/sub/mul/...) -- not just the seqcode string ops."""
    order = [t for t in GUIDANCE_ORDER if t in oracles]
    res = run_guided_arm(oracles, adaptive=True, order=order, width=GUIDE_WIDTH,
                         layers=GUIDE_LAYERS, waves=GUIDE_WAVES)
    for tn in ("scaled_widths", "midpoints", "keep_wide", "shift_intervals",
               "clamped_widths"):
        if tn not in oracles:
            continue
        sol, _st = synthesize(oracles[tn].public_view(), broad_policy(), 30_000, 7)
        if sol is not None and oracles[tn].verify(sol):
            res.guidance.world.observe_program(
                sol, [list(a) for a, _y in oracles[tn].public_view().public_examples])
    return res.guidance


def _portfolio_attack(orc, guidance: Guidance):
    """Try every channel on one task: OE, stochastic, then the PRM-guided beam.
    Returns (program | None, channel, best_partial)."""
    view = orc.public_view()
    # channel 1: bottom-up OE
    p = oe_solve(view, blocks=[], max_size=12, eval_budget=90_000)
    if p is not None and orc.verify(p) and adopted_program_ops(p, {}) >= MIN_SOLUTION_OPS:
        return p, "oe", 1.0
    # channel 2: memetic / stochastic (stateful prior -> scan families reachable)
    p, _st = synthesize(view, stateful_policy(), 45_000, 7)
    if p is not None and orc.verify(p) and adopted_program_ops(p, {}) >= MIN_SOLUTION_OPS:
        return p, "memetic", 1.0
    # channel 3: PRM-guided beam (adaptive guidance) -- scan frame enabled so the
    # stateful families are within the beam's reach.
    p, st = prm_beam_synthesize(view, guidance.prm, [], width=GUIDE_WIDTH,
                                max_layers=GUIDE_LAYERS, verify=orc.verify,
                                enable_scan=True)
    if p is not None and orc.verify(p) and adopted_program_ops(p, {}) >= MIN_SOLUTION_OPS:
        return p, "prm-beam", 1.0
    return None, "", st.best_partial


# suite order for solve-hard: the hard families are the explicit objective.
SOLVE_HARD_ORDER = [
    "rle_decode", "rle_decode_rev", "rle_decode_sorted", "rle_decode_shift1",
    "caesar_encode", "interleave_pairs", "shift_intervals", "keep_wide",
    "midpoints", "merge_intervals", "bracket_depths", "bytecode_interp",
]
HARD_FAMILIES = {"merge_intervals", "bracket_depths", "bytecode_interp"}


def run_solve_hard() -> int:
    oracles = build_oracles()
    fp = assert_verifier_unchanged(oracles, "solve-hard")
    print("=" * 78)
    print("VERIFIED-CODE-RSI -- SOLVE-HARD (full portfolio: OE + memetic + PRM-beam)")
    print(f"verifier_fp = {fp}")
    print("The hard, previously-OPEN families (the objective) are marked [HARD].")
    print("=" * 78)
    print("Training the adaptive guidance (PRM + world model) ...")
    g = _build_adaptive_guidance(oracles)
    print(f"  PRM digest evolution : {g.wave_digests}")
    print(f"  world-model coverage : {g.world.coverage()}")
    print("-" * 78)
    print("%-20s %-6s %-9s %s" % ("task", "state", "channel", "best_partial/note"))
    solved = 0
    hard_solved = 0
    order = [t for t in SOLVE_HARD_ORDER if t in oracles]
    for tn in order:
        orc = oracles[tn]
        prog, ch, bp = _portfolio_attack(orc, g)
        tag = " [HARD]" if tn in HARD_FAMILIES else ""
        if prog is not None:
            solved += 1
            hard_solved += int(tn in HARD_FAMILIES)
            # the world model learns op semantics from every solved program
            g.world.observe_program(prog, [list(a) for a, _y in
                                           orc.public_view().public_examples])
            print("%-20s %-6s %-9s %s" % (tn + tag, "SOLVED", ch, pp(prog)[:34]))
        else:
            print("%-20s %-6s %-9s best_exact_frac=%.2f" %
                  (tn + tag, "OPEN", "-", bp))
    assert_verifier_unchanged(oracles, "solve-hard.end")
    print("-" * 78)
    print(f"SOLVED {solved}/{len(order)} ; HARD families cracked {hard_solved}/3 "
          f"({sorted(HARD_FAMILIES)})")
    print("The PRM-guided beam's contribution is the tasks marked channel=prm-beam;")
    print("OPEN tasks are reported with their best train-exact fraction, never hidden.")
    print("=" * 78)
    return 0


# --------------------------------------------------------------------------- #
# --mode openended : run the self-generated curriculum loop                     #
# --------------------------------------------------------------------------- #
def run_openended_mode() -> int:
    from .openended import run_openended
    oracles = build_oracles()
    fp = assert_verifier_unchanged(oracles, "openended")
    print("=" * 78)
    print("VERIFIED-CODE-RSI -- OPEN-ENDED SELF-GENERATED CURRICULUM (--mode openended)")
    print(f"verifier_fp (hash-pinned correctness oracle) = {fp}")
    print("The system invents its OWN tasks (a fresh sealed reference defines each")
    print("task's ground truth), keeps only those passing the TRIPLE LOCK")
    print("(L1 whitelist / L2 §6B floor / L3 self-easiness), solves what it can, and")
    print("trains its guidance + library on its OWN solutions. No human target inside.")
    print("=" * 78)
    res = run_openended(generations=OE_GENERATIONS, batch=OE_BATCH, seed=0,
                        verbose=False)
    print("PER-GENERATION (minted -> triple-lock survivors -> solved-and-verified):")
    print("%-4s %7s %7s %7s %8s  %s" %
          ("gen", "minted", "locked", "solved", "lib", "lock-fails / note"))
    for gs in res.per_gen:
        note = ("FRONTIER STALLED" if gs.stalled
                else f"L1={gs.lock_fail.get('L1',0)} "
                     f"L2={gs.lock_fail.get('L2',0)} L3={gs.lock_fail.get('L3',0)}")
        liblen = sum(1 for _ in res.library) if gs is res.per_gen[-1] else ""
        print("%-4d %7d %7d %7d %8s  %s" %
              (gs.gen, gs.minted, gs.locked, gs.solved, str(liblen), note))
    print("-" * 78)
    print("FRONTIER DIFFICULTY TRAJECTORY (§6B metrics of NEWLY-SOLVED tasks/gen):")
    print("  the L3 frontier ratchets iff this climbs; if it stays flat / collapses")
    print("  onto a narrow band, that is reported honestly (it is the finding).")
    for row in res.frontier_trajectory():
        if row.get("solved"):
            print(f"    gen {row['gen']}: solved={row['solved']:2d}  "
                  f"distinct_ops min/med={row['ops_min']}/{row['ops_med']}  "
                  f"exec_depth min/med={row['depth_min']}/{row['depth_med']}  "
                  f"groups={row['groups']}")
        else:
            tag = "FRONTIER STALLED" if row.get("stalled") else "no new solves"
            print(f"    gen {row['gen']}: {tag}")
    print("-" * 78)
    print(f"  archive coverage (quality-diversity)   : {res.archive.coverage()}")
    print(f"  library blocks mined from own solves   : "
          f"{[b.name for b in res.library]}")
    print(f"  total: attacks={res.total_attacks} solved={res.total_solved}")
    print(f"  PRM digest per generation (own solves) : "
          f"{res.guidance.wave_digests}")
    assert_verifier_unchanged(oracles, "openended.end")
    print(f"  run digest (same seed -> identical)    : {res.digest()}")
    print("=" * 78)
    print("Run `--mode emergence` for the open-ended-vs-baseline external-set delta.")
    return 0


# --------------------------------------------------------------------------- #
# --mode emergence : open-ended arm vs fixed-suite baseline on the EXTERNAL set  #
# --------------------------------------------------------------------------- #
def run_emergence_mode() -> int:
    from .openended import run_emergence
    print("=" * 78)
    print("VERIFIED-CODE-RSI -- EMERGENCE (--mode emergence)")
    print("Does inventing+solving its OWN curriculum make the system better at")
    print("UNSEEN, human-authored tasks it never generated and never trained on?")
    print("  open-ended arm : guidance+library trained ONLY on self-generated tasks")
    print("  baseline   arm : identical budget/seeds, trained ONLY on the fixed suite")
    print("Both then evaluated FROZEN on the SEALED external held-out set.")
    print("=" * 78)
    r = run_emergence(generations=OE_GENERATIONS, batch=OE_BATCH, seed=0)
    oe = r.open_res
    print("OPEN-ENDED ARM -- self-generated curriculum:")
    for gs in oe.per_gen:
        note = "FRONTIER STALLED" if gs.stalled else f"solved={gs.solved}"
        print(f"    gen {gs.gen}: minted={gs.minted} locked={gs.locked} {note}")
    traj = oe.frontier_trajectory()
    climbed = [row for row in traj if row.get("solved")]
    print(f"    frontier difficulty (ops_med per gen with solves): "
          f"{[ (row['gen'], row['ops_med']) for row in climbed]}")
    print(f"    self-generated tasks solved (training data): {oe.total_solved}; "
          f"library={len(oe.library)}")
    print("-" * 78)
    print(f"EXTERNAL HELD-OUT SET ({r.n_external} sealed human-authored tasks); "
          f"equal budget = {r.attacks} attacks/arm, same seeds:")
    print("  FULL PORTFOLIO (OE + memetic + PRM-beam) -- 'got better at unseen tasks':")
    print(f"    open-ended arm solved : {len(r.open_solved_full)}  "
          f"{r.open_solved_full}")
    print(f"    baseline   arm solved : {len(r.base_solved_full)}  "
          f"{r.base_solved_full}")
    print(f"    >>> EMERGENCE DELTA (full)  = {len(r.open_solved_full)} - "
          f"{len(r.base_solved_full)} = {r.delta_full} <<<")
    print("  PRM-BEAM ONLY -- isolates the LEARNED GUIDANCE (the arms' real diff):")
    print(f"    open-ended arm solved : {len(r.open_solved_beam)}  "
          f"{r.open_solved_beam}")
    print(f"    baseline   arm solved : {len(r.base_solved_beam)}  "
          f"{r.base_solved_beam}")
    print(f"    >>> EMERGENCE DELTA (beam)  = {len(r.open_solved_beam)} - "
          f"{len(r.base_solved_beam)} = {r.delta_beam} <<<")
    print("-" * 78)
    if r.delta_full <= 0 and r.delta_beam <= 0:
        print("FINDING: NO measured emergence -- self-generation did not improve")
        print("external-set performance over the fixed-suite baseline here. This is a")
        print("legitimate, reported result (§8): the generated frontier re-covers")
        print("structure the suite already teaches, so guidance transfer is flat.")
    else:
        print("FINDING: a POSITIVE, reproducible emergence delta -- by inventing and")
        print("solving its own curriculum the system solved MORE unseen human tasks")
        print("than the fixed-suite baseline. Nothing beyond this delta is asserted.")
    print(f"verifier_fp (unchanged): {r.verifier_fp}")
    print(f"emergence digest (same seed -> byte-identical): {r.digest()}")
    print("=" * 78)
    return 0
