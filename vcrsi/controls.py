#!/usr/bin/env python3
"""Anti-cheat controls (§4 of the task), as RUNNABLE checks.

Each control is a falsifiable test of one specific way this system could be
faked. They are executed by ``--mode test``. A control that is missing, skipped
or trivially true is itself a failure; every control here either asserts a
non-trivial property or constructs the adversarial input it is guarding against.
"""
from __future__ import annotations

import inspect
import random
from typing import Callable, List, Tuple

from .ir import (Node, Block, PRIMS, pp, inline, MAX_LEN)
from .interp import run
from .oracle import (build_oracles, assert_verifier_unchanged,
                     verifier_fingerprint, SealedOracle, PublicView)
from .complexity import complexity_floor, adopted_program_ops, MIN_SOLUTION_OPS
from .tasks import BANNED_NAMES, SUITE_BY_NAME
from .library import default_policy, Policy
from . import search as search_mod
from .search import synthesize, problem_from_public, _Gen, _case_scores
from .rsi import run_arm, seed_for, run_arm  # noqa: F401
from .counterfactual import run_counterfactual


# --------------------------------------------------------------------------- #
# helpers                                                                       #
# --------------------------------------------------------------------------- #
def _b(op, *kids, const=None, rtype="V"):
    from .ir import PRIMS as P, COMB_RTYPE as C
    if op in C:
        rt = C[op]
    elif op in P:
        rt = P[op][0]
    else:
        rt = rtype
    return Node(op, rt, tuple(kids), const)


def _lin_scenario_oracles_order():
    return ["rle_decode", "rle_decode_rev", "rle_rev_palindrome",
            "rle_rev_palindrome_twice"]


# --------------------------------------------------------------------------- #
# §6B + §6A : complexity floor passes for every task; whitelist membership      #
# --------------------------------------------------------------------------- #
def ctl_complexity_floor(oracles) -> Tuple[bool, str]:
    bad = []
    for name, orc in oracles.items():
        ok, m = complexity_floor(orc)
        t = orc.task
        if t.family not in (1, 2, 3, 4, 5):
            bad.append(f"{name}: off-whitelist family {t.family}")
        if name in BANNED_NAMES:
            bad.append(f"{name}: banned scalar-reduction name")
        # input must be structured (a list or a string), never a flat scalar set
        if not any(at in ("L", "S") for at in t.arg_types):
            bad.append(f"{name}: input not structured ({t.arg_types})")
        if not ok:
            bad.append(f"{name}: floor fail {m}")
    return (not bad), ("all tasks clear §6A whitelist + §6B floor"
                       if not bad else "; ".join(bad))


# --------------------------------------------------------------------------- #
# §4.9 : no oracle leakage -- synthesizer source references neither the         #
# held-out battery nor the reference implementations                            #
# --------------------------------------------------------------------------- #
def ctl_no_leakage(oracles) -> Tuple[bool, str]:
    src = inspect.getsource(search_mod)
    # code-level leakage tokens (English words in comments do not count): the
    # synthesizer must not import the oracle/tasks modules nor name the reference
    # solutions or the held-out battery.
    forbidden = ["from .oracle", "from .tasks", "import oracle", "import tasks",
                 "build_oracles", "SUITE_BY_NAME", "SUITE", "._holdout",
                 "_make_example", "_ref_", ".reference", "task.reference"]
    hits = [w for w in forbidden if w in src]
    # the public view handed to search must not carry reference / held-out data
    pv_fields = set(PublicView.__dataclass_fields__.keys())
    leaked_fields = pv_fields & {"reference", "holdout", "_holdout", "battery"}
    ok = (not hits) and (not leaked_fields)
    detail = "synthesizer source is oracle-free; PublicView exposes only public data"
    if not ok:
        detail = f"leak: source hits={hits} pv_fields={leaked_fields}"
    return ok, detail


# --------------------------------------------------------------------------- #
# §4.10 : sandbox containment of a hostile candidate                           #
# --------------------------------------------------------------------------- #
def ctl_sandbox(oracles) -> Tuple[bool, str]:
    # (a) the IR has NO file/process/network/eval primitive -> escape is
    #     inexpressible, not merely blocked.
    dangerous = [k for k in PRIMS
                 if any(s in k.lower() for s in
                        ("open", "exec", "eval", "file", "proc", "net", "sys",
                         "import", "read", "write", "socket"))]
    litI = lambda v: _b("lit", const=v, rtype="I")
    acc = Node("var", "V", const="acc")
    # (b) a runaway computation (nested loops, ~16M steps) is cut off by the
    #     step budget and scored as failed, not left to hang the harness.
    inner = _b("map", _b("lrange", litI(4000)), litI(0))
    runaway = _b("map", _b("lrange", litI(4000)), inner)
    r1 = run(runaway, [[]], max_steps=50_000)
    # (c) a memory bomb (accumulator doubled each iteration) is cut off by the
    #     allocation cap.
    bomb = _b("foldl", _b("lrange", litI(40)),
              _b("lsingle", litI(0)), _b("lapp", acc, acc))
    r2 = run(bomb, [[]], max_steps=10**7)
    from .ir import StepLimit, AllocLimit
    contained = (isinstance(r1.error, StepLimit)
                 and isinstance(r2.error, AllocLimit)
                 and r1.steps <= 50_001)
    ok = (not dangerous) and contained
    detail = ("no I/O/exec primitive exists; runaway loop -> "
              f"{type(r1.error).__name__} @ {r1.steps} steps; memory bomb -> "
              f"{type(r2.error).__name__}")
    if not ok:
        detail = (f"dangerous prims={dangerous}; r1={type(r1.error).__name__}, "
                  f"r2={type(r2.error).__name__}")
    return ok, detail


# --------------------------------------------------------------------------- #
# §9 : the held-out gate rejects a program that overfits the public examples    #
# --------------------------------------------------------------------------- #
def ctl_holdout_rejects_overfit(oracles) -> Tuple[bool, str]:
    orc = oracles["rle_decode"]
    pub = orc.public_view().public_examples
    max_pub_pairs = max(len(args[0]) for args, _ in pub)
    # decode only the first (max_pub_pairs+1) pairs: correct on EVERY public
    # example, wrong on the larger held-out inputs -> an overfit.
    k = max_pub_pairs + 1
    body = _b("srepeat", _b("fst", Node("var", "V", const="it")),
              _b("snd", Node("var", "V", const="it")))
    overfit = _b("sconcat",
                 _b("map", _b("ltake", Node("arg", "L", const=0),
                              _b("lit", const=k, rtype="I")), body))
    passes_public = orc.passes_public(overfit)
    passes_holdout = orc.verify(overfit)
    ok = passes_public and (not passes_holdout)
    return ok, (f"overfit passes public={passes_public}, held-out rejects it="
                f"{not passes_holdout} (truncates at {k} pairs)")


# --------------------------------------------------------------------------- #
# §4.6 : reward-hacking floor -- remove the oracle, select on speed only, the   #
# winner is a WRONG-but-fast program; the oracle-gated winner is correct        #
# --------------------------------------------------------------------------- #
def ctl_reward_hacking(oracles) -> Tuple[bool, str]:
    orc = oracles["rle_decode"]
    view = orc.public_view()
    prob = problem_from_public(view)
    rng = random.Random(20240601)
    gen = _Gen(prob, default_policy(), rng)
    # speed-only "search": minimise runtime steps, NO correctness gate.
    best = None
    best_steps = float("inf")
    for _ in range(4000):
        p = gen.gen(prob.out_type, rng.randint(1, 5))
        tot = 0
        ok_run = True
        for args, _exp in prob.examples:
            r = run(p, list(args), max_steps=40_000)
            tot += r.steps
            ok_run = ok_run and r.ok
        if ok_run and tot < best_steps:
            best_steps, best = tot, p
    speed_winner_correct = orc.verify(best) if best is not None else False
    # oracle-gated search
    gated, _stats = synthesize(view, default_policy(), budget=8000, seed=1)
    gated_correct = gated is not None and orc.verify(gated)
    ok = (best is not None) and (not speed_winner_correct) and gated_correct
    return ok, (f"speed-only winner correct={speed_winner_correct} "
                f"(steps={best_steps}, prog={pp(best)[:40] if best else None}); "
                f"oracle-gated winner correct={gated_correct}")


# --------------------------------------------------------------------------- #
# §4.7 : distinct genomes produce distinct behaviour                           #
# --------------------------------------------------------------------------- #
def ctl_distinct_genome(oracles) -> Tuple[bool, str]:
    view = oracles["rle_decode"].public_view()
    p_default = default_policy()
    # a crippled genome: the key string op is essentially absent
    w = dict(p_default.weights)
    w["srepeat"] = 0.0001
    w["sconcat"] = 0.0001
    p_crippled = Policy(weights=w, blocks=[], block_prob=0.0)
    fp_diff = p_default.fingerprint() != p_crippled.fingerprint()
    a, _ = synthesize(view, p_default, budget=10000, seed=3)
    b, _ = synthesize(view, p_crippled, budget=10000, seed=3)
    sol_a = a is not None and oracles["rle_decode"].verify(a)
    sol_b = b is not None and oracles["rle_decode"].verify(b)
    behave_diff = sol_a != sol_b or (pp(a) if a else None) != (pp(b) if b else None)
    ok = fp_diff and behave_diff and sol_a and not sol_b
    return ok, (f"distinct genomes: fp_diff={fp_diff}; default solves={sol_a}, "
                f"crippled solves={sol_b} -> behaviour differs={behave_diff}")


# --------------------------------------------------------------------------- #
# §4.8 : lineage-depth -- a block whose body references an EARLIER block, both   #
# load-bearing, child first used in a strictly later round than the parent       #
# --------------------------------------------------------------------------- #
def ctl_lineage(oracles) -> Tuple[bool, str]:
    order = _lin_scenario_oracles_order()
    res = run_arm(oracles, adaptive=True, budget=16000, rounds=8,
                  gate_budget=16000, gate_frontier=4, max_gate_candidates=5,
                  learn_weights_on=False, encapsulate=True, task_order=order)
    names = {b.name for b in res.blocks}
    used = set()
    for a in res.adopted.values():
        used |= set(a.used_blocks)
    created = {b.name: b.created_round for b in res.blocks}
    pairs = []
    for b in res.blocks:
        for parent in b.calls():
            if parent in names and b.name in used and parent in used:
                # child created/used strictly later than the parent was created
                if created.get(b.name, -1) > created.get(parent, 99):
                    pairs.append((parent, created[parent], b.name, created[b.name]))
    ok = len(pairs) >= 1
    detail = ("no block-on-block lineage" if not ok else
              "; ".join(f"{p}(r{pr})->{c}(r{cr}), both load-bearing"
                        for p, pr, c, cr in pairs[:3]))
    return ok, detail


# --------------------------------------------------------------------------- #
# §4.11 : determinism -- same seed -> byte-identical adoption logs              #
# --------------------------------------------------------------------------- #
def ctl_determinism(oracles) -> Tuple[bool, str]:
    order = ["rle_decode", "rle_decode_rev", "rle_decode_twice"]
    o2 = build_oracles()
    a = run_arm(oracles, adaptive=True, budget=6000, rounds=3,
                gate_budget=6000, task_order=order, learn_weights_on=True)
    b = run_arm(o2, adaptive=True, budget=6000, rounds=3,
                gate_budget=6000, task_order=order, learn_weights_on=True)
    ok = a.adoption_digest() == b.adoption_digest()
    return ok, f"two runs, same seed: digests {a.adoption_digest()} == {b.adoption_digest()} -> {ok}"


# --------------------------------------------------------------------------- #
# §4.4 / §12 : the counterfactual delta is positive (adaptive > frozen)         #
# --------------------------------------------------------------------------- #
def ctl_counterfactual_delta(oracles) -> Tuple[bool, str]:
    order = ["rle_decode", "rle_decode_rev", "rle_decode_sorted",
             "rle_decode_twice", "rle_decode_palindrome"]
    cf = run_counterfactual(oracles, budget=6000, rounds=7, gate_budget=6000,
                            task_order=order)
    ok = cf.delta > 0
    return ok, (f"adaptive {cf.adaptive.solved_count()} - frozen "
                f"{cf.frozen.solved_count()} = delta {cf.delta} (>0: {ok})")


# --------------------------------------------------------------------------- #
# §4.5 : adopted solutions independently re-verify (no fabricated solved count)  #
# + adopted programs clear the inlined complexity floor                          #
# --------------------------------------------------------------------------- #
def ctl_recompute_solved(oracles) -> Tuple[bool, str]:
    order = ["rle_decode", "rle_decode_rev", "rle_decode_sorted",
             "rle_decode_twice"]
    res = run_arm(oracles, adaptive=True, budget=6000, rounds=5,
                  gate_budget=6000, task_order=order, learn_weights_on=True)
    bm = {b.name: b for b in res.blocks}
    recomputed = 0
    floor_ok = True
    for name, a in res.adopted.items():
        if oracles[name].verify(a.program, bm):      # independent re-execution
            recomputed += 1
        if adopted_program_ops(a.program, bm) < MIN_SOLUTION_OPS:
            floor_ok = False
    ok = (recomputed == res.solved_count()) and floor_ok
    return ok, (f"reported solved={res.solved_count()}, independently "
                f"re-verified={recomputed}; all clear inlined floor={floor_ok}")


# =========================================================================== #
# Phase B (v2) transfer- and mechanism-specific controls (§5)                  #
# =========================================================================== #
def ctl_family_diversity(oracles) -> Tuple[bool, str]:
    from collections import Counter
    groups = Counter(orc.task.group for orc in oracles.values())
    n_fam = len(groups)
    total = sum(groups.values())
    maxfrac = max(groups.values()) / total
    ok = n_fam >= 4 and maxfrac <= 0.40
    return ok, (f"{n_fam} families; largest is {maxfrac:.0%} of {total} tasks "
                f"(need >=4 families, none >40%) -> {dict(groups)}")


def ctl_transfer_load_bearing_and_socratic(oracles) -> Tuple[bool, str]:
    # positive control: the detector FIRES on a genuine planted cross-group
    # transfer (load-bearing AND Socratic) and REJECTS a spurious block.
    from .transfer import detector_self_test
    ok, det = detector_self_test()
    return ok, det


def ctl_transfer_socratic_rejects_spurious(oracles) -> Tuple[bool, str]:
    # build a block whose B-solution fits the PUBLIC examples but is semantically
    # wrong; the Socratic gate must find a distinguishing input and reject it.
    from .socratic import socratic_admit
    from .ir import Node
    orc = oracles["clamp_low"]
    task = orc.task
    pub = orc.public_view().public_examples
    # spurious: identity on the list (matches public iff no clamping needed)
    spurious = Node("arg", "L", const=0)
    fits_public = all(_run_eq(spurious, a, e) for a, e in pub)
    admit, detail = socratic_admit(spurious, task, {})
    # it should be rejected (a distinguishing input exists), regardless of public
    ok = (not admit)
    return ok, (f"spurious-block fits_public={fits_public}; Socratic admit="
                f"{admit} (must be False) -- {detail}")


def _run_eq(prog, args, exp) -> bool:
    r = run(prog, list(args))
    return r.ok and r.value == exp


def ctl_mining_is_B_blind(oracles) -> Tuple[bool, str]:
    # functional: mining for held-out B uses only families != B, so no mined
    # block's home family is B. + source check: no held-out symbols in miner.
    from .transfer import mine_blind, home_family_of, Mechanisms
    from .tasks import TRANSFER_FAMILIES
    from . import transfer as T, search_oe as OE, archive as A
    B = "select"
    mining = [f for f in TRANSFER_FAMILIES if f != B]
    arch = mine_blind(oracles, mining, Mechanisms(), budget=4000, rounds=1)
    homes = {home_family_of(b) for b in arch.blocks}
    no_B = B not in homes
    import inspect
    src = inspect.getsource(OE) + inspect.getsource(A)
    src_clean = ("_holdout" not in src and "_make_example" not in src)
    ok = no_B and src_clean
    return ok, (f"B={B} held out; mined block home-families={homes or '{}'} "
                f"(B absent={no_B}); OE/archive source holdout-free={src_clean}")


def ctl_normalizer_preserves_semantics(oracles) -> Tuple[bool, str]:
    from .normalize import normalize_block, _equiv, _fold
    from .ir import Block, Node, pp
    p0 = Node("param", "L", const=0)
    # a foldable / simplifiable block: lrev(lrev($0)) == $0
    blk = Block("T", ("L",), Node("lrev", "L", (Node("lrev", "L", (p0,)),)),
                "L", origin="fam:x")
    nb, changed = normalize_block(blk)
    simplified = changed and nb.body.op != "lrev"            # actually changed
    preserved = _equiv(blk.body, nb.body, ("L",))
    # a behaviour-CHANGING fake normalization must be rejected by _equiv
    fake = Node("lrev", "L", (p0,))                          # != lrev(lrev p0)
    fake_rejected = not _equiv(blk.body, fake, ("L",))
    ok = simplified and preserved and fake_rejected
    return ok, (f"normalized lrev(lrev x)->{pp(nb.body)} (changed={changed}); "
                f"semantics preserved={preserved}; behaviour-changing rewrite "
                f"rejected={fake_rejected}")


def ctl_oe_no_leakage(oracles) -> Tuple[bool, str]:
    from . import search_oe as OE
    from .search_oe import oe_solve
    import inspect
    src = inspect.getsource(OE)
    forbidden = ["_holdout", "_make_example", ".reference", "task.reference",
                 "from .tasks", "from .oracle"]
    hits = [w for w in forbidden if w in src]
    # functional: OE solves rle_decode on PUBLIC and the winner passes the SEALED
    # holdout (verified by the oracle, not by the enumerator).
    prog = oe_solve(oracles["rle_decode"].public_view(), blocks=[])
    held = prog is not None and oracles["rle_decode"].verify(prog)
    ok = (not hits) and held
    return ok, (f"OE source holdout-free={not hits}; OE winner passes SEALED "
                f"holdout via oracle={held}")


def ctl_archive_spread_is_real(oracles) -> Tuple[bool, str]:
    from .transfer import mine_blind, Mechanisms
    arch = mine_blind(oracles, ["seqcode", "select"], Mechanisms(), budget=14000,
                      rounds=1)
    cov = arch.coverage()
    ok = cov["families_spanned"] >= 2
    return ok, (f"archive coverage: {cov} (need cells spanning >=2 families)")


def ctl_ablation_runs(oracles) -> Tuple[bool, str]:
    # confirm all six configs are defined and each executes end-to-end on one
    # held-out family at a tiny budget (a skipped config = fail).
    from .transfer import transfer_on_family, Mechanisms
    configs = [Mechanisms(), Mechanisms(M1_oe=False), Mechanisms(M2_trigger=False),
               Mechanisms(M3_normalize=False), Mechanisms(M4_socratic=False),
               Mechanisms(M5_archive=False)]
    ran = 0
    for m in configs:
        transfer_on_family(oracles, "select", m, budget=2500)
        ran += 1
    ok = ran == 6
    return ok, f"all {ran}/6 ablation configurations executed end-to-end"


# --------------------------------------------------------------------------- #
# registry + runner                                                            #
# --------------------------------------------------------------------------- #
CONTROLS: List[Tuple[str, Callable]] = [
    ("complexity_floor_and_whitelist (§6A/§6B)", ctl_complexity_floor),
    ("no_oracle_leakage (§4.9)", ctl_no_leakage),
    ("sandbox_containment (§4.10)", ctl_sandbox),
    ("holdout_rejects_overfit (§9)", ctl_holdout_rejects_overfit),
    ("reward_hacking_floor (§4.6)", ctl_reward_hacking),
    ("distinct_genome_distinct_behaviour (§4.7)", ctl_distinct_genome),
    ("recompute_solved_count + adopted_floor (§4.5)", ctl_recompute_solved),
    ("determinism (§4.11)", ctl_determinism),
    ("lineage_block_on_block (§4.8)", ctl_lineage),
    ("counterfactual_delta_positive (§4.4)", ctl_counterfactual_delta),
    # --- Phase B (v2) transfer + mechanism controls (§5) --- #
    ("family_diversity (§2)", ctl_family_diversity),
    ("transfer_load_bearing+socratic / detector (§5.1)",
     ctl_transfer_load_bearing_and_socratic),
    ("transfer_socratic_rejects_spurious (§5.2)",
     ctl_transfer_socratic_rejects_spurious),
    ("mining_is_B_blind (§5.3)", ctl_mining_is_B_blind),
    ("normalizer_preserves_semantics (§5.4)", ctl_normalizer_preserves_semantics),
    ("oe_no_leakage (§5.5)", ctl_oe_no_leakage),
    ("archive_spread_is_real (§5.7)", ctl_archive_spread_is_real),
    ("ablation_runs (§5.8)", ctl_ablation_runs),
]


def run_controls(verbose: bool = True) -> int:
    oracles = build_oracles()
    fp0 = assert_verifier_unchanged(oracles, "controls.start")
    print("=" * 78)
    print("ANTI-CHEAT CONTROLS  (--mode test)")
    print(f"verifier_fp (hash-pinned correctness oracle) = {fp0}")
    print("=" * 78)
    n_pass = 0
    for name, fn in CONTROLS:
        try:
            ok, detail = fn(oracles)
        except Exception as e:  # a crashing control is a failure, never silent
            ok, detail = False, f"EXCEPTION: {type(e).__name__}: {e}"
        n_pass += int(ok)
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        print(f"       {detail}")
    fp1 = assert_verifier_unchanged(oracles, "controls.end")
    print("-" * 78)
    print(f"verifier_fp unchanged: {fp0 == fp1} ({fp0} -> {fp1})")
    print(f"controls passed: {n_pass}/{len(CONTROLS)}")
    print("=" * 78)
    return 0 if n_pass == len(CONTROLS) else 1
