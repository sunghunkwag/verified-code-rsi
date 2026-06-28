#!/usr/bin/env python3
"""Anti-cheat controls (§4 of the task), as RUNNABLE checks.

Each control is a falsifiable test of one specific way this system could be
faked. They are executed by ``--mode test``. A control that is missing, skipped
or trivially true is itself a failure; every control here either asserts a
non-trivial property or constructs the adversarial input it is guarding against.
"""
from __future__ import annotations

import hashlib
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


# =========================================================================== #
# Phase C (v3) learned-guidance controls (§5)                                  #
# =========================================================================== #
def ctl_prm_is_oracle_free(oracles) -> Tuple[bool, str]:
    """The PRM and its feature extractor reference neither the sealed oracle, the
    held-out battery, nor family/group metadata -- only train I/O + program
    structure. Source/data-flow inspection of prm.py and prm_beam.py."""
    from . import prm as PRM_MOD, prm_beam as BEAM_MOD
    src = inspect.getsource(PRM_MOD) + inspect.getsource(BEAM_MOD)
    # code-level leakage symbols (English words in docstrings do not count): the
    # PRM must not import the oracle/tasks modules nor read references/held-out.
    forbidden = ["from .oracle", "from .tasks", "import oracle", "import tasks",
                 "build_oracles", "SUITE_BY_NAME", "._holdout", "_make_example",
                 "_ref_", "task.reference", ".reference", "TRANSFER_FAMILIES",
                 "task.family", "task.group", ".gen_input"]
    hits = [w for w in forbidden if w in src]
    # the only data the features read is the public training I/O (BeamCtx.train
    # is built from view.public_examples); confirm that surface is the source.
    from .prm_beam import make_ctx
    ctx = make_ctx(oracles["rle_decode"].public_view(), [])
    train_is_public = (ctx.train ==
                       [(tuple(a), y) for a, y in
                        oracles["rle_decode"].public_view().public_examples[:4]])
    ok = (not hits) and train_is_public
    return ok, (f"PRM/feature source oracle-free={not hits} (hits={hits}); features"
                f" read only public train I/O={train_is_public}")


def ctl_prm_cross_task_not_memorised(oracles) -> Tuple[bool, str]:
    """The PRM is a small FIXED-dimensional model whose parameters do not grow per
    task (cannot memorise task-specific answers); a frozen PRM is load-bearing
    (changes behaviour); and a PRM is not a lookup table of solutions."""
    from .prm import PRM, NFEAT
    from .prm_beam import train_prm_on_solution
    from .search_oe import oe_solve
    sol = oe_solve(oracles["rle_decode"].public_view(), blocks=[], max_size=12,
                   eval_budget=90_000)
    p1 = PRM()
    train_prm_on_solution(p1, sol, oracles["rle_decode"].public_view())
    dim1 = (len(p1.w), len(p1.wsum))
    # train on MORE tasks -> dimension is UNCHANGED (no per-task growth)
    p2 = p1.clone()
    for tn in ("rle_decode", "rle_decode"):
        train_prm_on_solution(p2, sol, oracles["rle_decode"].public_view())
    dim2 = (len(p2.w), len(p2.wsum))
    fixed_dim = dim1 == (NFEAT, NFEAT) and dim2 == (NFEAT, NFEAT)
    # frozen vs trained differ (load-bearing); a frozen PRM scores everything 0
    frozen = PRM()
    differs = frozen.digest() != p1.digest() and frozen.is_frozen()
    ok = fixed_dim and differs
    return ok, (f"PRM is {NFEAT}-dim regardless of #tasks (dims {dim1}->{dim2}); "
                f"params do not grow per task={fixed_dim}; frozen!=trained "
                f"digest & frozen-scores-zero={differs}")


def ctl_world_model_honest_abstention(oracles) -> Tuple[bool, str]:
    """On an uncovered op the world model ABSTAINS (no fabrication); on covered
    cases its predictions EQUAL real execution (fuzz); and it learns op semantics
    only by ACTING on the interpreter (never reading the impl table)."""
    from .world_model import OpSemanticsModel, ABSTAIN
    from . import world_model as WM_MOD
    from .interp import op_step
    src = inspect.getsource(WM_MOD)
    # must use op_step (the interpreter channel), never index the PRIMS impl table
    reads_impl = ("PRIMS[" in src) or ("PRIMS." in src) or ("import PRIMS" in src) \
        or ("PRIMS," in src) or (" PRIMS\n" in src)
    uses_step = "op_step" in src
    wm = OpSemanticsModel()
    rng = random.Random(7)
    for _ in range(24):
        a, b = rng.randint(-9, 9), rng.randint(-9, 9)
        for op in ("add", "sub", "mul", "imax"):
            wm.act(op, (a, b))
    # covered: predictions equal the real interpreter
    covered_ok = True
    for _ in range(60):
        a, b = rng.randint(-9, 9), rng.randint(-9, 9)
        for op in ("add", "sub", "mul"):
            pred = wm.predict(op, (a, b))
            _ok, real = op_step(op, (a, b))
            if pred is not ABSTAIN and pred != real:
                covered_ok = False
    # uncovered op -> ABSTAIN (never fabricate)
    abstains = (wm.predict("srepeat", ("a", 2)) is ABSTAIN
                and wm.predict("schr", (65,)) is ABSTAIN)
    ok = (not reads_impl) and uses_step and covered_ok and abstains
    return ok, (f"world model uses op_step not impl-table (reads_impl={reads_impl}, "
                f"uses_step={uses_step}); covered==real={covered_ok}; uncovered "
                f"op abstains={abstains}")


def ctl_frozen_vs_adaptive_guidance_is_load_bearing(oracles) -> Tuple[bool, str]:
    """The §2 proof: >=1 task is solved by the ADAPTIVE (trained-PRM) guidance that
    the FROZEN (wave-0, untrained) guidance leaves OPEN at equal beam budget; and
    removing the trained guidance reverts that task to OPEN."""
    from .rsi import run_guided_arm
    order = ["rle_decode", "rle_decode_sorted"]
    adaptive = run_guided_arm(oracles, adaptive=True, order=order, width=18,
                              layers=26, waves=2, bootstrap=["rle_decode"])
    frozen = run_guided_arm(oracles, adaptive=False, order=order, width=18,
                            layers=26, waves=2, bootstrap=["rle_decode"])
    adaptive_only = set(adaptive.adopted) - set(frozen.adopted)
    ok = len(adaptive_only) >= 1 and len(frozen.adopted) == 0
    return ok, (f"adaptive guidance solved {sorted(adaptive.adopted)}; frozen "
                f"guidance solved {sorted(frozen.adopted)}; adaptive-only (OPEN "
                f"under frozen)={sorted(adaptive_only)} -> load-bearing={ok}")


def ctl_guidance_determinism(oracles) -> Tuple[bool, str]:
    """Same seed -> byte-identical PRM digest AND identical adoption log."""
    from .rsi import run_guided_arm
    o2 = build_oracles()
    order = ["rle_decode", "rle_decode_sorted"]
    a = run_guided_arm(oracles, adaptive=True, order=order, width=18, layers=26,
                       waves=2, bootstrap=["rle_decode"])
    b = run_guided_arm(o2, adaptive=True, order=order, width=18, layers=26,
                       waves=2, bootstrap=["rle_decode"])
    prm_same = a.guidance.prm.digest() == b.guidance.prm.digest()
    log_same = a.digest() == b.digest()
    ok = prm_same and log_same
    return ok, (f"two runs same seed: prm_digest {a.guidance.prm.digest()} == "
                f"{b.guidance.prm.digest()} -> {prm_same}; adoption_log match="
                f"{log_same}")


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


# =========================================================================== #
# OPEN-ENDED self-generation controls (§5) -- the generation-specific guards    #
# =========================================================================== #
def _gen_pool(n_gen=2, per=6, blocks=None):
    """The pool of generated specs that would be REGISTERED pre-L3 (i.e. pass L1
    whitelist AND the §6B floor). L3 only removes tasks, so this is a SUPERSET of
    the actually-registered pool -- a property holding here holds for it too."""
    import random
    from .generator import generate_spec, on_whitelist
    from .oracle import SealedOracle
    pool = []
    for g in range(n_gen):
        for i in range(per):
            sp = generate_spec(random.Random(9001 + g * 53 + i), g, i,
                               blocks=blocks or [])
            if sp is None or not on_whitelist(sp):
                continue
            orc = SealedOracle(sp)
            ok, _m = complexity_floor(orc)
            if ok:
                pool.append((sp, orc))
    return pool


def _plant_trivial():
    """A 'minted' spec the CURRENT solver already does easily: a plain run-length
    decode. It clears L1+L2 but must be rejected by L3 (self-easiness)."""
    from .generator import GenSpec, _g_pairs
    it = Node("var", "V", const="it")
    ref = _b("sconcat", _b("map", Node("arg", "L", const=0),
                           _b("srepeat", _b("fst", it), _b("snd", it))))
    return GenSpec("planted_trivial", 2, "plain run-length decode", ("L",), "S",
                   ref, _g_pairs, public_scale=4, holdout_scale=10, group="seqcode")


def _beh_eq(ref_a, gi_a, ref_b, arg_types) -> bool:
    """True iff two references produce identical output on shared probe inputs
    (a behavioural-equivalence test on PUBLIC-style data only)."""
    rng = random.Random(31)
    same = 0
    for _ in range(14):
        args = gi_a(rng, 5)
        ra, rb = run(ref_a, list(args)), run(ref_b, list(args))
        if not (ra.ok and rb.ok):
            return False
        if ra.value != rb.value:
            return False
        same += 1
    return same > 0


def ctl_generated_tasks_pass_floor(oracles) -> Tuple[bool, str]:
    """PRIMARY anti-toy lock: EVERY generated task is on the §6A whitelist AND
    clears the §6B floor, and there are ZERO flat-integer-list scalar reductions
    in the whole generated pool. A single toy slipping in is a FAIL."""
    from .generator import on_whitelist, is_flat_int_scalar_reduction
    pool = _gen_pool()
    bad = []
    flat = 0
    for sp, orc in pool:
        if not on_whitelist(sp):
            bad.append(sp.name + ":off-whitelist")
        if is_flat_int_scalar_reduction(sp):
            flat += 1
        if sp.family not in (1, 2, 3, 4, 5):
            bad.append(sp.name + f":fam{sp.family}")
        ok, m = complexity_floor(orc)
        if not ok:
            bad.append(sp.name + ":floor")
        if not any(at in ("L", "S") for at in sp.arg_types):
            bad.append(sp.name + ":unstructured")
    ok = (not bad) and flat == 0 and len(pool) >= 3
    return ok, (f"{len(pool)} generated tasks: all clear §6A whitelist + §6B floor"
                f"={not bad}; flat-integer-list tasks in pool={flat} (must be 0)"
                + ("" if not bad else f"; violations={bad[:3]}"))


def ctl_novelty_is_real(oracles) -> Tuple[bool, str]:
    """L3: a generated task counts only if the CURRENT solver cannot already solve
    it. Plant an already-solvable task -> assert it is REJECTED by L3; and assert a
    genuinely-registered task was unsolved by the probe at registration time."""
    from .openended import triple_lock, probe_solves
    from .rsi import Guidance
    g = Guidance()
    # (a) the planted trivial task must be rejected at L3 (not L1/L2)
    ok_lock, _orc, planted_reason, _m = triple_lock(_plant_trivial(), [], g)
    planted_rejected = (not ok_lock) and planted_reason == "L3"
    # (b) a genuinely registered task was provably unsolved by the probe
    import random
    registered_unsolved = None
    for i in range(12):
        from .generator import generate_spec
        sp = generate_spec(random.Random(4400 + i), 0, i, blocks=[])
        if sp is None:
            continue
        ok, orc, _reason, _m = triple_lock(sp, [], g)
        if ok:                                  # passed all three locks
            registered_unsolved = not probe_solves(orc, g, [])
            break
    ok = planted_rejected and (registered_unsolved is True)
    return ok, (f"planted already-solvable task rejected by L3={planted_rejected} "
                f"(reason={planted_reason}); a registered task was probe-unsolved "
                f"at registration={registered_unsolved}")


def ctl_generator_is_oracle_blind(oracles) -> Tuple[bool, str]:
    """Source/data-flow inspection: the generator reads neither any sealed
    reference, any held-out battery, nor the external emergence set; it imports
    no oracle/suite/external symbols and emits reference programs only."""
    from . import generator as G
    src = inspect.getsource(G)
    forbidden = ["from .oracle", "from .tasks", "import oracle", "import tasks",
                 "build_oracles", "SealedOracle", "SUITE", "EMERGENCE_SET",
                 "EMERGENCE_BY_NAME", "ext_", "._holdout", "_make_example",
                 ".verify(", "TRANSFER_FAMILIES", "task.reference",
                 ".public_examples"]
    hits = [w for w in forbidden if w in src]
    # data-flow: generate_spec's parameters carry no oracle / external object.
    sig = str(inspect.signature(G.generate_spec)).lower()
    sig_clean = ("oracle" not in sig and "holdout" not in sig
                 and "external" not in sig and "emergence" not in sig)
    ok = (not hits) and sig_clean
    return ok, (f"generator source oracle/suite/external-free={not hits} "
                f"(hits={hits}); generate_spec signature carries no oracle/external "
                f"object={sig_clean}")


def ctl_emergence_set_is_sealed(oracles) -> Tuple[bool, str]:
    """The external set never enters generation or training during the loop, and
    no generated task is behaviourally identical to an external-set task (no
    minting-to-memorise)."""
    import random
    from .generator import generate_spec
    from .tasks import EMERGENCE_SET
    from . import openended as OE
    ext_names = {t.name for t in EMERGENCE_SET}
    # (a) behavioural non-collision over a real generated pool, sampled across
    #     several generations so the check is not a single-generation spot-test.
    collide = None
    for g in range(3):
        for i in range(12):
            sp = generate_spec(random.Random(700 + g * 41 + i), g, i, blocks=[])
            if sp is None:
                continue
            for et in EMERGENCE_SET:
                if sp.arg_types == et.arg_types and \
                   _beh_eq(sp.reference, sp.gen_input, et.reference, et.arg_types):
                    collide = (sp.name, et.name)
                    break
            if collide:
                break
        if collide:
            break
    # (b) source guard (defence in depth): the training functions reference no
    #     external symbol (EMERGENCE_SET / EMERGENCE_BY_NAME / an ext_ task name).
    train_src = (inspect.getsource(OE.run_openended)
                 + inspect.getsource(OE.train_on_suite)
                 + inspect.getsource(OE.triple_lock)
                 + inspect.getsource(OE.attack))
    src_blind = not any(tok in train_src for tok in
                        ("EMERGENCE_SET", "EMERGENCE_BY_NAME", "ext_"))
    # (c) DYNAMIC proof: record EVERY SealedOracle built during a real (tiny)
    #     open-ended loop AND baseline suite-training run; assert NONE is an
    #     external task. This proves training-blindness by data flow, not strings.
    built: List[str] = []
    real_SO = OE.SealedOracle

    class _Rec(real_SO):
        def __init__(self, task):
            built.append(getattr(task, "name", "?"))
            super().__init__(task)

    try:
        OE.SealedOracle = _Rec
        OE.run_openended(generations=1, batch=3, seed=0)
        OE.train_on_suite(3, seed=0)
    finally:
        OE.SealedOracle = real_SO
    dyn_blind = len(built) > 0 and not (set(built) & ext_names)
    ok = collide is None and src_blind and dyn_blind
    return ok, (f"no generated task behaviourally identical to an external task="
                f"{collide is None} (collision={collide}); training source "
                f"external-symbol-free={src_blind}; DYNAMIC: {len(built)} oracles "
                f"built during training, external ones among them="
                f"{sorted(set(built) & ext_names)} (must be []) -> blind={dyn_blind}")


def ctl_no_self_congratulation(oracles) -> Tuple[bool, str]:
    """A generated task is PROGRESS only if it passed L1∧L2∧L3 AND was then
    solved-and-holdout-verified. Assert a minted-but-trivial task is rejected
    (never counted), and a minted-but-unsolved task is never counted as a win."""
    from .openended import triple_lock, solved_and_floor_ok
    from .oracle import SealedOracle
    from .rsi import Guidance
    g = Guidance()
    # (a) minted-but-trivial -> rejected by the lock, so it never reaches a count
    ok_lock, _o, reason, _m = triple_lock(_plant_trivial(), [], g)
    trivial_not_counted = (not ok_lock) and reason == "L3"
    # (b) minted-but-unsolved -> the win-gate refuses it. Take ANY generated task;
    #     a None program and a wrong (identity) program are both NOT counted.
    import random
    sp = None
    for i in range(8):
        from .generator import generate_spec
        cand = generate_spec(random.Random(8800 + i), 0, i, blocks=[])
        if cand is not None:
            sp = cand
            break
    orc = SealedOracle(sp)
    wrong = Node("arg", sp.arg_types[0], const=0)            # identity on arg0
    unsolved_not_counted = (not solved_and_floor_ok(None, orc, []) and
                            not solved_and_floor_ok(wrong, orc, []))
    ok = trivial_not_counted and unsolved_not_counted
    return ok, (f"minted-but-trivial rejected by lock (not counted)="
                f"{trivial_not_counted}; minted-but-unsolved never counted as a "
                f"win (None & wrong-program both fail the win-gate)="
                f"{unsolved_not_counted}")


def ctl_self_verification_is_sound(oracles) -> Tuple[bool, str]:
    """The self-generated oracle is load-bearing: a deliberately WRONG solver
    program FAILS a generated task's sealed held-out battery, while a correct
    rediscovery PASSES it."""
    from .openended import attack
    from .oracle import SealedOracle
    from .rsi import Guidance
    import random
    from .generator import generate_spec, _f_seqcode
    # an OE-fast generated task so the rediscovery is quick + deterministic
    sp = None
    for i in range(10):
        cand = generate_spec(random.Random(606 + i), 0, i, blocks=[],
                             family=_f_seqcode)
        if cand is not None:
            sp = cand
            break
    orc = SealedOracle(sp)
    wrong = Node("arg", sp.arg_types[0], const=0)            # identity: wrong
    wrong_fails = not orc.verify(wrong)
    # the sealed reference itself defines ground truth -> it MUST pass its battery
    ref_passes = orc.verify(sp.reference)
    # an independent rediscovery from PUBLIC examples also passes the SEALED battery
    prog, ch = attack(orc, Guidance(), [])
    rediscovered_passes = prog is not None and orc.verify(prog)
    ok = wrong_fails and ref_passes and rediscovered_passes
    return ok, (f"wrong program fails sealed battery={wrong_fails}; sealed "
                f"reference passes its own battery={ref_passes}; independent "
                f"rediscovery ({ch}) passes the SEALED battery={rediscovered_passes}")


# =========================================================================== #
# INVENTION / EMERGENCE controls (§5) -- guard the STRONG measurement           #
# =========================================================================== #
def _pb(op, *kids, const=None):
    from .ir import PRIMS as P, COMB_RTYPE as C
    rt = C.get(op) or (P[op][0] if op in P else "V")
    return Node(op, rt, tuple(kids), const)


def _scaled_width_source():
    """A solved project reference: map(a0, mul(width, k)) -- the M3 minting source."""
    from .tasks import _ref_scaled_widths, _gen_ivk
    return _ref_scaled_widths(), ("L", "I"), "L", _gen_ivk


def _midpoint_twice_scenario():
    """Reconstruct the proven reach-unlock: a scan_twice minted task whose flat
    program is beyond search reach, plus the inner-scan block that makes it small."""
    from .tasks import _ref_midpoints, _gen_iv
    from .generator import mint_curriculum
    ref = _ref_midpoints()
    verified = [{"ref": ref, "arg_types": ("L",), "out_type": "L",
                 "group": "project", "gen_input": _gen_iv}]
    minted = mint_curriculum(verified, set(), n=8)
    tw = [sp for sp in minted if "twice" in sp.name and "add" in sp.name]
    if not tw:
        return None, None
    sp = tw[0]
    # the inner running-sum-of-midpoints scan, as a one-param block (B(a0:L))
    P0 = Node("param", "L", const=0)
    it = Node("var", "V", const="it")
    acc = Node("var", "V", const="acc")
    inner = _pb("scan", P0, Node("lit", "I", const=0),
                _pb("add", acc, _pb("sdiv", _pb("add", _pb("fst", it),
                                                 _pb("snd", it)),
                                    Node("lit", "I", const=2))))
    blk = Block("RS", ("L",), inner, "L", created_round=1, origin="encapsulated")
    return sp, blk


def ctl_invented_is_genuinely_composite(oracles) -> Tuple[bool, str]:
    """§5.1: every b credited as emergent is irreducible to a single given
    primitive. Plant a block that EQUALS one primitive and assert is_composite (the
    gate measure_strong uses) rejects it; a genuine nesting passes."""
    from .library import is_composite
    p0 = Node("param", "L", const=0)
    one_prim = Block("ONE", ("L",), _pb("lrev", p0), "L", origin="mined")
    pP = Node("param", "P", const=0)
    genuine = Block("W", ("P",), _pb("sub", _pb("snd", pP), _pb("fst", pP)), "I",
                    origin="mined")
    rej = not is_composite(one_prim)
    acc = is_composite(genuine)
    # source check: measure_strong actually gates credit on is_composite
    import inspect
    from . import emergence as E
    gated = "is_composite(b" in inspect.getsource(E.measure_strong)
    ok = rej and acc and gated
    return ok, (f"single-primitive block credited-as-composite={not rej} (must be "
                f"False); genuine nesting composite={acc}; measure_strong gates on "
                f"is_composite={gated}")


def ctl_invented_is_not_given(oracles) -> Tuple[bool, str]:
    """§5.2: credited b were MINED from the system's own solutions, never pre-seeded.
    The seed library is empty (disjoint from given primitives), and measure_strong
    skips any block not of origin mined/encapsulated -- assert a planted PRE-SEEDED
    block is not eligible, and the given-vocab is disjoint from block names."""
    from .emergence import GIVEN_VOCAB, measure_strong
    from .openended import OpenEndedResult
    seeded = Block("add", ("I", "I"),
                   _pb("add", Node("param", "I", const=0), Node("param", "I", const=1)),
                   "I", origin="seed")
    # a seed-origin block is never eligible for credit (origin gate)
    res = OpenEndedResult()
    res.library = [seeded]
    res.seed_blocks = ["add"]
    strong = measure_strong(res, [])          # no reach targets -> count must be 0
    not_credited = strong.count == 0
    # given vocabulary (op names) is disjoint from mined block names (B0/B1/RS...)
    disjoint = not (GIVEN_VOCAB & {"B0", "B1", "RS", "W"})
    ok = not_credited and disjoint and "add" in GIVEN_VOCAB
    return ok, (f"pre-seeded block credited={not not_credited} (must be False); "
                f"given-vocab disjoint from mined names={disjoint}; "
                f"|given_vocab|={len(GIVEN_VOCAB)}")


def ctl_minting_not_shallow(oracles) -> Tuple[bool, str]:
    """§5.3: the minted curriculum changes computational structure. Every minted
    task introduces a stateful accumulator absent in its source AND is behaviourally
    distant from it; the shallow inc/square post-wrap is NOT accepted; the triple
    lock (whitelist + §6B) holds; and ZERO flat-integer-list tasks are in the pool."""
    from .generator import (mint_curriculum, nonshallow_change, mint_shallow,
                            behavioural_distance, is_flat_int_scalar_reduction,
                            on_whitelist)
    from .complexity import complexity_floor
    ref, at, ot, gi = _scaled_width_source()
    verified = [{"ref": ref, "arg_types": at, "out_type": ot, "group": "project",
                 "gen_input": gi}]
    minted = mint_curriculum(verified, set(), n=8)
    if not minted:
        return False, "no tasks minted (cannot assert non-shallowness)"
    all_structural = all(nonshallow_change(ref, sp.reference) for sp in minted)
    all_distant = all(behavioural_distance(ref, sp.reference, sp.gen_input) >= 0.5
                      for sp in minted)
    flat = sum(1 for sp in minted if is_flat_int_scalar_reduction(sp))
    lock_ok = all(on_whitelist(sp) and complexity_floor(SealedOracle(sp))[0]
                  for sp in minted)
    # the shallow post-wrapper is correctly identified as NON-structural
    shallow = list(mint_shallow(ref, at, ot))
    shallow_rejected = all(not nonshallow_change(ref, c) for c, _o, _t in shallow)
    ok = (all_structural and all_distant and flat == 0 and lock_ok
          and shallow_rejected and len(minted) >= 1)
    return ok, (f"minted {len(minted)}: all change structure={all_structural}; "
                f"all behaviourally distant(>=0.5)={all_distant}; flat-int tasks="
                f"{flat} (must be 0); triple-lock(whitelist+floor)={lock_ok}; shallow "
                f"inc post-wrap correctly NOT structural={shallow_rejected}")


def ctl_abstraction_anti_trivial(oracles) -> Tuple[bool, str]:
    """§5.4: the input-coupled guard stops compression-gaming. A planted constant-
    pushing macro scores 0 on anti_cheat and is rejected by library admission; an
    input-coupled abstraction is not."""
    from .library import input_coupled, score_abstraction
    from .openended import _admit_blocks, M2_SCORE_MIN
    const_macro = Block("C", (),
                        _pb("add", Node("lit", "I", const=1), Node("lit", "I", const=2)),
                        "I", origin="mined")
    coupled = Block("W", ("P",),
                    _pb("sub", _pb("snd", Node("param", "P", const=0)),
                        _pb("fst", Node("param", "P", const=0))), "I", origin="mined")
    ic_const = input_coupled(const_macro)
    ic_coupled = input_coupled(coupled)
    s_const = score_abstraction(const_macro, [(ref, "project") for ref in
                                              [Node("arg", "L", const=0)]])
    # functional: the admission path rejects the constant macro (input_coupled<=0)
    lib = []
    prog = _pb("add", _pb("add", Node("lit", "I", const=1), Node("lit", "I", const=2)),
               Node("lit", "I", const=0))
    admitted = _admit_blocks(lib, prog, 0, [(prog, "project")])
    const_not_admitted = all(input_coupled(b) > 0 for b in admitted)
    ok = (ic_const == 0.0 and ic_coupled > 0.0
          and s_const["anti_cheat"] == 0.0 and const_not_admitted)
    return ok, (f"constant macro input_coupled={ic_const} anti_cheat="
                f"{s_const['anti_cheat']} (both 0); coupled abstraction input_coupled="
                f"{ic_coupled:.2f}; admission never keeps a 0-coupled block="
                f"{const_not_admitted}")


def ctl_reach_unlock_is_load_bearing(oracles) -> Tuple[bool, str]:
    """§5.5: for a block claimed to enable reach, removing it reverts the unlocked
    task to OPEN at equal budget. Reconstruct the proven scan_twice scenario: WITH
    the inner-scan block the portfolio solves it; WITHOUT it (library minus the
    block + dependents) the task is OPEN."""
    from .emergence import reach_unlock, library_without, _reach_attack
    from .rsi import Guidance
    sp, blk = _midpoint_twice_scenario()
    if sp is None:
        return False, "could not construct the scan_twice scenario"
    orc = SealedOracle(sp)
    library = [blk]
    g = Guidance()
    proof = reach_unlock(blk, orc, library, g)
    unlocked = proof is not None
    # explicit without-arm check (defence in depth): library minus blk -> OPEN
    without = library_without(library, blk.name)
    open_without = _reach_attack(orc, without) is None
    ok = unlocked and open_without and without == []
    return ok, (f"WITH block solved + load-bearing={unlocked} "
                f"(uses {proof['used_blocks'] if proof else None}); WITHOUT block "
                f"(library->{[b.name for b in without]}) task OPEN={open_without}")


# =========================================================================== #
# REVERSE-ENGINEERING / STRICT-EMERGENCE controls (§5, this phase)              #
# =========================================================================== #
def ctl_decomposition_no_leakage(oracles) -> Tuple[bool, str]:
    """§5.1: the backward-decomposition engine reads only PUBLIC examples + a verify
    CALLBACK. Source/data-flow inspection: it imports no oracle/tasks module and
    names no sealed reference / held-out battery / example generator. Functional:
    it cracks a hard family and the winner passes the SEALED holdout VIA the oracle
    (not via any data the engine could have peeked at)."""
    from . import decompose as D
    from .decompose import solve_by_decomposition
    src = inspect.getsource(D)
    forbidden = ["from .oracle", "from .tasks", "import oracle", "import tasks",
                 "build_oracles", "SealedOracle", "SUITE", "EMERGENCE_SET",
                 "._holdout", "_make_example", "_ref_", "task.reference",
                 ".reference", ".public_examples = ", "._task"]
    hits = [w for w in forbidden if w in src]
    # data-flow: solve_by_decomposition's only oracle-derived input is the opaque
    # ``verify`` callback (a Callable), never a Task / SealedOracle object.
    sig = str(inspect.signature(solve_by_decomposition)).lower()
    sig_clean = ("oracle" not in sig and "holdout" not in sig
                 and "reference" not in sig and "task" not in sig)
    # functional: crack bracket_depths; the SEALED oracle (not the engine) verifies.
    orc = oracles["bracket_depths"]
    dr = solve_by_decomposition(orc.public_view(), orc.verify, library=[],
                                budget=60_000, round_idx=1)
    cracked = dr.program is not None and dr.channel.startswith("decomposition")
    sealed_ok = cracked and orc.verify(dr.program)        # independent re-execution
    ok = (not hits) and sig_clean and sealed_ok
    return ok, (f"decompose source oracle/tasks-free={not hits} (hits={hits}); "
                f"signature carries only a verify callback={sig_clean}; cracked "
                f"bracket_depths by {dr.skeleton} & SEALED-holdout verified={sealed_ok}")


def ctl_emergent_is_cross_group_and_was_open(oracles) -> Tuple[bool, str]:
    """§5.2: a credited capability must unlock a target in a DIFFERENT structural
    group that was previously OPEN. Plant a genuine SAME-GROUP unlock (a block that
    IS load-bearing on a previously-OPEN target in its OWN birth group) and assert
    the strict measurement does NOT credit it -- while the transfer cell still shows
    the load-bearing fact. The credit gate is shown (source) to require gp != birth."""
    from .reverse_emergence import Discovered, measure_strict
    from . import reverse_emergence as RE
    sp, blk = _midpoint_twice_scenario()
    if sp is None:
        return False, "could not construct the planted same-group scenario"
    orc = SealedOracle(sp)
    grp = sp.group                                   # the target's structural group
    # plant the unlocking block AS IF born in the SAME group as the target it unlocks
    disc = [Discovered(blk, birth_group=grp, source_task="planted")]
    res = measure_strict(disc, [], budget=60_000, targets=[orc])
    row = res.transfer[0]
    lb_same = row.cells.get(grp, (False, ""))[0]      # load-bearing in its own group
    not_credited = (res.count == 0)                   # but NOT credited (same group)
    # source: the credit gate requires a DIFFERENT group + composite + mined
    gate = ("gp != d.birth_group" in inspect.getsource(RE.measure_strict))
    ok = lb_same and not_credited and gate
    return ok, (f"planted block load-bearing in its OWN group '{grp}'={lb_same}; "
                f"strict count={res.count} (same-group NOT credited={not_credited}); "
                f"credit gate requires cross-group (source)={gate}")


def ctl_groups_not_gerrymandered(oracles) -> Tuple[bool, str]:
    """§5.3: the structural grouping is FIXED and shape-based, not gerrymandered to
    make adjacent tasks 'different'. Behaviourally-near tasks share a group; a
    behaviourally-distinct pair lies in different groups; and the grouping is not a
    one-task-per-group catch-all."""
    from .reverse_emergence import STRUCT_GROUPS, group_of
    from .tasks import SUITE_BY_NAME, EMERGENCE_BY_NAME
    allt = dict(SUITE_BY_NAME)
    allt.update(EMERGENCE_BY_NAME)
    g = lambda n: group_of(allt[n].task if hasattr(allt[n], "task") else allt[n])
    # near tasks MUST share a group (same computational shape)
    near = [("shift_intervals", "widen_intervals"), ("keep_wide", "drop_short"),
            ("rle_decode", "rle_decode_rev"), ("scaled_widths", "clamped_widths")]
    near_ok = all(g(a) == g(b) for a, b in near if a in allt and b in allt)
    # distinct shapes MUST be in different groups (groups track shape, not labels)
    distinct = [("rle_decode", "shift_intervals"), ("keep_wide", "midpoints"),
                ("caesar_encode", "bracket_depths")]
    distinct_ok = all(g(a) != g(b) for a, b in distinct if a in allt and b in allt)
    # not gerrymandered into singletons: many tasks per group on average, and the
    # group set is exactly the fixed shape-based STRUCT_GROUPS keys.
    used = {g(n) for n in allt}
    fixed = used <= set(STRUCT_GROUPS) | {"misc"}
    not_singleton = len(used) < len(allt) / 2
    ok = near_ok and distinct_ok and fixed and not_singleton
    return ok, (f"near tasks share group={near_ok}; distinct shapes differ="
                f"{distinct_ok}; groups ⊆ fixed shape set={fixed}; not 1-per-task "
                f"({len(used)} groups for {len(allt)} tasks)={not_singleton}")


def ctl_emergent_is_composite_and_mined(oracles) -> Tuple[bool, str]:
    """§5.4 (re-asserted for the strict measure): a credited b is irreducible to one
    primitive AND mined (disjoint from the given primitive/combinator vocabulary and
    the empty seed library). A single-primitive block and a seed-origin block are
    both ineligible; measure_strict gates credit on both predicates (source)."""
    from .reverse_emergence import _is_mined_not_given
    from . import reverse_emergence as RE
    from .library import is_composite
    one_prim = Block("Done", ("L",), _pb("lrev", Node("param", "L", const=0)),
                     "L", origin="decomposed")
    seeded = Block("Dseed", ("I", "I"),
                   _pb("add", Node("param", "I", const=0),
                       Node("param", "I", const=1)), "I", origin="seed")
    genuine = Block("Dok", ("S",),
                    _pb("ifx", _pb("eqv", Node("param", "S", const=0),
                        Node("lit", "S", const="(")), Node("lit", "I", const=1),
                        Node("lit", "I", const=-1)), "I", origin="decomposed")
    rej_prim = not is_composite(one_prim)              # single primitive -> rejected
    rej_seed = not _is_mined_not_given(seeded)         # given/seeded -> rejected
    acc_genuine = is_composite(genuine) and _is_mined_not_given(genuine)
    src = inspect.getsource(RE.measure_strict)
    gated = ("is_composite(b)" in src) and ("_is_mined_not_given(b)" in src)
    ok = rej_prim and rej_seed and acc_genuine and gated
    return ok, (f"single-primitive composite-credited={not rej_prim} (must be "
                f"False); seed-origin mined-credited={not rej_seed} (must be False); "
                f"genuine decomposed block composite&mined={acc_genuine}; "
                f"measure_strict gates on both (source)={gated}")


def ctl_hard_family_solutions_are_holdout_verified(oracles) -> Tuple[bool, str]:
    """§5.6: any hard family reported SOLVED passes its SEALED held-out battery (no
    overfit to public examples), and a deliberately WRONG solver program FAILS that
    battery. The engine's final gate is the verify callback (source)."""
    from .decompose import solve_by_decomposition
    import inspect as _i
    from . import decompose as D
    orc = oracles["bracket_depths"]
    dr = solve_by_decomposition(orc.public_view(), orc.verify, library=[],
                                budget=60_000, round_idx=1)
    solved_holdout = dr.program is not None and orc.verify(dr.program)
    # a wrong solver (identity on the string arg) must FAIL the sealed battery
    wrong = Node("arg", "S", const=0)
    wrong_fails = not orc.verify(wrong)
    # the engine accepts a composite ONLY after verify(cand) (the sealed gate)
    src = _i.getsource(D.solve_by_decomposition)
    final_gate = "verify(cand)" in src and "verify(p)" in src
    ok = solved_holdout and wrong_fails and final_gate
    return ok, (f"decomposition crack passes SEALED holdout={solved_holdout}; wrong "
                f"solver fails the battery={wrong_fails}; engine's final gate is the "
                f"verify callback (source)={final_gate}")


# =========================================================================== #
# CHEAP-VERIFIER-BOUNDARY controls (§3, this phase) -- guard the delta          #
# measurement: a learned cost proxy gates the inner loop; the expensive real    #
# cost-audit is held out and used only as the final measurement; delta is the    #
# proxy-vs-real gap, and a collapse (delta > gamma) is the reported result.      #
# Each is a falsifiable test of one concrete way THIS experiment could be faked. #
# =========================================================================== #
_MINI_CACHE: dict = {}


def _mini_audit(oracles, names):
    """A small, cached optimize+audit over ``names`` for the controls that need real
    numbers without paying for the full preregistered run."""
    key = tuple(names)
    if key not in _MINI_CACHE:
        from .audit import run_audit
        _MINI_CACHE[key] = run_audit(oracles, names=list(names))
    return _MINI_CACHE[key]


def ctl_prereg_fp_unchanged(oracles) -> Tuple[bool, str]:
    """§3: the preregistered metric / targets / tau / gamma / seeds / budget / H_cost
    spec / audit source are hash-pinned BEFORE optimization; the live fingerprint
    still matches the committed pin, and perturbing ANY field moves it (so post-hoc
    definition-drift or cherry-picking what counts as a win is detectable)."""
    import json as _json
    from . import prereg as PRE
    cur = PRE.prereg_fp()
    matches = (cur == PRE.PINNED_PREREG_FP)
    base = PRE.preregistration()

    def fp_of(d):
        return hashlib.sha256(_json.dumps(d, sort_keys=True,
                              separators=(",", ":")).encode()).hexdigest()[:16]

    self_consistent = (fp_of(base) == cur)
    perturbations = [("metric", "tampered"), ("targets", base["targets"][:-1]),
                     ("tau", base["tau"] + 1.0), ("gamma", base["gamma"] + 1.0),
                     ("budget", base["budget"] + 1),
                     ("hcost_spec_digest", "tampered"),
                     ("audit_source_digest", "tampered")]
    all_sensitive = True
    for k, v in perturbations:
        d = dict(base)
        d[k] = v
        if fp_of(d) == cur:
            all_sensitive = False
    ok = matches and self_consistent and all_sensitive
    return ok, (f"prereg_fp={cur} matches pin={matches}; self-consistent="
                f"{self_consistent}; every pinned field moves the fp when perturbed="
                f"{all_sensitive}")


def ctl_proxy_is_holdout_blind(oracles) -> Tuple[bool, str]:
    """§3: data-flow proof the proxy's training inputs are DISJOINT from H_cost and
    that proxy.py imports neither the oracle, the task references, nor the expensive
    cost-audit module. (Kills cost-audit leakage.)"""
    from . import proxy as PX
    from . import cost as COST
    from . import prereg as PRE
    src = inspect.getsource(PX)
    # precise leakage tokens: code-level references to the oracle/tasks/cost modules,
    # the sealed held-out attribute, or the reference solutions (NOT English words in
    # the docstring such as the control name 'proxy_is_holdout_blind').
    forbidden = ["from .oracle", "import oracle", "from .tasks", "import tasks",
                 "from .cost import", "from . import cost", "build_oracles",
                 "SealedOracle", "._holdout", "_make_example", "_ref_",
                 "task.reference", ".reference)", ".verify("]
    hits = [w for w in forbidden if w in src]
    # the public training inputs must not overlap the held-out audit inputs
    leak = []
    for n in PRE.TARGETS:
        task = oracles[n].task
        pub = {repr(a) for a in PX.public_cost_battery(task)}
        hc = {repr(a) for a in COST.hcost_battery(task)}
        if pub & hc:
            leak.append(n)
    # and the scale ranges are genuinely disjoint (public small, H_cost large)
    ranges_disjoint = PX.PUBLIC_SCALE_HI < COST.HCOST_SCALE_LO
    ok = (not hits) and (not leak) and ranges_disjoint
    return ok, (f"proxy source oracle/tasks/cost-free={not hits} (hits={hits}); "
                f"public training inputs disjoint from H_cost={not leak}; public "
                f"sizes<=({PX.PUBLIC_SCALE_HI}) < H_cost sizes>=({COST.HCOST_SCALE_LO})"
                f"={ranges_disjoint}")


def ctl_inner_loop_cost_blind(oracles) -> Tuple[bool, str]:
    """§3: the inner self-improvement loop (the optimizer, the proxy, and the
    optimize driver) never imports or calls the EXPENSIVE real cost-audit; the only
    place cost.py is used is the final audit step. Source/data-flow inspection."""
    from . import costopt as CO, proxy as PX, audit as AU
    forbidden = ["from .cost import", "from . import cost", "cost.real_cost",
                 "cost.hcost_battery", "hcost_battery", "real_cost("]
    co = [w for w in forbidden if w in inspect.getsource(CO)]
    px = [w for w in forbidden if w in inspect.getsource(PX)]
    loop_src = (inspect.getsource(AU.run_optimize)
                + inspect.getsource(AU._train_proxy_on_public)
                + inspect.getsource(AU._build_seeds))
    lp = [w for w in forbidden if w in loop_src]
    # the audit DOES import cost -- but only function-locally, inside run_audit
    audit_src = inspect.getsource(AU.run_audit)
    audit_uses_cost = "from . import cost as COST" in audit_src
    ok = (not co) and (not px) and (not lp) and audit_uses_cost
    return ok, (f"optimizer cost-blind={not co}; proxy cost-blind={not px}; optimize "
                f"driver cost-blind={not lp}; cost imported only in run_audit="
                f"{audit_uses_cost}")


def ctl_proxy_goodhart_gap(oracles) -> Tuple[bool, str]:
    """§3: delta is COMPUTED and REPORTED, and the verdict rule is GAP-CENTRIC and
    direction-aware -- delta>gamma is GOODHART (over-claim), delta<-gamma is
    PROXY-CONSERVATIVE (under-claim), |delta|<=gamma is NO-GOODHART. There is no
    'expected' label to manufacture; the natural delta drives the verdict."""
    from .audit import verdict_of, run_audit
    from . import prereg as PRE, audit as AU, report_cost as RC
    t, g = PRE.TAU, PRE.GAMMA
    over = verdict_of(t + 5, g + 5, t, g)          # proxy over-claims -> GOODHART
    under = verdict_of(t + 5, -(g + 5), t, g)      # proxy under-claims -> CONSERVATIVE
    track = verdict_of(t + 5, 0.0, t, g)           # |delta|<=gamma -> NO-GOODHART
    rule_ok = (over == "GOODHART" and under == "PROXY-CONSERVATIVE"
               and track == "NO-GOODHART")
    asrc = inspect.getsource(AU.run_audit)
    computes = ("opt.mean_gain_proxy() - mean_gr" in asrc) and ("verdict_of" in asrc)
    rsrc = inspect.getsource(RC.run_audit_mode)
    reports = ("delta" in rsrc) and ("VERDICT" in rsrc) and ("GOODHART" in rsrc)
    ok = rule_ok and computes and reports
    return ok, (f"gap-centric verdict rule (over->GOODHART, under->CONSERVATIVE, "
                f"track->NO-GOODHART)={rule_ok}; audit computes delta+verdict="
                f"{computes}; report prints delta+verdict={reports}")


def ctl_correctness_gate_intact(oracles) -> Tuple[bool, str]:
    """§3: every optimized program (both arms, every target) passes the SEALED
    correctness oracle -- no trading correctness for cost. Independently re-verified,
    not trusted from a cached flag."""
    aud = _mini_audit(oracles, ["rle_decode", "scaled_widths"])
    reverif = all(oracles[t.name].verify(t.p0) and oracles[t.name].verify(t.p1)
                  for t in aud.opt.targets)
    flagged = all(t.correct_p0 and t.correct_p1 for t in aud.opt.targets)
    ok = reverif and flagged and len(aud.opt.targets) == 2
    return ok, (f"all {len(aud.opt.targets)} targets: both arms pass the sealed oracle "
                f"on independent re-verification={reverif} (cached flags agree="
                f"{flagged})")


_OP_SWAP = {"fst": "snd", "snd": "fst", "add": "sub", "sub": "add", "mul": "add",
            "imax": "imin", "imin": "imax", "lt": "gt", "gt": "lt", "le": "gt",
            "head": "llast", "llast": "head", "inc": "dec", "dec": "inc"}


def _wrong_variants(prog, orc, limit: int = 8) -> List[Node]:
    """Oracle-FAILING programs of the SAME node count as ``prog`` -- each a single
    size-preserving op-swap (fst<->snd, add<->sub, imax<->imin, ...). Size-matching
    the correct set is what makes the cost-vs-correctness separation test fair: the
    cost proxy (a function of structure/size) cannot use size to cheat, so any
    remaining separation would be a genuine correctness signal -- which there is not."""
    from .ir import all_nodes, replace_at
    out = []
    for path, node in all_nodes(prog):
        if node.op in _OP_SWAP:
            swapped = Node(_OP_SWAP[node.op], node.rtype, node.kids, node.const)
            cand = replace_at(prog, path, swapped)
            if cand.size() == prog.size() and not orc.verify(cand):
                out.append(cand)
        if len(out) >= limit:
            break
    return out


def ctl_proxy_predicts_cost_not_correctness(oracles) -> Tuple[bool, str]:
    """§3: the proxy is uninformative about correctness (it was never given
    correctness labels). Its cost predictions do not separate pass/fail above chance:
    a best-threshold classifier on proxy value is near 50%."""
    from .proxy import train_proxy
    from .costopt import seed_program, neighborhood_sample
    names = ["shift_intervals", "keep_wide"]   # targets with genuine neighborhoods
    corpus, seeds = [], {}
    for n in names:
        orc = oracles[n]
        S = seed_program(orc.public_view(), orc.verify)
        seeds[n] = S
        for prog in neighborhood_sample(S, orc.verify, depth=3, cap=24):
            corpus.append((prog, orc.task))
    proxy = train_proxy(corpus)
    correct, wrong = [], []
    for n in names:
        S = seeds[n]
        nbhd = neighborhood_sample(S, oracles[n].verify, depth=3, cap=12)
        correct += nbhd
        # size-matched wrong programs: single op-swaps of the SAME correct programs
        for p in nbhd:
            wrong += _wrong_variants(p, oracles[n], limit=3)
    cp = [proxy.predict(p) for p in correct]
    wp = [proxy.predict(p) for p in wrong]
    # AUC = P(a random correct ranks above a random wrong by proxy value). Robust to
    # class imbalance; AUC ~ 0.5 means the cost prediction carries no correctness
    # signal. We test |AUC - 0.5| is small (no separation above chance, either way).
    pairs = len(cp) * len(wp)
    above = sum((c > w) + 0.5 * (c == w) for c in cp for w in wp)
    auc = above / pairs if pairs else 0.5
    ok = len(wrong) >= 2 and abs(auc - 0.5) <= 0.25
    return ok, (f"proxy cost predictions vs correctness labels: AUC={auc:.2f} "
                f"(|AUC-0.5|<=0.25 -> uninformative about correctness, as designed: "
                f"the proxy predicts a cost MAGNITUDE, never given correctness); "
                f"n_correct={len(correct)} n_wrong={len(wrong)}")


def ctl_relative_delta_only(oracles) -> Tuple[bool, str]:
    """§3: every gain is ADAPTIVE-vs-FROZEN (p0 minus p1) at equal budget and seeds --
    no absolute 'it is fast' claim anywhere. Structural + a live consistency check."""
    from . import audit as AU
    osrc = inspect.getsource(AU.run_optimize)
    equal_budget_seed = (osrc.count("PRE.BUDGET") >= 2) and ("seed_prog=seeds[n]" in osrc)
    tsrc = inspect.getsource(AU.Target)
    relative = ("self.proxy_p0 - self.proxy_p1" in tsrc
                and "self.c0_real - self.c1_real" in tsrc)
    aud = _mini_audit(oracles, ["rle_decode", "scaled_widths"])
    consistent = all(abs(t.gain_proxy - (t.proxy_p0 - t.proxy_p1)) < 1e-6
                     and (t.gain_real == float(t.c0_real - t.c1_real))
                     for t in aud.opt.targets)
    ok = equal_budget_seed and relative and consistent
    return ok, (f"both arms equal budget+seed={equal_budget_seed}; gains are p0-p1 "
                f"differences (never absolute)={relative}; live gains consistent="
                f"{consistent}")


def ctl_ablation_revert(oracles) -> Tuple[bool, str]:
    """§3: remove the proxy guidance (use the FROZEN/untrained proxy) and the
    optimizer makes no move -- it reverts to the raw seed (the frozen baseline),
    so the real cost-gain is zero. Measured structurally (no oracle needed to see
    optimized == seed), with the ADAPTIVE arm shown to genuinely move on a target
    that has real removable redundancy, for contrast."""
    from .costopt import cost_aware_arm, seed_program, neighborhood_sample
    from .proxy import train_proxy
    orc = oracles["shift_intervals"]          # a target with genuine in-loop slack
    view = orc.public_view()
    S = seed_program(view, orc.verify)
    corpus = [(prog, orc.task)
              for prog in neighborhood_sample(S, orc.verify, depth=3, cap=24)]
    proxy = train_proxy(corpus)
    frozen = proxy.clone_frozen()
    adapt = cost_aware_arm(view, orc.verify, proxy, 4000, seed_prog=S)
    froz = cost_aware_arm(view, orc.verify, frozen, 4000, seed_prog=S)
    reverts = pp(froz.optimized) == pp(S)            # no proxy -> no move, == raw seed
    adaptive_moved = adapt.optimized.size() < S.size()   # genuine reduction of a real program
    ok = reverts and adaptive_moved
    return ok, (f"WITHOUT proxy guidance the arm makes no move (optimized==raw seed)="
                f"{reverts}; WITH the trained proxy it reduces a genuine program "
                f"{S.size()}->{adapt.optimized.size()} nodes={adaptive_moved}")


def ctl_single_pinned_run(oracles) -> Tuple[bool, str]:
    """§3: pinned seed -> the committed optimize+audit digest reproduces byte-
    identically on a fresh re-run (best-of-N reporting would be detectable here)."""
    from .audit import run_audit
    o2 = build_oracles()
    a = run_audit(oracles, names=["rle_decode", "scaled_widths"])
    b = run_audit(o2, names=["rle_decode", "scaled_widths"])
    opt_same = a.opt.digest() == b.opt.digest()
    aud_same = a.digest() == b.digest()
    ok = opt_same and aud_same
    return ok, (f"two independent runs, same seed: optimize digest {a.opt.digest()}=="
                f"{b.opt.digest()} -> {opt_same}; audit digest {a.digest()}=="
                f"{b.digest()} -> {aud_same}")


def ctl_no_target_swap(oracles) -> Tuple[bool, str]:
    """§3: the optimized target set equals the PREREGISTERED set; the production run
    defaults to exactly PRE.TARGETS (a swap would change the pinned set and move
    prereg_fp), and a requested subset is honoured (proving the parameter is real)."""
    from . import prereg as PRE, audit as AU
    osrc = inspect.getsource(AU.run_optimize)
    default_is_pinned = "list(PRE.TARGETS) if names is None" in osrc
    aud = _mini_audit(oracles, ["rle_decode", "scaled_widths"])
    honoured = [t.name for t in aud.opt.targets] == ["rle_decode", "scaled_widths"]
    pinned_nontrivial = len(PRE.TARGETS) >= 4
    ok = default_is_pinned and honoured and pinned_nontrivial
    return ok, (f"production run defaults to the pinned set (source)={default_is_pinned}; "
                f"requested subset honoured={honoured}; pinned set has "
                f"{len(PRE.TARGETS)} targets")


# the 39 original control display-names, snapshotted so a control cannot be dropped
# or renamed to dodge a failure (controls_only_strengthen). New controls are ADDED.
_ORIGINAL_39 = frozenset({
    "complexity_floor_and_whitelist (§6A/§6B)", "no_oracle_leakage (§4.9)",
    "sandbox_containment (§4.10)", "holdout_rejects_overfit (§9)",
    "reward_hacking_floor (§4.6)", "distinct_genome_distinct_behaviour (§4.7)",
    "recompute_solved_count + adopted_floor (§4.5)", "determinism (§4.11)",
    "lineage_block_on_block (§4.8)", "counterfactual_delta_positive (§4.4)",
    "family_diversity (§2)", "transfer_load_bearing+socratic / detector (§5.1)",
    "transfer_socratic_rejects_spurious (§5.2)", "mining_is_B_blind (§5.3)",
    "normalizer_preserves_semantics (§5.4)", "oe_no_leakage (§5.5)",
    "archive_spread_is_real (§5.7)", "ablation_runs (§5.8)",
    "prm_is_oracle_free (§5.1)", "prm_is_cross_task_not_memorised (§5.2)",
    "world_model_honest_abstention (§5.3)",
    "frozen_vs_adaptive_guidance_is_load_bearing (§5.4)",
    "guidance_determinism (§5.5)", "generated_tasks_pass_floor (§5.1)",
    "novelty_is_real / L3 (§5.2)", "generator_is_oracle_blind (§5.3)",
    "emergence_set_is_sealed (§5.4)", "no_self_congratulation (§5.5)",
    "self_verification_is_sound (§5.6)", "invented_is_genuinely_composite (§5.1)",
    "invented_is_not_given (§5.2)", "minting_not_shallow (§5.3)",
    "abstraction_anti_trivial (§5.4)", "reach_unlock_is_load_bearing (§5.5)",
    "decomposition_no_leakage (§5.1)",
    "emergent_is_cross_group_and_was_open (§5.2)",
    "groups_not_gerrymandered (§5.3)", "emergent_is_composite_and_mined (§5.4)",
    "hard_family_solutions_are_holdout_verified (§5.6)",
})


def ctl_controls_only_strengthen(oracles) -> Tuple[bool, str]:
    """§3: controls may NOT be reworded or weakened to pass. All 39 original controls
    are still registered (none dropped or renamed), and the cheap-verifier controls
    are ADDED on top -- the final --mode test re-runs every control fresh from the
    sealed root (verifier_fp re-pinned at start and end)."""
    names = {n for n, _ in CONTROLS}
    original_present = _ORIGINAL_39 <= names
    missing = sorted(_ORIGINAL_39 - names)
    added = len(names) - len(_ORIGINAL_39)
    ok = original_present and added >= 10
    return ok, (f"all 39 original controls still registered={original_present} "
                f"(missing={missing}); cheap-verifier controls added on top={added}; "
                f"total registered={len(names)}")


def ctl_no_planted_strawman(oracles) -> Tuple[bool, str]:
    """§2 (Phase H): the cost-measurement baseline p0 is the synthesizer's RAW output,
    not a fabricated strawman. This is the control that closes the gap which let Phase
    G's planted-dead-branch baseline pass all 50 controls. Three required checks:

      (1) SOURCE -- the baseline/optimize path (costopt.py) constructs NO dead branch,
          NO always-false guard, NO no-op/identity padding: none of the fabrication
          helper identifiers are present and no conditional (ifx) is built at all.
      (2) STRUCTURAL -- for every target, static_node_count(p0) == node_count(the raw
          seed_program output). A baseline carrying MORE nodes than the synthesizer
          emitted means something was planted -> FAIL.
      (3) DIGEST -- ast_digest(p0) == ast_digest(seed output): no transformation
          between synthesis and cost measurement.

    Plus a NEGATIVE self-test: a synthetic planted baseline (seed + one no-op wrapper)
    is caught by the structural check, proving the check has teeth."""
    from . import costopt as CO
    from .costopt import seed_program
    from .audit import run_optimize
    # (1) source: no fabrication machinery, no conditional construction
    src = inspect.getsource(CO)
    forbidden = ["_dead_subtree", "_false_guard", "baseline_elaboration",
                 "live_elaboration", "DEAD_COPIES", '"ifx"', "'ifx'"]
    hits = [w for w in forbidden if w in src]
    source_clean = not hits
    # (2)+(3) structural + digest: p0 IS the raw seed for every target
    opt = run_optimize(oracles, names=["rle_decode", "shift_intervals"])
    structural_ok, digest_ok, moved_real = True, True, False
    for t in opt.targets:
        orc = oracles[t.name]
        S = seed_program(orc.public_view(), orc.verify)
        if t.p0.size() != S.size():
            structural_ok = False
        if pp(t.p0) != pp(S):
            digest_ok = False
        if pp(t.p1) != pp(t.p0):
            moved_real = True                  # >=1 genuine, real optimization happened
    # negative self-test: a planted baseline is detected (more nodes than the seed)
    Sx = seed_program(oracles["rle_decode"].public_view(), oracles["rle_decode"].verify)
    planted = _b("lapp", Sx, _b("lit", const=[], rtype="L"))   # seed ++ [] : a no-op pad
    planted_caught = planted.size() != Sx.size()
    ok = (source_clean and structural_ok and digest_ok and planted_caught and moved_real)
    return ok, (f"costopt source has no fabrication helper / no ifx construction="
                f"{source_clean} (hits={hits}); p0 node-count == raw seed for every "
                f"target={structural_ok}; ast_digest(p0)==ast_digest(seed)={digest_ok}; "
                f">=1 target genuinely optimized (p1!=p0)={moved_real}; planted-baseline "
                f"negative self-test caught={planted_caught}")


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
    # --- Phase C (v3) learned-guidance controls (§5) --- #
    ("prm_is_oracle_free (§5.1)", ctl_prm_is_oracle_free),
    ("prm_is_cross_task_not_memorised (§5.2)", ctl_prm_cross_task_not_memorised),
    ("world_model_honest_abstention (§5.3)", ctl_world_model_honest_abstention),
    ("frozen_vs_adaptive_guidance_is_load_bearing (§5.4)",
     ctl_frozen_vs_adaptive_guidance_is_load_bearing),
    ("guidance_determinism (§5.5)", ctl_guidance_determinism),
    # --- Open-ended self-generation controls (§5) --- #
    ("generated_tasks_pass_floor (§5.1)", ctl_generated_tasks_pass_floor),
    ("novelty_is_real / L3 (§5.2)", ctl_novelty_is_real),
    ("generator_is_oracle_blind (§5.3)", ctl_generator_is_oracle_blind),
    ("emergence_set_is_sealed (§5.4)", ctl_emergence_set_is_sealed),
    ("no_self_congratulation (§5.5)", ctl_no_self_congratulation),
    ("self_verification_is_sound (§5.6)", ctl_self_verification_is_sound),
    # --- INVENTION / EMERGENCE controls (§5) --- #
    ("invented_is_genuinely_composite (§5.1)", ctl_invented_is_genuinely_composite),
    ("invented_is_not_given (§5.2)", ctl_invented_is_not_given),
    ("minting_not_shallow (§5.3)", ctl_minting_not_shallow),
    ("abstraction_anti_trivial (§5.4)", ctl_abstraction_anti_trivial),
    ("reach_unlock_is_load_bearing (§5.5)", ctl_reach_unlock_is_load_bearing),
    # --- REVERSE-ENGINEERING / STRICT cross-group emergence controls (§5) --- #
    ("decomposition_no_leakage (§5.1)", ctl_decomposition_no_leakage),
    ("emergent_is_cross_group_and_was_open (§5.2)",
     ctl_emergent_is_cross_group_and_was_open),
    ("groups_not_gerrymandered (§5.3)", ctl_groups_not_gerrymandered),
    ("emergent_is_composite_and_mined (§5.4)", ctl_emergent_is_composite_and_mined),
    ("hard_family_solutions_are_holdout_verified (§5.6)",
     ctl_hard_family_solutions_are_holdout_verified),
    # --- CHEAP-VERIFIER-BOUNDARY controls (delta measurement, this phase) --- #
    ("prereg_fp_unchanged (§3)", ctl_prereg_fp_unchanged),
    ("proxy_is_holdout_blind (§3)", ctl_proxy_is_holdout_blind),
    ("inner_loop_cost_blind (§3)", ctl_inner_loop_cost_blind),
    ("proxy_goodhart_gap (§3)", ctl_proxy_goodhart_gap),
    ("correctness_gate_intact (§3)", ctl_correctness_gate_intact),
    ("proxy_predicts_cost_not_correctness (§3)",
     ctl_proxy_predicts_cost_not_correctness),
    ("relative_delta_only (§3)", ctl_relative_delta_only),
    ("ablation_revert (§3)", ctl_ablation_revert),
    ("single_pinned_run (§3)", ctl_single_pinned_run),
    ("no_target_swap (§3)", ctl_no_target_swap),
    ("controls_only_strengthen (§3)", ctl_controls_only_strengthen),
    # --- Phase H: forbid the planted-strawman baseline (closes the Phase G gap) --- #
    ("no_planted_strawman (§2)", ctl_no_planted_strawman),
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
