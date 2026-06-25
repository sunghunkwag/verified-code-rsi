#!/usr/bin/env python3
"""Self-generated task synthesis (the open-ended generator) -- ORACLE-BLIND.

To invent a task the system synthesises a NEW reference program in the IR; that
program's input->output behaviour DEFINES the task's ground truth (§1). This
module produces those reference programs and packages them as ``GenSpec`` objects
(Task-shaped: name/family/spec/types/reference/input-generator), which the loop
then wraps in the EXISTING sealed oracle to build a held-out battery. The solver
must rediscover a behaviourally-equivalent program from PUBLIC examples only.

Oracle-blindness (control ``generator_is_oracle_blind``): this module imports
``ir`` / ``interp`` only. It never imports the oracle, the fixed suite, or the
external emergence set; it never reads any *other* task's sealed reference or
held-out battery. It emits reference programs and their own input generators --
nothing else. (A GenSpec carries its OWN freshly-synthesised reference; that is
self-verification, not leakage -- the generator created it.)

References are deliberately BLOCK-FREE pure IR: the sealed oracle runs a reference
standalone (no library in scope), so a ``call`` node would crash it. Library
blocks still influence generation -- a block may be INLINED (its body grafted,
params substituted) as a structural building block, which keeps the reference
runnable while genuinely reusing discovered structure (§1 "composition over the
IR + existing library blocks").

The whole behavioural space is large and the families never touch flat-integer
arrays, so the triple lock (whitelist / §6B floor / self-easiness) in
``openended.py`` is what decides which generated tasks actually count.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Set, Tuple

from .ir import Node, PRIMS, COMB_RTYPE, inline

# Ops that build a NEW container (mirrors complexity._CONSTRUCTORS); used by the
# cheap floor pre-check so generate_spec rarely emits a floor-failing reference.
_CONSTRUCTORS = {"cons", "lapp", "lsingle", "lrange", "pair", "sconcat",
                 "srepeat", "schars", "map", "filter", "scan", "ltake", "ldrop"}
_LOOPS = {"map", "filter", "foldl", "scan", "iterate"}
_PLUMBING = {"lit", "arg", "var", "param"}


# --------------------------------------------------------------------------- #
# IR builder helpers (local; the generator does not borrow tasks.py's DSL)      #
# --------------------------------------------------------------------------- #
def _lit(v: Any) -> Node:
    if isinstance(v, bool):
        t = "B"
    elif isinstance(v, int):
        t = "I"
    elif isinstance(v, str):
        t = "S"
    elif isinstance(v, list):
        t = "L"
    elif isinstance(v, tuple):
        t = "P"
    else:
        t = "V"
    return Node("lit", t, const=v)


def _arg(i: int, t: str) -> Node:
    return Node("arg", t, const=i)


def _it() -> Node:
    return Node("var", "V", const="it")


def _acc() -> Node:
    return Node("var", "V", const="acc")


def _b(op: str, *kids: Node) -> Node:
    if op in COMB_RTYPE:
        rt = COMB_RTYPE[op]
    elif op in PRIMS:
        rt = PRIMS[op][0]
    else:
        rt = "V"
    return Node(op, rt, tuple(kids))


# --------------------------------------------------------------------------- #
# A Task-shaped specification (carries only its OWN reference + input generator) #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GenSpec:
    name: str
    family: int                       # §6A whitelist family id (1..5)
    spec: str
    arg_types: Tuple[str, ...]
    out_type: str
    reference: Node                   # freshly synthesised, block-free, SEALED
    gen_input: Callable[[random.Random, int], Tuple[Any, ...]]
    public_scale: int = 4
    holdout_scale: int = 10
    n_public: int = 6
    n_holdout: int = 20
    group: str = ""
    note: str = ""
    roundtrip_with: Optional[str] = None   # present for Task-shape compatibility


# --------------------------------------------------------------------------- #
# Input generators (structured inputs only -- never a flat integer array)        #
# --------------------------------------------------------------------------- #
_ALPHA = "abcde"


def _g_iv(rng: random.Random, scale: int) -> Tuple[Any, ...]:
    k = rng.randint(max(2, scale - 1), scale + 2)
    ivs = []
    for _ in range(k):
        a = rng.randint(-6, 18)
        ivs.append((a, a + rng.randint(0, 7)))
    return (ivs,)


def _g_ivk(rng: random.Random, scale: int) -> Tuple[Any, ...]:
    return (_g_iv(rng, scale)[0], rng.randint(1, 4))


def _g_ivk2(rng: random.Random, scale: int) -> Tuple[Any, ...]:
    lo = rng.randint(-4, 6)
    return (_g_iv(rng, scale)[0], lo, lo + rng.randint(2, 10))


def _g_pairs(rng: random.Random, scale: int) -> Tuple[Any, ...]:
    k = rng.randint(max(2, scale - 1), scale + 2)
    return ([(rng.choice(_ALPHA), rng.randint(1, 4)) for _ in range(k)],)


def _g_str_shift(rng: random.Random, scale: int) -> Tuple[Any, ...]:
    k = rng.randint(max(3, scale), scale + 3)
    return ("".join(rng.choice(_ALPHA) for _ in range(k)), rng.randint(1, 5))


def _g_brackets(rng: random.Random, scale: int) -> Tuple[Any, ...]:
    k = rng.randint(max(4, scale), scale + 4)
    return ("".join(rng.choice("()") for _ in range(k)),)


# --------------------------------------------------------------------------- #
# Random typed expression synthesis (the source of behavioural novelty)         #
# --------------------------------------------------------------------------- #
_IOPS = ("add", "sub", "mul", "imax", "imin")
_EASY_IOPS = ("add", "sub")
_COPS = ("gt", "lt", "le")

# Difficulty controls the synthesised reference's arithmetic depth + op palette,
# giving the loop a SPREAD: easy refs are solvable at the probe (-> rejected by
# L3), hard refs need the full attack or stay OPEN (the honest frontier).
def _ops_for(hard: bool):
    return (_IOPS, 2) if hard else (_EASY_IOPS, 1)


def _rand_int(rng: random.Random, leaves: List[Node], depth: int,
              ops=_IOPS) -> Node:
    pool = list(leaves) + [_lit(c) for c in (0, 1, 2, -1)]
    if depth <= 0 or rng.random() < 0.45:
        return rng.choice(pool)
    op = rng.choice(ops)
    return _b(op, _rand_int(rng, leaves, depth - 1, ops),
              _rand_int(rng, leaves, depth - 1, ops))


# --------------------------------------------------------------------------- #
# Optional library-block grafting (kept runnable by INLINING)                   #
# --------------------------------------------------------------------------- #
def _graft_pre(rng: random.Random, blocks, a0: Node) -> Node:
    """Use an L->L library block (inlined over ``a0``) as a structural pre-pass.
    Inlining substitutes the block's param with ``a0`` and removes the call, so
    the reference stays block-free and runnable. Falls back to ``a0``."""
    cands = [bk for bk in (blocks or [])
             if bk.rtype == "L" and bk.ptypes == ("L",)]
    if cands and rng.random() < 0.5:
        bk = rng.choice(cands)
        try:
            return inline(Node("call", "L", (a0,), bk.name), {bk.name: bk})
        except Exception:
            return a0
    return a0


# --------------------------------------------------------------------------- #
# Generation families (each over STRUCTURED input; each clears §6B by design)    #
# --------------------------------------------------------------------------- #
def _f_interval(rng, blocks, hard):
    iops, depth = _ops_for(hard)
    knob = rng.random() < 0.5
    arg_types = ("L", "I") if knob else ("L",)
    a0 = _graft_pre(rng, blocks, _arg(0, "L")) if hard else _arg(0, "L")
    pre = rng.choice([a0, _b("lrev", a0), _b("lsort", a0)]) if hard else a0
    f, s = _b("fst", _it()), _b("snd", _it())
    leaves = [f, s, _b("sub", s, f)] + ([_arg(1, "I")] if knob else [])
    lo = _b(rng.choice(iops), f, _rand_int(rng, leaves, depth - 1, iops))
    hi = _b(rng.choice(iops), s, _rand_int(rng, leaves, depth - 1, iops))
    ref = _b("map", pre, _b("pair", lo, hi))
    return (4, "interval", "map each interval to a derived interval",
            arg_types, "L", _g_ivk if knob else _g_iv, 4, ref)


def _f_project(rng, blocks, hard):
    iops, depth = _ops_for(hard)
    knob = rng.random() < 0.6
    arg_types = ("L", "I") if knob else ("L",)
    f, s = _b("fst", _it()), _b("snd", _it())
    width = _b("sub", s, f)
    leaves = [f, s, width] + ([_arg(1, "I")] if knob else [])
    body = _b(rng.choice(("add", "mul", "imax", "imin") if hard else ("add", "mul")),
              width, _rand_int(rng, leaves, depth - 1, iops))
    ref = _b("map", _arg(0, "L"), body)
    return (4, "project", "map each interval to a derived scalar",
            arg_types, "L", _g_ivk if knob else _g_iv, 4, ref)


def _f_select(rng, blocks, hard):
    two = rng.random() < (0.5 if hard else 0.3)
    arg_types = ("L", "I", "I") if two else ("L", "I")
    f, s = _b("fst", _it()), _b("snd", _it())
    width = _b("sub", s, f)
    leaves = [f, s, width, _arg(1, "I")] + ([_arg(2, "I")] if two else [])
    if two and rng.random() < 0.7:
        pred = _b("and", _b("le", _arg(1, "I"), f), _b("le", s, _arg(2, "I")))
    else:
        pred = _b(rng.choice(_COPS), width if not hard else rng.choice([width, f, s]),
                  _arg(1, "I") if not hard else rng.choice(leaves))
    ref = _b("filter", _arg(0, "L"), pred)
    return (3, "select", "keep intervals matching a predicate",
            arg_types, "L", _g_ivk2 if two else _g_ivk, 4, ref)


def _f_codec(rng, blocks, hard):
    op = rng.choice(("add", "sub"))
    inner = _b(op, _b("sord", _it()), _arg(1, "I"))
    if hard and rng.random() < 0.6:
        inner = _b(rng.choice(("inc", "dec")), inner)
    ref = _b("sconcat", _b("map", _b("schars", _arg(0, "S")), _b("schr", inner)))
    return (2, "codec", "per-character codepoint shift codec",
            ("S", "I"), "S", _g_str_shift, 5, ref)


def _f_seqcode(rng, blocks, hard):
    a0 = _graft_pre(rng, blocks, _arg(0, "L")) if hard else _arg(0, "L")
    pre = rng.choice([a0, _b("lrev", a0), _b("lsort", a0)]) if hard else a0
    expand = _b("map", pre, _b("srepeat", _b("fst", _it()), _b("snd", _it())))
    mode = rng.choice(("plain", "twice", "palindrome", "shift")) if hard \
        else rng.choice(("plain", "twice"))
    if mode == "plain":
        ref = _b("sconcat", expand)
    elif mode == "twice":
        ref = _b("sconcat", _b("lapp", expand, expand))
    elif mode == "palindrome":
        ref = _b("sconcat", _b("lapp", expand, _b("lrev", expand)))
    else:
        dec = _b("sconcat", expand)
        ref = _b("sconcat", _b("map", _b("schars", dec),
                              _b("schr", _b(rng.choice(("inc", "dec")),
                                            _b("sord", _it())))))
    return (2, "seqcode", "run-length decode variant",
            ("L",), "S", _g_pairs, 4, ref)


def _f_scan(rng, blocks, hard):
    """A running-accumulator scan over interval widths, written with the Unlock-A
    ``scan`` primitive: scan(a0, 0, COMBINE(acc, width)). The OE channel cannot
    reach a state-threading body, so these stay on the stateful frontier -- but the
    memetic + scan-enabled beam CAN solve them, so they enter the curriculum (the
    previous foldl/bracket variants were effectively unsolvable, which is partly
    why the stateful families never entered reach)."""
    # running MAXIMUM of (interval width + 1), init 0. imax over the non-negative
    # (width+1) keeps the task non-degenerate AND within the search's reach (a
    # running-min from 0 would collapse to a constant, and a prefix-SUM landscape
    # is beyond the stochastic search). The +1 shift makes this behaviourally
    # DISTINCT from every sealed external held-out task (no minting-to-memorise),
    # while still USING the width sub-program so the loop mines the width atom and
    # encapsulates the scan-on-width capability that seeds the stateful lineage.
    width = _b("sub", _b("snd", _it()), _b("fst", _it()))
    ref = _b("scan", _arg(0, "L"), _lit(0),
             _b("imax", _acc(), _b("add", width, _lit(1))))
    return (4, "scan", "running maximum of (interval width + 1) (scan accumulator)",
            ("L",), "L", _g_iv, 4, ref)


FAMILIES = [_f_interval, _f_project, _f_select, _f_codec, _f_seqcode, _f_scan]
FAMILY_NAMES = ["interval", "project", "select", "codec", "seqcode", "scan"]


# =========================================================================== #
# M3 -- NON-SHALLOW MINTING (compose VERIFIED solved references)               #
# --------------------------------------------------------------------------- #
# The previous emergence run measured zero partly because minting was SHALLOW:  #
# wrapping a solved function's OUTPUT with square/inc is post-composition -- a   #
# behaviourally-near variant in the SAME family that re-covers known territory.  #
# Genuine novelty changes the COMPUTATIONAL STRUCTURE. Every operator here       #
# introduces a STATEFUL accumulator (scan) that the source program did not have: #
#   scanify   wrap a solved map's per-element body in a running accumulator      #
#   chain     thread a solved function's whole OUTPUT through a running scan      #
# A minted task therefore needs structure no source task taught elementwise, and #
# the source's per-element body is exactly the sub-program a mined block         #
# captures -- so the composite is reachable WITH that block and deep without it.  #
# (The shallow inc/square post-wrap is provided only so the minting_not_shallow  #
# control can demonstrate it is REJECTED -- it introduces no accumulator.)        #
# =========================================================================== #
def _node_ops(n: Node) -> Set[str]:
    acc: Set[str] = set()
    _ops(n, acc)
    return acc


_STATEFUL = {"scan", "foldl", "iterate"}


_INT_HEAD = ("add", "sub", "mul", "imax", "imin", "sdiv", "smod", "inc",
             "dec", "llen", "slen", "sord")


def _int_typed_map_body(ref: Node) -> Optional[Tuple[Node, Node]]:
    """Find the first map(src, body) SUBTREE (anywhere in ``ref``) whose body
    returns an int -- the 'project' shape we scanify. Searching subtrees (not just
    the root) makes minting robust to the harmless wrappers a search adds, e.g.
    lrev(lrev(map(...)))."""
    def walk(n: Node) -> Optional[Tuple[Node, Node]]:
        if n.op == "map" and len(n.kids) == 2 and n.kids[1].op in _INT_HEAD:
            return n.kids[0], n.kids[1]
        for k in n.kids:
            r = walk(k)
            if r is not None:
                return r
        return None
    return walk(ref)


def mint_scanify(ref: Node, arg_types, out_type):
    """Wrap a solved map's per-element body in a running accumulator (map -> scan).
    Yields (new_ref, new_out_type, tag). Introduces a scan the source lacked."""
    mb = _int_typed_map_body(ref)
    if mb is None:
        return
    src, body = mb
    for combine in ("add", "imax"):
        new = _b("scan", src, _lit(0), _b(combine, _acc(), body))
        yield new, "L", f"scanify:{combine}"


def mint_chain(ref: Node, arg_types, out_type):
    """Thread a solved function's whole OUTPUT through a running scan (chain two
    computations through an accumulator). ``ref`` must produce a list of ints."""
    if out_type != "L":
        return
    mb = _int_typed_map_body(ref)
    if mb is None:                      # only chain list-of-int producers
        return
    for combine in ("add", "imax"):
        new = _b("scan", ref, _lit(0), _b(combine, _acc(), _it()))
        yield new, "L", f"chain:{combine}"


def _count_stateful(n: Node) -> int:
    c = 1 if n.op in _STATEFUL else 0
    return c + sum(_count_stateful(k) for k in n.kids)


def mint_loop_twice(ref: Node, arg_types, out_type):
    """Given a solved program that ALREADY contains a stateful loop and produces a
    list, emit it FOLLOWED BY its reverse (lapp(ref, lrev(ref))). The stateful sub-
    computation now appears TWICE, so the flat program is beyond search reach, but
    a block that captures the loop (encapsulated from the SAME solution) makes it
    small -- a guaranteed-matching depth-2 reach target. This is the stateful
    analogue of rle_rev_palindrome_twice, built from the system's own scan solves."""
    if out_type != "L" or not (_node_ops(ref) & _STATEFUL):
        return
    yield _b("lapp", ref, _b("lrev", ref)), "L", "loop_twice"


def mint_shallow(ref: Node, arg_types, out_type):
    """A SHALLOW post-wrapper (increment every output element). Provided ONLY for
    the minting_not_shallow control to assert it is rejected: it introduces NO
    accumulator -- the computational structure is unchanged (still an elementwise
    map), so ``introduces_accumulator`` is False."""
    if out_type != "L" or _int_typed_map_body(ref) is None:
        return
    yield _b("map", ref, _b("inc", _it())), "L", "shallow:inc"


def mint_scan_twice(ref: Node, arg_types, out_type):
    """Scanify a solved map, then emit the running-aggregate list FOLLOWED BY its
    reverse (a 'twice/palindrome' shape). The inner scan therefore appears TWICE,
    so the flat program is large and beyond the memetic/beam reach -- but a block
    that captures the scan-running-aggregate (itself built on the source's per-
    element body) makes it small. This is the stateful analogue of the proven
    rle_rev_palindrome_twice depth-2 lineage: the deep gap that makes a composite
    abstraction LOAD-BEARING and reach-unlocking (§3(4))."""
    mb = _int_typed_map_body(ref)
    if mb is None:
        return
    src, body = mb
    for combine in ("add", "imax"):
        r = _b("scan", src, _lit(0), _b(combine, _acc(), body))
        yield _b("lapp", r, _b("lrev", r)), "L", f"scan_twice:{combine}"


# loop_twice (deep reach target from a solved scan) leads -- it is the robust,
# guaranteed-matching reach target; scanify/scan_twice add project-derived variants.
NON_SHALLOW_MINT_OPS = [mint_loop_twice, mint_scanify, mint_scan_twice, mint_chain]


def introduces_accumulator(src_ref: Node, minted_ref: Node) -> bool:
    """True iff ``minted_ref`` uses a stateful combinator (scan/foldl/iterate) that
    ``src_ref`` did not -- one of the two structural-change tests."""
    return bool((_node_ops(minted_ref) & _STATEFUL)
                - (_node_ops(src_ref) & _STATEFUL))


def nonshallow_change(src_ref: Node, minted_ref: Node) -> bool:
    """The non-shallow test (§M3): a minted task changes computational structure if
    it EITHER introduces a stateful accumulator the source lacked OR DUPLICATES a
    stateful sub-computation (the loop appears strictly more often than in the
    source). A plain arithmetic post-wrap does neither."""
    return (introduces_accumulator(src_ref, minted_ref)
            or _count_stateful(minted_ref) > _count_stateful(src_ref))


def behavioural_distance(ref_a: Node, ref_b: Node, gen_input, scale: int = 6,
                         trials: int = 10) -> float:
    """Fraction of shared probe inputs on which two references differ (both must
    run). 1.0 = totally different behaviour; ~0 = a near-variant."""
    from .interp import run
    rng = random.Random(4242)
    diff = ok = 0
    for _ in range(trials):
        args = list(gen_input(rng, scale))
        ra, rb = run(ref_a, args), run(ref_b, args)
        if not (ra.ok and rb.ok):
            continue
        ok += 1
        if ra.value != rb.value:
            diff += 1
    return (diff / ok) if ok else 0.0


def mint_curriculum(verified: List[dict], registry: Set[str], n: int,
                    blocks=None) -> List[GenSpec]:
    """Compose the system's OWN verified solved references into NEW, structurally-
    novel tasks (§M3). ``verified`` is a list of dicts {ref, arg_types, out_type,
    group, gen_input}. Each survivor changes computational structure (introduces an
    accumulator), is behaviourally distant from its source, is deduped by signature,
    and passes L1 whitelist + the §6B floor. L3 self-easiness is enforced later by
    the loop's triple lock. Returns up to ``n`` sealed GenSpecs."""
    out: List[GenSpec] = []
    vrng = random.Random(13)
    for v in verified:
        ref, gi = v["ref"], v["gen_input"]
        for op in NON_SHALLOW_MINT_OPS:
            for cand, new_out, tag in op(ref, v["arg_types"], v["out_type"]):
                if not nonshallow_change(ref, cand):
                    continue
                sig = _behav_sig(cand, gi)
                if sig is None or sig in registry:
                    continue
                if behavioural_distance(ref, cand, gi) < 0.5:
                    continue
                sp = GenSpec(
                    name=f"mint_{len(registry)}_{tag.replace(':','_')}",
                    family=4, spec=f"non-shallow composite ({tag})",
                    arg_types=v["arg_types"], out_type=new_out, reference=cand,
                    gen_input=gi, public_scale=4, holdout_scale=10,
                    group="scan", note=f"minted:{tag}:from={v['group']}")
                # cheap, oracle-FREE gates (the authoritative §6B floor / L1 / L3
                # run in the loop's triple_lock; the generator never imports the
                # oracle, keeping generator_is_oracle_blind intact).
                if not on_whitelist(sp) or not _floor_precheck(sp, vrng):
                    continue
                registry.add(sig)
                out.append(sp)
                if len(out) >= n:
                    return out
    return out


def _behav_sig(ref: Node, gen_input) -> Optional[str]:
    """A behavioural signature: outputs on a fixed probe battery (dedup key)."""
    from .interp import run
    rng = random.Random(20240131)
    parts = []
    for _ in range(6):
        r = run(ref, list(gen_input(rng, 6)))
        if not r.ok:
            return None
        parts.append(repr(r.value))
    return "|".join(parts)


# --------------------------------------------------------------------------- #
# Cheap structural checks (the authoritative §6B floor lives in complexity.py)   #
# --------------------------------------------------------------------------- #
def _ops(n: Node, acc: Set[str]) -> None:
    if n.op not in _PLUMBING:
        acc.add(n.op)
    for k in n.kids:
        _ops(k, acc)


def is_flat_int_scalar_reduction(sp: GenSpec) -> bool:
    """The permanently-banned shape: a single flat integer LIST reduced to a
    scalar. Detected structurally (out type is a scalar AND the one list arg
    holds plain ints). The generator never produces this; the check is a genuine
    guard, asserted against by the ``generated_tasks_pass_floor`` control."""
    if sp.out_type not in ("I", "B"):
        return False
    list_args = [i for i, t in enumerate(sp.arg_types) if t == "L"]
    if len(list_args) != 1:
        return False
    args = sp.gen_input(random.Random(1), sp.public_scale)
    lst = args[list_args[0]]
    if not isinstance(lst, list) or not lst:
        return False
    return all(isinstance(e, int) and not isinstance(e, bool) for e in lst)


def on_whitelist(sp: GenSpec) -> bool:
    """L1: §6A family, STRUCTURED input, and NOT a banned flat-int reduction."""
    structured = any(t in ("L", "S") for t in sp.arg_types)
    return (sp.family in (1, 2, 3, 4, 5) and structured
            and not is_flat_int_scalar_reduction(sp))


def _floor_precheck(sp: GenSpec, rng: random.Random) -> bool:
    """A fast surrogate for §6B so generate_spec mostly emits floor-clearing
    specs; runs the reference on samples and checks distinct-ops/loop/structure/
    exec-depth + output VARIETY (reject constant-output degenerates)."""
    ops: Set[str] = set()
    _ops(sp.reference, ops)
    if len(ops) < 5 or not (ops & _LOOPS) or not (ops & _CONSTRUCTORS):
        return False
    outs = []
    top = sp.public_scale + 4
    for scale in (top, sp.holdout_scale, sp.holdout_scale + 6):
        from .interp import run
        r = run(sp.reference, list(sp.gen_input(rng, scale)))
        if not r.ok:
            return False
        outs.append(repr(r.value))
    from .interp import run
    if run(sp.reference, list(sp.gen_input(rng, top))).iters < 6:
        return False
    return len(set(outs)) >= 2          # not a constant-output task


# --------------------------------------------------------------------------- #
# Top-level: synthesise one valid GenSpec                                       #
# --------------------------------------------------------------------------- #
def generate_spec(rng: random.Random, gen: int, idx: int, blocks=None,
                  family: Optional[Callable] = None) -> Optional[GenSpec]:
    """Synthesise one self-verifying task spec (L1-clean, floor-clearing, runs on
    its inputs). Returns None if no valid reference was found in the attempt
    budget (the caller treats this as 'no candidate this slot')."""
    vrng = random.Random(987 + gen * 131 + idx)
    for _ in range(60):
        fam = family or rng.choice(FAMILIES)
        hard = rng.random() < 0.5
        fam_id, group, spec, arg_types, out_type, gi, pscale, ref = \
            fam(rng, blocks, hard)
        sp = GenSpec(name=f"gen_g{gen}_{idx}", family=fam_id, spec=spec,
                     arg_types=arg_types, out_type=out_type, reference=ref,
                     gen_input=gi, public_scale=pscale, holdout_scale=pscale + 6,
                     group=group,
                     note=f"generated:{group}:{'hard' if hard else 'easy'}")
        if on_whitelist(sp) and _floor_precheck(sp, vrng):
            return sp
    return None
