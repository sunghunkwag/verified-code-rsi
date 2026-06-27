#!/usr/bin/env python3
"""UNLOCK B -- the reverse-engineering / backward-decomposition solver channel.

When the forward portfolio (OE + memetic + PRM-beam) cannot crack a hard target,
this engine works BACKWARD: it hypothesises a structural DECOMPOSITION of the
target's behaviour into sub-functions that are individually in reach, solves the
sub-pieces, and composes them -- exactly how a human writes an interpreter
(tokenise -> classify -> evaluate). The discovered sub-functions become library
abstractions and are the candidates for the strict emergence test (§3).

THE KEY DISCIPLINE -- a skeleton with TYPED HOLES, filled from PUBLIC data only.
A skeleton fixes the combinator scaffold (a map, a running scan, a filter, a
two-stage pipe) and leaves a small typed HOLE (the per-element step/predicate, or
the second stage). ``propose_intermediate_io`` derives each hole's input/output
examples from the target's PUBLIC examples + the skeleton's shape ALONE -- e.g.
the per-step deltas of a running-accumulator output are the first-differences of
that output; the intermediate of a sort-then-process pipe is ``lsort`` of the
input. It NEVER reads the target's sealed reference or held-out battery.

ANTI-LEAKAGE (control ``decomposition_no_leakage``). This module receives a public
view (public examples only) and a ``verify`` CALLBACK (the sealed oracle's pass or
fail, an opaque black box). It imports neither the oracle module nor the task
module; it never names a sealed solution, a held-out battery, or the example
generator. The sealed solution defines ground truth for the FINAL holdout check
only (through ``verify``); decomposition is a SEARCH STRATEGY, not a hint pipe.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .ir import Block, Node, pp, PRIMS, COMB_RTYPE
from .interp import run
from .library import Policy, stateful_policy
from .search import synthesize, shrink, problem_from_public
from .search_oe import oe_solve


# --------------------------------------------------------------------------- #
# tiny typed IR builders (local, so this module imports no task DSL)            #
# --------------------------------------------------------------------------- #
def _b(op: str, *kids: Node) -> Node:
    rt = COMB_RTYPE.get(op) or (PRIMS[op][0] if op in PRIMS else "V")
    return Node(op, rt, tuple(kids))


def _arg(i: int, t: str) -> Node:
    return Node("arg", t, const=i)


def _lit(v: Any) -> Node:
    t = ("B" if isinstance(v, bool) else "I" if isinstance(v, int) else
         "S" if isinstance(v, str) else "L" if isinstance(v, list) else
         "P" if isinstance(v, tuple) else "V")
    return Node("lit", t, const=v)


def _it() -> Node:
    return Node("var", "V", const="it")


def _acc() -> Node:
    return Node("var", "V", const="acc")


def _param(i: int, t: str) -> Node:
    return Node("param", t, const=i)


def _subst_arg0(n: Node, repl: Node) -> Node:
    """Replace every ``arg(0)`` leaf with ``repl`` (turns a hole solved over its
    own arg0 into a body over the iterated ``it``, or into a block parameter)."""
    if n.op == "arg" and n.const == 0:
        return repl
    if not n.kids:
        return n
    return Node(n.op, n.rtype, tuple(_subst_arg0(k, repl) for k in n.kids), n.const)


# --------------------------------------------------------------------------- #
# value typing helpers (public-only)                                           #
# --------------------------------------------------------------------------- #
def _vt(v: Any) -> str:
    if isinstance(v, bool):
        return "B"
    if isinstance(v, int):
        return "I"
    if isinstance(v, str):
        return "S"
    if isinstance(v, tuple):
        return "P"
    if isinstance(v, list):
        return "L"
    return "V"


@dataclass(frozen=True)
class _SubView:
    """A minimal public-view for a sub-piece. Carries ONLY public data (the same
    surface the synthesizer reads from a real PublicView): name/family/spec/types/
    public examples. Defined here so this module imports NOTHING from oracle/tasks
    -- the no-leakage control proves decomposition cannot touch a sealed reference."""
    name: str
    family: int
    spec: str
    arg_types: Tuple[str, ...]
    out_type: str
    public_examples: Tuple


def _public_view(name: str, arg_types: Tuple[str, ...], out_type: str,
                 examples: List[Tuple[Tuple[Any, ...], Any]]):
    return _SubView(name, 1, "decomposition sub-piece", arg_types, out_type,
                    tuple(examples))


def _harvest_lits(view) -> Tuple:
    """Distinct atomic literals (single chars + small ints) appearing in the
    PUBLIC inputs -- legitimately available to the synthesizer (public data), and
    what a per-element classifier needs (e.g. the '(' a bracket step compares to).
    Capped so the grammar does not bloat."""
    chars: set = set()
    ints: set = set()

    def scan(v: Any) -> None:
        if isinstance(v, str):
            for c in v:
                chars.add(c)
        elif isinstance(v, bool):
            pass
        elif isinstance(v, int):
            if -9 <= v <= 9:
                ints.add(v)
        elif isinstance(v, tuple):
            for e in v:
                scan(e)
        elif isinstance(v, list):
            for e in v:
                scan(e)

    for args, _y in view.public_examples:
        for a in args:
            scan(a)
    lits = sorted(chars)[:8] + sorted(ints)[:6]
    return tuple(lits)


def _decomp_policy(view, blocks: List[Block]) -> Policy:
    """A stateful policy augmented with the public-input literals and a stronger
    prior on the conditional/comparison ops a per-element classifier needs."""
    pol = stateful_policy()
    pol.blocks = list(blocks)
    pol.block_prob = 0.4 if blocks else 0.0
    pol.example_lits = _harvest_lits(view)
    pol.weights = dict(pol.weights)
    pol.weights.update({"ifx": 4.0, "eqv": 4.0, "eqi": 3.0, "lit_example": 3.0,
                        "gt": 2.0, "lt": 2.0, "le": 2.0, "and": 1.6,
                        "lit_int": 1.6})
    return pol


# --------------------------------------------------------------------------- #
# hole solving (a small forward search on the sub-piece's derived I/O)          #
# --------------------------------------------------------------------------- #
def _solve_hole(name: str, arg_types: Tuple[str, ...], out_type: str,
                examples: List[Tuple[Tuple[Any, ...], Any]],
                lits: Tuple, blocks: List[Block], budget: int,
                weights: Optional[dict] = None) -> Optional[Node]:
    """Solve one sub-piece on its derived public examples (OE first, then the
    literal-augmented memetic search). Returns a program over the hole's args, or
    None. The examples here are DERIVED FROM PUBLIC DATA -- never the reference."""
    if not examples:
        return None
    v = _public_view(name, arg_types, out_type, examples)
    # channel A: bottom-up OE (fast for literal-free elementwise maps)
    p = oe_solve(v, blocks=blocks, max_size=11, eval_budget=max(8000, budget // 4))
    if p is not None and _hole_exact(p, examples, blocks):
        return shrink(p, examples, {b.name: b for b in blocks})
    # channel B: literal-augmented memetic (reaches conditional classifiers)
    pol = stateful_policy()
    pol.blocks = list(blocks)
    pol.block_prob = 0.35 if blocks else 0.0
    pol.example_lits = lits
    pol.weights = dict(pol.weights)
    pol.weights.update(weights or {"ifx": 4.0, "eqv": 4.0, "eqi": 3.0,
                                   "lit_example": 3.0, "gt": 2.0, "lt": 2.0,
                                   "le": 2.0, "lit_int": 1.6})
    p, _st = synthesize(v, pol, budget, seed=7)
    if p is not None and _hole_exact(p, examples, blocks):
        return shrink(p, examples, {b.name: b for b in blocks})
    return None


def _hole_exact(prog: Node, examples, blocks: List[Block]) -> bool:
    bm = {b.name: b for b in blocks}
    for args, exp in examples:
        r = run(prog, list(args), bm, max_steps=60_000)
        if not r.ok or r.value != exp:
            return False
    return True


def _lift_block(name: str, prog: Node, arg_type: str, out_type: str,
                round_idx: int) -> Block:
    """Lift a sub-piece program (over its own ``arg(0)``) into a one-param library
    block: ``arg(0) -> param(0)``. The block is the discovered abstraction."""
    body = _subst_arg0(prog, _param(0, arg_type))
    return Block(name=name, ptypes=(arg_type,), body=body, rtype=out_type,
                 created_round=round_idx, origin="decomposed")


# --------------------------------------------------------------------------- #
# source-list helpers (the per-element view of the target)                     #
# --------------------------------------------------------------------------- #
def _source_of(view) -> Optional[Tuple[Node, str, Callable[[Any], list]]]:
    """The list the target iterates over and how to obtain it from arg0:
       string arg0 -> schars(arg0), elements are single-char strings;
       list   arg0 -> arg0,         elements are the list's items.
    Returns (src_node, elem_type, getter) or None."""
    if not view.arg_types:
        return None
    t0 = view.arg_types[0]
    if t0 == "S":
        def get_s(a):
            return list(a) if isinstance(a, str) else []
        return _b("schars", _arg(0, "S")), "S", get_s
    if t0 == "L":
        def get_l(a):
            return list(a) if isinstance(a, list) else []
        et = "V"
        for args, _y in view.public_examples:
            lst = args[0]
            if isinstance(lst, list) and lst:
                et = _vt(lst[0])
                break
        return _arg(0, "L"), et, get_l
    return None


# =========================================================================== #
# THE DECOMPOSITION SKELETONS                                                  #
# --------------------------------------------------------------------------- #
# Each skeleton: given the PUBLIC view, propose zero or more HYPOTHESES. A
# hypothesis carries the derived hole examples + a ``build(holes)`` that assembles
# the final IR (and the list of mined sub-function blocks). propose_* reads ONLY
# public (input, output) pairs and structure-preserving transforms of them.
# =========================================================================== #
@dataclass
class Hypothesis:
    skeleton: str
    holes: Dict[str, Tuple[Tuple[str, ...], str, List]]   # name -> (argtypes,out,examples)
    build: Callable[[Dict[str, Node], int], Tuple[Node, List[Block]]]


def _skel_scan_step(view) -> List[Hypothesis]:
    """scan-with-step: output is a RUNNING ACCUMULATOR (numeric list, one value
    per source element). The per-step delta is the first-difference of the output;
    the step function maps an element to its delta. Compose as a pipe of a
    delta-map and a running-sum scan -- the shape that cracks ``bracket_depths``."""
    src = _source_of(view)
    if src is None or view.out_type != "L":
        return []
    src_node, elem_t, getter = src
    step_ex: List[Tuple[Tuple[Any, ...], Any]] = []
    for args, y in view.public_examples:
        elems = getter(args[0])
        if (not isinstance(y, list) or len(y) != len(elems) or not y
                or not all(isinstance(v, int) and not isinstance(v, bool) for v in y)):
            return []
        diffs = [y[0]] + [y[i] - y[i - 1] for i in range(1, len(y))]
        for e, d in zip(elems, diffs):
            step_ex.append(((e,), d))
    if not step_ex:
        return []

    def build(holes: Dict[str, Node], rnd: int):
        step = holes["step"]
        step_blk = _lift_block(f"Dstep{rnd}", step, elem_t, "I", rnd)
        # the running-sum scan stage as a standalone one-param block (L -> L)
        sum_blk = Block(f"Dscan{rnd}", ("L",),
                        _b("scan", _param(0, "L"), _lit(0),
                           _b("add", _acc(), _it())), "L", rnd, "decomposed")
        stage1 = _b("map", src_node, _subst_arg0(step, _it()))
        stage2 = _b("scan", _arg(0, "L"), _lit(0), _b("add", _acc(), _it()))
        return Node("pipe", "L", (stage1, stage2)), [step_blk, sum_blk]

    return [Hypothesis("scan_step", {"step": ((elem_t,), "I", step_ex)}, build)]


def _skel_map_step(view) -> List[Hypothesis]:
    """map-step: output is a list, same length as the source, each element a
    function of the corresponding source element (an elementwise transform whose
    per-element rule may need an input-derived literal)."""
    src = _source_of(view)
    if src is None or view.out_type != "L":
        return []
    src_node, elem_t, getter = src
    out_elem = "V"
    step_ex: List[Tuple[Tuple[Any, ...], Any]] = []
    for args, y in view.public_examples:
        elems = getter(args[0])
        if not isinstance(y, list) or len(y) != len(elems):
            return []
        for e, o in zip(elems, y):
            step_ex.append(((e,), o))
            out_elem = _vt(o)
    if not step_ex:
        return []

    def build(holes: Dict[str, Node], rnd: int):
        step = holes["step"]
        blk = _lift_block(f"Dmap{rnd}", step, elem_t, out_elem, rnd)
        body = _subst_arg0(step, _it())
        return _b("map", src_node, body), [blk]

    return [Hypothesis("map_step", {"step": ((elem_t,), out_elem, step_ex)}, build)]


def _skel_filter_step(view) -> List[Hypothesis]:
    """filter-step: output is a SUBSEQUENCE of the input list; the predicate keeps
    an element iff it is present in the output (derivable from public I/O)."""
    if not view.arg_types or view.arg_types[0] != "L" or view.out_type != "L":
        return []
    elem_t = "V"
    pred_ex: List[Tuple[Tuple[Any, ...], Any]] = []
    for args, y in view.public_examples:
        lst = args[0]
        if not isinstance(lst, list) or not isinstance(y, list):
            return []
        # subsequence check: y must be the kept elements of lst in order
        j = 0
        for e in lst:
            keep = j < len(y) and y[j] == e
            if keep:
                j += 1
            pred_ex.append(((e,), keep))
            if lst:
                elem_t = _vt(lst[0])
        if j != len(y):
            return []                       # output is not a subsequence -> N/A
    if not pred_ex or all(not k for _a, k in pred_ex) or all(k for _a, k in pred_ex):
        return []

    def build(holes: Dict[str, Node], rnd: int):
        pred = holes["pred"]
        blk = _lift_block(f"Dpred{rnd}", pred, elem_t, "B", rnd)
        return _b("filter", _arg(0, "L"), _subst_arg0(pred, _it())), [blk]

    return [Hypothesis("filter_step", {"pred": ((elem_t,), "B", pred_ex)}, build)]


# preprocess transforms whose intermediate is DERIVABLE from the input alone
_PREPROCESS = [
    ("sort", lambda x: sorted(x, key=_sortkey) if isinstance(x, list) else None,
     lambda: _b("lsort", _arg(0, "L"))),
    ("rev", lambda x: list(reversed(x)) if isinstance(x, list) else None,
     lambda: _b("lrev", _arg(0, "L"))),
    ("sortrev",
     lambda x: list(reversed(sorted(x, key=_sortkey))) if isinstance(x, list) else None,
     lambda: _b("lrev", _b("lsort", _arg(0, "L")))),
]


def _sortkey(x):
    from .ir import _sortkey as sk
    return sk(x)


def _skel_preprocess(view) -> List[Hypothesis]:
    """g . f pipe (incl. split-process-merge): apply a derivable preprocessing f
    (sort / reverse / sort+reverse) to arg0, then solve the SECOND stage g on the
    transformed input -> output pairs with the forward portfolio. The intermediate
    is f(public_input) -- read off the public input, never the reference."""
    if not view.arg_types or view.arg_types[0] != "L":
        return []
    out: List[Hypothesis] = []
    for fname, ftrans, fnode in _PREPROCESS:
        g_ex: List[Tuple[Tuple[Any, ...], Any]] = []
        ok = True
        for args, y in view.public_examples:
            inter = ftrans(args[0])
            if inter is None:
                ok = False
                break
            rest = tuple(args[1:])
            g_ex.append(((inter,) + rest, y))
        if not ok or not g_ex:
            continue
        argtypes = view.arg_types

        def build(holes: Dict[str, Node], rnd: int, _fnode=fnode, _fname=fname):
            g = holes["g"]
            blk = _lift_block(f"Dpre_{_fname}{rnd}", g, "L", view.out_type, rnd) \
                if len(argtypes) == 1 else None
            stage1 = _fnode()
            composed = Node("pipe", view.out_type, (stage1, g))
            return composed, ([blk] if blk is not None else [])

        out.append(Hypothesis("preprocess:" + fname,
                              {"g": (view.arg_types, view.out_type, g_ex)}, build))
    return out


def _skel_map_fold(view) -> List[Hypothesis]:
    """map-then-fold: output is a SCALAR obtained by folding a per-element mapped
    value with a small fixed reducer (sum / max / min / last). The map step is the
    hole; the reducer is enumerated. Covers 'aggregate a derived quantity'."""
    src = _source_of(view)
    if src is None or view.out_type not in ("I", "V"):
        return []
    src_node, elem_t, getter = src
    out: List[Hypothesis] = []
    reducers = [
        ("sum", _lit(0), lambda: _b("add", _acc(), _it()), sum),
        ("max", None, lambda: _b("imax", _acc(), _it()), max),
        ("last", None, lambda: _it(), lambda xs: xs[-1]),
    ]
    for rname, init, body_fn, pyred in reducers:
        step_ex: List[Tuple[Tuple[Any, ...], Any]] = []
        ok = True
        for args, y in view.public_examples:
            elems = getter(args[0])
            if not elems or not isinstance(y, int) or isinstance(y, bool):
                ok = False
                break
            # we cannot invert the reducer per-element; instead require the map to
            # be IDENTITY-recoverable only when the reducer over raw elements works.
            # So this skeleton proposes step=identity-derived (elem -> y contribution
            # unknown); we approximate by trying the map hole on (elem, elem) which
            # the forward search can specialise. Hole IO is left to the map_step
            # search over the FULL task instead -> only emit when elements are ints.
            if not all(isinstance(e, int) and not isinstance(e, bool) for e in elems):
                ok = False
                break
            step_ex.append((tuple([args[0]]), y))
        if not ok:
            continue
        # map_fold is only attempted as foldl(arg0, init, reducer) over int lists
        # (a genuinely-stateful aggregate); the map hole defaults to identity.
        def build(holes: Dict[str, Node], rnd: int, _body_fn=body_fn, _init=init,
                  _rname=rname):
            init_node = _init if _init is not None else _b("head", _arg(0, "L"))
            prog = _b("foldl", _arg(0, "L"), init_node, _body_fn())
            return prog, []
        out.append(Hypothesis("map_fold:" + rname, {}, build))
    return out


DECOMP_SKELETONS: List[Callable[[Any], List[Hypothesis]]] = [
    _skel_scan_step, _skel_map_step, _skel_filter_step, _skel_preprocess,
    _skel_map_fold,
]


def propose_intermediate_io(view, skeleton) -> List[Hypothesis]:
    """Public entry: the hypotheses a skeleton derives from the target's PUBLIC
    examples + the skeleton's shape (never the reference / held-out battery)."""
    return skeleton(view)


# =========================================================================== #
# THE BACKWARD-DECOMPOSITION SOLVER (the fourth portfolio channel)             #
# =========================================================================== #
@dataclass
class DecompResult:
    program: Optional[Node]
    mined: List[Block] = field(default_factory=list)
    skeleton: str = ""
    channel: str = ""                 # 'forward' | 'decomposition' | ''
    best_partial: float = 0.0


def _public_exact(prog: Node, view, blocks: List[Block]) -> bool:
    bm = {b.name: b for b in blocks}
    for args, exp in view.public_examples:
        r = run(prog, list(args), bm, max_steps=120_000)
        if not r.ok or r.value != exp:
            return False
    return True


def solve_by_decomposition(view, verify: Callable[[Node], bool],
                           library: Optional[List[Block]] = None,
                           budget: int = 120_000,
                           round_idx: int = 0,
                           forward_first: bool = True) -> DecompResult:
    """Solve ``view`` by reverse-engineering. (1) optionally try the forward
    portfolio; (2) else try each structural skeleton: derive hole I/O from PUBLIC
    examples, solve the sub-pieces, compose, and accept ONLY if the composite
    passes the full public train AND the sealed holdout (``verify``). Returns the
    solving program + the mined sub-function blocks, or a best-partial.

    ``verify`` is the sole holdout channel and is used ONLY as the final gate."""
    library = list(library or [])

    # (1) forward first (cheap; OE cannot build conditionals so it will NOT pre-empt
    #     a genuine decomposition of a classifier-bearing target).
    if forward_first:
        fb = max(8000, int(budget * 0.4))
        p = oe_solve(view, blocks=library, max_size=12,
                     eval_budget=min(fb, 60_000))
        if p is not None and _public_exact(p, view, library) and verify(p):
            return DecompResult(p, [], "forward-oe", "forward", 1.0)
        pol = _decomp_policy(view, library)
        p, _st = synthesize(view, pol, fb, seed=7)
        if p is not None and _public_exact(p, view, library) and verify(p):
            return DecompResult(p, [], "forward-memetic", "forward", 1.0)

    # (2) backward: each skeleton, each hypothesis. Equal hole budget per piece.
    lits = _harvest_lits(view)
    hole_budget = max(12_000, int(budget * 0.25))
    for skeleton in DECOMP_SKELETONS:
        for hyp in propose_intermediate_io(view, skeleton):
            solved: Dict[str, Node] = {}
            ok = True
            for hname, (argtypes, out_t, examples) in hyp.holes.items():
                hp = _solve_hole(f"{hyp.skeleton}:{hname}", argtypes, out_t,
                                 examples, lits, library, hole_budget)
                if hp is None:
                    ok = False
                    break
                solved[hname] = hp
            if not ok:
                continue
            cand, mined = hyp.build(solved, round_idx)
            if _public_exact(cand, view, library + mined) and verify(cand):
                return DecompResult(cand, mined, hyp.skeleton, "decomposition", 1.0)
    return DecompResult(None, [], "", "", 0.0)
