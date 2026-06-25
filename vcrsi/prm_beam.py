#!/usr/bin/env python3
"""M2 -- PRM-guided beam search over program PREFIXES (``prm_beam_synthesize``).

A third solver channel (alongside the memetic search and the bottom-up OE solver)
that builds programs incrementally and keeps the top-``width`` type-valid prefixes
ranked by the learned process-reward model (``prm.PRM``).

Prefix representation (bottom-up postfix). A prefix is a token sequence whose
left-fold yields a STACK of completed sub-expressions plus at most one open
combinator FRAME. A token is one of:

  * a leaf  -> push a 0-arg sub (arg, int const, [], "", pair, False; and -- only
              while a frame is open -- the bound vars it/acc and their accessors);
  * a prim  -> pop arity(op) subs, push op(...);  (library ``call`` blocks too)
  * open    -> pop a list src (and, for foldl, an init), push a Frame; the bound
              variables it/acc become legal leaves ONLY inside the frame -- this
              is what makes stateful foldl/map BODIES reachable at all;
  * close   -> pop the single body sub, emit the combinator, pop the Frame.

The static type/shape gate lives in ``_expand``: ill-typed tokens are never
generated, so the beam never spends a run on garbage, and the gate is identical
for the frozen and adaptive arms (the only difference is RANKING).

Running a partial program (the crux of the features). ``candidate_programs``
closes every open obligation with the best IN-SCOPE bound variable of the needed
type (an identity fold uses ``acc``; a per-element int hole grabs ``snd(it)``),
falling back to a typed literal only when no scoped var fits. This scope-aware
completion -- not a zero stub -- is what gives a monotone feature gradient: a
half-built correct body completes near the target, a wrong one does not. For
foldl bodies an additional trace-sampled probe (real mid-fold (it,acc) states)
keeps a state-threading body observable instead of crashing at an empty acc.

Determinism. No RNG anywhere; the beam sort key (-score, #tokens, canonical
digest) is a strict total order, so the same PRM gives byte-identical beams.

Sealing. This module imports only ir / interp / prm / search-type-inference; it
references neither the oracle, the held-out battery, nor task/family metadata
(the ``prm_is_oracle_free`` control inspects this source).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

from .ir import Block, Node, PRIMS, pp, tmatch
from .interp import run
from .prm import (PRM, CRASH, classify, features_from_outputs, near_of, vtype)
from .search import problem_from_public, elem_type, _scope_types

INT_CONSTS = (0, 1, 2, -1)
MAX_TOKENS = 48          # hard depth cap (the length feature normalises by this)
K_TRAIN = 4              # number of train pairs the features look at
_PROBE_MAX_STEPS = 20_000


# --------------------------------------------------------------------------- #
# Prefix state                                                                  #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Sub:
    node: Node
    rtype: str
    freevars: FrozenSet[str]          # subset of {'it','acc'} referenced, unbound


@dataclass(frozen=True)
class Frame:
    comb: str                          # 'map' | 'filter' | 'foldl'
    src: Sub
    init: Optional[Sub]
    it_t: str
    acc_t: str
    base: int                          # stack height when the frame opened

    @property
    def body_rtype(self) -> str:
        return "B" if self.comb == "filter" else "V"

    def binds(self) -> FrozenSet[str]:
        return frozenset({"it", "acc"}) if self.comb == "foldl" else frozenset({"it"})


@dataclass(frozen=True)
class Prefix:
    stack: Tuple[Sub, ...]
    frames: Tuple[Frame, ...]
    ntok: int
    used_block: bool

    def complete(self) -> bool:
        return (len(self.stack) == 1 and not self.frames
                and not self.stack[0].freevars)

    def region_base(self) -> int:
        return self.frames[-1].base if self.frames else 0

    def digest(self) -> str:
        parts = [pp(s.node) for s in self.stack]
        parts.append("F:" + "|".join(f"{f.comb}:{pp(f.src.node)}" for f in self.frames))
        return hashlib.sha256(";".join(parts).encode()).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# Search context (public-only)                                                  #
# --------------------------------------------------------------------------- #
@dataclass
class BeamCtx:
    arg_types: Tuple[str, ...]
    out_type: str
    arg_elem: Tuple[str, ...]
    arg_is_list: Tuple[bool, ...]
    comps: Tuple[str, str]
    blocks: List[Block]
    block_map: Dict[str, Block]
    train: List[Tuple[Tuple[Any, ...], Any]]


def make_ctx(view, blocks: Optional[List[Block]] = None) -> BeamCtx:
    prob = problem_from_public(view)
    blocks = list(blocks or [])
    return BeamCtx(
        arg_types=prob.arg_types, out_type=prob.out_type,
        arg_elem=prob.arg_elem,
        arg_is_list=tuple(t == "L" for t in prob.arg_types),
        comps=prob.pair_comps, blocks=blocks,
        block_map={b.name: b for b in blocks},
        train=[(tuple(a), y) for a, y in view.public_examples[:K_TRAIN]])


# --------------------------------------------------------------------------- #
# Scope: which bound-var leaves are legal, and their types                       #
# --------------------------------------------------------------------------- #
def _scope_leaves(prefix: Prefix, ctx: BeamCtx) -> List[Sub]:
    """it / acc accessor leaves legal at the current point (a frame must be open;
    nesting is capped at 1 so the innermost frame fully determines scope)."""
    if not prefix.frames:
        return []
    fr = prefix.frames[-1]
    out: List[Sub] = []
    it = Node("var", fr.it_t, const="it")
    out.append(Sub(it, fr.it_t, frozenset({"it"})))
    if fr.it_t == "P":
        out.append(Sub(Node("fst", ctx.comps[0], (it,)), ctx.comps[0], frozenset({"it"})))
        out.append(Sub(Node("snd", ctx.comps[1], (it,)), ctx.comps[1], frozenset({"it"})))
    if fr.comb == "foldl":
        ac = Node("var", fr.acc_t, const="acc")
        out.append(Sub(ac, fr.acc_t, frozenset({"acc"})))
        if fr.acc_t == "P":
            out.append(Sub(Node("fst", "V", (ac,)), "V", frozenset({"acc"})))
            out.append(Sub(Node("snd", "V", (ac,)), "V", frozenset({"acc"})))
    return out


def _base_leaves(ctx: BeamCtx) -> List[Sub]:
    out: List[Sub] = []
    for c in INT_CONSTS:
        out.append(Sub(Node("lit", "I", const=c), "I", frozenset()))
    out.append(Sub(Node("lit", "L", const=[]), "L", frozenset()))
    out.append(Sub(Node("lit", "S", const=""), "S", frozenset()))
    out.append(Sub(Node("lit", "B", const=False), "B", frozenset()))
    out.append(Sub(Node("lit", "P", const=(0, 0)), "P", frozenset()))
    for i, t in enumerate(ctx.arg_types):
        out.append(Sub(Node("arg", t, const=i), t, frozenset()))
    return out


# --------------------------------------------------------------------------- #
# Expansion: every type-valid next token (the static gate)                       #
# --------------------------------------------------------------------------- #
def _avail(prefix: Prefix) -> Tuple[Sub, ...]:
    """The sub-expressions consumable by the next op (those above the innermost
    frame's base -- ops may not reach below into the frame's outer operands)."""
    return prefix.stack[prefix.region_base():]


def _push(prefix: Prefix, sub: Sub, is_block: bool = False) -> Prefix:
    return Prefix(prefix.stack + (sub,), prefix.frames, prefix.ntok + 1,
                  prefix.used_block or is_block)


def _consume(prefix: Prefix, k: int, new: Sub, is_block: bool = False) -> Prefix:
    stack = prefix.stack[:len(prefix.stack) - k] + (new,)
    return Prefix(stack, prefix.frames, prefix.ntok + 1,
                  prefix.used_block or is_block)


def _expand(prefix: Prefix, ctx: BeamCtx) -> List[Prefix]:
    if prefix.ntok >= MAX_TOKENS:
        return []
    out: List[Prefix] = []
    region = _avail(prefix)
    bound = prefix.frames[-1].binds() if prefix.frames else frozenset()

    # --- leaves (base, then scoped) --------------------------------------- #
    for s in _base_leaves(ctx):
        out.append(_push(prefix, s))
    for s in _scope_leaves(prefix, ctx):
        out.append(_push(prefix, s))

    # --- primitive ops (fixed sorted order) ------------------------------- #
    for name in sorted(PRIMS):
        rt, ats, _fn = PRIMS[name]
        k = len(ats)
        if len(region) < k:
            continue
        args = region[len(region) - k:]
        if all(tmatch(a.rtype, at) for a, at in zip(args, ats)):
            node = Node(name, rt, tuple(a.node for a in args))
            fv = frozenset().union(*(a.freevars for a in args)) if args else frozenset()
            out.append(_consume(prefix, k, Sub(node, rt, fv)))

    # --- library block calls (prim-like; sets has_reused_block) ----------- #
    for b in ctx.blocks:
        k = len(b.ptypes)
        if len(region) < k:
            continue
        args = region[len(region) - k:]
        if all(tmatch(a.rtype, pt) for a, pt in zip(args, b.ptypes)):
            node = Node("call", b.rtype, tuple(a.node for a in args), b.name)
            fv = frozenset().union(*(a.freevars for a in args)) if args else frozenset()
            out.append(_consume(prefix, k, Sub(node, b.rtype, fv), is_block=True))

    # --- open a combinator frame (nesting capped at 1) -------------------- #
    if not prefix.frames:
        if region and tmatch(region[-1].rtype, "L") and not region[-1].freevars:
            src = region[-1]
            it_t = elem_type(src.node, {}, ctx.arg_elem, ctx.arg_is_list)
            base = len(prefix.stack) - 1
            for comb in ("map", "filter"):
                fr = Frame(comb, src, None, it_t, "", base)
                out.append(Prefix(prefix.stack[:-1], (fr,), prefix.ntok + 1,
                                  prefix.used_block))
        if (len(region) >= 2 and tmatch(region[-2].rtype, "L")
                and not region[-2].freevars and not region[-1].freevars):
            src, init = region[-2], region[-1]
            it_t = elem_type(src.node, {}, ctx.arg_elem, ctx.arg_is_list)
            base = len(prefix.stack) - 2
            fr = Frame("foldl", src, init, it_t, init.rtype, base)
            out.append(Prefix(prefix.stack[:-2], (fr,), prefix.ntok + 1,
                              prefix.used_block))

    # --- close the open frame --------------------------------------------- #
    if prefix.frames:
        fr = prefix.frames[-1]
        if len(region) == 1:
            body = region[0]
            if tmatch(body.rtype, fr.body_rtype) and body.freevars <= fr.binds():
                comb_node, comb_rt = _build_comb(fr, body)
                fv = ((fr.src.freevars | (fr.init.freevars if fr.init else frozenset())
                       | body.freevars) - fr.binds())
                stack = prefix.stack[:fr.base] + (Sub(comb_node, comb_rt, fv),)
                out.append(Prefix(stack, (), prefix.ntok + 1, prefix.used_block))
    return out


def _build_comb(fr: Frame, body: Sub) -> Tuple[Node, str]:
    if fr.comb == "map":
        return Node("map", "L", (fr.src.node, body.node)), "L"
    if fr.comb == "filter":
        return Node("filter", "L", (fr.src.node, body.node)), "L"
    return (Node("foldl", fr.acc_t, (fr.src.node, fr.init.node, body.node)),
            fr.acc_t)


# --------------------------------------------------------------------------- #
# candidate_programs: turn a partial prefix into runnable closed programs        #
# --------------------------------------------------------------------------- #
_LIT = {"I": Node("lit", "I", const=0), "B": Node("lit", "B", const=False),
        "S": Node("lit", "S", const=""), "L": Node("lit", "L", const=[]),
        "P": Node("lit", "P", const=(0, 0))}

# A single canonical one-step coercion bridging a sub's type to a wanted type, so
# a near-complete prefix completes to the REAL target instead of a dead default
# (the list-of-strings one `sconcat` from the answer must read as near-exact, not
# as a type mismatch). Deterministic; oracle-free; one op only.
_COERCE = {("L", "S"): "sconcat", ("S", "L"): "schars",
           ("L", "I"): "llen", ("S", "I"): "slen"}


def _coerce(node: Node, have: str, want: str) -> Optional[Node]:
    if tmatch(have, want):
        return node
    op = _COERCE.get((have, want))
    if op is not None:
        return Node(op, want, (node,))
    return None


def _scope_fillers(want: str, fr: Frame, comps: Tuple[str, str]) -> List[Node]:
    """A small FAN of in-scope completions for a hole of type ``want`` inside a
    frame (so a just-opened combinator is scored by its BEST plausible body, not
    one arbitrary default that may crash). Bounded and deterministic."""
    out: List[Node] = []
    it = Node("var", fr.it_t, const="it")
    if fr.it_t == "P":
        if tmatch(comps[0], want):
            out.append(Node("fst", comps[0], (it,)))
        if tmatch(comps[1], want):
            out.append(Node("snd", comps[1], (it,)))
    if tmatch(fr.it_t, want):
        out.append(it)
    if fr.comb == "foldl":
        ac = Node("var", fr.acc_t, const="acc")
        if tmatch(fr.acc_t, want):
            out.append(ac)
    out.append(_LIT.get(want, _LIT["I"]))
    return out


def _combine_lookahead(subs: Tuple[Sub, ...], want: str) -> List[Node]:
    """A bounded ONE-OP lookahead: nodes of type ``want`` formed by applying a
    single type-valid primitive to the top one/two region subs. This credits a
    bottom-up intermediate (two operands on the stack) for being one op away from
    a good value -- without it, building any multi-argument op looks like a
    regression and the beam stalls before it can combine its operands."""
    out: List[Node] = []
    for name in sorted(PRIMS):
        rt, ats, _fn = PRIMS[name]
        k = len(ats)
        if k == 0 or k > len(subs) or k > 2:
            continue
        args = subs[len(subs) - k:]
        if all(tmatch(a.rtype, at) for a, at in zip(args, ats)):
            node = Node(name, rt, tuple(a.node for a in args))
            c = _coerce(node, rt, want)
            if c is not None:
                out.append(c)
    return out


def _complete_region(subs: Tuple[Sub, ...], want: str, fr: Optional[Frame],
                     comps: Tuple[str, str]) -> List[Node]:
    """Candidate closed nodes of type ``want`` for a (possibly empty) region:
    the top sub coerced, a one-op lookahead over the top subs, and (for an empty
    body region) the scope-aware fan of fillers."""
    out: List[Node] = []
    if subs:
        top = subs[-1]
        c = _coerce(top.node, top.rtype, want)
        if c is not None:
            out.append(c)
        out += _combine_lookahead(subs, want)
    elif fr is not None:
        out += _scope_fillers(want, fr, comps)
    if not out:
        out.append(_LIT.get(want, _LIT["I"]))
    # dedupe (stable); NO alphabetical truncation -- prefix_features screens these
    # by real output proximity so a useful late-alphabet op (srepeat) is not lost.
    seen, uniq = set(), []
    for n in out:
        key = pp(n)
        if key not in seen:
            seen.add(key)
            uniq.append(n)
    return uniq


def candidate_programs(prefix: Prefix, ctx: BeamCtx) -> List[Node]:
    """Up to a handful of closed, runnable programs approximating the prefix: the
    body hole (if a frame is open) is filled by its scope-aware fan, and the
    result coerced to the output type. The features take the BEST over these."""
    progs: List[Node] = []
    if prefix.frames:
        fr = prefix.frames[-1]
        region = prefix.stack[fr.base:]
        for body in _complete_region(region, fr.body_rtype, fr, ctx.comps):
            comb_node, comb_rt = _build_comb(fr, Sub(body, fr.body_rtype, frozenset()))
            stack = prefix.stack[:fr.base] + (Sub(comb_node, comb_rt, frozenset()),)
            progs.append(_complete_region(stack, ctx.out_type, None, ctx.comps)[0])
    else:
        progs = _complete_region(prefix.stack, ctx.out_type, None, ctx.comps)
    return progs or [_LIT.get(ctx.out_type, _LIT["I"])]


def _innermost_body(prefix: Prefix, ctx: BeamCtx) -> Optional[Tuple[Node, Frame]]:
    if not prefix.frames:
        return None
    fr = prefix.frames[-1]
    region = prefix.stack[fr.base:]
    body = _complete_region(region, fr.body_rtype, fr, ctx.comps)
    return body, fr


# --------------------------------------------------------------------------- #
# Trace-sampled probe envs (keep a state-threading foldl body observable)        #
# --------------------------------------------------------------------------- #
def _probe_states(fr: Frame, xs: Tuple[Any, ...], bm: Dict[str, Block]
                  ) -> List[Tuple[Any, Any]]:
    """Run the frame's source on this input, then thread an append-accumulator to
    recover realistic mid-fold (it, acc) states (the wave-0 trajectory)."""
    rs = run(fr.src.node, list(xs), bm, max_steps=_PROBE_MAX_STEPS)
    if not rs.ok or not isinstance(rs.value, list) or not rs.value:
        return []
    elems = rs.value
    ri = run(fr.init.node, list(xs), bm, max_steps=_PROBE_MAX_STEPS) if fr.init else None
    acc = ri.value if (ri and ri.ok) else []
    states: List[Tuple[Any, Any]] = []
    for e in elems:
        states.append((e, acc))
        if isinstance(acc, list):
            acc = acc + [e]
    n = len(states)
    idxs = sorted({0, n // 2, n - 1})
    return [states[i] for i in idxs]


def _foldl_probe(prefix: Prefix, ctx: BeamCtx, bm: Dict[str, Block]
                 ) -> Optional[Tuple[float, float]]:
    """Return (crash_rate, typed_rate) of the partial foldl body on real
    mid-fold states -- a state-consistency signal that survives an empty acc.
    typed = body returns a value of the accumulator type (state-preserving)."""
    info = _innermost_body(prefix, ctx)
    if info is None or info[1].comb != "foldl":
        return None
    body, fr = info
    runs = crash = typed = 0
    for xs, _y in ctx.train:
        for (it_v, acc_v) in _probe_states(fr, xs, bm):
            r = run(body, list(xs), bm, max_steps=_PROBE_MAX_STEPS,
                    env={"it": it_v, "acc": acc_v})
            runs += 1
            if not r.ok:
                crash += 1
            elif vtype(r.value) == fr.acc_t:
                typed += 1
    if runs == 0:
        return None
    return crash / runs, typed / runs


# --------------------------------------------------------------------------- #
# prefix_features: featurise a partial program by its REAL behaviour            #
# --------------------------------------------------------------------------- #
_SCREEN_KEEP = 5         # completions kept after the cheap single-input screen


def _run_dist(prog: Node, xs: Tuple[Any, ...], y: Any, bm: Dict[str, Block]):
    from .prm import value_distance
    r = run(prog, list(xs), bm, max_steps=40_000)
    if not r.ok:
        return float("inf"), CRASH
    return value_distance(r.value, y), r.value


def prefix_features(prefix: Prefix, ctx: BeamCtx) -> List[float]:
    bm = ctx.block_map
    progs = candidate_programs(prefix, ctx)
    # cheap screen: rank completions by proximity on the FIRST train pair, keep a
    # few, then take the best per-input over those (so a useful op anywhere in the
    # fan is found without paying a full K-input run for every completion).
    xs0, y0 = ctx.train[0]
    screened = sorted(((_run_dist(p, xs0, y0, bm)[0], i, p)
                       for i, p in enumerate(progs)), key=lambda t: (t[0], t[1]))
    keep = [t[2] for t in screened[:_SCREEN_KEEP]]
    outputs: List[Any] = []
    targets: List[Any] = []
    for xs, y in ctx.train:
        best_d, best_v = float("inf"), CRASH
        for prog in keep:
            d, v = _run_dist(prog, xs, y, bm)
            if d < best_d:
                best_d, best_v = d, v
        outputs.append(best_v)
        targets.append(y)
    length_ratio = prefix.ntok / MAX_TOKENS
    feats = features_from_outputs(outputs, targets, length_ratio, prefix.used_block)
    # foldl-body supplement: a body that threads state cleanly on REAL mid-fold
    # states should not be punished for the whole-program completion crashing on
    # an empty initial acc.
    probe = _foldl_probe(prefix, ctx, bm)
    if probe is not None:
        crash_rate, typed_rate = probe
        feats[4] = min(feats[4], crash_rate)            # crash
        feats[1] = max(feats[1], typed_rate)            # typed
    return feats


# --------------------------------------------------------------------------- #
# The deterministic PRM-guided beam                                            #
# --------------------------------------------------------------------------- #
@dataclass
class BeamStats:
    layers: int
    solved: bool
    best_score: float
    pos_feats: List[List[float]]          # +1 training prefixes (solution path)
    neg_feats: List[List[float]]          # -1 training prefixes (dead ends)
    best_partial: float = 0.0             # best exact/n seen on an OPEN task


def _exact_train(prog: Node, ctx: BeamCtx) -> bool:
    for xs, y in ctx.train:
        r = run(prog, list(xs), ctx.block_map, max_steps=120_000)
        if not r.ok or r.value != y:
            return False
    return True


def prm_beam_synthesize(view, prm: PRM, blocks: Optional[List[Block]] = None, *,
                        width: int = 8, max_layers: int = 28,
                        verify=None, collect: bool = True
                        ) -> Tuple[Optional[Node], BeamStats]:
    """Build programs token-by-token, ranking type-valid prefixes by ``prm``.
    A full candidate is admitted only when it is train-exact AND passes the sealed
    holdout (``verify``) -- identical to every other channel's gate. Returns the
    solving program (or None) and stats incl. PRM training prefixes."""
    ctx = make_ctx(view, blocks)
    start = Prefix((), (), 0, False)
    beam: List[Prefix] = [start]
    parent: Dict[str, Optional[str]] = {_key(start): None}
    by_key: Dict[str, Prefix] = {_key(start): start}
    feat_cache: Dict[str, List[float]] = {}
    pos: List[List[float]] = []
    neg: List[List[float]] = []
    best_score = 0.0
    best_exact = 0.0
    solution: Optional[Node] = None

    def feats(p: Prefix) -> List[float]:
        k = _key(p)
        if k not in feat_cache:
            feat_cache[k] = prefix_features(p, ctx)
        return feat_cache[k]

    for layer in range(max_layers):
        scored: List[Tuple[float, int, str, Prefix]] = []
        for p in beam:
            for c in _expand(p, ctx):
                k = _key(c)
                if k in by_key:
                    continue
                by_key[k] = c
                parent[k] = _key(p)
                f = feats(c)
                best_exact = max(best_exact, f[0])
                s = prm.score(f)
                scored.append((s, c.ntok, k, c))
        if not scored:
            break
        scored.sort(key=lambda t: (-t[0], t[1], t[2]))
        best_score = max(best_score, scored[0][0])
        # short-circuit: admit the first complete, train-exact, holdout-passing
        win_key: Optional[str] = None
        for s, _n, k, c in scored:
            if c.complete():
                prog = c.stack[0].node
                if _exact_train(prog, ctx) and (verify is None or verify(prog)):
                    solution, win_key = prog, k
                    break
        if solution is not None:
            if collect:
                pos = _collect_path(parent, feat_cache, win_key)
            break
        beam = [t[3] for t in scored[:width]]
        if collect:
            for t in scored[width:width * 3]:
                neg.append(feat_cache[t[2]])

    return solution, BeamStats(layer + 1, solution is not None, best_score,
                               pos, neg, best_exact)


def _key(p: Prefix) -> str:
    return p.digest() + f"#{p.ntok}"


# --------------------------------------------------------------------------- #
# Replaying a known solution into the +1 / -1 prefix episodes that train the PRM #
# --------------------------------------------------------------------------- #
def _leaf_sub(node: Node, fr: Optional[Frame], ctx: BeamCtx) -> Sub:
    if node.op == "var":
        if node.const == "acc" and fr is not None:
            return Sub(Node("var", fr.acc_t, const="acc"), fr.acc_t, frozenset({"acc"}))
        it_t = fr.it_t if fr is not None else "V"
        return Sub(Node("var", it_t, const="it"), it_t, frozenset({"it"}))
    if node.op == "arg":
        return Sub(node, ctx.arg_types[node.const], frozenset())
    return Sub(node, node.rtype, frozenset())


def replay_path(prog: Node, ctx: BeamCtx) -> Optional[List[Prefix]]:
    """Reconstruct the postfix derivation of a known solution as the ordered list
    of prefixes the beam would pass through to build it (the +1 episode). Returns
    None if the program uses a shape the beam representation cannot express
    (e.g. nested combinator frames -- capped at depth 1)."""
    states: List[Prefix] = []

    def emit(node: Node, st: Prefix) -> Prefix:
        op = node.op
        fr = st.frames[-1] if st.frames else None
        if op in ("lit", "arg", "var"):
            st2 = Prefix(st.stack + (_leaf_sub(node, fr, ctx),), st.frames,
                         st.ntok + 1, st.used_block)
            states.append(st2)
            return st2
        if op in ("map", "filter", "foldl"):
            if st.frames:
                raise _Unsupported()                      # nested frame > depth 1
            st = emit(node.kids[0], st)                   # source
            if op == "foldl":
                st = emit(node.kids[1], st)               # init
            src = st.stack[-2 if op == "foldl" else -1]
            it_t = elem_type(src.node, {}, ctx.arg_elem, ctx.arg_is_list)
            npop = 2 if op == "foldl" else 1
            init = st.stack[-1] if op == "foldl" else None
            fr2 = Frame(op, src, init, it_t,
                        init.rtype if init else "", len(st.stack) - npop)
            st = Prefix(st.stack[:len(st.stack) - npop], (fr2,), st.ntok + 1,
                        st.used_block)
            states.append(st)
            body_idx = 2 if op == "foldl" else 1
            st = emit(node.kids[body_idx], st)            # body (it/acc in scope)
            body = st.stack[-1]
            comb_node, comb_rt = _build_comb(fr2, body)
            stack = st.stack[:fr2.base] + (Sub(comb_node, comb_rt, frozenset()),)
            st = Prefix(stack, (), st.ntok + 1, st.used_block)
            states.append(st)
            return st
        # prim or library call: emit kids, then consume
        for k in node.kids:
            st = emit(k, st)
        nk = len(node.kids)
        args = st.stack[len(st.stack) - nk:] if nk else ()
        fv = frozenset().union(*(a.freevars for a in args)) if args else frozenset()
        rt = node.rtype if node.op == "call" else PRIMS.get(op, (node.rtype,))[0]
        new = Sub(Node(op, rt, tuple(a.node for a in args), node.const), rt, fv)
        st2 = Prefix(st.stack[:len(st.stack) - nk] + (new,), st.frames,
                     st.ntok + 1, st.used_block or op == "call")
        states.append(st2)
        return st2

    try:
        final = emit(prog, Prefix((), (), 0, False))
    except _Unsupported:
        return None
    if not final.complete():
        return None
    return states


class _Unsupported(Exception):
    pass


def _clean_label(f: List[float]) -> Optional[int]:
    """Classify a prefix into a CLEANLY-separable training example, or None for
    the ambiguous middle (which carries no learnable per-prefix signal):
      +1  the best completion reaches / closely approaches the target
      -1  the completion crashes or collapses to a scalar (a dead end)
    Training only on the separable extremes is what lets the averaged perceptron
    recover the right direction (exact/near up, crash/single down)."""
    if f[0] >= 0.5 or f[2] >= 0.45:                   # exact or strongly near
        return +1
    if f[4] >= 0.5 or f[3] >= 0.5:                    # crashes or collapses
        return -1
    return None


def train_prm_on_solution(prm: PRM, prog: Node, view,
                          blocks: Optional[List[Block]] = None,
                          neg_per_step: int = 5) -> bool:
    """Train the PRM on ONE solved program. The averaged perceptron sees only the
    cleanly-separable prefixes (``_clean_label``): the solution path's
    exact/near-reaching prefixes as +1, and crashing/collapsing siblings as -1.
    The ambiguous early prefixes (an in-progress body that has not yet reached the
    target) are skipped -- they are featurally indistinguishable from dead ends
    and would only inject noise."""
    ctx = make_ctx(view, blocks)
    path = replay_path(prog, ctx)
    if path is None:
        return False
    path_keys = {_key(p) for p in path}
    n = 0
    prev = Prefix((), (), 0, False)
    for p in path:
        pf = prefix_features(p, ctx)
        lab = _clean_label(pf)
        if lab is not None:
            prm.update(pf, lab)
            n += 1
        sibs = [c for c in _expand(prev, ctx) if _key(c) not in path_keys]
        sibs.sort(key=lambda c: c.digest())
        taken = 0
        for c in sibs:
            cf = prefix_features(c, ctx)
            if _clean_label(cf) == -1:                # clean dead ends only
                prm.update(cf, -1)
                n += 1
                taken += 1
                if taken >= neg_per_step:
                    break
        prev = p
    return n > 0


def _collect_path(parent, feat_cache, key) -> List[List[float]]:
    """The +1 derivation path: feature vectors of every prefix from the root to
    the admitted solution (in build order)."""
    feats: List[List[float]] = []
    seen = set()
    while key is not None and key not in seen:
        seen.add(key)
        if key in feat_cache:
            feats.append(feat_cache[key])
        key = parent.get(key)
    feats.reverse()
    return feats
