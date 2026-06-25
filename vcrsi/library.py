#!/usr/bin/env python3
"""The evolvable policy genome: what the search behaviour depends on.

The synthesizer's behaviour is a pure function of (a) a weight vector over IR
operators / leaf kinds and (b) a library of mined subroutines (``Block``s), plus
the probability of emitting a library call. This whole object is DATA -- it can
be cloned, fingerprinted, A/B-tested, and serialized. Improving it is the only
way the system improves its own search (there is no other hidden state), so
"self-improvement" is literally "a better genome", measured against a frozen
copy of the same genome.

Two responsibilities live here:
  * ``Policy``            the genome itself (+ default prior, +fingerprint).
  * block mining          propose candidate subroutines from solved programs,
                          abstracting free variables / input args into params,
                          and preferring fragments that reference EARLIER blocks
                          (which is what produces a depth-2 lineage).

Nothing here reads tasks, references or held-out batteries.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .ir import Block, Node, pp, PRIMS, COMB_RTYPE, COMBINATORS, inline


# --------------------------------------------------------------------------- #
# The genome                                                                    #
# --------------------------------------------------------------------------- #
@dataclass
class Policy:
    weights: Dict[str, float] = field(default_factory=dict)
    blocks: List[Block] = field(default_factory=list)
    block_prob: float = 0.0
    version: int = 0

    def w(self, key: str) -> float:
        return self.weights.get(key, 1.0)

    def block_map(self) -> Dict[str, Block]:
        return {b.name: b for b in self.blocks}

    def clone(self) -> "Policy":
        return Policy(dict(self.weights), list(self.blocks), self.block_prob,
                     self.version)

    def fingerprint(self) -> str:
        h = hashlib.sha256()
        for k in sorted(self.weights):
            h.update(f"{k}={round(self.weights[k], 4)};".encode())
        for b in self.blocks:
            h.update(f"|{b.name}:{pp(b.body)}".encode())
        h.update(f"#bp={round(self.block_prob, 4)}".encode())
        return h.hexdigest()[:16]


def default_policy() -> Policy:
    """A generic, domain-general prior shared by BOTH arms at start. It is NOT
    tuned to any task -- it mildly favours the structural combinators and the
    common accessors that any list/string program needs, which is enough to
    bootstrap the very easiest tasks but not the harder ones. Everything beyond
    that must be LEARNED (and the frozen arm never learns it)."""
    w = {
        # structural combinators: any program over structured data needs these
        "map": 6.0, "foldl": 5.0, "filter": 1.4, "ifx": 1.4,
        # accessors / constructors common to list/string/pair transforms
        "fst": 6.0, "snd": 6.0, "head": 1.6, "tail": 1.4, "cons": 2.0,
        "lapp": 2.6, "lsingle": 2.6, "lrev": 2.4,
        # string processing ops
        "sconcat": 6.0, "srepeat": 6.0, "schars": 4.0, "schr": 4.0, "sord": 4.0,
        # arithmetic that shows up in codecs / counters
        "add": 2.6, "sub": 2.6, "inc": 1.4, "dec": 1.2,
        # leaves -- bare it/acc are kept LOW vs accessors
        "arg": 4.0, "it": 1.6, "acc": 1.2, "lit_int": 0.8,
        # down-weight rarely-useful generic ops so they don't flood the grammar
        "eqv": 0.4, "eqi": 0.4, "and": 0.4, "or": 0.4, "not": 0.4, "gt": 0.5,
        "lt": 0.5, "le": 0.5, "lit_bool": 0.3, "lit_nil": 0.5, "lit_estr": 0.5,
        "lit_pair": 0.3, "imax": 0.4, "imin": 0.4, "sdiv": 0.3, "smod": 0.3,
        "mul": 0.6, "lrange": 0.4, "ltake": 0.6, "ldrop": 0.6, "slen": 0.5,
        "llen": 0.5, "snth": 0.6, "nth": 0.6, "llast": 1.2, "linit": 1.2,
        "lsort": 0.8, "lempty": 0.6, "pair": 1.0,
    }
    return Policy(weights=w, blocks=[], block_prob=0.0, version=0)


def broad_policy() -> Policy:
    """A FAMILY-NEUTRAL prior used only by the Phase-B transfer experiment, where
    the suite spans string AND int-pair families. It balances integer/pair/
    comparison ops up to string ops so int-pair tasks are reachable. (The Phase-A
    default above stays string-biased so the RLE counterfactual is unaffected.)"""
    w = dict(default_policy().weights)
    w.update({"filter": 4.0, "pair": 4.0, "add": 4.0, "sub": 4.0, "mul": 2.6,
              "imax": 2.4, "imin": 2.0, "sdiv": 1.6, "gt": 2.4, "lt": 2.4,
              "le": 2.4, "and": 1.8, "or": 1.2, "eqi": 1.0})
    return Policy(weights=w, blocks=[], block_prob=0.0, version=0)


def stateful_policy() -> Policy:
    """broad_policy + a stronger prior on the stateful combinators and the int/pair
    accessors their bodies need (acc, imax/imin, sub/add). Used by the open-ended
    loop and solve-hard so the scan families are reliably within the memetic
    channel's reach; the global broad_policy (used by the transfer experiment) is
    left untouched so those measured results are unchanged."""
    w = dict(broad_policy().weights)
    w.update({"scan": 4.5, "iterate": 1.2, "acc": 2.4, "imax": 3.2, "imin": 2.8,
              "add": 4.0, "sub": 4.0, "mul": 3.2, "sdiv": 2.4, "llast": 1.8,
              "lempty": 1.2})
    return Policy(weights=w, blocks=[], block_prob=0.0, version=0)


# --------------------------------------------------------------------------- #
# Free-variable analysis (a var is free in T if its binder is outside T)        #
# --------------------------------------------------------------------------- #
def _binder_scope(op: str, i: int, bound: frozenset) -> frozenset:
    """Extend ``bound`` with the loop variables a combinator binds in child ``i``.
    map/filter bind ``it`` in the body (kid 1); foldl/scan/iterate bind ``it`` AND
    ``acc`` in the body (kid 2)."""
    if op in ("map", "filter") and i == 1:
        return bound | {"it"}
    if op in ("foldl", "scan", "iterate") and i == 2:
        return bound | {"it", "acc"}
    return bound


def free_vars(n: Node, bound: frozenset = frozenset()) -> set:
    if n.op == "var" and n.const not in bound:
        return {n.const}
    res: set = set()
    for i, k in enumerate(n.kids):
        res |= free_vars(k, _binder_scope(n.op, i, bound))
    return res


def _block_calls(n: Node) -> set:
    out = set()
    if n.op == "call":
        out.add(n.const)
    for k in n.kids:
        out |= _block_calls(k)
    return out


# --------------------------------------------------------------------------- #
# Mining: turn a solved program into candidate reusable blocks                  #
# --------------------------------------------------------------------------- #
def _abstractable_leaves(n: Node, bound: frozenset, acc: List[Node]) -> None:
    """Collect leaves that should become params: free vars (it/acc not bound
    within the fragment) and input-arg references."""
    if n.op == "arg":
        acc.append(n)
        return
    if n.op == "var" and n.const not in bound:
        acc.append(n)
        return
    for i, k in enumerate(n.kids):
        _abstractable_leaves(k, _binder_scope(n.op, i, bound), acc)


def _leaf_key(leaf: Node) -> Tuple:
    return (leaf.op, leaf.const, leaf.rtype)


def abstract_fragment(frag: Node) -> Optional[Tuple[Node, Tuple[str, ...], Tuple[Node, ...]]]:
    """Turn a fragment into (body_with_params, ptypes, call_args).

    Distinct abstractable leaves (free vars + arg refs) become params $0,$1,...
    in first-seen order; the call args are the original leaves. Returns None if
    there is nothing to abstract (a closed constant) or it is trivial."""
    leaves: List[Node] = []
    _abstractable_leaves(frag, frozenset(), leaves)
    if not leaves:
        return None
    # distinct leaves in first-seen order
    seen: Dict[Tuple, int] = {}
    order: List[Node] = []
    for lf in leaves:
        k = _leaf_key(lf)
        if k not in seen:
            seen[k] = len(order)
            order.append(lf)
    ptypes = tuple(lf.rtype for lf in order)

    def rebuild(n: Node, bound: frozenset) -> Node:
        if (n.op == "arg") or (n.op == "var" and n.const not in bound):
            idx = seen[_leaf_key(n)]
            return Node("param", n.rtype, const=idx)
        if not n.kids:
            return n
        kids = []
        for i, k in enumerate(n.kids):
            kids.append(rebuild(k, _binder_scope(n.op, i, bound)))
        return Node(n.op, n.rtype, tuple(kids), n.const)

    body = rebuild(frag, frozenset())
    return body, ptypes, tuple(order)


def _useful_fragments(prog: Node) -> List[Node]:
    """Candidate fragments to abstract: every subtree that is non-trivial in
    size and is a genuine operator application (not a bare leaf)."""
    out: List[Node] = []

    def walk(n: Node):
        if n.op not in ("lit", "arg", "var", "param") and n.size() >= 3:
            out.append(n)
        for k in n.kids:
            walk(k)
    walk(prog)
    return out


MAX_BLOCK_SIZE = 5   # keep blocks small so they stay reusable ATOMS; a block
# that captured a whole solution would be so powerful nothing built ON it could
# ever be NEEDED -- and the depth-2 lineage requires composed blocks to be needed.


def mine_blocks(prog: Node, existing: List[Block], round_idx: int,
                max_new: int = 4) -> List[Block]:
    """Propose new candidate blocks from one solved program.

    Each candidate abstracts a small useful fragment's free leaves into params.
    Fragments that reference EARLIER blocks are preferred (they create lineage:
    a new block whose body calls an older block). Smaller atoms are preferred so
    composition (and thus a need for higher-level blocks) actually arises."""
    have = {pp(b.body) for b in existing}
    have_names = {b.name for b in existing}
    cands: List[Tuple[Tuple[int, int], Block]] = []
    next_id = len(existing)
    for frag in _useful_fragments(prog):
        res = abstract_fragment(frag)
        if res is None:
            continue
        body, ptypes, _args = res
        calls = _block_calls(body)
        # flat atoms stay small; a fragment that CALLS an earlier block may be a
        # bit larger (it is a genuine higher-level composition -> lineage)
        size_cap = MAX_BLOCK_SIZE + 4 if calls else MAX_BLOCK_SIZE
        if body.size() > size_cap:
            continue
        key = pp(body)
        if key in have:
            continue
        lineage_bonus = 1 if calls else 0
        # prefer (1) fragments that call earlier blocks, then (2) SMALLER atoms
        score = (lineage_bonus, -body.size())
        rtype = frag.rtype if frag.rtype != "V" else _infer_rtype(frag)
        name = _fresh_name(next_id, have_names)
        next_id += 1
        have_names.add(name)
        blk = Block(name=name, ptypes=ptypes, body=body, rtype=rtype,
                    created_round=round_idx, origin="mined")
        cands.append((score, blk))
        have.add(key)
    cands.sort(key=lambda x: x[0], reverse=True)
    return [b for _, b in cands[:max_new]]


# --------------------------------------------------------------------------- #
# Encapsulation operator (port of SECTION 24's idea): rewrite a program to CALL  #
# existing blocks where their body-pattern occurs, then freeze a recurring       #
# block-containing pattern as a NEW block. A block frozen this way references the #
# earlier block in its body -> this is what produces a depth-2 lineage.          #
# --------------------------------------------------------------------------- #
def _node_eq(a: Node, b: Node) -> bool:
    return pp(a) == pp(b)


def _match(subtree: Node, body: Node, binding: Dict[int, Node]) -> bool:
    if body.op == "param":
        i = body.const
        if i in binding:
            return _node_eq(binding[i], subtree)
        binding[i] = subtree
        return True
    if (subtree.op != body.op or subtree.const != body.const
            or len(subtree.kids) != len(body.kids)):
        return False
    return all(_match(s, b, binding) for s, b in zip(subtree.kids, body.kids))


def _nparams(body: Node) -> int:
    mx = [-1]

    def w(n):
        if n.op == "param":
            mx[0] = max(mx[0], n.const)
        for k in n.kids:
            w(k)
    w(body)
    return mx[0] + 1


def re_encapsulate(node: Node, blocks: List[Block]) -> Node:
    """Bottom-up rewrite: wherever a block's body-pattern matches, replace the
    subtree with a call to that block. Larger blocks are tried first so the most
    compressive encapsulation wins."""
    node = Node(node.op, node.rtype,
                tuple(re_encapsulate(k, blocks) for k in node.kids), node.const)
    for blk in sorted(blocks, key=lambda b: -b.body.size()):
        binding: Dict[int, Node] = {}
        if _match(node, blk.body, binding):
            npar = _nparams(blk.body)
            if set(binding.keys()) == set(range(npar)):
                args = tuple(binding[i] for i in range(npar))
                return Node("call", blk.rtype, args, blk.name)
    return node


def propose_encapsulations(progs: List[Node], existing: List[Block],
                           round_idx: int, max_new: int = 3) -> List[Block]:
    """From already-encapsulated solutions, propose NEW blocks whose body CALLS
    an existing block (the higher-level compositions). These are the depth-2
    lineage candidates."""
    have = {pp(b.body) for b in existing}
    have_names = {b.name for b in existing}
    counts: Dict[str, Tuple[int, Block]] = {}
    next_id = len(existing)
    for prog in progs:
        for frag in _useful_fragments(prog):
            if not _block_calls(frag):
                continue
            res = abstract_fragment(frag)
            if res is None:
                continue
            body, ptypes, _args = res
            if not _block_calls(body):
                continue
            key = pp(body)
            if key in have:
                continue
            if key in counts:
                c, blk = counts[key]
                counts[key] = (c + 1, blk)
            else:
                rtype = frag.rtype if frag.rtype != "V" else _infer_rtype(frag)
                name = _fresh_name(next_id, have_names)
                next_id += 1
                have_names.add(name)
                counts[key] = (1, Block(name=name, ptypes=ptypes, body=body,
                                        rtype=rtype, created_round=round_idx,
                                        origin="encapsulated"))
    # prefer the largest composed pattern (most compression), then most reused
    ranked = sorted(counts.values(), key=lambda cb: (cb[1].body.size(), cb[0]),
                    reverse=True)
    return [blk for _c, blk in ranked[:max_new]]


def _infer_rtype(n: Node) -> str:
    if n.op in COMB_RTYPE:
        return COMB_RTYPE[n.op]
    if n.op in PRIMS:
        return PRIMS[n.op][0]
    return "V"


def _fresh_name(idx: int, taken: set) -> str:
    while True:
        nm = f"B{idx}"
        if nm not in taken:
            return nm
        idx += 1


# =========================================================================== #
# M1 -- ABSTRACTION-FIRST ENUMERATION ORDER                                    #
# --------------------------------------------------------------------------- #
# When the solver enumerates programs it should reach the NEWEST learned blocks #
# before the base primitives, so a block built from earlier blocks is reached    #
# as ONE unit and higher-order / nested structure accumulates instead of being   #
# re-derived from scratch. This helper returns that order; the OE and memetic     #
# channels consult it (see search_oe.grow / search._Gen).                         #
# =========================================================================== #
def build_enumeration_order(blocks: List[Block], base_ops: List[str]) -> List[str]:
    """Newest library blocks first (by created_round desc, then reverse insertion),
    then the base ops. Returns a flat list of NAMES (block names + op names)."""
    ranked = sorted(range(len(blocks)),
                    key=lambda i: (blocks[i].created_round, i), reverse=True)
    return [blocks[i].name for i in ranked] + list(base_ops)


# =========================================================================== #
# M2 -- MULTI-OBJECTIVE ABSTRACTION SCORE                                      #
# --------------------------------------------------------------------------- #
# Score a candidate abstraction on three terms (compression / transfer / anti-  #
# cheat). The open-ended loop admits a block to the library only on this score   #
# AND the empirical META-GATE (a held-out reach gain with no regression). The     #
# anti-cheat (input-coupled) term is load-bearing: it stops the system "inventing" #
# trivial constant macros to inflate compression. Family labels are used for       #
# MEASUREMENT only (curation), never inside the search.                            #
# =========================================================================== #
_PLUMBING = frozenset({"lit", "arg", "var", "param"})


def _op_nodes(n: Node) -> int:
    """Count of genuine operator nodes (a primitive or combinator application),
    excluding plumbing leaves and (already-inlined) calls."""
    c = 1 if (n.op not in _PLUMBING and n.op != "call") else 0
    return c + sum(_op_nodes(k) for k in n.kids)


def expand_block(block: Block, block_map: Dict[str, Block]) -> Node:
    """The block's body with every nested ``call`` inlined -> its real, irreducible
    computation (the same expansion the complexity floor measures over)."""
    try:
        return inline(block.body, block_map)
    except Exception:
        return block.body


def is_composite(block: Block, block_map: Optional[Dict[str, Block]] = None) -> bool:
    """§3(1): ``block`` is COMPOSITE iff it is NOT expressible as a single given IR
    primitive -- its inlined body nests >= 2 operator applications. A block whose
    body is one primitive over its params (e.g. ``add($0,$1)``) is NOT composite."""
    flat = expand_block(block, block_map or {})
    return _op_nodes(flat) >= 2


def _structural_size(n: Node) -> int:
    """Count of non-plumbing nodes in a body WITHOUT inlining -- a ``call`` counts
    as one node. Distinguishes a genuine composition from a trivial alias."""
    c = 0 if n.op in _PLUMBING else 1
    return c + sum(_structural_size(k) for k in n.kids)


def is_nontrivial_abstraction(block: Block) -> bool:
    """Reject trivial library entries: a block whose body is a single primitive/
    combinator or a bare alias of ONE other block (e.g. ``B5($0,$1)`` or
    ``lrev(B7($0,$1))``) adds no structure -- it only pollutes the enumeration
    order. A useful abstraction composes >= 2 structural nodes."""
    return _structural_size(block.body) >= 2 and block.body.op != "call"


def _leaf_kinds(n: Node, lits: List[int], params: List[int]) -> None:
    if n.op == "lit":
        lits.append(1)
    elif n.op in ("param", "arg", "var"):
        # param/arg consume the input argument; var (it/acc) consumes the iterated
        # element / threaded accumulator -- both are INPUT-derived, NOT constants.
        params.append(1)
    for k in n.kids:
        _leaf_kinds(k, lits, params)


def input_coupled(block: Block) -> float:
    """Fraction of the block body's LEAVES that consume an input (a param/arg, or
    the loop variables it/acc which thread the iterated input) rather than a pushed
    constant. A constant-pushing macro scores 0 -> its anti_cheat term is 0 -> it
    cannot be credited (control abstraction_anti_trivial)."""
    lits: List[int] = []
    params: List[int] = []
    _leaf_kinds(block.body, lits, params)
    tot = len(lits) + len(params)
    return (len(params) / tot) if tot else 0.0


def block_occurs_in(block: Block, prog: Node) -> bool:
    """True iff ``block``'s body-pattern matches SOME subtree of ``prog`` (the
    token-stream of the solution CONTAINS the block)."""
    def walk(n: Node) -> bool:
        binding: Dict[int, Node] = {}
        if _match(n, block.body, binding):
            npar = _nparams(block.body)
            if set(binding.keys()) <= set(range(npar)):
                return True
        return any(walk(k) for k in n.kids)
    return walk(prog)


def score_abstraction(block: Block, adopted: List[Tuple[Node, str]],
                      block_map: Optional[Dict[str, Block]] = None) -> dict:
    """M2 multi-objective score of one candidate abstraction. ``adopted`` is the
    list of (solution_program, family) pairs the loop has accumulated. Returns the
    component terms + the final score in [0,1]."""
    bm = block_map or {}
    if not adopted:
        return {"capture": 0.0, "diversity": 0.0, "short": 0.0, "compression": 0.0,
                "transfer": 0.0, "input_coupled": input_coupled(block),
                "anti_cheat": input_coupled(block), "score": 0.0, "n_fams": 0,
                "captured": 0}
    captured = [(p, fam) for p, fam in adopted if block_occurs_in(block, p)]
    capture = len(captured) / len(adopted)
    fams = {fam for _p, fam in captured}
    all_fams = {fam for _p, fam in adopted}
    diversity = min(1.0, len(fams) / 5.0)
    short = 1.0 / (1.0 + math.log1p(expand_block(block, bm).size()) / 10.0)
    compression = min(1.0, 0.65 * capture + 0.20 * diversity + 0.15 * short)
    transfer = (len(fams) / len(all_fams)) if all_fams else 0.0
    ic = input_coupled(block)
    anti_cheat = ic
    score = 0.5 * compression + 0.3 * transfer + 0.2 * anti_cheat
    return {"capture": capture, "diversity": diversity, "short": short,
            "compression": compression, "transfer": transfer,
            "input_coupled": ic, "anti_cheat": anti_cheat, "score": score,
            "n_fams": len(fams), "captured": len(captured)}
