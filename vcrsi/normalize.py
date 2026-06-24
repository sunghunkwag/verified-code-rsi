#!/usr/bin/env python3
"""M3 -- verified normalizer (canonicalise so blocks CAN transfer).

A block mined from family-A solutions is often entangled with incidental surface
structure that blocks reuse. The normalizer rewrites a block body into a
canonical form via semantics-preserving rules (constant folding; algebraic
identities such as lrev(lrev x)->x, lsort(lsort x)->lsort x, not(not x)->x,
add(x,0)->x; param re-indexing to first-occurrence order). It is NOT a relabel
no-op: it must change the tree (fold/simplify) to count.

Anti-gaming (control ``normalizer_preserves_semantics``): every normalisation is
ACCEPTED only after a kernel-equivalence check -- the normalized body must
produce identical outputs on a probe set of bindings, judged by re-execution
(the sealed kernel). A behaviour-changing rewrite is rejected (original kept).
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

from .ir import Block, Node, pp
from .interp import run


def _fold(n: Node) -> Node:
    kids = tuple(_fold(k) for k in n.kids)
    n = Node(n.op, n.rtype, kids, n.const)
    # algebraic identities (semantics-preserving)
    if n.op in ("lrev",) and kids and kids[0].op == "lrev":
        return kids[0].kids[0]
    if n.op == "lsort" and kids and kids[0].op == "lsort":
        return kids[0]
    if n.op == "not" and kids and kids[0].op == "not":
        return kids[0].kids[0]
    if n.op == "add" and len(kids) == 2:
        for a, b in ((0, 1), (1, 0)):
            if kids[a].op == "lit" and kids[a].const == 0:
                return kids[b]
    if n.op == "mul" and len(kids) == 2:
        for a, b in ((0, 1), (1, 0)):
            if kids[a].op == "lit" and kids[a].const == 1:
                return kids[b]
    # constant folding: an op whose children are all literals
    if (n.op in ("add", "sub", "mul", "sdiv", "smod", "inc", "dec", "imax",
                 "imin") and all(k.op == "lit" for k in kids)):
        r = run(n, [])
        if r.ok and isinstance(r.value, int):
            return Node("lit", "I", const=r.value)
    return n


def _reindex_params(n: Node) -> Tuple[Node, int]:
    mapping: Dict[int, int] = {}

    def walk(x: Node) -> Node:
        if x.op == "param":
            if x.const not in mapping:
                mapping[x.const] = len(mapping)
            return Node("param", x.rtype, const=mapping[x.const])
        return Node(x.op, x.rtype, tuple(walk(k) for k in x.kids), x.const)
    out = walk(n)
    return out, len(mapping)


def _probe_args(ptypes: Tuple[str, ...], rng: random.Random) -> List:
    def mk(t):
        if t == "I":
            return rng.randint(-4, 6)
        if t == "B":
            return bool(rng.getrandbits(1))
        if t == "S":
            return "".join(rng.choice("abc") for _ in range(rng.randint(1, 4)))
        if t == "P":
            return (rng.randint(-3, 5), rng.randint(-3, 5))
        if t == "L":
            return [(rng.randint(-3, 5), rng.randint(0, 5))
                    for _ in range(rng.randint(2, 5))]
        return rng.randint(0, 5)
    return [mk(t) for t in ptypes]


def _equiv(a: Node, b: Node, ptypes: Tuple[str, ...], seed: int = 7) -> bool:
    """Kernel-equivalence: a and b agree on a probe set of param bindings."""
    rng = random.Random(seed)
    agree = 0
    for _ in range(24):
        args = _probe_args(ptypes, rng)
        env = {("param_%d" % i): v for i, v in enumerate(args)}
        ra = _run_body(a, args)
        rb = _run_body(b, args)
        if (ra[0], ra[1]) != (rb[0], rb[1]):
            return False
        agree += 1
    return agree > 0


def _run_body(body: Node, params: List) -> Tuple[bool, object]:
    # substitute params then run with no args
    def sub(x):
        if x.op == "param":
            v = params[x.const]
            t = ("I" if isinstance(v, bool) or isinstance(v, int) else
                 "S" if isinstance(v, str) else "P" if isinstance(v, tuple)
                 else "L")
            return Node("lit", t, const=v)
        return Node(x.op, x.rtype, tuple(sub(k) for k in x.kids), x.const)
    r = run(sub(body), [], max_steps=40_000)
    return (r.ok, r.value if r.ok else None)


def normalize_block(b: Block) -> Tuple[Block, bool]:
    """Return (normalized_block, changed). The normalization is applied only if
    it is verified semantics-preserving by the kernel-equivalence check."""
    folded = _fold(b.body)
    reixed, npar = _reindex_params(folded)
    changed = pp(reixed) != pp(b.body)
    if not changed:
        return b, False
    if not _equiv(b.body, reixed, b.ptypes):
        return b, False          # behaviour changed -> reject, keep original
    nb = Block(name=b.name, ptypes=b.ptypes[:npar] if npar <= len(b.ptypes)
               else b.ptypes, body=reixed, rtype=b.rtype,
               created_round=b.created_round, origin="norm:" + b.origin)
    return nb, True
