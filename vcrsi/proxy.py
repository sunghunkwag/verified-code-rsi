#!/usr/bin/env python3
"""The LEARNED, CHEAP proxy cost-model (static features; NO execution at predict).

This is the self-learned cheaper verifier the system trusts INSIDE the loop, in
place of the expensive real cost-audit (``cost.py``). It predicts a program's
executed-step cost from STATIC features of its IR alone -- it never runs the
program when predicting:

  * total node count
  * loop-node count and max static loop-nesting depth
  * a static exec-depth bound (sum of loop nesting depths -- a structural stand-in
    for "how nested the iteration is")
  * an op-type histogram over the fixed IR vocabulary

It is trained by ridge regression on a PUBLIC cost battery: a corpus of correct
programs labelled with their real step-cost on SMALL public inputs. Those input
sizes are DISJOINT from the held-out audit battery ``H_cost`` (in ``cost.py``), so
the proxy is calibrated only at small scale and has never seen audit-scale cost --
which is exactly the blind spot the audit exposes.

SEALING (controls ``proxy_is_holdout_blind`` / ``proxy_predicts_cost_not_correctness``
/ ``inner_loop_cost_blind``):
  * this module imports NEITHER the sealed correctness oracle, the task references,
    NOR ``cost.py`` (the held-out audit). Its training labels are real step-counts
    on PUBLIC inputs only, computed via the interpreter here.
  * it predicts a cost MAGNITUDE; it is given no correctness labels and is shown
    (empirically) to be uninformative about pass/fail.
"""
from __future__ import annotations

import hashlib
import random
from typing import Any, Dict, List, Optional, Tuple

from .ir import Node, PRIMS, COMBINATORS
from .interp import run

# PUBLIC training-input sizes: SMALL, and disjoint from cost.HCOST_SCALE_* (30..46).
PUBLIC_SCALE_LO = 2
PUBLIC_SCALE_HI = 4
PUBLIC_N = 10
PUBLIC_SEED_BASE = 0x9111
PUBLIC_MAX_STEPS = 400_000

_LOOPS = ("map", "filter", "foldl", "scan", "iterate")
# fixed, sorted op vocabulary for the histogram (combinators + primitives + leaves)
_VOCAB: Tuple[str, ...] = tuple(sorted(
    set(PRIMS) | COMBINATORS | {"ifx", "pipe", "call", "lit", "arg", "var", "param"}))
_VOCAB_IDX = {o: i for i, o in enumerate(_VOCAB)}
# feature layout: [nodes, loop_nodes, max_loop_depth, exec_depth_bound, *histogram, 1]
NFEAT = 4 + len(_VOCAB) + 1


def features(prog: Node) -> List[float]:
    """STATIC feature vector (no execution). Pure function of the AST."""
    hist = [0.0] * len(_VOCAB)
    agg = {"nodes": 0.0, "loop_nodes": 0.0, "max_depth": 0.0, "depth_sum": 0.0}

    def walk(n: Node, loop_depth: int) -> None:
        agg["nodes"] += 1
        d = loop_depth
        if n.op in _LOOPS:
            agg["loop_nodes"] += 1
            d = loop_depth + 1
            agg["max_depth"] = max(agg["max_depth"], d)
            agg["depth_sum"] += d
        idx = _VOCAB_IDX.get(n.op)
        if idx is not None:
            hist[idx] += 1.0
        for k in n.kids:
            walk(k, d)

    walk(prog, 0)
    return ([agg["nodes"], agg["loop_nodes"], agg["max_depth"], agg["depth_sum"]]
            + hist + [1.0])


# --------------------------------------------------------------------------- #
# Ridge regression (tiny, deterministic; Gaussian elimination)                #
# --------------------------------------------------------------------------- #
def _ridge(X: List[List[float]], y: List[float], lam: float) -> List[float]:
    m = len(X[0])
    A = [[0.0] * m for _ in range(m)]
    bvec = [0.0] * m
    for xi, yi in zip(X, y):
        for i in range(m):
            bvec[i] += xi[i] * yi
            Ai = A[i]
            xii = xi[i]
            for j in range(m):
                Ai[j] += xii * xi[j]
    for i in range(m):
        A[i][i] += lam            # regularise (also makes the system non-singular)
    # Gauss-Jordan
    for i in range(m):
        piv = A[i][i] if abs(A[i][i]) > 1e-12 else 1e-12
        inv = 1.0 / piv
        for j in range(i, m):
            A[i][j] *= inv
        bvec[i] *= inv
        for k in range(m):
            if k != i and A[k][i] != 0.0:
                fac = A[k][i]
                for j in range(i, m):
                    A[k][j] -= fac * A[i][j]
                bvec[k] -= fac * bvec[i]
    return bvec


class CostProxy:
    """The learned cheap cost-model. UNTRAINED instances predict a constant 0 (the
    'frozen, wave-0' copy used as the counterfactual baseline arm); a TRAINED
    instance predicts the ridge model over static features."""

    __slots__ = ("w", "trained", "lam")

    def __init__(self, lam: float = 5.0):
        self.w: List[float] = [0.0] * NFEAT
        self.trained = False
        self.lam = lam

    def train(self, programs: List[Node], costs: List[float]) -> "CostProxy":
        X = [features(p) for p in programs]
        self.w = _ridge(X, costs, self.lam)
        self.trained = True
        return self

    def predict(self, prog: Node) -> float:
        f = features(prog)
        return sum(wi * fi for wi, fi in zip(self.w, f))

    def is_trained(self) -> bool:
        return self.trained

    def clone_frozen(self) -> "CostProxy":
        """A FROZEN (untrained, wave-0) copy: same class, no learned weights."""
        return CostProxy(self.lam)

    def digest(self) -> str:
        h = hashlib.sha256()
        h.update(b"trained" if self.trained else b"frozen")
        for wi in self.w:
            h.update(f"{wi:.6f};".encode())
        return h.hexdigest()[:16]


# --------------------------------------------------------------------------- #
# PUBLIC cost battery (small inputs) -> training labels                        #
# --------------------------------------------------------------------------- #
def _name_seed(name: str) -> int:
    return int(hashlib.sha256(("pub|" + name).encode()).hexdigest(), 16) % (2 ** 31)


def _ramped(n: int, lo: int, hi: int) -> List[int]:
    if n <= 1:
        return [hi]
    step = (hi - lo) / (n - 1)
    return [int(round(lo + i * step)) for i in range(n)]


def public_cost_battery(task) -> List[Tuple[Any, ...]]:
    """SMALL public inputs (disjoint in size from H_cost) used ONLY to label the
    proxy's training corpus. Deterministic."""
    scales = _ramped(PUBLIC_N, PUBLIC_SCALE_LO, PUBLIC_SCALE_HI)
    base = _name_seed(task.name)
    out: List[Tuple[Any, ...]] = []
    for i, sc in enumerate(scales):
        rng = random.Random((base + 1009 * i) ^ PUBLIC_SEED_BASE)
        out.append(task.gen_input(rng, sc))
    return out


def public_cost(prog: Node, battery: List[Tuple[Any, ...]],
                blocks=None) -> Optional[float]:
    """Real executed-step cost of ``prog`` on the SMALL public battery -- the proxy's
    training LABEL. (Runs the interpreter on PUBLIC inputs only; never H_cost.)"""
    total = 0
    for args in battery:
        r = run(prog, list(args), blocks, max_steps=PUBLIC_MAX_STEPS)
        if not r.ok:
            return None
        total += r.steps
    return float(total)


def train_proxy(corpus: List[Tuple[Node, "object"]], lam: float = 5.0) -> CostProxy:
    """Train the proxy on a corpus of (program, task) pairs: each program's static
    features regressed onto its real step-cost on that task's PUBLIC battery. The
    corpus is the system's OWN correct programs over the PUBLIC targets -- never the
    held-out battery, never any reference."""
    progs: List[Node] = []
    labels: List[float] = []
    for prog, task in corpus:
        c = public_cost(prog, public_cost_battery(task))
        if c is None:
            continue
        progs.append(prog)
        labels.append(c)
    p = CostProxy(lam)
    if progs:
        p.train(progs, labels)
    return p
