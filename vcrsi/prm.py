#!/usr/bin/env python3
"""M1 -- the Process-Reward Model (PRM): an averaged perceptron over a small
fixed feature vector of a PARTIAL program (a ``StepScorer``).

This is the learned search guidance that Phase C recursively improves. It is
deterministic, oracle-free and cross-task: it scores program PREFIXES from a
fixed 8-dimensional feature vector computed ONLY from the public training I/O and
the program's own structure -- never the sealed oracle, the held-out battery, or
any family/reference metadata (the ``prm_is_oracle_free`` control inspects this
module's source to prove it).

The model is an averaged perceptron (§1 M1 of the task):

    score(f)        = dot(wsum, f) / seen        (seen>0; else dot(w, f) == 0)
    update(f, y)    = if y*dot(w,f) <= 0: w += y*f ;  wsum += w ; seen += 1

A FROZEN/wave-0 PRM has never been trained (seen==0, w==0), so it scores every
prefix 0.0 -> the beam degenerates to its deterministic structural tiebreak
(blind breadth-first). A PRM trained through wave N up-weights the features that
distinguished adopted-solution prefixes from dead ends, so it ranks productive
partial programs higher and reaches solutions a frozen PRM misses at equal budget
-- that gap is the measured solver-self-improvement (§2).

The feature classification helpers (``classify`` / ``value_distance``) live here
too; the prefix featuriser that actually RUNS a partial program is in
``prm_beam.py`` (it is tied to the postfix prefix representation).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, List, Sequence, Tuple

NFEAT = 8                       # exact, typed, near, single, crash, len, reuse, bias
_BIG = 1.0e6                    # distance assigned to a type mismatch (near ~ 0)


# A sentinel for "this (partial) program crashed / abstained on this input".
class _Crash:
    __slots__ = ()
    def __repr__(self): return "CRASH"


CRASH = _Crash()


# --------------------------------------------------------------------------- #
# Value typing + a domain-appropriate distance on the IR's value type          #
# --------------------------------------------------------------------------- #
def vtype(v: Any) -> str:
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


def _seqsim(a: Sequence, b: Sequence) -> float:
    """Longest-common-subsequence ratio in [0, 1] (1.0 == identical). A single,
    fixed, deterministic similarity used for both strings and lists."""
    la, lb = len(a), len(b)
    if la == 0 and lb == 0:
        return 1.0
    if la == 0 or lb == 0:
        return 0.0
    if la * lb > 120_000:                      # cheap fallback on huge inputs
        common = sum(1 for x, y in zip(a, b) if x == y)
        return common / max(la, lb)
    prev = [0] * (lb + 1)
    for i in range(1, la + 1):
        cur = [0] * (lb + 1)
        ai = a[i - 1]
        for j in range(1, lb + 1):
            if ai == b[j - 1]:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = prev[j] if prev[j] >= cur[j - 1] else cur[j - 1]
        prev = cur
    return prev[lb] / max(la, lb)


def value_distance(o: Any, y: Any) -> float:
    """A deterministic distance between a produced value ``o`` and the target
    ``y`` on the IR's value type. ONE fixed metric, kept stable: absolute
    difference on ints, mismatch on bools, (1 - LCS-ratio)*len on strings/lists,
    componentwise on pairs, and a large constant across a type mismatch."""
    to, ty = vtype(o), vtype(y)
    if to != ty:
        return _BIG
    if ty == "I":
        return float(abs(o - y))
    if ty == "B":
        return 0.0 if o == y else 1.0
    if ty in ("S", "L"):
        sim = _seqsim(o, y)
        return (1.0 - sim) * max(len(o), len(y), 1)
    if ty == "P":
        return value_distance(o[0], y[0]) + value_distance(o[1], y[1])
    return _BIG


def near_of(o: Any, y: Any) -> float:
    """Graded partial credit 1/(1+dist): 1.0 on an exact match, ->0 as ``o``
    diverges from ``y``."""
    return 1.0 / (1.0 + value_distance(o, y))


def classify(o: Any, y: Any) -> Tuple[str, float]:
    """Bucket a produced value against the target, returning (bucket, near).
    Buckets: 'exact' (==y), 'typed' (same type, not exact), 'single' (a collapsed
    scalar where the target is a container), 'crash', or 'other' (type mismatch)."""
    if o is CRASH:
        return "crash", 0.0
    if o == y:
        return "exact", 1.0
    if vtype(o) == vtype(y):
        return "typed", near_of(o, y)
    if vtype(y) in ("L", "S") and vtype(o) in ("I", "B", "P"):
        return "single", 0.0
    return "other", 0.0


def features_from_outputs(outputs: List[Any], targets: List[Any],
                          length_ratio: float, has_reuse: bool) -> List[float]:
    """Assemble the 8-feature vector from per-input (output, target) pairs plus
    the two structural features. ``outputs[i]`` may be the CRASH sentinel."""
    n = max(1, len(outputs))
    exact = typed = near = single = crash = 0.0
    for o, y in zip(outputs, targets):
        bucket, nr = classify(o, y)
        if bucket == "exact":
            exact += 1.0
            near += nr
        elif bucket == "typed":
            typed += 1.0
            near += nr
        elif bucket == "single":
            single += 1.0
        elif bucket == "crash":
            crash += 1.0
    return [exact / n, typed / n, near / n, single / n, crash / n,
            min(1.0, length_ratio), 1.0 if has_reuse else 0.0, 1.0]


# --------------------------------------------------------------------------- #
# The averaged perceptron (the learned, persisted, recursively-improved state)  #
# --------------------------------------------------------------------------- #
@dataclass
class PRM:
    w: List[float] = field(default_factory=lambda: [0.0] * NFEAT)
    wsum: List[float] = field(default_factory=lambda: [0.0] * NFEAT)
    seen: int = 0

    def score(self, f: Sequence[float]) -> float:
        """Rank/inference score. Uses the AVERAGED weights once trained; an
        untrained (frozen) PRM scores every prefix exactly 0.0."""
        if self.seen > 0:
            return sum(ws * fi for ws, fi in zip(self.wsum, f)) / self.seen
        return sum(wi * fi for wi, fi in zip(self.w, f))

    def update(self, f: Sequence[float], label: int) -> None:
        """One averaged-perceptron step. ``label`` is +1 (good prefix) or -1
        (dead end). The margin condition ``label*dot(w,f) <= 0`` updates only on
        a mistake; ``wsum`` accumulates for averaging."""
        margin = sum(wi * fi for wi, fi in zip(self.w, f))
        if label * margin <= 0:
            self.w = [wi + label * fi for wi, fi in zip(self.w, f)]
        self.wsum = [ws + wi for ws, wi in zip(self.wsum, self.w)]
        self.seen += 1

    def train_episode(self, prefix_feats: List[Sequence[float]], label: int) -> None:
        """Train on EVERY prefix of one episode (a solved program -> +1, a dead
        end -> -1). ``prefix_feats`` is the ordered list of feature vectors."""
        for f in prefix_feats:
            self.update(f, label)

    def clone(self) -> "PRM":
        p = PRM(list(self.w), list(self.wsum), self.seen)
        return p

    def avg(self) -> List[float]:
        if self.seen == 0:
            return [0.0] * NFEAT
        return [ws / self.seen for ws in self.wsum]

    def is_frozen(self) -> bool:
        return self.seen == 0

    def digest(self) -> str:
        """A pure function of the trained state: byte-identical across same-seed
        runs (the §5 determinism-of-guidance control checks this)."""
        h = hashlib.sha256()
        h.update(f"seen={self.seen};".encode())
        for x in self.avg():
            h.update(f"{round(x, 6):.6f};".encode())
        return h.hexdigest()[:16]
