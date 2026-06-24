#!/usr/bin/env python3
"""The sealed correctness oracle (verifier-first).

This module is built BEFORE the generator and is the immutable root of the whole
system. For each task it:

  * generates a small set of PUBLIC training examples (the only thing the
    synthesizer is ever handed), and
  * generates a larger HIDDEN held-out battery at unseen input sizes by running
    the task's reference solution -- this battery IS the correctness oracle.

A candidate program is correct iff it reproduces the reference's output exactly
on the FULL held-out battery (plus the public examples, and -- for round-trip
codecs -- a composition-to-identity check).

Sealing:
  * ``PublicView`` exposes ONLY name/family/spec/types/public examples. The
    synthesizer receives a ``PublicView``; it never receives the ``Task``
    object, the reference solution, or the held-out battery.
  * ``verifier_fingerprint()`` hashes every reference + battery + the verify
    procedure source. ``assert_verifier_unchanged()`` pins it and aborts the run
    on any drift (tamper detection).
"""
from __future__ import annotations

import hashlib
import inspect
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .ir import Node, pp
from .interp import run
from .tasks import Task, SUITE, SUITE_BY_NAME


# --------------------------------------------------------------------------- #
# Public view handed to the synthesizer (no reference, no held-out)            #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PublicView:
    name: str
    family: int
    spec: str
    arg_types: Tuple[str, ...]
    out_type: str
    public_examples: Tuple[Tuple[Tuple[Any, ...], Any], ...]


def _ramped_scales(n: int, lo: int, hi: int) -> List[int]:
    if n == 1:
        return [hi]
    step = (hi - lo) / (n - 1)
    return [int(round(lo + i * step)) for i in range(n)]


class SealedOracle:
    """Wraps one task. Generates public + held-out from the reference; exposes a
    sealed verify(). The reference and held-out battery are private."""

    def __init__(self, task: Task):
        self._task = task
        rng = random.Random(_seed_for(task.name))
        # public examples at ramped small sizes; the LONGEST drives the
        # complexity floor's dynamic-depth metric.
        self._public: List[Tuple[Tuple[Any, ...], Any]] = []
        for sc in _ramped_scales(task.n_public, 2, task.public_scale + 4):
            self._public.append(self._make_example(task, rng, sc))
        # held-out battery at larger, unseen sizes
        self._holdout: List[Tuple[Tuple[Any, ...], Any]] = []
        for sc in _ramped_scales(task.n_holdout, task.holdout_scale,
                                 task.holdout_scale + 6):
            self._holdout.append(self._make_example(task, rng, sc))

    @staticmethod
    def _make_example(task, rng, scale) -> Tuple[Tuple[Any, ...], Any]:
        args = task.gen_input(rng, scale)
        r = run(task.reference, list(args))
        if not r.ok:
            raise RuntimeError(f"reference for {task.name} failed on a generated "
                               f"input: {r.error}")
        return (args, r.value)

    # ---- public surface (safe to hand to search) ------------------------- #
    def public_view(self) -> PublicView:
        return PublicView(self._task.name, self._task.family, self._task.spec,
                          self._task.arg_types, self._task.out_type,
                          tuple(self._public))

    # ---- sealed verification (the oracle) -------------------------------- #
    def verify(self, prog: Node, blocks: Optional[Dict] = None) -> bool:
        """Exact pass/fail on the FULL battery: public + held-out, re-executed
        independently. For round-trip codecs, also require composition-to-id."""
        for args, exp in self._public:
            r = run(prog, list(args), blocks)
            if not r.ok or r.value != exp:
                return False
        for args, exp in self._holdout:
            r = run(prog, list(args), blocks)
            if not r.ok or r.value != exp:
                return False
        return True

    def passes_public(self, prog: Node, blocks: Optional[Dict] = None) -> bool:
        for args, exp in self._public:
            r = run(prog, list(args), blocks)
            if not r.ok or r.value != exp:
                return False
        return True

    # ---- accessors used only by sealed tooling (complexity, fingerprint) -- #
    @property
    def task(self) -> Task:
        return self._task

    def longest_public_args(self) -> Tuple[Any, ...]:
        return max((ex[0] for ex in self._public),
                   key=lambda a: _input_size(a))

    def fingerprint(self) -> str:
        h = hashlib.sha256()
        h.update(pp(self._task.reference).encode())
        h.update(repr(self._public).encode())
        h.update(repr(self._holdout).encode())
        return h.hexdigest()[:16]


def _input_size(args: Tuple[Any, ...]) -> int:
    n = 0
    for a in args:
        if isinstance(a, (list, str)):
            n += len(a)
        else:
            n += 1
    return n


def _seed_for(name: str) -> int:
    return int(hashlib.sha256(name.encode()).hexdigest(), 16) % (2 ** 31)


# --------------------------------------------------------------------------- #
# Build the sealed oracle set once (deterministic)                            #
# --------------------------------------------------------------------------- #
def build_oracles() -> Dict[str, SealedOracle]:
    return {t.name: SealedOracle(t) for t in SUITE}


# --------------------------------------------------------------------------- #
# Verifier fingerprint + tamper pin (the immutable correctness root)          #
# --------------------------------------------------------------------------- #
def verifier_fingerprint(oracles: Dict[str, SealedOracle]) -> str:
    h = hashlib.sha256()
    for name in sorted(oracles):
        h.update(name.encode())
        h.update(oracles[name].fingerprint().encode())
    # fold in the verify procedure source: if the gate itself is edited, the
    # fingerprint moves and the run aborts.
    h.update(inspect.getsource(SealedOracle.verify).encode())
    h.update(inspect.getsource(SealedOracle._make_example).encode())
    return h.hexdigest()[:16]


_FP_FROZEN: Optional[str] = None


def assert_verifier_unchanged(oracles: Dict[str, SealedOracle], where: str) -> str:
    global _FP_FROZEN
    fp = verifier_fingerprint(oracles)
    if _FP_FROZEN is None:
        _FP_FROZEN = fp
    elif fp != _FP_FROZEN:
        raise RuntimeError(f"verifier_fp changed at {where}: {_FP_FROZEN} -> {fp}")
    return fp
