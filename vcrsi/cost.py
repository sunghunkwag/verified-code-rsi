#!/usr/bin/env python3
"""The REAL execution-cost verifier (EXPENSIVE, audit-only).

This is the second objective introduced across the cheap-verifier boundary (§0 of
the task). Correctness stays gated by the sealed oracle (unchanged). COST is a
DIFFERENT objective whose real verifier is EXPENSIVE *by fiat*: to know a program's
true cost you must RUN it on a held-out input battery and count executed steps.

  metric  = executed-step count (the interpreter's ``RunResult.steps``)
  cost(p) = sum of that count over the SEALED held-out battery ``H_cost``.

``H_cost`` is generated deterministically from each target's own input generator at
LARGE, unseen sizes -- in scale and seed ranges DISJOINT from the PUBLIC cost
battery that the proxy (``proxy.py``) trains on. It is reproducible to a
byte-identical digest, so the preregistration can hash-pin it.

CONTAINMENT OF THE EXPENSE: the inner self-improvement loop is FORBIDDEN from
importing or calling this module. Only the final audit (``audit.py``) does. The
control ``inner_loop_cost_blind`` proves, by source inspection, that the optimizer
(``costopt.py``) and the proxy (``proxy.py``) never import ``cost``.

Nothing here trains anything and nothing here is consulted inside the loop; this is
purely the expensive ground-truth used once, at the end, to measure how far the
self-learned proxy's claimed cost-improvement actually survives.
"""
from __future__ import annotations

import hashlib
import inspect
import random
from typing import Any, List, Optional, Tuple

from .interp import run
from .ir import Node

METRIC = "executed_step_count"

# H_cost is LARGE + unseen. These ranges are DISJOINT from proxy.PUBLIC_SCALE_*
# (2..4): a held-out audit input is an order of magnitude bigger than any public
# training input, so the proxy -- calibrated on small inputs -- has never observed
# cost at this scale (the distribution shift the audit exposes).
HCOST_SCALE_LO = 30
HCOST_SCALE_HI = 46
HCOST_N = 12                  # inputs per target in the held-out battery
HCOST_SEED_BASE = 0xC057      # reserved seed namespace for the audit battery
HCOST_MAX_STEPS = 6_000_000   # generous; a runaway candidate is simply uncosted


def _name_seed(name: str) -> int:
    return int(hashlib.sha256(("hcost|" + name).encode()).hexdigest(), 16) % (2 ** 31)


def _ramped(n: int, lo: int, hi: int) -> List[int]:
    if n <= 1:
        return [hi]
    step = (hi - lo) / (n - 1)
    return [int(round(lo + i * step)) for i in range(n)]


def hcost_battery(task) -> List[Tuple[Any, ...]]:
    """The SEALED held-out cost battery for one target: deterministic args-tuples at
    large, unseen sizes. Identical across runs (so the audit digest is stable)."""
    scales = _ramped(HCOST_N, HCOST_SCALE_LO, HCOST_SCALE_HI)
    base = _name_seed(task.name)
    out: List[Tuple[Any, ...]] = []
    for i, sc in enumerate(scales):
        rng = random.Random((base + 1009 * i) ^ HCOST_SEED_BASE)
        out.append(task.gen_input(rng, sc))
    return out


def real_cost(prog: Node, battery: List[Tuple[Any, ...]],
              blocks=None, max_steps: int = HCOST_MAX_STEPS) -> Optional[int]:
    """EXPENSIVE: execute ``prog`` on every held-out input and sum executed steps.
    Returns None if the program errors or exceeds the step budget on any input
    (an uncostable program -- correctness is gated separately by the oracle)."""
    total = 0
    for args in battery:
        r = run(prog, list(args), blocks, max_steps=max_steps)
        if not r.ok:
            return None
        total += r.steps
    return total


def battery_digest(battery: List[Tuple[Any, ...]]) -> str:
    h = hashlib.sha256()
    for args in battery:
        h.update(repr(args).encode())
    return h.hexdigest()[:16]


def hcost_spec_digest(tasks) -> str:
    """A digest over the H_cost SPEC: the metric, the scale/size parameters, and the
    per-target battery digests. The preregistration folds this in; any change to the
    audit inputs moves it and aborts the run."""
    h = hashlib.sha256()
    h.update(METRIC.encode())
    h.update(f"{HCOST_SCALE_LO},{HCOST_SCALE_HI},{HCOST_N},{HCOST_SEED_BASE}".encode())
    for t in tasks:
        h.update(t.name.encode())
        h.update(battery_digest(hcost_battery(t)).encode())
    return h.hexdigest()[:16]


def audit_source_digest() -> str:
    """Hash of the real-cost audit PROCEDURE source (the cost function + battery
    builder). Pinned by the preregistration so the audit itself cannot be quietly
    rewritten between prereg and audit."""
    h = hashlib.sha256()
    h.update(inspect.getsource(real_cost).encode())
    h.update(inspect.getsource(hcost_battery).encode())
    h.update(inspect.getsource(_ramped).encode())
    return h.hexdigest()[:16]
