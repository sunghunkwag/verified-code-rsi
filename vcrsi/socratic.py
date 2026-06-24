#!/usr/bin/env python3
"""M4 -- Socratic transfer gate (counterexample-validated promotion / CEGIS).

The load-bearing check (removal -> OPEN) proves a block is USED. It does not
prove the block is RIGHT: a block can fit a B-task's public + held-out examples
by accident and still be semantically wrong. The Socratic gate asks a QUESTIONER
to search for a DISTINGUISHING INPUT -- a fresh input on which the adopted
B-solution diverges from the sealed reference (the kernel). If one is found the
transfer is spurious and REJECTED; only a block whose B-solution survives the
counterexample search is admitted to the cross-family library.

This is the primary defence against fake transfer (control
``transfer_passes_socratic``: a deliberately spurious block is rejected here).
The sealed reference is the judge; the questioner never sees nor edits it.
"""
from __future__ import annotations

import random
from typing import Optional, Tuple

from .ir import Node
from .interp import run


def find_distinguishing_input(prog: Node, task, blocks, n_probes: int = 120,
                              seed: int = 12345) -> Optional[Tuple]:
    """Search for an input where ``prog`` diverges from the task's sealed
    reference. Returns the counterexample args, or None if none found."""
    rng = random.Random(seed)
    ref = task.reference
    scales = [task.public_scale, task.holdout_scale, task.holdout_scale + 4,
              task.holdout_scale + 8]
    for i in range(n_probes):
        scale = scales[i % len(scales)] + (i // len(scales))
        try:
            args = task.gen_input(rng, scale)
        except Exception:
            continue
        rr = run(ref, list(args), max_steps=200_000)
        rp = run(prog, list(args), blocks, max_steps=200_000)
        if not rr.ok:
            continue                      # reference itself failed; skip probe
        if (not rp.ok) or rp.value != rr.value:
            return args                   # distinguishing input found
    return None


def socratic_admit(prog: Node, task, blocks, seed: int = 12345) -> Tuple[bool, str]:
    """A transfer is Socratically admitted iff NO distinguishing input exists."""
    ce = find_distinguishing_input(prog, task, blocks, seed=seed)
    if ce is None:
        return True, "no distinguishing input found over 120 fresh probes"
    return False, f"distinguishing input found -> spurious (e.g. {repr(ce)[:60]})"
