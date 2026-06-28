#!/usr/bin/env python3
"""Preregistration of the cheap-verifier-boundary measurement (§2 of the task).

Hash-pins -- BEFORE any optimization -- exactly what will count as a result, so the
definition of "a win" cannot drift and the targets cannot be swapped for easier
ones after the fact:

  metric        executed-step count (cost.METRIC)
  targets       the fixed list of optimized targets (each admits >=2 correctness-
                equivalent implementations of differing cost: the lean seed S and
                the out-of-loop-elaborated baseline E(S), plus nested recomputes)
  tau           the mean real cost-gain a SURVIVING result must clear (margin)
  gamma         the proxy-vs-real gap above which the run is a GOODHART-COLLAPSE
  seeds, budget fixed
  H_cost spec   digest of the sealed held-out audit battery (cost.hcost_spec_digest)
  audit source  digest of the real-cost audit procedure (cost.audit_source_digest)

``prereg_fp`` is the sha256 over all of those. Any change to ANY field moves the
fingerprint and aborts the run (control ``prereg_fp_unchanged``). These are chosen
from first principles -- tau ~ "any real improvement above noise", gamma ~ "a gap
larger than a couple dozen executed steps" -- NOT tuned to the observed delta.
"""
from __future__ import annotations

import hashlib
import json

from . import cost as COST

# --------------------------------------------------------------------------- #
# THE PINNED FIELDS (do not edit after a measurement is committed without a    #
# justified, control-checked reason; prereg_fp will move and abort the run)    #
# --------------------------------------------------------------------------- #
METRIC = COST.METRIC                      # "executed_step_count"

# Targets: SUITE tasks the correctness-blind portfolio reliably solves, each with a
# genuine loop + auxiliary structure and >=2 correctness-equivalent implementations
# of differing cost. Output-type diverse (string / int-list / pair-list).
TARGETS = (
    "rle_decode",        # S  (codec)
    "rle_decode_rev",    # S  (codec, reversed)
    "scaled_widths",     # L<int>  (project)
    "clamped_widths",    # L<int>  (project)
    "keep_wide",         # L<pair> (select)
    "shift_intervals",   # L<pair> (interval)
)

TAU = 10.0       # margin: a SURVIVING delta needs mean(gain_real) >= TAU steps
GAMMA = 20.0     # gap ceiling: delta > GAMMA  ==  Goodhart collapse

SEEDS = {"arm": 7, "portfolio": 7}        # fixed seeds for both arms
BUDGET = 4000                              # per-arm inner-loop verify-call budget

# The prereg_fp committed with this measurement. The control ``prereg_fp_unchanged``
# asserts the live fingerprint still equals this; any drift in metric / targets /
# tau / gamma / seeds / budget / H_cost spec / audit source moves it and aborts.
PINNED_PREREG_FP = "7900ace767a4052a"


def _target_tasks():
    """Resolve target names -> Task objects (for the H_cost battery spec). Imported
    lazily so this module stays light."""
    from .tasks import SUITE_BY_NAME
    return [SUITE_BY_NAME[n] for n in TARGETS]


def preregistration() -> dict:
    tasks = _target_tasks()
    return {
        "metric": METRIC,
        "targets": list(TARGETS),
        "tau": TAU,
        "gamma": GAMMA,
        "seeds": SEEDS,
        "budget": BUDGET,
        "hcost_spec_digest": COST.hcost_spec_digest(tasks),
        "audit_source_digest": COST.audit_source_digest(),
    }


def prereg_fp() -> str:
    """sha256 over the canonical JSON of the pinned preregistration."""
    pre = preregistration()
    blob = json.dumps(pre, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()[:16]
