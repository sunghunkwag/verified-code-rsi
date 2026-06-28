#!/usr/bin/env python3
"""verified-code-rsi :: entry point.

Verification-grounded recursive self-improvement of genuinely multi-step
programs, LLM-free. This file is a thin dispatcher; the system lives in the
``vcrsi`` package (see README.md and vcrsi/__init__.py for the architecture).

The previous contents of this file -- a ~50,000-line monolith whose recursive
self-improvement was real but whose DOMAIN was a toy (integer-list arithmetic
solved by 2-5 op bytecode) -- have been deleted entirely. What survives is the
verify-then-measure discipline of that monolith's one genuine component (its
native-kernel section): a hash-pinned correctness oracle, a physical executor,
candidates that are data, improvements proven as a measured delta over a frozen
counterfactual. That discipline is now applied to the synthesis of multi-step
programs over structured inputs.

    python rsi_core.py --mode demo            # adaptive run + complexity table + lineage
    python rsi_core.py --mode counterfactual  # adaptive vs frozen, equal budget/seeds
    python rsi_core.py --mode test            # the anti-cheat controls (§4)

Cheap-verifier-boundary measurement (this phase -- observational RSI across the
proxy-cost boundary; the deliverable is exactly one number, delta):

    python rsi_core.py --mode prereg          # hash-pin the preregistration; emit prereg_fp
    python rsi_core.py --mode optimize        # proxy-guided cost optimization (oracle-gated)
    python rsi_core.py --mode audit           # expensive held-out real-cost audit; delta; verdict
"""
from __future__ import annotations

import argparse

from vcrsi.report import (run_demo, run_counterfactual_mode, run_transfer_mode,
                          run_ablation_mode, run_solve_hard,
                          run_openended_mode, run_emergence_mode,
                          run_transfer_matrix_mode)
from vcrsi.report_cost import (run_prereg_mode, run_optimize_mode, run_audit_mode)
from vcrsi.controls import run_controls


def main() -> int:
    ap = argparse.ArgumentParser(description="verified-code-rsi")
    ap.add_argument("--mode", choices=("demo", "counterfactual", "test",
                                       "transfer", "ablation", "solve-hard",
                                       "openended", "emergence",
                                       "transfer-matrix",
                                       "prereg", "optimize", "audit"),
                    default="demo")
    args = ap.parse_args()
    if args.mode == "prereg":
        return run_prereg_mode()
    if args.mode == "optimize":
        return run_optimize_mode()
    if args.mode == "audit":
        return run_audit_mode()
    if args.mode == "demo":
        return run_demo()
    if args.mode == "counterfactual":
        return run_counterfactual_mode()
    if args.mode == "solve-hard":
        return run_solve_hard()
    if args.mode == "transfer":
        return run_transfer_mode()
    if args.mode == "ablation":
        return run_ablation_mode()
    if args.mode == "openended":
        return run_openended_mode()
    if args.mode == "emergence":
        return run_emergence_mode()
    if args.mode == "transfer-matrix":
        return run_transfer_matrix_mode()
    if args.mode == "test":
        return run_controls()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
