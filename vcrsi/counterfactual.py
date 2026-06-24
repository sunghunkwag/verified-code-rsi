#!/usr/bin/env python3
"""The counterfactual: adaptive arm vs frozen arm at EQUAL budget and seeds.

This is the ONLY place "self-improvement" is reported, and it is reported as a
single number: the solved-count delta between an arm that improves its own search
policy (weights + mined subroutines) and an otherwise-identical arm whose policy
is frozen. Both arms attack the same task stream, with the same per-attempt
budget and the same per-(task, round) seeds, so the delta is attributable to
adaptation alone. Run twice with the same seed -> byte-identical adoption logs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from .oracle import SealedOracle, assert_verifier_unchanged, verifier_fingerprint
from .rsi import run_arm, ArmResult


# canonical task order (easy -> hard); a curriculum so learned structure from
# easy tasks can lift harder ones.
DEFAULT_ORDER = [
    "rle_decode",                                            # bootstrap (base-solvable)
    "rle_decode_rev", "rle_decode_sorted",                  # adaptive: reuse EXPAND
    "rle_decode_twice", "rle_decode_palindrome",            # adaptive: reuse EXPAND
    "rle_decode_rev_twice", "rle_rev_palindrome",           # adaptive: deep / lineage
    "caesar_encode",                                        # honest OPEN (char-shift)
    "merge_intervals", "bracket_depths", "bytecode_interp",  # honest OPEN frontier
]


@dataclass
class CFResult:
    adaptive: ArmResult
    frozen: ArmResult
    delta: int
    verifier_fp: str

    def to_dict(self) -> dict:
        return {
            "verifier_fp": self.verifier_fp,
            "adaptive_solved": self.adaptive.solved_count(),
            "frozen_solved": self.frozen.solved_count(),
            "delta": self.delta,
            "adaptive_tasks": sorted(self.adaptive.adopted.keys()),
            "frozen_tasks": sorted(self.frozen.adopted.keys()),
            "adaptive_only": sorted(set(self.adaptive.adopted)
                                    - set(self.frozen.adopted)),
            "adaptive_digest": self.adaptive.adoption_digest(),
            "frozen_digest": self.frozen.adoption_digest(),
            "blocks_adopted": [b.name for b in self.adaptive.blocks],
            "lineage_events": self.adaptive.lineage,
            "adaptive_evals": self.adaptive.total_evals,
            "frozen_evals": self.frozen.total_evals,
        }


def run_counterfactual(oracles: Dict[str, SealedOracle], *, budget: int = 25000,
                       rounds: int = 3, gate_budget: int = 10000,
                       task_order: Optional[List[str]] = None,
                       verbose: bool = False) -> CFResult:
    order = task_order or [t for t in DEFAULT_ORDER if t in oracles]
    fp = assert_verifier_unchanged(oracles, "counterfactual.start")
    frozen = run_arm(oracles, adaptive=False, budget=budget, rounds=rounds,
                     gate_budget=gate_budget, task_order=order, verbose=verbose)
    adaptive = run_arm(oracles, adaptive=True, budget=budget, rounds=rounds,
                       gate_budget=gate_budget, learn_weights_on=True,
                       task_order=order, verbose=verbose)
    assert_verifier_unchanged(oracles, "counterfactual.end")
    delta = adaptive.solved_count() - frozen.solved_count()
    return CFResult(adaptive, frozen, delta, fp)
