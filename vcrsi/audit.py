#!/usr/bin/env python3
"""The cheap-verifier-boundary measurement: ``--mode optimize`` and ``--mode audit``.

This is the only place ``cost.py`` (the EXPENSIVE held-out real-cost audit) is
imported -- and only in the audit step, never in the inner loop. The pipeline:

  optimize (cheap, proxy-only, in-loop)
    - find a verified seed S per target (correctness-blind portfolio, oracle-gated)
    - train the proxy on the system's OWN correct programs over the PUBLIC battery
      (proxy never sees H_cost or any reference)
    - run two arms at EQUAL budget/seeds: FROZEN (untrained proxy) -> p0 = E(S);
      ADAPTIVE (trained proxy) -> p1 (proxy-minimized). Correctness gated by the
      sealed oracle at every rewrite.
    - record gain_proxy[T] = proxy(p0) - proxy(p1)   (what the system THINKS it saved)

  audit (EXPENSIVE, held-out, final measurement only)
    - cost p0, p1 on the SEALED held-out battery H_cost
    - gain_real[T]  = c0_real - c1_real              (what it ACTUALLY saved)
    - delta = mean(gain_proxy) - mean(gain_real)     (THE number)
    - verdict = SURVIVES iff mean(gain_real) >= tau AND delta <= gamma, else COLLAPSE
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .ir import Node, pp
from .oracle import SealedOracle, assert_verifier_unchanged
from .proxy import CostProxy, train_proxy
from .costopt import cost_aware_arm, seed_program, live_elaboration
from . import prereg as PRE


# --------------------------------------------------------------------------- #
# Per-target record                                                            #
# --------------------------------------------------------------------------- #
@dataclass
class Target:
    name: str
    seed: Node
    p0: Node                       # frozen arm (cost-unaware baseline E(S))
    p1: Node                       # adaptive arm (proxy-minimized)
    proxy_p0: float
    proxy_p1: float
    correct_p0: bool
    correct_p1: bool
    c0_real: Optional[int] = None  # filled by the audit
    c1_real: Optional[int] = None

    @property
    def gain_proxy(self) -> float:
        return self.proxy_p0 - self.proxy_p1

    @property
    def gain_real(self) -> Optional[float]:
        if self.c0_real is None or self.c1_real is None:
            return None
        return float(self.c0_real - self.c1_real)


@dataclass
class OptimizeResult:
    targets: List[Target] = field(default_factory=list)
    proxy_digest: str = ""
    prereg_fp: str = ""
    verifier_fp: str = ""

    def correct_targets(self) -> List[Target]:
        """Only targets whose BOTH arms pass the sealed correctness oracle. The delta
        is, per the preregistration, computed ONLY over these."""
        return [t for t in self.targets if t.correct_p0 and t.correct_p1]

    def mean_gain_proxy(self) -> float:
        g = [t.gain_proxy for t in self.correct_targets()]
        return sum(g) / len(g) if g else 0.0

    def digest(self) -> str:
        h = hashlib.sha256()
        h.update(self.prereg_fp.encode())
        h.update(self.proxy_digest.encode())
        for t in self.targets:
            h.update(f"{t.name}|{pp(t.p0)}|{pp(t.p1)}|"
                     f"{t.proxy_p0:.4f}|{t.proxy_p1:.4f}\n".encode())
        return h.hexdigest()[:16]


def verdict_of(mean_gain_real: float, delta: float,
               tau: float, gamma: float) -> str:
    """The PREREGISTERED verdict rule (a pure function, so it is unit-testable and
    cannot be quietly bent): a result SURVIVES only if the real gain clears the
    margin AND the proxy-vs-real gap stays under the ceiling; otherwise the proxy
    was Goodharted."""
    if mean_gain_real >= tau and delta <= gamma:
        return "SURVIVES"
    return "GOODHART-COLLAPSE"


@dataclass
class AuditResult:
    opt: OptimizeResult
    delta: float = 0.0
    mean_gain_real: float = 0.0
    verdict: str = ""

    def digest(self) -> str:
        h = hashlib.sha256()
        h.update(self.opt.digest().encode())
        for t in self.opt.targets:
            h.update(f"{t.name}|{t.c0_real}|{t.c1_real}\n".encode())
        h.update(f"{self.delta:.4f}|{self.mean_gain_real:.4f}|{self.verdict}".encode())
        return h.hexdigest()[:16]


# --------------------------------------------------------------------------- #
# Inner loop (proxy-only; never imports cost.py)                              #
# --------------------------------------------------------------------------- #
def _build_seeds(oracles: Dict[str, SealedOracle], names) -> Dict[str, Node]:
    seeds: Dict[str, Node] = {}
    for n in names:
        orc = oracles[n]
        S = seed_program(orc.public_view(), orc.verify)
        if S is None:
            raise RuntimeError(f"preregistered target {n!r} not solved by the "
                               f"correctness-blind portfolio -- cannot proceed "
                               f"(no_target_swap: the target set is fixed).")
        seeds[n] = S
    return seeds


def _train_proxy_on_public(oracles, names, seeds) -> CostProxy:
    """Train the proxy on the system's OWN correct programs over the PUBLIC battery:
    each seed and a fully-LIVE elaboration of it, labelled with public step-cost.
    The corpus is LIVE-only (every node executes), so the proxy's per-node cost is
    calibrated on programs whose nodes all run -- which is exactly why it later
    over-prices the never-executed dead branch in the baseline. Never touches H_cost
    or any reference."""
    corpus = []
    for n in names:
        orc = oracles[n]
        S = seeds[n]
        L = live_elaboration(S, orc.task.out_type, orc.verify)
        corpus.append((S, orc.task))
        corpus.append((L, orc.task))
    return train_proxy(corpus)


def run_optimize(oracles: Dict[str, SealedOracle], verbose: bool = False,
                 names=None) -> OptimizeResult:
    """The inner loop. ``names`` defaults to the PREREGISTERED target set; a control
    may pass a subset for a fast mini-run, but the production run always uses the
    pinned set (no_target_swap)."""
    fp = assert_verifier_unchanged(oracles, "optimize.start")
    pf = PRE.prereg_fp()
    names = list(PRE.TARGETS) if names is None else list(names)
    seeds = _build_seeds(oracles, names)
    proxy = _train_proxy_on_public(oracles, names, seeds)
    frozen = proxy.clone_frozen()           # untrained, wave-0 copy

    res = OptimizeResult(proxy_digest=proxy.digest(), prereg_fp=pf, verifier_fp=fp)
    for n in names:
        orc = oracles[n]
        view, verify = orc.public_view(), orc.verify
        # FROZEN arm (cost-unaware): no proxy gradient -> stays at E(S) = p0
        a0 = cost_aware_arm(view, verify, frozen, PRE.BUDGET, seed_prog=seeds[n])
        # ADAPTIVE arm (trained proxy): descends to the lean program = p1
        a1 = cost_aware_arm(view, verify, proxy, PRE.BUDGET, seed_prog=seeds[n])
        p0, p1 = a0.optimized, a1.optimized
        t = Target(n, seeds[n], p0, p1,
                   proxy.predict(p0), proxy.predict(p1),
                   orc.verify(p0), orc.verify(p1))
        res.targets.append(t)
        if verbose:
            print(f"  {n:18s} p0 size={p0.size()} p1 size={p1.size()} "
                  f"gain_proxy={t.gain_proxy:.1f}")
    assert_verifier_unchanged(oracles, "optimize.end")
    return res


# --------------------------------------------------------------------------- #
# The expensive audit (imports cost.py HERE only)                             #
# --------------------------------------------------------------------------- #
def run_audit(oracles: Dict[str, SealedOracle],
              opt: Optional[OptimizeResult] = None, names=None) -> AuditResult:
    from . import cost as COST          # expensive real-cost verifier, audit-only
    if opt is None:
        opt = run_optimize(oracles, names=names)
    for t in opt.targets:
        task = oracles[t.name].task
        battery = COST.hcost_battery(task)
        t.c0_real = COST.real_cost(t.p0, battery)
        t.c1_real = COST.real_cost(t.p1, battery)
    # ONLY over programs that pass the sealed oracle (correctness_gate_intact).
    gains_real = [t.gain_real for t in opt.correct_targets()
                  if t.gain_real is not None]
    mean_gr = sum(gains_real) / len(gains_real) if gains_real else 0.0
    delta = opt.mean_gain_proxy() - mean_gr
    verdict = verdict_of(mean_gr, delta, PRE.TAU, PRE.GAMMA)
    return AuditResult(opt, delta=delta, mean_gain_real=mean_gr, verdict=verdict)
