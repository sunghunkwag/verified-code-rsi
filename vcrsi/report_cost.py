#!/usr/bin/env python3
"""--mode prereg / optimize / audit drivers for the cheap-verifier-boundary
measurement (§0-§5 of this phase). The headline is exactly one number, delta, with
a preregistered verdict and the located blind spot."""
from __future__ import annotations

from .ir import pp
from .oracle import build_oracles, assert_verifier_unchanged
from . import prereg as PRE
from .audit import run_optimize, run_audit


def run_prereg_mode() -> int:
    oracles = build_oracles()
    fp = assert_verifier_unchanged(oracles, "prereg")
    pre = PRE.preregistration()
    pf = PRE.prereg_fp()
    print("=" * 78)
    print("VERIFIED-CODE-RSI -- PREREGISTRATION (hash-pinned BEFORE optimization)")
    print(f"verifier_fp (sealed correctness oracle) = {fp}")
    print("=" * 78)
    print(f"  metric              : {pre['metric']}")
    print(f"  targets ({len(pre['targets'])})         : {pre['targets']}")
    print(f"  tau   (margin)      : {pre['tau']}   "
          "(a SURVIVING result needs mean(gain_real) >= tau)")
    print(f"  gamma (gap ceiling) : {pre['gamma']}   "
          "(delta > gamma  ==  Goodhart collapse)")
    print(f"  seeds               : {pre['seeds']}")
    print(f"  budget (per arm)    : {pre['budget']}")
    print(f"  H_cost spec digest  : {pre['hcost_spec_digest']}  "
          "(sealed held-out audit battery)")
    print(f"  audit source digest : {pre['audit_source_digest']}  "
          "(real-cost procedure)")
    print("-" * 78)
    print(f"  >>> prereg_fp = {pf} <<<")
    print("  Any change to ANY pinned field moves this fingerprint and aborts the")
    print("  run (control prereg_fp_unchanged + no_target_swap).")
    print("=" * 78)
    return 0


def _print_optimize(opt) -> None:
    print("%-18s %5s %5s %11s %11s %11s" %
          ("target", "p0sz", "p1sz", "proxy(p0)", "proxy(p1)", "gain_proxy"))
    for t in opt.targets:
        print("%-18s %5d %5d %11.1f %11.1f %11.1f" %
              (t.name, t.p0.size(), t.p1.size(), t.proxy_p0, t.proxy_p1,
               t.gain_proxy))
    print("-" * 78)
    print(f"  mean gain_proxy (what the system THINKS it saved) = "
          f"{opt.mean_gain_proxy():.2f}")
    print(f"  proxy digest = {opt.proxy_digest}   optimize digest = {opt.digest()}")


def run_optimize_mode() -> int:
    oracles = build_oracles()
    fp = assert_verifier_unchanged(oracles, "optimize.mode")
    print("=" * 78)
    print("VERIFIED-CODE-RSI -- OPTIMIZE (proxy-guided cost minimization, in-loop)")
    print("The correctness gate is the SEALED oracle (unchanged). The SECOND")
    print("objective -- executed-step cost -- is minimized using ONLY the learned")
    print("cheap PROXY (static IR features, trained on the PUBLIC battery). The")
    print("expensive real cost-audit is NOT consulted here; the proxy gates the loop.")
    print(f"verifier_fp = {fp}   prereg_fp = {PRE.prereg_fp()}")
    print("=" * 78)
    opt = run_optimize(oracles, verbose=False)
    _print_optimize(opt)
    bad = [t.name for t in opt.targets if not (t.correct_p0 and t.correct_p1)]
    print(f"  correctness gate (sealed oracle) intact for every program: "
          f"{not bad}" + (f" -- FAILED on {bad}" if bad else ""))
    assert_verifier_unchanged(oracles, "optimize.mode.end")
    print("=" * 78)
    print("Run `--mode audit` for the EXPENSIVE held-out real-cost audit + delta.")
    return 0


def run_audit_mode() -> int:
    oracles = build_oracles()
    fp = assert_verifier_unchanged(oracles, "audit.mode")
    pf = PRE.prereg_fp()
    print("=" * 78)
    print("VERIFIED-CODE-RSI -- AUDIT (expensive held-out real-cost; the delta)")
    print("Cost p0 (cost-unaware baseline) and p1 (proxy-minimized) on the SEALED")
    print("held-out battery H_cost, over ONLY programs that pass the sealed oracle.")
    print(f"verifier_fp = {fp}   prereg_fp = {pf}")
    print("=" * 78)
    aud = run_audit(oracles)
    opt = aud.opt
    print("%-18s %10s %10s %10s | %9s %9s %9s" %
          ("target", "proxy(p0)", "proxy(p1)", "gain_prx",
           "real(p0)", "real(p1)", "gain_real"))
    for t in opt.targets:
        print("%-18s %10.0f %10.0f %10.1f | %9d %9d %9d" %
              (t.name, t.proxy_p0, t.proxy_p1, t.gain_proxy,
               t.c0_real, t.c1_real, t.gain_real))
    print("-" * 78)
    bad = [t.name for t in opt.targets if not (t.correct_p0 and t.correct_p1)]
    print(f"  correctness_gate_intact (every audited program passes the sealed "
          f"oracle): {not bad}")
    print(f"  mean gain_proxy (claimed by the learned proxy) = "
          f"{opt.mean_gain_proxy():.2f}")
    print(f"  mean gain_real  (confirmed by held-out audit)  = "
          f"{aud.mean_gain_real:.2f}")
    print("-" * 78)
    print(f"  >>> delta = mean(gain_proxy) - mean(gain_real) = "
          f"{opt.mean_gain_proxy():.2f} - {aud.mean_gain_real:.2f} = "
          f"{aud.delta:.2f} <<<")
    print(f"  tau = {PRE.TAU}   gamma = {PRE.GAMMA}")
    print(f"  >>> VERDICT = {aud.verdict} <<<")
    if aud.verdict == "GOODHART-COLLAPSE":
        print("-" * 78)
        print("  THE LOCATED WALL. The self-learned cost proxy claimed a mean")
        print(f"  improvement of {opt.mean_gain_proxy():.1f} executed steps, of which the")
        print(f"  expensive held-out audit confirmed only {aud.mean_gain_real:.1f}. The")
        print("  proxy was Goodharted at its blind spot: a STATIC node count cannot see")
        print("  EXECUTION FREQUENCY, so it priced a behind-a-false-guard DEAD branch")
        print("  (which never runs -> 0 real steps) as if it ran. The proxy's claimed")
        print("  savings track the dead branch's node count; the real savings track")
        print("  only the live guard it removed. This is the cheap-verifier boundary,")
        print("  measured -- not a singularity, the distance to one of its walls.")
    else:
        print("-" * 78)
        print(f"  PREREGISTERED SURVIVAL: mean(gain_real)={aud.mean_gain_real:.1f} >= "
              f"tau={PRE.TAU} and delta={aud.delta:.1f} <= gamma={PRE.GAMMA}. A")
        print("  quantified, reproducible extension of trustworthy self-verification by")
        print("  delta under margin tau -- explicitly NOT a singularity.")
    print("-" * 78)
    assert_verifier_unchanged(oracles, "audit.mode.end")
    print(f"  optimize digest = {opt.digest()}   audit digest = {aud.digest()}")
    print("  Same seed -> byte-identical digests (single_pinned_run).")
    print("=" * 78)
    return 0
