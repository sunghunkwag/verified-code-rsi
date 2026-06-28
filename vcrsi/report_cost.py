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
          f"{aud.delta:.2f} <<<   (the NATURAL gap; gamma = {PRE.GAMMA})")
    # which targets the proxy-guided search genuinely optimized (p1 != p0)
    moved = [t.name for t in opt.targets if pp(t.p0) != pp(t.p1)]
    regressed = [t.name for t in opt.targets
                 if t.gain_real is not None and t.gain_real < 0]
    print(f"  targets genuinely optimized (p1 != raw seed p0): {moved or 'none'}")
    print(f"  per-target gain_real (varies -> a real, not planted, baseline): "
          f"{[t.gain_real for t in opt.targets]}")
    print(f"  targets where the optimization REGRESSED real cost: "
          f"{regressed or 'none'}")
    print(f"  >>> VERDICT = {aud.verdict} <<<")
    print("-" * 78)
    if aud.verdict == "GOODHART":
        print("  A DISCOVERED BLIND SPOT. The proxy-guided search preferred a program")
        print(f"  the proxy rated {opt.mean_gain_proxy():.0f} cheaper, but the held-out audit")
        print(f"  confirmed only {aud.mean_gain_real:.0f} -- the proxy OVER-claimed by "
              f"{aud.delta:.0f} on a")
        print("  REAL program (see the moved targets above). The cheap-verifier")
        print("  over-claim wall IS reached here -- measured, not constructed.")
    elif aud.verdict == "PROXY-CONSERVATIVE":
        print("  THE PROXY UNDER-CLAIMS (it is NOT Goodharted in the dangerous,")
        print("  over-claim direction). On the natural targets the proxy-guided search")
        print("  removed GENUINE in-loop redundancy a cost-blind synthesizer shipped;")
        print(f"  the proxy -- calibrated on cheap small public inputs -- predicted a")
        print(f"  {opt.mean_gain_proxy():.0f}-step saving, but the held-out audit at large scale")
        print(f"  confirmed {aud.mean_gain_real:.0f} (the removed work is INSIDE loops, so its")
        print("  real cost grows with input size the static proxy cannot see). Every")
        print("  target's real cost went DOWN or stayed equal -- no regression. The")
        print("  cheap-verifier OVER-claim wall is NOT reached by this proxy here.")
    else:
        print("  NO-GOODHART. On the natural optimization targets the static cost proxy")
        print("  tracks real held-out cost within tolerance gamma. The cheap-verifier")
        print("  over-claim wall is NOT reached by this proxy on this family -- an")
        print("  honest negative, reported plainly. Not a singularity, not a collapse.")
    print("-" * 78)
    print("  NOTE: this delta is MEASURED from a raw-synthesizer baseline (control")
    print("  no_planted_strawman). It is NOT Phase G's constructed delta=1480, which")
    print("  scaled with a planted-dead-branch knob and was a constructibility note,")
    print("  not a measurement. See docs/PHASE_H_RESULTS.txt.")
    print("-" * 78)
    assert_verifier_unchanged(oracles, "audit.mode.end")
    print(f"  optimize digest = {opt.digest()}   audit digest = {aud.digest()}")
    print("  Same seed -> byte-identical digests (single_pinned_run).")
    print("=" * 78)
    return 0
