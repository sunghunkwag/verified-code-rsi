"""verified-code-rsi: verification-grounded recursive self-improvement of
genuinely multi-step programs (LLM-free).

See README.md for the architecture and the measured deltas. The package layout:

  ir.py           typed IR (programs as data) + library blocks + inlining
  interp.py       resource-budgeted, side-effect-free executor (the physical root)
  tasks.py        the §6A whitelist task suite + reference solutions
  oracle.py       the sealed, hash-pinned correctness oracle (the verifier root)
  complexity.py   the machine-checked complexity floor (§6B)
  search.py       LLM-free stochastic/genetic synthesizer over the IR
  library.py      the evolvable policy genome (op weights + mined subroutines)
  rsi.py          the recursive-self-improvement loop + META-GATE + lineage
  counterfactual.py  adaptive vs frozen arms at equal budget/seeds
  controls.py     the anti-cheat controls (§4) as runnable tests
  report.py       --mode demo / counterfactual / test drivers
"""
