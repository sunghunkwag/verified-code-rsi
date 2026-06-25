"""verified-code-rsi: verification-grounded recursive self-improvement of
genuinely multi-step programs (LLM-free).

See README.md for the architecture and the measured deltas. The package layout:

  ir.py           typed IR (programs as data) + library blocks + inlining
  interp.py       resource-budgeted, side-effect-free executor (the physical root)
  tasks.py        the §6A whitelist task suite + reference solutions + the SEALED
                  external emergence held-out set (kept out of SUITE)
  oracle.py       the sealed, hash-pinned correctness oracle (the verifier root)
  complexity.py   the machine-checked complexity floor (§6B)
  search.py       LLM-free stochastic/genetic synthesizer over the IR
  search_oe.py    bottom-up observational-equivalence synthesizer (a 2nd channel)
  library.py      the evolvable policy genome (op weights + mined subroutines)
  prm.py          Phase C M1: the process-reward model (averaged perceptron) + features
  prm_beam.py     Phase C M2: the PRM-guided beam search over program prefixes
  world_model.py  Phase C M3: a learned, honestly-abstaining model of op semantics
  rsi.py          the RSI loop + META-GATE + lineage + the learned-guidance arm
  counterfactual.py  adaptive vs frozen arms (macro library AND learned guidance)
  generator.py    Phase D: ORACLE-BLIND self-generation of new reference programs
                  (the system invents its own self-verifying tasks)
  openended.py    Phase D: the open-ended loop (triple lock + ratcheting frontier)
                  and the EMERGENCE measurement (open-ended vs fixed-suite baseline)
  controls.py     the anti-cheat controls (§4/§5) as runnable tests
  report.py       --mode demo / counterfactual / solve-hard / openended /
                  emergence / test drivers
"""
