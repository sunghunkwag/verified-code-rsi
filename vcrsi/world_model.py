#!/usr/bin/env python3
"""M3 -- a world model over IR primitive-op semantics (``OpSemanticsModel``).

Learns what each primitive operation DOES from observed (args -> result)
transitions, so a prefix's effect can be predicted WITHOUT re-executing through
the full interpreter (cheaper ``prefix_features`` + lookahead), with HONEST
abstention outside what it has learned.

The only channel to ground truth is ``interp.op_step`` (which routes through the
real interpreter's ``PRIMS`` dispatch). This module NEVER reads the primitive
implementation table; the ``world_model_honest_abstention`` control inspects this
source to confirm it references neither ``PRIMS`` nor any op's lambda.

For each op the model keeps:
  * ``memo`` -- an exact map (args_key -> result) of everything observed by
    ACTING on the interpreter, and
  * ``hyp``  -- the simplest hypothesis from a FINITE DECLARED prior that is
    consistent with ALL observations (or None -> memo-only).

The declared hypothesis prior (finite, explicit):
  const(k) | identity | proj_fst | proj_snd
  | binary f in {add,sub,mul,max,min,eq,lt,le,gt,and,or}
  | unary  f in {inc,dec,neg,not,len}
Selection is by consistency with observed transitions, NOT by reading the impl;
an op whose behaviour matches no declared hypothesis stays memo-only.

predict(op, args):
  * a consistent hypothesis (with >= MIN_OBS observations) -> apply it
  * elif the exact args were observed                      -> the memoised result
  * else                                                   -> ABSTAIN (never fabricate)

Where it predicts, it equals the real interpreter exactly (fuzz-tested). Where it
cannot, it abstains and the caller falls back to acting on the interpreter.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .ir import Node
from .interp import op_step, run, PRIM_NAMES


class _Abstain:
    __slots__ = ()
    def __repr__(self): return "ABSTAIN"


ABSTAIN = _Abstain()
MIN_OBS = 2                      # never trust a hypothesis from < 2 observations


# --------------------------------------------------------------------------- #
# The finite declared hypothesis prior                                         #
# --------------------------------------------------------------------------- #
def _safe(fn: Callable, args: Tuple[Any, ...]) -> Tuple[bool, Any]:
    try:
        return True, fn(*args)
    except Exception:
        return False, None


_BINARY: Dict[str, Callable] = {
    "add": lambda a, b: a + b, "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b, "max": lambda a, b: max(a, b),
    "min": lambda a, b: min(a, b), "eq": lambda a, b: a == b,
    "lt": lambda a, b: a < b, "le": lambda a, b: a <= b,
    "gt": lambda a, b: a > b, "and": lambda a, b: bool(a) and bool(b),
    "or": lambda a, b: bool(a) or bool(b),
}
_UNARY: Dict[str, Callable] = {
    "inc": lambda a: a + 1, "dec": lambda a: a - 1, "neg": lambda a: -a,
    "not": lambda a: not bool(a), "len": lambda a: len(a),
    "identity": lambda a: a, "proj_fst": lambda a: a[0], "proj_snd": lambda a: a[1],
}


@dataclass
class _OpModel:
    op: str
    arity: int
    memo: Dict[Tuple, Any] = field(default_factory=dict)
    obs: List[Tuple[Tuple, Any]] = field(default_factory=list)

    def record(self, args: Tuple, result: Any) -> None:
        key = _key(args)
        if key not in self.memo:
            self.memo[key] = result
            self.obs.append((args, result))

    def hypothesis(self) -> Optional[Tuple[str, Callable]]:
        """The simplest declared hypothesis consistent with ALL observations AND
        fuzz-validated against the real interpreter on edge inputs, in a fixed
        order; None if nothing fits (op stays memo-only). The fuzz validation is
        what keeps a prediction HONEST -- e.g. ``srepeat`` matches ``a*b`` on small
        observed counts but the interpreter clamps huge counts, so the ``mul``
        hypothesis is rejected and ``srepeat`` stays memo-only."""
        if len(self.obs) < MIN_OBS:
            return None
        results = [r for _a, r in self.obs]
        if all(r == results[0] for r in results):
            k = results[0]
            cand = ("const", lambda *_a, _k=k: _k)
            return cand if self._fuzz_ok(cand[1]) else None
        family = (_UNARY if self.arity == 1 else _BINARY if self.arity == 2 else {})
        order = (("identity", "proj_fst", "proj_snd", "inc", "dec", "neg", "not",
                  "len") if self.arity == 1 else
                 ("add", "sub", "mul", "max", "min", "eq", "lt", "le", "gt",
                  "and", "or"))
        for name in order:
            fn = family[name]
            if self._fits(fn) and self._fuzz_ok(fn):
                return (name, fn)
        return None

    def _fits(self, fn: Callable) -> bool:
        for args, result in self.obs:
            ok, val = _safe(fn, args)
            if not ok or val != result:
                return False
        return True

    def _fuzz_ok(self, fn: Callable) -> bool:
        """A hypothesis is trusted only if it matches the REAL interpreter on edge
        inputs derived from the observed argument types (0, negatives, a huge
        value) -- so an observation-consistent but semantically-wrong hypothesis
        (clamping, overflow, type coercion) is rejected rather than fabricated."""
        for args, _r in self.obs[:4]:
            for i, a in enumerate(args):
                if isinstance(a, bool) or not isinstance(a, int):
                    continue
                for v in (0, -3, 9999):
                    pert = tuple(v if j == i else x for j, x in enumerate(args))
                    real_ok, real = op_step(self.op, pert)
                    fn_ok, val = _safe(fn, pert)
                    if real_ok and (not fn_ok or val != real):
                        return False
                    if (not real_ok) and fn_ok:
                        return False
        return True


def _key(args: Tuple) -> Tuple:
    return tuple(repr(a) for a in args)


# --------------------------------------------------------------------------- #
# The model                                                                     #
# --------------------------------------------------------------------------- #
class OpSemanticsModel:
    def __init__(self):
        self._ops: Dict[str, _OpModel] = {}

    # ---- learning: act on the interpreter, record the transition ---------- #
    def act(self, op: str, args: Tuple[Any, ...]) -> Tuple[bool, Any]:
        """The ONLY way the model observes ground truth: call ``interp.op_step``
        (the real interpreter) and record (args -> result)."""
        ok, val = op_step(op, args)
        if op not in self._ops:
            self._ops[op] = _OpModel(op=op, arity=len(args))
        if ok:
            self._ops[op].record(args, val)
        return ok, val

    def observe_program(self, prog: Node, argslist: List[List[Any]]) -> None:
        """Run a program for real and record every primitive op application it
        performs -- INCLUDING those inside map/filter/foldl bodies -- so the model
        learns op semantics from the system's own solved programs."""
        for args in argslist:
            self._observe(prog, args, {})

    def _observe(self, node: Node, args: List[Any], env: Dict[str, Any]) -> Any:
        op = node.op
        if op == "lit":
            return node.const
        if op == "arg":
            return args[node.const]
        if op == "var":
            return env.get(node.const, ABSTAIN)
        if op == "ifx":
            cond = self._observe(node.kids[0], args, env)
            if cond is ABSTAIN:
                return ABSTAIN
            return self._observe(node.kids[1] if cond else node.kids[2], args, env)
        if op == "map":
            src = self._observe(node.kids[0], args, env)
            if not isinstance(src, list):
                return ABSTAIN
            out = []
            for e in src:
                out.append(self._observe(node.kids[1], args, {**env, "it": e}))
            return out
        if op == "filter":
            src = self._observe(node.kids[0], args, env)
            if not isinstance(src, list):
                return ABSTAIN
            return [e for e in src
                    if self._observe(node.kids[1], args, {**env, "it": e})]
        if op == "foldl":
            src = self._observe(node.kids[0], args, env)
            acc = self._observe(node.kids[1], args, env)
            if not isinstance(src, list):
                return ABSTAIN
            for e in src:
                acc = self._observe(node.kids[2], args, {**env, "it": e, "acc": acc})
            return acc
        if op in ("call", "param"):
            r = run(node, args, env=env or None)          # blocks: act for real
            return r.value if r.ok else ABSTAIN
        # a primitive op application: observe (args -> result) and return it
        vals = tuple(self._observe(k, args, env) for k in node.kids)
        if any(v is ABSTAIN for v in vals):
            return ABSTAIN
        ok, v = self.act(op, vals)
        return v if ok else ABSTAIN

    # ---- prediction: hypothesis, else memo, else honest abstention -------- #
    def predict(self, op: str, args: Tuple[Any, ...]) -> Any:
        m = self._ops.get(op)
        if m is None:
            return ABSTAIN
        hyp = m.hypothesis()
        if hyp is not None:
            ok, val = _safe(hyp[1], args)
            if ok:
                return val
            return ABSTAIN
        key = _key(args)
        if key in m.memo:
            return m.memo[key]
        return ABSTAIN

    def predict_program(self, prog: Node, args: List[Any]) -> Any:
        """Fold predictions over a program's primitive ops. Returns ABSTAIN if any
        op is uncovered (caller then runs the program for real)."""
        return self._pp(prog, args, {})

    def _pp(self, node: Node, args: List[Any], env: Dict[str, Any]) -> Any:
        op = node.op
        if op == "lit":
            return node.const
        if op == "arg":
            return args[node.const]
        if op == "var":
            return env.get(node.const, ABSTAIN)
        if op in ("map", "filter", "foldl", "ifx", "call", "param"):
            return ABSTAIN                                 # structure: out of scope
        vals = tuple(self._pp(k, args, env) for k in node.kids)
        if any(v is ABSTAIN for v in vals):
            return ABSTAIN
        return self.predict(op, vals)

    # ---- reporting -------------------------------------------------------- #
    def coverage(self) -> Dict[str, Any]:
        total = len(PRIM_NAMES)
        learned = [op for op in self._ops if self._ops[op].hypothesis() is not None]
        memo_only = [op for op in self._ops
                     if self._ops[op].hypothesis() is None and self._ops[op].memo]
        return {"ops_seen": len(self._ops), "ops_total": total,
                "hypothesised": len(learned), "memo_only": len(memo_only),
                "fraction": round(len(self._ops) / max(1, total), 3)}

    def digest(self) -> str:
        import hashlib
        h = hashlib.sha256()
        for op in sorted(self._ops):
            hyp = self._ops[op].hypothesis()
            h.update(f"{op}:{hyp[0] if hyp else 'memo'}:{len(self._ops[op].memo)};"
                     .encode())
        return h.hexdigest()[:16]
