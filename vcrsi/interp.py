#!/usr/bin/env python3
"""The executor: a resource-budgeted, side-effect-free interpreter for the IR.

This is one half of the irreducible root (the other is the correctness oracle).
It is the *physical executor*: it turns a program + inputs into an output, under
hard CPU-step, allocation and recursion budgets. Because the IR has no I/O,
process or eval primitives (see ``ir.py``), and because every run is bounded,
a hostile candidate cannot read the held-out files, spawn a process, hang the
harness, or exhaust memory -- it is simply scored as failed (see §4.10).

The interpreter also records an *execution trace depth*: the maximum nesting of
loop iterations / block-call frames actually exercised on a given input. The
complexity floor uses this to reject constant-size straight-line programs
masquerading as multi-step ones.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .ir import (Block, IRError, StepLimit, AllocLimit, Node, PRIMS, MAX_LEN, _l)

# Failures we convert into "this candidate is wrong", never a harness crash.
_CONTROLLED = (IRError, RecursionError, ZeroDivisionError, IndexError,
               TypeError, ValueError, KeyError, OverflowError, AttributeError)

DEFAULT_STEPS = 200_000
MAX_CALL_DEPTH = 60


class RunResult:
    __slots__ = ("value", "error", "steps", "trace_depth", "iters")

    def __init__(self, value, error, steps, trace_depth, iters):
        self.value = value
        self.error = error          # None on success, else exception
        self.steps = steps
        self.trace_depth = trace_depth   # max NESTING of loops/calls
        self.iters = iters               # total loop-body executions + call frames

    @property
    def ok(self) -> bool:
        return self.error is None


class _Ctx:
    __slots__ = ("args", "blocks", "env", "params_stack", "steps", "max_steps",
                 "loop_depth", "max_trace_depth", "call_depth", "iters")

    def __init__(self, args, blocks, max_steps):
        self.args = args
        self.blocks: Dict[str, Block] = blocks or {}
        self.env: Dict[str, Any] = {}
        self.params_stack: List[Tuple[Any, ...]] = []
        self.steps = 0
        self.max_steps = max_steps
        self.loop_depth = 0
        self.max_trace_depth = 0
        self.call_depth = 0
        self.iters = 0              # loop-body executions + call frames exercised


def _ev(node: Node, ctx: _Ctx) -> Any:
    ctx.steps += 1
    if ctx.steps > ctx.max_steps:
        raise StepLimit("step budget exceeded")
    op = node.op

    if op == "lit":
        return node.const
    if op == "arg":
        return ctx.args[node.const]
    if op == "var":
        if node.const not in ctx.env:
            raise IRError("unbound var " + str(node.const))
        return ctx.env[node.const]
    if op == "param":
        if not ctx.params_stack:
            raise IRError("param outside block")
        frame = ctx.params_stack[-1]
        if node.const >= len(frame):
            raise IRError("bad param index")
        return frame[node.const]

    if op == "ifx":
        cond = _ev(node.kids[0], ctx)
        return _ev(node.kids[1], ctx) if bool(cond) else _ev(node.kids[2], ctx)

    if op == "map":
        lst = _l(_ev(node.kids[0], ctx))
        body = node.kids[1]
        out: List[Any] = []
        ctx.loop_depth += 1
        ctx.max_trace_depth = max(ctx.max_trace_depth, ctx.loop_depth)
        save = ctx.env.get("it")
        for e in lst:
            ctx.iters += 1
            ctx.env["it"] = e
            out.append(_ev(body, ctx))
            if len(out) > MAX_LEN:
                raise AllocLimit("map output too large")
        ctx.env["it"] = save
        ctx.loop_depth -= 1
        return out

    if op == "filter":
        lst = _l(_ev(node.kids[0], ctx))
        body = node.kids[1]
        out = []
        ctx.loop_depth += 1
        ctx.max_trace_depth = max(ctx.max_trace_depth, ctx.loop_depth)
        save = ctx.env.get("it")
        for e in lst:
            ctx.iters += 1
            ctx.env["it"] = e
            if bool(_ev(body, ctx)):
                out.append(e)
        ctx.env["it"] = save
        ctx.loop_depth -= 1
        return out

    if op == "foldl":
        lst = _l(_ev(node.kids[0], ctx))
        acc = _ev(node.kids[1], ctx)
        body = node.kids[2]
        ctx.loop_depth += 1
        ctx.max_trace_depth = max(ctx.max_trace_depth, ctx.loop_depth)
        sit, sacc = ctx.env.get("it"), ctx.env.get("acc")
        for e in lst:
            ctx.iters += 1
            ctx.env["it"] = e
            ctx.env["acc"] = acc
            acc = _ev(body, ctx)
        ctx.env["it"], ctx.env["acc"] = sit, sacc
        ctx.loop_depth -= 1
        return acc

    if op == "call":
        blk = ctx.blocks.get(node.const)
        if blk is None:
            raise IRError("unknown block " + str(node.const))
        ctx.call_depth += 1
        if ctx.call_depth > MAX_CALL_DEPTH:
            raise IRError("call depth exceeded")
        argv = tuple(_ev(k, ctx) for k in node.kids)
        ctx.params_stack.append(argv)
        # a block call is a frame; count it toward exercised depth
        ctx.iters += 1
        ctx.loop_depth += 1
        ctx.max_trace_depth = max(ctx.max_trace_depth, ctx.loop_depth)
        try:
            res = _ev(blk.body, ctx)
        finally:
            ctx.params_stack.pop()
            ctx.loop_depth -= 1
            ctx.call_depth -= 1
        return res

    spec = PRIMS.get(op)
    if spec is None:
        raise IRError("unknown op " + str(op))
    vals = [_ev(k, ctx) for k in node.kids]
    return spec[2](*vals)


def run(prog: Node, args: List[Any], blocks: Optional[Dict[str, Block]] = None,
        max_steps: int = DEFAULT_STEPS, env: Optional[Dict[str, Any]] = None
        ) -> RunResult:
    """Execute ``prog`` on ``args``. Never raises on candidate misbehaviour;
    returns a RunResult whose ``error`` is set on any controlled failure.
    ``env`` optionally preseeds loop variables (it/acc) -- used by the bottom-up
    OE synthesizer to evaluate combinator BODIES on probe element values."""
    ctx = _Ctx(args, blocks, max_steps)
    if env:
        ctx.env.update(env)
    try:
        val = _ev(prog, ctx)
        return RunResult(val, None, ctx.steps, ctx.max_trace_depth, ctx.iters)
    except _CONTROLLED as e:
        return RunResult(None, e, ctx.steps, ctx.max_trace_depth, ctx.iters)
