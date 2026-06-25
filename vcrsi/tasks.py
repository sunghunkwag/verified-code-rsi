#!/usr/bin/env python3
"""The task suite.

Every task is drawn from the APPROVED WHITELIST (§6A) of genuinely multi-step
programming problems whose *input is structured* (a string to parse, a list of
pairs/intervals, an adjacency structure) -- never a flat integer array reduced
to a scalar. Each task ships:

  * a public SPEC and a few public training examples (all the synthesizer sees);
  * a reference solution written in the IR itself (used ONLY to (a) machine-check
    the complexity floor and (b) generate the sealed held-out battery -- the
    synthesizer never receives it);
  * deterministic input generators (small inputs for public examples, larger /
    unseen sizes for the held-out battery).

WHITELIST FAMILY IDS (closed list):
  1 parsing/interpreting   2 encoding transforms (round-trip)
  3 graph/structure        4 interval/sequence restructuring
  5 small state machines

To add a task family you must extend §6A in the prompt first; this module may
not invent its own families.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Tuple

from .ir import Node

# Banned-forever scalar reductions of flat integer arrays (the deleted toy
# domain). Asserted against in the test suite so a toy cannot creep back in.
BANNED_NAMES = {"sum", "count", "max", "min", "mean", "parity", "sum_of_powers",
                "second_largest", "argmax", "longest_increasing_run"}


# --------------------------------------------------------------------------- #
# Tiny DSL for writing reference solutions as IR trees                          #
# --------------------------------------------------------------------------- #
def lit(v: Any) -> Node:
    if isinstance(v, bool):
        t = "B"
    elif isinstance(v, int):
        t = "I"
    elif isinstance(v, str):
        t = "S"
    elif isinstance(v, list):
        t = "L"
    elif isinstance(v, tuple):
        t = "P"
    else:
        t = "V"
    return Node("lit", t, const=v)


def arg(i: int, t: str) -> Node:
    return Node("arg", t, const=i)


def it() -> Node:
    return Node("var", "V", const="it")


def acc() -> Node:
    return Node("var", "V", const="acc")


# return-type lookup so the DSL builder stamps the right rtype
from .ir import PRIMS, COMB_RTYPE


def b(op: str, *kids: Node) -> Node:
    if op in COMB_RTYPE:
        rt = COMB_RTYPE[op]
    elif op in PRIMS:
        rt = PRIMS[op][0]
    else:
        rt = "V"
    return Node(op, rt, tuple(kids))


# --------------------------------------------------------------------------- #
# Task definition                                                               #
# --------------------------------------------------------------------------- #
@dataclass
class Task:
    name: str
    family: int                       # whitelist family id (1..5)
    spec: str
    arg_types: Tuple[str, ...]
    out_type: str
    reference: Node                   # IR reference solution (SEALED from search)
    gen_input: Callable[[random.Random, int], Tuple[Any, ...]]
    public_scale: int = 3
    holdout_scale: int = 9            # larger / unseen sizes
    n_public: int = 6
    n_holdout: int = 24
    roundtrip_with: Optional[str] = None   # family-2 identity partner
    note: str = ""
    group: str = ""                   # structural-shape family (for transfer §4)


# --------------------------------------------------------------------------- #
# Input generators                                                              #
# --------------------------------------------------------------------------- #
_ALPHA = "abcde"


def _gen_pairs(rng: random.Random, scale: int) -> Tuple[Any, ...]:
    k = rng.randint(max(2, scale - 1), scale + 2)
    pairs = [(rng.choice(_ALPHA), rng.randint(1, 4)) for _ in range(k)]
    return (pairs,)


def _gen_str(rng: random.Random, scale: int) -> Tuple[Any, ...]:
    k = rng.randint(max(2, scale), scale + 3)
    s = "".join(rng.choice(_ALPHA) for _ in range(k))
    return (s,)


def _gen_str_shift(rng: random.Random, scale: int) -> Tuple[Any, ...]:
    k = rng.randint(max(2, scale), scale + 3)
    s = "".join(rng.choice(_ALPHA) for _ in range(k))
    shift = rng.randint(1, 5)
    return (s, shift)


def _gen_pairs_int(rng: random.Random, scale: int) -> Tuple[Any, ...]:
    k = rng.randint(max(2, scale - 1), scale + 2)
    pairs = [(rng.randint(0, 9), rng.randint(1, 4)) for _ in range(k)]
    return (pairs,)


def _gen_intervals(rng: random.Random, scale: int) -> Tuple[Any, ...]:
    k = rng.randint(max(2, scale - 1), scale + 2)
    ivs = []
    for _ in range(k):
        a = rng.randint(0, 18)
        bb = a + rng.randint(0, 6)
        ivs.append((a, bb))
    return (ivs,)


def _gen_brackets(rng: random.Random, scale: int) -> Tuple[Any, ...]:
    # random but mostly-shallow paren strings, mix of valid/invalid
    k = rng.randint(max(2, scale), scale + 4)
    s = "".join(rng.choice("()") for _ in range(k))
    return (s,)


def _gen_bytecode(rng: random.Random, scale: int) -> Tuple[Any, ...]:
    # program = list of (opcode, operand): 0=PUSH n, 1=ADD, 2=MUL
    n = rng.randint(max(3, scale), scale + 3)
    prog = [(0, rng.randint(1, 5))]  # always start with a push
    stack = 1
    for _ in range(n):
        if stack >= 2 and rng.random() < 0.5:
            prog.append((rng.choice([1, 2]), 0))
            stack -= 1
        else:
            prog.append((0, rng.randint(1, 5)))
            stack += 1
    return (prog,)


def _gen_charlist(rng: random.Random, scale: int) -> Tuple[Any, ...]:
    k = rng.randint(max(2, scale), scale + 3)
    return ([rng.choice(_ALPHA) for _ in range(k)],)


# -- Phase B generators: int-pair families (intervals) ---------------------- #
def _gen_ivk(rng: random.Random, scale: int) -> Tuple[Any, ...]:
    """A list of (a, a+w) intervals (a may be negative) plus one int knob."""
    k = rng.randint(max(2, scale - 1), scale + 2)
    ivs = [( (a := rng.randint(-6, 18)), a + rng.randint(0, 7)) for _ in range(k)]
    return (ivs, rng.randint(1, 4))


def _gen_ivk2(rng: random.Random, scale: int) -> Tuple[Any, ...]:
    k = rng.randint(max(2, scale - 1), scale + 2)
    ivs = [( (a := rng.randint(-6, 18)), a + rng.randint(0, 7)) for _ in range(k)]
    lo = rng.randint(-4, 6)
    return (ivs, lo, lo + rng.randint(2, 10))


def _gen_iv(rng: random.Random, scale: int) -> Tuple[Any, ...]:
    k = rng.randint(max(2, scale - 1), scale + 2)
    ivs = [( (a := rng.randint(-6, 18)), a + rng.randint(0, 7)) for _ in range(k)]
    return (ivs,)


# --------------------------------------------------------------------------- #
# Reference solutions + task list                                              #
# --------------------------------------------------------------------------- #
def _ref_rle_decode() -> Node:
    # sconcat(map(a0, srepeat(fst(it), snd(it))))
    return b("sconcat", b("map", arg(0, "L"),
                          b("srepeat", b("fst", it()), b("snd", it()))))


def _ref_rle_decode_rev() -> Node:
    # decode with pair order reversed first
    return b("sconcat", b("map", b("lrev", arg(0, "L")),
                          b("srepeat", b("fst", it()), b("snd", it()))))


def _ref_caesar_encode() -> Node:
    # sconcat(map(schars(a0), schr(add(sord(it), a1))))
    return b("sconcat", b("map", b("schars", arg(0, "S")),
                          b("schr", b("add", b("sord", it()), arg(1, "I")))))


def _ref_caesar_decode() -> Node:
    return b("sconcat", b("map", b("schars", arg(0, "S")),
                          b("schr", b("sub", b("sord", it()), arg(1, "I")))))


def _ref_interleave_pairs() -> Node:
    # flatten a list of pairs into a flat list: [a,b, c,d, ...]
    # foldl(a0, [], lapp(acc, lapp(lsingle(fst(it)), lsingle(snd(it)))))
    return b("foldl", arg(0, "L"), lit([]),
             b("lapp", acc(),
               b("lapp", b("lsingle", b("fst", it())),
                 b("lsingle", b("snd", it())))))


def _ref_rle_decode_caps() -> Node:
    # decode, then shift every produced char by +1 (codec composition):
    # sconcat(map(schars(sconcat(map(a0, srepeat(fst,snd)))), schr(inc(sord(it)))))
    decode = b("map", arg(0, "L"), b("srepeat", b("fst", it()), b("snd", it())))
    return b("sconcat", b("map", b("schars", b("sconcat", decode)),
                          b("schr", b("inc", b("sord", it())))))


def _expand() -> Node:
    # map(a0, srepeat(fst(it), snd(it)))  -- "expand every (char,count) pair"
    return b("map", arg(0, "L"), b("srepeat", b("fst", it()), b("snd", it())))


def _expand_of(src: Node) -> Node:
    return b("map", src, b("srepeat", b("fst", it()), b("snd", it())))


def _ref_rle_decode_sorted() -> Node:
    # sort the pairs, then decode
    return b("sconcat", _expand_of(b("lsort", arg(0, "L"))))


def _ref_rle_decode_twice() -> Node:
    # decoded string concatenated with itself
    return b("sconcat", b("lapp", _expand(), _expand()))


def _ref_rle_decode_palindrome() -> Node:
    # decoded run-list followed by the reverse of the run-list
    return b("sconcat", b("lapp", _expand(), b("lrev", _expand())))


def _ref_rle_decode_rev_twice() -> Node:
    # reverse-then-decode, twice (its best solution reuses a reverse-decode block)
    rd = _expand_of(b("lrev", arg(0, "L")))
    return b("sconcat", b("lapp", rd, rd))


def _ref_rle_rev_palindrome() -> Node:
    # reverse-decode the run-list, then append the CHARACTER-reverse of that
    # string. The reverse-decoded string appears twice, so a solution without a
    # reverse-decode subroutine is large; one WITH such a subroutine (itself
    # built on the expand subroutine) is small -> drives a depth-2 lineage.
    rd = b("sconcat", _expand_of(b("lrev", arg(0, "L"))))
    return b("sconcat", b("lapp", b("schars", rd), b("lrev", b("schars", rd))))


def _ref_rle_rev_palindrome_twice() -> Node:
    # the reverse-decode palindrome, output twice -- so deep that flat search
    # (even with learned weights) cannot reach it; only a reverse-decode-shaped
    # subroutine (itself built on the expand atom) makes it small. Drives the
    # depth-2 lineage: a block whose body calls an earlier block, used later.
    rd = b("sconcat", _expand_of(b("lrev", arg(0, "L"))))
    rp = b("sconcat", b("lapp", b("schars", rd), b("lrev", b("schars", rd))))
    return b("sconcat", b("lapp", b("schars", rp), b("schars", rp)))


def _ref_merge_intervals() -> Node:
    last = b("llast", acc())
    overlap = b("le", b("fst", it()), b("snd", last))
    newend = b("ifx", b("gt", b("snd", it()), b("snd", last)),
               b("snd", it()), b("snd", last))
    merged = b("lapp", b("linit", acc()),
               b("lsingle", b("pair", b("fst", last), newend)))
    appended = b("lapp", acc(), b("lsingle", it()))
    body = b("ifx", b("lempty", acc()),
             b("lsingle", it()),
             b("ifx", overlap, merged, appended))
    return b("foldl", b("lsort", arg(0, "L")), lit([]), body)


def _ref_bytecode_interp() -> Node:
    # interpret PUSH/ADD/MUL bytecode; result = top of final stack
    # acc = stack (list). step: if op==0 push operand; if op==1 add top2; else mul
    op = b("fst", it())
    operand = b("snd", it())
    top = b("head", acc())
    nxt = b("head", b("tail", acc()))
    rest = b("tail", b("tail", acc()))
    push = b("cons", operand, acc())
    addv = b("cons", b("add", top, nxt), rest)
    mulv = b("cons", b("mul", top, nxt), rest)
    body = b("ifx", b("eqi", op, lit(0)), push,
             b("ifx", b("eqi", op, lit(1)), addv, mulv))
    return b("head", b("foldl", arg(0, "L"), lit([]), body))


def _ref_bracket_depths() -> Node:
    # running nesting depth after each char of a '()' string
    cur = b("ifx", b("lempty", acc()), lit(0), b("llast", acc()))
    delta = b("ifx", b("eqv", it(), lit("(")), lit(1), lit(-1))
    return b("foldl", b("schars", arg(0, "S")), lit([]),
             b("lapp", acc(), b("lsingle", b("add", cur, delta))))


# -- Phase B references: genuinely distinct SOLVABLE families --------------- #
# F2 "interval": int-pair -> int-pair, map combinator
def _ref_shift_intervals() -> Node:
    return b("map", arg(0, "L"),
             b("pair", b("add", b("fst", it()), arg(1, "I")),
               b("add", b("snd", it()), arg(1, "I"))))


def _ref_widen_intervals() -> Node:
    return b("map", arg(0, "L"),
             b("pair", b("sub", b("fst", it()), arg(1, "I")),
               b("add", b("snd", it()), arg(1, "I"))))


def _ref_clamp_low() -> Node:
    return b("map", arg(0, "L"),
             b("pair", b("imax", b("fst", it()), lit(0)),
               b("imax", b("snd", it()), lit(0))))


# F3 "select": int-pair filter combinator
def _ref_keep_wide() -> Node:
    return b("filter", arg(0, "L"),
             b("gt", b("sub", b("snd", it()), b("fst", it())), arg(1, "I")))


def _ref_drop_short() -> Node:
    return b("filter", arg(0, "L"),
             b("le", b("sub", b("snd", it()), b("fst", it())), arg(1, "I")))


def _ref_keep_band() -> Node:
    return b("filter", arg(0, "L"),
             b("and", b("le", arg(1, "I"), b("fst", it())),
               b("le", b("snd", it()), arg(2, "I"))))


# F4 "project": int-pair -> int (scalar) list, map combinator
def _ref_scaled_widths() -> Node:
    return b("map", arg(0, "L"),
             b("mul", b("sub", b("snd", it()), b("fst", it())), arg(1, "I")))


def _ref_midpoints() -> Node:
    return b("map", arg(0, "L"),
             b("sdiv", b("add", b("fst", it()), b("snd", it())), lit(2)))


def _ref_clamped_widths() -> Node:
    return b("map", arg(0, "L"),
             b("imax", b("sub", b("snd", it()), b("fst", it())), arg(1, "I")))


SUITE: List[Task] = [
    # --- family 2: encoding transforms (round-trip) ----------------------- #
    Task("rle_decode", 2,
         "Decode a run-length list of (char,count) pairs into the expanded string.",
         ("L",), "S", _ref_rle_decode(), _gen_pairs,
         roundtrip_with="rle_encode",
         note="easiest multi-step task; reachable baseline"),
    Task("rle_decode_rev", 2,
         "Run-length decode, but expand the pairs in reverse order first.",
         ("L",), "S", _ref_rle_decode_rev(), _gen_pairs),
    Task("rle_decode_sorted", 2,
         "Sort the (char,count) pairs, then run-length decode them.",
         ("L",), "S", _ref_rle_decode_sorted(), _gen_pairs),
    Task("rle_decode_twice", 2,
         "Run-length decode and output the decoded string concatenated twice.",
         ("L",), "S", _ref_rle_decode_twice(), _gen_pairs),
    Task("rle_decode_palindrome", 2,
         "Run-length decode, then append the run-list expanded in reverse.",
         ("L",), "S", _ref_rle_decode_palindrome(), _gen_pairs),
    Task("rle_decode_rev_twice", 2,
         "Reverse-then-decode, output the result twice (reuses reverse-decode).",
         ("L",), "S", _ref_rle_decode_rev_twice(), _gen_pairs),
    Task("rle_rev_palindrome", 2,
         "Reverse-decode the run-list, then append its character-reverse.",
         ("L",), "S", _ref_rle_rev_palindrome(), _gen_pairs,
         note="deep composition: needs a reverse-decode block built on expand"),
    Task("rle_rev_palindrome_twice", 2,
         "The reverse-decode palindrome, output twice.",
         ("L",), "S", _ref_rle_rev_palindrome_twice(), _gen_pairs,
         note="very deep: only reachable via a block built on an earlier block"),
    Task("caesar_encode", 2,
         "Caesar substitution codec: shift every character up by arg1 codepoints.",
         ("S", "I"), "S", _ref_caesar_encode(), _gen_str_shift,
         roundtrip_with="caesar_decode"),
    Task("caesar_decode", 2,
         "Inverse Caesar codec: shift every character down by arg1 codepoints.",
         ("S", "I"), "S", _ref_caesar_decode(), _gen_str_shift,
         roundtrip_with="caesar_encode"),
    Task("rle_decode_shift1", 2,
         "Run-length decode then apply a +1 substitution to every output char.",
         ("L",), "S", _ref_rle_decode_caps(), _gen_pairs,
         note="composes the decode pattern with a substitution -> block reuse"),
    # --- family 4: interval / sequence restructuring ---------------------- #
    Task("interleave_pairs", 4,
         "Flatten a list of pairs into a flat sequence [a0,b0,a1,b1,...].",
         ("L",), "L", _ref_interleave_pairs(), _gen_pairs_int),
    Task("merge_intervals", 4,
         "Sort intervals and merge all overlapping ones into a minimal list.",
         ("L",), "L", _ref_merge_intervals(), _gen_intervals,
         note="hard frontier: ~30-node reference solution"),
    Task("bracket_depths", 4,
         "Given a '()' string, output the running nesting depth after each char.",
         ("S",), "L", _ref_bracket_depths(), _gen_brackets,
         note="hard frontier: stateful scan"),
    # --- family 1: parsing / interpreting --------------------------------- #
    Task("bytecode_interp", 1,
         "Interpret a tiny stack bytecode (PUSH/ADD/MUL); return the final top.",
         ("L",), "V", _ref_bytecode_interp(), _gen_bytecode,
         note="hard frontier: stack-machine interpreter"),

    # === Phase B: genuinely distinct SOLVABLE families (for transfer §4) === #
    # F2 interval: int-pair -> int-pair (map)
    Task("shift_intervals", 4, "Shift every interval (a,b) by arg1: (a+k,b+k).",
         ("L", "I"), "L", _ref_shift_intervals(), _gen_ivk),
    Task("widen_intervals", 4, "Widen every interval (a,b) to (a-k, b+k).",
         ("L", "I"), "L", _ref_widen_intervals(), _gen_ivk),
    Task("clamp_low", 4, "Clamp both endpoints of every interval up to >= 0.",
         ("L",), "L", _ref_clamp_low(), _gen_iv),
    # F3 select: int-pair filter
    Task("keep_wide", 4, "Keep only intervals strictly wider than arg1.",
         ("L", "I"), "L", _ref_keep_wide(), _gen_ivk),
    Task("drop_short", 4, "Keep only intervals whose width is <= arg1.",
         ("L", "I"), "L", _ref_drop_short(), _gen_ivk),
    Task("keep_band", 3, "Keep intervals lying within the band [arg1, arg2].",
         ("L", "I", "I"), "L", _ref_keep_band(), _gen_ivk2),
    # F4 project: int-pair -> int (map)
    Task("scaled_widths", 4, "List of interval widths each multiplied by arg1.",
         ("L", "I"), "L", _ref_scaled_widths(), _gen_ivk),
    Task("midpoints", 4, "List of interval midpoints (a+b)/2.",
         ("L",), "L", _ref_midpoints(), _gen_iv),
    Task("clamped_widths", 4, "List of interval widths each clamped up to >= arg1.",
         ("L", "I"), "L", _ref_clamped_widths(), _gen_ivk),
]

# Structural-shape family ("group") of each task -- for transfer (§4). This is a
# finer, behaviour-based grouping than the §6A whitelist 'family' number.
_GROUP_OF = {
    "rle_decode": "seqcode", "rle_decode_rev": "seqcode",
    "rle_decode_sorted": "seqcode", "rle_decode_twice": "seqcode",
    "rle_decode_palindrome": "seqcode", "rle_decode_rev_twice": "seqcode",
    "rle_rev_palindrome": "seqcode", "rle_rev_palindrome_twice": "seqcode",
    "rle_decode_shift1": "seqcode", "caesar_encode": "codec",
    "caesar_decode": "codec", "interleave_pairs": "interval",
    "merge_intervals": "interval", "bracket_depths": "scan",
    "bytecode_interp": "parse",
    "shift_intervals": "interval", "widen_intervals": "interval",
    "clamp_low": "interval", "keep_wide": "select", "drop_short": "select",
    "keep_band": "select", "scaled_widths": "project",
    "midpoints": "project", "clamped_widths": "project",
}
for _t in SUITE:
    _t.group = _GROUP_OF.get(_t.name, "misc")

SUITE_BY_NAME = {t.name: t for t in SUITE}

# The transfer experiment (§4) operates on a balanced set of SOLVABLE families;
# each is genuinely distinct (data type / combinator / output structure). 'parse'
# (fold-based) is included in the suite for diversity but is OPEN, so it cannot
# host adopted solutions and is excluded from the solvable transfer matrix.
TRANSFER_FAMILIES = {
    "seqcode":  ["rle_decode", "rle_decode_rev"],     # string, map  (OE-solvable)
    "interval": ["shift_intervals", "clamp_low"],     # int-pair -> int-pair, map
    "select":   ["keep_wide", "drop_short"],          # int-pair filter
    "project":  ["clamped_widths", "scaled_widths"],  # int-pair -> int, map
}
