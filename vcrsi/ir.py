#!/usr/bin/env python3
"""Typed intermediate representation (IR) for the program synthesizer.

This is the language candidate programs are written in. It is deliberately
richer than the deleted toy stack-VM: typed values (int, bool, string, list,
pair), bounded iteration combinators (map / filter / foldl) that build
auxiliary data structures, conditionals, and *callable library subroutines*
(``call`` nodes) that the recursive-self-improvement loop discovers and reuses.

Design constraints this module is responsible for:

  * The IR is a restricted, side-effect-free expression language. There is NO
    primitive for file access, process spawning, network, ``eval``/``exec`` or
    any host interaction. A candidate program therefore *cannot express* a
    sandbox escape -- containment is structural, not bolted on (see §4.10 of
    the task; the executor lives in ``interp.py`` and adds resource budgets).
  * Programs are plain data (``Node`` trees), so they can be mutated, mined for
    recurring sub-structure, fingerprinted, and serialized deterministically.
  * Complexity of a program is *computed* from its AST + an execution trace
    (``complexity.py``), never asserted -- the floor gate depends on it.

Nothing in this module imports task data, reference solutions or held-out
batteries, so the synthesizer (which builds on this) cannot peek at the oracle.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple


# --------------------------------------------------------------------------- #
# Runtime errors / limits (enforced by the interpreter)                       #
# --------------------------------------------------------------------------- #
class IRError(Exception):
    """A controlled, expected failure during evaluation (type error, bad index,
    unbound variable). Such a program simply scores as wrong; it never crashes
    the harness."""


class StepLimit(IRError):
    """Candidate exceeded its evaluation step budget (e.g. a runaway loop)."""


class AllocLimit(IRError):
    """Candidate tried to build a list/string larger than the allocation cap."""


MAX_LEN = 4000  # hard cap on any list/string a candidate may build at runtime


# --------------------------------------------------------------------------- #
# The AST                                                                       #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Node:
    """One IR expression node.

    op       operator name (a key of ``PRIMS``), a combinator
             ('map'/'filter'/'foldl'/'ifx'), or one of the special leaf/binder
             ops 'lit' | 'arg' | 'var' | 'param' | 'call'.
    rtype    declared return type tag (see ``TYPES``); used to keep search
             type-consistent.
    kids     child expressions.
    const    payload for leaves: literal value ('lit'), arg index ('arg'),
             variable name ('var': 'it'/'acc'), param index ('param'), or the
             callee block name ('call').
    """
    op: str
    rtype: str
    kids: Tuple["Node", ...] = ()
    const: Any = None

    def size(self) -> int:
        return 1 + sum(k.size() for k in self.kids)

    def height(self) -> int:
        return 1 + max((k.height() for k in self.kids), default=0)


# --------------------------------------------------------------------------- #
# Type lattice (used to keep generation / crossover type-consistent).          #
# 'V' is the top type (any value); it unifies with everything.                 #
# --------------------------------------------------------------------------- #
TYPES = ("I", "B", "S", "L", "P", "V")


def tmatch(have: str, want: str) -> bool:
    return have == want or have == "V" or want == "V"


# --------------------------------------------------------------------------- #
# Primitive coercions (raise IRError on the dynamic type mismatches that       #
# loose 'V'-typing inevitably produces)                                        #
# --------------------------------------------------------------------------- #
def _i(x: Any) -> int:
    if isinstance(x, bool):
        return int(x)
    if isinstance(x, int):
        return x
    raise IRError("expected int")


def _l(x: Any) -> list:
    if isinstance(x, list):
        return x
    raise IRError("expected list")


def _s(x: Any) -> str:
    if isinstance(x, str):
        return x
    raise IRError("expected str")


def _p(x: Any) -> tuple:
    if isinstance(x, tuple) and len(x) == 2:
        return x
    raise IRError("expected pair")


def _cap(x: Any) -> Any:
    if isinstance(x, (list, str)) and len(x) > MAX_LEN:
        raise AllocLimit("value too large")
    return x


def _err(msg: str):
    raise IRError(msg)


def _sortkey(x: Any):
    """A total order across every value kind, so ``lsort`` can never raise on
    mixed lists (it just produces a deterministic order)."""
    if isinstance(x, tuple):
        return (3, tuple(_sortkey(e) for e in x))
    if isinstance(x, list):
        return (2, tuple(_sortkey(e) for e in x))
    if isinstance(x, str):
        return (1, (x,))
    if isinstance(x, bool):
        return (0, (int(x),))
    if isinstance(x, int):
        return (0, (x,))
    return (4, (repr(x),))


# --------------------------------------------------------------------------- #
# Primitive registry: name -> (return_type, arg_types, eval_fn).               #
# Combinators and special leaves are NOT here (handled by the interpreter).    #
# Every fn is total or raises IRError; none touches the host environment.      #
# --------------------------------------------------------------------------- #
PRIMS: Dict[str, Tuple[str, Tuple[str, ...], Callable[..., Any]]] = {
    # arithmetic ----------------------------------------------------------- #
    "add":  ("I", ("I", "I"), lambda a, b: _i(a) + _i(b)),
    "sub":  ("I", ("I", "I"), lambda a, b: _i(a) - _i(b)),
    "mul":  ("I", ("I", "I"), lambda a, b: _i(a) * _i(b)),
    "sdiv": ("I", ("I", "I"), lambda a, b: _i(a) // _i(b) if _i(b) != 0 else 0),
    "smod": ("I", ("I", "I"), lambda a, b: _i(a) % _i(b) if _i(b) != 0 else 0),
    "inc":  ("I", ("I",), lambda a: _i(a) + 1),
    "dec":  ("I", ("I",), lambda a: _i(a) - 1),
    "imax": ("I", ("I", "I"), lambda a, b: max(_i(a), _i(b))),
    "imin": ("I", ("I", "I"), lambda a, b: min(_i(a), _i(b))),
    # comparison ----------------------------------------------------------- #
    "eqi":  ("B", ("I", "I"), lambda a, b: _i(a) == _i(b)),
    "lt":   ("B", ("I", "I"), lambda a, b: _i(a) < _i(b)),
    "le":   ("B", ("I", "I"), lambda a, b: _i(a) <= _i(b)),
    "gt":   ("B", ("I", "I"), lambda a, b: _i(a) > _i(b)),
    "eqv":  ("B", ("V", "V"), lambda a, b: a == b),
    # boolean -------------------------------------------------------------- #
    "and":  ("B", ("B", "B"), lambda a, b: bool(a) and bool(b)),
    "or":   ("B", ("B", "B"), lambda a, b: bool(a) or bool(b)),
    "not":  ("B", ("B",), lambda a: not bool(a)),
    # list ----------------------------------------------------------------- #
    "cons":   ("L", ("V", "L"), lambda x, l: _cap([x] + _l(l))),
    "head":   ("V", ("L",), lambda l: _l(l)[0] if _l(l) else _err("head[]")),
    "tail":   ("L", ("L",), lambda l: _l(l)[1:]),
    "llen":   ("I", ("L",), lambda l: len(_l(l))),
    "nth":    ("V", ("L", "I"),
               lambda l, i: _l(l)[_i(i)] if 0 <= _i(i) < len(_l(l)) else _err("nth")),
    "lapp":   ("L", ("L", "L"), lambda a, b: _cap(_l(a) + _l(b))),
    "lrev":   ("L", ("L",), lambda l: list(reversed(_l(l)))),
    "lempty": ("B", ("L",), lambda l: len(_l(l)) == 0),
    "llast":  ("V", ("L",), lambda l: _l(l)[-1] if _l(l) else _err("last[]")),
    "linit":  ("L", ("L",), lambda l: _l(l)[:-1]),
    "lsort":  ("L", ("L",), lambda l: sorted(_l(l), key=_sortkey)),
    "lsingle": ("L", ("V",), lambda x: [x]),
    "lrange": ("L", ("I",), lambda n: list(range(min(max(_i(n), 0), MAX_LEN)))),
    "ltake":  ("L", ("L", "I"), lambda l, n: _l(l)[:max(_i(n), 0)]),
    "ldrop":  ("L", ("L", "I"), lambda l, n: _l(l)[max(_i(n), 0):]),
    # pair ----------------------------------------------------------------- #
    "pair": ("P", ("V", "V"), lambda a, b: (a, b)),
    "fst":  ("V", ("P",), lambda p: _p(p)[0]),
    "snd":  ("V", ("P",), lambda p: _p(p)[1]),
    # string (a string is a genuine text value, parsed/produced structurally) #
    "schars":  ("L", ("S",), lambda s: list(_s(s))),
    "sconcat": ("S", ("L",), lambda l: _cap("".join(_chk_str(x) for x in _l(l)))),
    "srepeat": ("S", ("S", "I"),
                lambda s, n: _cap(_s(s) * min(max(_i(n), 0), MAX_LEN))),
    "slen":    ("I", ("S",), lambda s: len(_s(s))),
    "snth":    ("S", ("S", "I"),
                lambda s, i: _s(s)[_i(i)] if 0 <= _i(i) < len(_s(s)) else _err("snth")),
    "sord":    ("I", ("S",), lambda s: ord(_s(s)[0]) if _s(s) else _err("ord[]")),
    "schr":    ("S", ("I",), lambda i: chr(min(max(_i(i), 0), 0x10FFFF))),
}


def _chk_str(x: Any) -> str:
    if isinstance(x, str):
        return x
    raise IRError("sconcat: non-string element")


COMBINATORS = {"map", "filter", "foldl", "ifx"}
COMB_RTYPE = {"map": "L", "filter": "L", "foldl": "V", "ifx": "V"}


# --------------------------------------------------------------------------- #
# Library subroutines (blocks). A block is a named, parameterised IR fragment  #
# discovered by the RSI loop. It may itself ``call`` earlier blocks -> lineage.#
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Block:
    name: str
    ptypes: Tuple[str, ...]      # parameter types
    body: Node                   # uses Node('param', t, const=i) for params
    rtype: str
    created_round: int = -1      # bookkeeping for lineage / counterfactual
    origin: str = "mined"

    def calls(self) -> List[str]:
        """Names of other blocks this block's body references (direct deps)."""
        out: List[str] = []
        _collect_calls(self.body, out)
        return out


def _collect_calls(n: Node, out: List[str]) -> None:
    if n.op == "call":
        out.append(n.const)
    for k in n.kids:
        _collect_calls(k, out)


# --------------------------------------------------------------------------- #
# Inlining: expand every ``call`` so complexity is measured on the real,       #
# irreducible computation (a block that merely *is* the answer still inlines    #
# to the full multi-step program, so the complexity floor cannot be gamed by    #
# hiding work inside a subroutine).                                             #
# --------------------------------------------------------------------------- #
def inline(n: Node, blocks: Dict[str, Block]) -> Node:
    if n.op == "call":
        blk = blocks[n.const]
        args = tuple(inline(k, blocks) for k in n.kids)
        return _subst_params(inline(blk.body, blocks), args)
    if not n.kids:
        return n
    return Node(n.op, n.rtype, tuple(inline(k, blocks) for k in n.kids), n.const)


def _subst_params(body: Node, args: Tuple[Node, ...]) -> Node:
    if body.op == "param":
        return args[body.const]
    if not body.kids:
        return body
    return Node(body.op, body.rtype,
                tuple(_subst_params(k, args) for k in body.kids), body.const)


# --------------------------------------------------------------------------- #
# Structural helpers used by search and mining                                 #
# --------------------------------------------------------------------------- #
def all_nodes(n: Node, path: Tuple[int, ...] = ()) -> List[Tuple[Tuple[int, ...], Node]]:
    res = [(path, n)]
    for i, k in enumerate(n.kids):
        res += all_nodes(k, path + (i,))
    return res


def replace_at(root: Node, path: Tuple[int, ...], new: Node) -> Node:
    if not path:
        return new
    i = path[0]
    kids = list(root.kids)
    kids[i] = replace_at(kids[i], path[1:], new)
    return Node(root.op, root.rtype, tuple(kids), root.const)


def references_param(n: Node) -> bool:
    if n.op == "param":
        return True
    return any(references_param(k) for k in n.kids)


def references_var(n: Node, names=("it", "acc")) -> bool:
    if n.op == "var" and n.const in names:
        return True
    return any(references_var(k, names) for k in n.kids)


# --------------------------------------------------------------------------- #
# Canonical, deterministic serialization (for fingerprints / dedup / logs)     #
# --------------------------------------------------------------------------- #
def pp(n: Node) -> str:
    if n.op == "lit":
        return repr(n.const)
    if n.op == "arg":
        return f"a{n.const}"
    if n.op == "var":
        return str(n.const)
    if n.op == "param":
        return f"$%d" % n.const
    if n.op == "call":
        return f"{n.const}(" + ", ".join(pp(k) for k in n.kids) + ")"
    if not n.kids:
        return n.op
    return f"{n.op}(" + ", ".join(pp(k) for k in n.kids) + ")"
