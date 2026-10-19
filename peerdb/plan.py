"""Execution plans: the operator program the model emits.

A plan is a short register program. Each line runs one operator expert and
stores its result in a register (working memory); the last line returns a
register. Example:

    r0 = scan(singer, t1)
    r1 = scan(concert, t2)
    r2 = join(r0, r1, eq(t1.singer_id, t2.singer_id))
    r3 = group(r2, t1.name)
    r4 = sort(r3, desc(count_star()))
    r5 = limit(r4, 1)
    r6 = project(r5, t1.name)
    return r6

Operators (experts): scan, scanreg, join, leftjoin, filter, group, sort, limit,
project, distinct, union, unionall, intersect, except.

Expressions are prefix calls over column references (``alias.column``),
literals (numbers, 'strings', null), positional output references ``pos(i)``
and registers (``scalar(r3)``, ``insub(x, r3)``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Name:
    """A bare identifier: table name, alias, or register (r0, r1, ...)."""

    text: str


@dataclass(frozen=True)
class Col:
    alias: str
    name: str


@dataclass(frozen=True)
class Lit:
    value: object  # int | float | str | None


@dataclass(frozen=True)
class Call:
    fn: str
    args: tuple = field(default_factory=tuple)


@dataclass
class Step:
    dest: int | None  # None for the final ``return``
    op: Call


Plan = list[Step]

_REG = re.compile(r"^r(\d+)$")
_SIMPLE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def reg_index(x) -> int:
    if isinstance(x, Name):
        m = _REG.match(x.text)
        if m:
            return int(m.group(1))
    raise ValueError(f"expected a register, got {x!r}")


# --- printing -----------------------------------------------------------------

def _ident(s: str) -> str:
    if _SIMPLE.match(s):
        return s
    return "`" + s.replace("`", "``") + "`"


def _lit(v) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return str(int(v))
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        r = repr(v)
        return r if ("." in r or "e" in r or "n" in r) else r + ".0"
    return "'" + str(v).replace("'", "''") + "'"


def format_expr(e) -> str:
    if isinstance(e, Name):
        return _ident(e.text)
    if isinstance(e, Col):
        return f"{_ident(e.alias)}.{_ident(e.name)}"
    if isinstance(e, Lit):
        return _lit(e.value)
    if isinstance(e, Call):
        return f"{e.fn}({', '.join(format_expr(a) for a in e.args)})"
    raise TypeError(f"cannot format {e!r}")


def format_plan(plan: Plan) -> str:
    lines = []
    for s in plan:
        if s.dest is None:
            lines.append(f"return {format_expr(s.op.args[0])}")
        else:
            lines.append(f"r{s.dest} = {format_expr(s.op)}")
    return "\n".join(lines)


# --- parsing ------------------------------------------------------------------

_TOKEN = re.compile(
    r"""
    (?P<ws>\s+)
  | (?P<str>'(?:[^']|'')*')
  | (?P<num>-?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?)
  | (?P<qid>`(?:[^`]|``)*`)
  | (?P<id>[^\W\d][\w]*)
  | (?P<punct>[(),.=])
    """,
    re.VERBOSE | re.UNICODE,
)


class PlanSyntaxError(ValueError):
    pass


def _tokenize(text: str):
    pos, out = 0, []
    while pos < len(text):
        m = _TOKEN.match(text, pos)
        if not m:
            raise PlanSyntaxError(f"unexpected character {text[pos]!r} at {pos}")
        pos = m.end()
        kind = m.lastgroup
        if kind == "ws":
            continue
        val = m.group(kind)
        if kind == "str":
            out.append(("str", val[1:-1].replace("''", "'")))
        elif kind == "num":
            out.append(("num", val))
        elif kind == "qid":
            out.append(("id", val[1:-1].replace("``", "`")))
        elif kind == "id":
            out.append(("id", val))
        else:
            out.append(("p", val))
    return out


class _Parser:
    def __init__(self, tokens):
        self.t = tokens
        self.i = 0

    def peek(self, k=0):
        j = self.i + k
        return self.t[j] if j < len(self.t) else (None, None)

    def take(self, kind=None, val=None):
        tok = self.peek()
        if tok[0] is None or (kind and tok[0] != kind) or (val is not None and tok[1] != val):
            raise PlanSyntaxError(f"expected {val or kind}, got {tok[1]!r}")
        self.i += 1
        return tok

    def expr(self):
        kind, val = self.peek()
        if kind == "str":
            self.i += 1
            return Lit(val)
        if kind == "num":
            self.i += 1
            if re.fullmatch(r"-?\d+", val):
                return Lit(int(val))
            return Lit(float(val))
        if kind == "id":
            self.i += 1
            nk, nv = self.peek()
            if nk == "p" and nv == "(":
                self.i += 1
                args = []
                if self.peek() != ("p", ")"):
                    args.append(self.expr())
                    while self.peek() == ("p", ","):
                        self.i += 1
                        args.append(self.expr())
                self.take("p", ")")
                return Call(val, tuple(args))
            if nk == "p" and nv == ".":
                self.i += 1
                _, col = self.take("id")
                return Col(val, col)
            if val == "null":
                return Lit(None)
            return Name(val)
        raise PlanSyntaxError(f"unexpected token {val!r}")


def parse_plan(text: str) -> Plan:
    plan: Plan = []
    for raw in text.strip().splitlines():
        line = raw.strip()
        if not line:
            continue
        p = _Parser(_tokenize(line))
        kind, val = p.peek()
        if kind == "id" and val == "return":
            p.i += 1
            plan.append(Step(None, Call("return", (p.expr(),))))
        else:
            _, dest = p.take("id")
            m = _REG.match(dest)
            if not m:
                raise PlanSyntaxError(f"bad register {dest!r}")
            p.take("p", "=")
            op = p.expr()
            if not isinstance(op, Call):
                raise PlanSyntaxError("operator call expected")
            plan.append(Step(int(m.group(1)), op))
        if p.i != len(p.t):
            raise PlanSyntaxError(f"trailing tokens in line {line!r}")
    if not plan or plan[-1].dest is not None:
        raise PlanSyntaxError("plan must end with return")
    return plan
