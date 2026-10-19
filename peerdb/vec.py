"""Typed value vectors: the in-memory form of a column inside the data weights.

Every value is stored as a tensor row. A column can hold a mix of SQLite
storage classes, so each row carries its class:

    cls   0 = NULL, 1 = numeric, 2 = text
    num   float64 value for numeric rows
    isint True when the numeric value is an integer (then ``ival`` is exact)
    ival  int64 value for integer rows
    code  index into the database string table for text rows (-1 otherwise)

Comparisons follow SQLite's ordering: NULL < numeric < text, text compared by
code point (BINARY collation).
"""

from __future__ import annotations

import math
import re
from bisect import bisect_left
from dataclasses import dataclass

import torch

NULL, NUM, TEXT = 0, 1, 2


class Unsupported(Exception):
    """Raised for SQL features the executor does not implement.

    The system reports these as errors instead of guessing, so anything it
    does answer is exact.
    """


class Strings:
    """Per-database string table. Text values are stored as integer codes."""

    def __init__(self):
        self.values: list[str] = []
        self.index: dict[str, int] = {}
        self._rank: torch.Tensor | None = None
        self._sorted: list[str] | None = None
        self._numeric: tuple | None = None

    def code(self, s: str) -> int:
        c = self.index.get(s)
        if c is None:
            c = len(self.values)
            self.values.append(s)
            self.index[s] = c
            self._rank = None
            self._sorted = None
            self._numeric = None
        return c

    def rank(self) -> torch.Tensor:
        """Rank of every code in BINARY collation order."""
        if self._rank is None:
            order = sorted(range(len(self.values)), key=lambda i: self.values[i])
            rank = torch.empty(len(self.values), dtype=torch.int64)
            rank[torch.tensor(order, dtype=torch.int64)] = torch.arange(len(order), dtype=torch.int64)
            self._rank = rank
        return self._rank

    def numeric(self):
        """Per code: whether the string is a well-formed number, and its value."""
        if self._numeric is None:
            n = len(self.values)
            ok = torch.zeros(n, dtype=torch.bool)
            num = torch.zeros(n, dtype=torch.float64)
            isint = torch.zeros(n, dtype=torch.bool)
            ival = torch.zeros(n, dtype=torch.int64)
            for i, s in enumerate(self.values):
                parsed = parse_number(s)
                if parsed is not None:
                    ok[i] = True
                    num[i] = float(parsed)
                    if isinstance(parsed, int):
                        isint[i] = True
                        ival[i] = parsed
            self._numeric = (ok, num, isint, ival)
        return self._numeric


_INT_RE = re.compile(r"^\s*[+-]?\d+\s*$")
_REAL_RE = re.compile(r"^\s*[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?\s*$")


def parse_number(s: str):
    """SQLite NUMERIC affinity conversion of a string, or None."""
    if _INT_RE.match(s):
        v = int(s)
        if -(2**63) <= v < 2**63:
            return v
        return float(v)
    if _REAL_RE.match(s):
        f = float(s)
        if f.is_integer() and abs(f) < 2**63 and "e" not in s.lower():
            return int(f)
        return f
    return None


def format_real(f: float) -> str:
    """Approximates SQLite's text rendering of a REAL value."""
    if math.isinf(f):
        return "Inf" if f > 0 else "-Inf"
    s = "%.15g" % f
    if "e" in s:
        mant, exp = s.split("e")
        if "." not in mant:
            mant += ".0"
        sign = exp[0]
        digits = exp[1:].lstrip("0") or "0"
        return f"{mant}e{sign}{digits.zfill(2)}"
    if "." not in s and "n" not in s:
        s += ".0"
    return s


@dataclass
class Vec:
    cls: torch.Tensor
    num: torch.Tensor
    isint: torch.Tensor
    ival: torch.Tensor
    code: torch.Tensor

    def __len__(self):
        return self.cls.shape[0]

    @staticmethod
    def empty(n: int) -> "Vec":
        return Vec(
            torch.zeros(n, dtype=torch.int8),
            torch.zeros(n, dtype=torch.float64),
            torch.zeros(n, dtype=torch.bool),
            torch.zeros(n, dtype=torch.int64),
            torch.full((n,), -1, dtype=torch.int64),
        )

    @staticmethod
    def from_python(values: list, strings: Strings) -> "Vec":
        n = len(values)
        cls, num, isint, ival, code = [0] * n, [0.0] * n, [False] * n, [0] * n, [-1] * n
        for i, v in enumerate(values):
            if v is None:
                continue
            if isinstance(v, bool):
                v = int(v)
            if isinstance(v, int):
                if not -(2**63) <= v < 2**63:
                    raise Unsupported("integer out of int64 range")
                cls[i], num[i], isint[i], ival[i] = NUM, float(v), True, v
            elif isinstance(v, float):
                if math.isnan(v):
                    continue
                cls[i], num[i] = NUM, v
            elif isinstance(v, str):
                cls[i], code[i] = TEXT, strings.code(v)
            elif isinstance(v, (bytes, bytearray, memoryview)):
                raise Unsupported("BLOB values")
            else:
                raise Unsupported(f"value type {type(v).__name__}")
        return Vec(
            torch.tensor(cls, dtype=torch.int8),
            torch.tensor(num, dtype=torch.float64),
            torch.tensor(isint, dtype=torch.bool),
            torch.tensor(ival, dtype=torch.int64),
            torch.tensor(code, dtype=torch.int64),
        )

    @staticmethod
    def scalar(v, strings: Strings) -> "Vec":
        return Vec.from_python([v], strings)

    def take(self, idx: torch.Tensor) -> "Vec":
        return Vec(self.cls[idx], self.num[idx], self.isint[idx], self.ival[idx], self.code[idx])

    def expand(self, n: int) -> "Vec":
        if len(self) == n:
            return self
        if len(self) != 1:
            raise ValueError("cannot broadcast vector")
        return Vec(self.cls.expand(n), self.num.expand(n), self.isint.expand(n), self.ival.expand(n), self.code.expand(n))

    @staticmethod
    def cat(vecs: list["Vec"]) -> "Vec":
        return Vec(
            torch.cat([v.cls for v in vecs]),
            torch.cat([v.num for v in vecs]),
            torch.cat([v.isint for v in vecs]),
            torch.cat([v.ival for v in vecs]),
            torch.cat([v.code for v in vecs]),
        )

    def to_python(self, strings: Strings) -> list:
        out = []
        cls, num, isint, ival, code = (
            self.cls.tolist(), self.num.tolist(), self.isint.tolist(), self.ival.tolist(), self.code.tolist(),
        )
        for i in range(len(cls)):
            if cls[i] == NULL:
                out.append(None)
            elif cls[i] == NUM:
                out.append(ival[i] if isint[i] else num[i])
            else:
                out.append(strings.values[code[i]])
        return out


@dataclass
class Bool:
    """Three-valued logic result (TRUE / FALSE / NULL)."""

    val: torch.Tensor
    null: torch.Tensor

    def __len__(self):
        return self.val.shape[0]

    def is_true(self) -> torch.Tensor:
        return self.val & ~self.null

    def to_vec(self) -> Vec:
        n = len(self)
        v = Vec.empty(n)
        cls = torch.where(self.null, torch.tensor(NULL, dtype=torch.int8), torch.tensor(NUM, dtype=torch.int8))
        ival = self.val.to(torch.int64)
        return Vec(cls, ival.to(torch.float64), ~self.null, ival, v.code)


def as_bool(x) -> Bool:
    if isinstance(x, Bool):
        return x
    # SQLite: numeric non-zero is true; text is converted to a number first.
    null = x.cls == NULL
    if (x.cls == TEXT).any():
        raise Unsupported("text used as a boolean")
    return Bool((x.num != 0) & ~null, null)


# --- affinity -------------------------------------------------------------

def affinity_of(decl: str | None) -> str:
    """SQLite column affinity from a declared type."""
    t = (decl or "").upper()
    if "INT" in t:
        return "INTEGER"
    if "CHAR" in t or "CLOB" in t or "TEXT" in t:
        return "TEXT"
    if "BLOB" in t or t == "":
        return "BLOB"
    if "REAL" in t or "FLOA" in t or "DOUB" in t:
        return "REAL"
    return "NUMERIC"


NUMERIC_AFF = {"INTEGER", "REAL", "NUMERIC"}


def apply_numeric(v: Vec, strings: Strings) -> Vec:
    """Convert text rows that look like numbers into numeric rows."""
    is_text = v.cls == TEXT
    if not is_text.any():
        return v
    ok, num, isint, ival = strings.numeric()
    code = v.code.clamp(min=0)
    conv = is_text & ok[code]
    return Vec(
        torch.where(conv, torch.tensor(NUM, dtype=torch.int8), v.cls),
        torch.where(conv, num[code], v.num),
        torch.where(conv, isint[code], v.isint),
        torch.where(conv, ival[code], v.ival),
        torch.where(conv, torch.tensor(-1, dtype=torch.int64), v.code),
    )


def apply_text(v: Vec, strings: Strings) -> Vec:
    """Convert numeric rows into their text rendering."""
    is_num = v.cls == NUM
    if not is_num.any():
        return v
    idx = torch.nonzero(is_num).flatten().tolist()
    code = v.code.clone()
    nums, isints, ivals = v.num.tolist(), v.isint.tolist(), v.ival.tolist()
    for i in idx:
        s = str(ivals[i]) if isints[i] else format_real(nums[i])
        code[i] = strings.code(s)
    return Vec(
        torch.where(is_num, torch.tensor(TEXT, dtype=torch.int8), v.cls),
        torch.where(is_num, torch.tensor(0.0, dtype=torch.float64), v.num),
        torch.where(is_num, torch.tensor(False), v.isint),
        torch.where(is_num, torch.tensor(0, dtype=torch.int64), v.ival),
        code,
    )


def coerce_pair(a: Vec, aff_a: str | None, b: Vec, aff_b: str | None, strings: Strings):
    """SQLite's affinity rules applied to the operands of a comparison."""
    a_num, b_num = aff_a in NUMERIC_AFF, aff_b in NUMERIC_AFF
    if a_num and not b_num:
        b = apply_numeric(b, strings)
    elif b_num and not a_num:
        a = apply_numeric(a, strings)
    elif aff_a == "TEXT" and aff_b is None:
        b = apply_text(b, strings)
    elif aff_b == "TEXT" and aff_a is None:
        a = apply_text(a, strings)
    return a, b


def bcast_len(*lens: int) -> int:
    """Common length for elementwise ops; length-1 operands broadcast."""
    others = {n for n in lens if n != 1}
    if len(others) > 1:
        raise ValueError(f"length mismatch {sorted(others)}")
    return others.pop() if others else 1


def compare(a: Vec, b: Vec, strings: Strings) -> tuple[torch.Tensor, torch.Tensor]:
    """Elementwise three-way comparison. Returns (sign in {-1,0,1}, null mask)."""
    n = bcast_len(len(a), len(b))
    a, b = a.expand(n), b.expand(n)
    null = (a.cls == NULL) | (b.cls == NULL)
    sign = torch.sign(a.cls.to(torch.int64) - b.cls.to(torch.int64))

    both_num = (a.cls == NUM) & (b.cls == NUM)
    both_int = both_num & a.isint & b.isint
    num_sign = torch.sign(a.num - b.num).to(torch.int64)
    int_sign = torch.sign(a.ival - b.ival)
    num_sign = torch.where(both_int, int_sign, num_sign)
    sign = torch.where(both_num, num_sign, sign)

    both_text = (a.cls == TEXT) & (b.cls == TEXT)
    if both_text.any():
        rank = strings.rank()
        ra, rb = rank[a.code.clamp(min=0)], rank[b.code.clamp(min=0)]
        sign = torch.where(both_text, torch.sign(ra - rb), sign)
    return sign, null


def text_rank_of_literal(s: str, strings: Strings) -> int:
    if strings._sorted is None:
        strings._sorted = sorted(strings.values)
    return bisect_left(strings._sorted, s)


def row_keys(vecs: list[Vec]) -> torch.Tensor:
    """Exact per-row identity key, used for grouping, DISTINCT and joins.

    Numeric values compare by value (so 1 and 1.0 are the same key), text by code.
    """
    cols = []
    for v in vecs:
        num = torch.where(v.cls == NUM, v.num + 0.0, torch.zeros_like(v.num))
        cols.append(v.cls.to(torch.int64))
        cols.append(num.view(torch.int64))
        cols.append(torch.where(v.cls == TEXT, v.code, torch.full_like(v.code, -1)))
    if not cols:
        n = 0
        return torch.zeros((n, 0), dtype=torch.int64)
    return torch.stack(cols, dim=1)
