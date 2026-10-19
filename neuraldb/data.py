"""Synthetic tables and queries for the end-to-end neural database demo.

Every batch is a fresh random table, so the model cannot memorise data: it has
to read the rows it is given. Labels are computed exactly (and checked against
SQLite in tests/test_neuraldb.py).

Query forms (P = PAGE_SIZE rows per page):
  SELECT * [WHERE conds] [ORDER BY col ASC|DESC] PAGE k   -> row pointers + has_more
  COUNT ( * ) [WHERE conds]                               -> integer
  SUM ( col ) [WHERE conds]                               -> integer
  AVG ( col ) [WHERE conds]                               -> floor(100 * avg), or NULL
  MIN ( col ) / MAX ( col ) [WHERE conds]                 -> pointer to the row holding it
conds: 1-3 comparisons joined by AND or OR; =, !=, <, <=, >, >=, BETWEEN.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

PAGE_SIZE = 10
DIGITS = 6  # cells hold integers below 10**6; category ids are below 1000
N_WORDS = 500
COL_NAMES = [
    "age", "price", "qty", "score", "year", "city", "dept", "level", "rank", "stock", "weight", "height",
    "color", "brand", "region", "status", "kind", "grade", "team", "size", "zone", "floor", "salary", "bonus",
    "tax", "fee", "rate", "count", "views", "likes", "speed", "power", "area", "depth", "width", "length",
    "month", "day", "hour", "code", "group", "tier", "shift", "lane", "batch", "lot", "unit", "slot",
]

SPECIAL = ["<pad>", "<cls>", "<rowcls>"]
KEYWORDS = [
    "SELECT", "*", "COUNT", "SUM", "AVG", "MIN", "MAX", "(", ")", "WHERE", "AND", "OR",
    "ORDER", "BY", "ASC", "DESC", "PAGE", "=", "!=", "<", "<=", ">", ">=", "BETWEEN",
]
VOCAB = SPECIAL + KEYWORDS + [f"d{i}" for i in range(10)] + COL_NAMES + [f"w{i:03d}" for i in range(N_WORDS)]
TOK = {t: i for i, t in enumerate(VOCAB)}
PAD, CLS, ROWCLS = TOK["<pad>"], TOK["<cls>"], TOK["<rowcls>"]
D0 = TOK["d0"]
W0 = TOK["w000"]

TYPES = ["page", "count", "sum", "avg", "min", "max"]
INT_OPS = ["=", "!=", "<", "<=", ">", ">=", "BETWEEN"]
CAT_OPS = ["=", "!="]


@dataclass
class Table:
    names: list[str]
    is_int: list[bool]
    values: np.ndarray  # (N, K) int64; category columns hold word ids
    hi: list[int]  # exclusive upper bound per column (for boundary weighting)


@dataclass
class Query:
    qtype: str
    conds: list[tuple]  # (col, op, lit) or (col, "BETWEEN", (lo, hi))
    conj: str = "AND"
    target: int | None = None  # aggregated / ordered column
    desc: bool = False
    page: int = 0


@dataclass
class Example:
    table: Table
    query: Query
    q_tokens: list[int]
    sel: np.ndarray  # (N,) bool: rows satisfying WHERE
    order: np.ndarray  # row ids in output order (selected rows only)
    target_vals: np.ndarray | None  # (N,) values of the target column
    boundary: np.ndarray  # (N,) float: row sits on a condition boundary
    answer: object
    extras: dict = field(default_factory=dict)


def gen_table(rng: np.random.Generator, n: int) -> Table:
    k = int(rng.integers(2, 7))
    names = list(rng.choice(COL_NAMES, size=k, replace=False))
    is_int = [bool(rng.random() < 0.65) for _ in range(k)]
    is_int[int(rng.integers(k))] = True  # at least one numeric column
    vals, his = np.zeros((n, k), dtype=np.int64), []
    for j in range(k):
        if is_int[j]:
            hi = int(rng.choice([10, 100, 1000, 100_000]))
            vals[:, j] = rng.integers(0, hi, n)
        else:
            size = int(rng.choice([2, 5, 20, 100]))
            words = rng.choice(N_WORDS, size=size, replace=False)
            vals[:, j] = rng.choice(words, n)
            hi = N_WORDS
        his.append(hi)
    return Table(names, is_int, vals, his)


def _literal(rng, t: Table, j: int) -> int:
    col = t.values[:, j]
    if len(col) and rng.random() < 0.75:
        v = int(col[int(rng.integers(len(col)))])  # an existing value: boundary rows exist
        if t.is_int[j] and rng.random() < 0.3:
            v = max(0, v + int(rng.choice([-1, 1])))
        return v
    return int(rng.integers(0, t.hi[j]))


def gen_query(rng: np.random.Generator, t: Table, qtype: str | None = None) -> Query:
    qtype = qtype or str(rng.choice(TYPES, p=[0.3, 0.15, 0.15, 0.1, 0.15, 0.15]))
    k = len(t.names)
    ints = [j for j in range(k) if t.is_int[j]]
    ncond = int(rng.choice([0, 1, 2, 3], p=[0.15, 0.4, 0.3, 0.15]))
    conds = []
    for _ in range(ncond):
        j = int(rng.integers(k))
        if t.is_int[j]:
            op = str(rng.choice(INT_OPS))
            if op == "BETWEEN":
                a, b = sorted((_literal(rng, t, j), _literal(rng, t, j)))
                conds.append((j, op, (a, b)))
            else:
                conds.append((j, op, _literal(rng, t, j)))
        else:
            conds.append((j, str(rng.choice(CAT_OPS)), _literal(rng, t, j)))
    q = Query(qtype, conds, conj=str(rng.choice(["AND", "OR"])))
    if qtype in ("sum", "avg", "min", "max"):
        q.target = int(rng.choice(ints))
    elif qtype == "page":
        if rng.random() < 0.6:
            q.target = int(rng.integers(k))
            q.desc = bool(rng.random() < 0.5)
        q.page = int(rng.choice([0, 1, 2], p=[0.6, 0.25, 0.15]))
    q.desc = q.desc or qtype == "max"
    return q


def _num_tokens(v: int) -> list[int]:
    return [D0 + int(c) for c in str(v)]


def _cell_text(t: Table, j: int, v: int) -> list[int]:
    return _num_tokens(v) if t.is_int[j] else [W0 + v]


def query_tokens(t: Table, q: Query) -> list[int]:
    toks = [CLS]
    if q.qtype == "page":
        toks += [TOK["SELECT"], TOK["*"]]
    elif q.qtype == "count":
        toks += [TOK["COUNT"], TOK["("], TOK["*"], TOK[")"]]
    else:
        toks += [TOK[q.qtype.upper()], TOK["("], TOK[t.names[q.target]], TOK[")"]]
    if q.conds:
        toks.append(TOK["WHERE"])
        for i, (j, op, lit) in enumerate(q.conds):
            if i:
                toks.append(TOK[q.conj])
            toks += [TOK[t.names[j]], TOK[op]]
            if op == "BETWEEN":
                toks += _cell_text(t, j, lit[0]) + [TOK["AND"]] + _cell_text(t, j, lit[1])
            else:
                toks += _cell_text(t, j, lit)
    if q.qtype == "page":
        if q.target is not None:
            toks += [TOK["ORDER"], TOK["BY"], TOK[t.names[q.target]], TOK["DESC" if q.desc else "ASC"]]
        toks += [TOK["PAGE"]] + _num_tokens(q.page)
    return toks


def _cond_mask(t: Table, c) -> tuple[np.ndarray, np.ndarray]:
    j, op, lit = c
    col = t.values[:, j]
    if op == "BETWEEN":
        lo, hi = lit
        m = (col >= lo) & (col <= hi)
        near = (np.abs(col - lo) <= 1) | (np.abs(col - hi) <= 1)
    else:
        m = {"=": col == lit, "!=": col != lit, "<": col < lit, "<=": col <= lit,
             ">": col > lit, ">=": col >= lit}[op]
        near = np.abs(col - lit) <= 1 if t.is_int[j] else col == lit
    return m, near


def label(t: Table, q: Query) -> Example:
    n = t.values.shape[0]
    sel = np.ones(n, dtype=bool) if not q.conds or q.conj == "AND" else np.zeros(n, dtype=bool)
    boundary = np.zeros(n, dtype=np.float32)
    for c in q.conds:
        m, near = _cond_mask(t, c)
        sel = (sel & m) if q.conj == "AND" else (sel | m)
        boundary = np.maximum(boundary, near.astype(np.float32))
    rows = np.arange(n)
    tv = t.values[:, q.target] if q.target is not None else None
    if tv is not None:
        key = -tv if q.desc else tv
        order_all = np.lexsort((rows, key))  # stable: ties keep row order
    else:
        order_all = rows
    order = order_all[sel[order_all]]
    if q.qtype == "page":
        off = q.page * PAGE_SIZE
        answer = (order[off: off + PAGE_SIZE].tolist(), bool(len(order) > off + PAGE_SIZE))
    elif q.qtype == "count":
        answer = int(sel.sum())
    elif q.qtype == "sum":
        answer = int(tv[sel].sum()) if sel.any() else None
    elif q.qtype == "avg":
        answer = (int(tv[sel].sum()) * 100) // int(sel.sum()) if sel.any() else None
    else:
        answer = int(order[0]) if len(order) else None
    return Example(t, q, query_tokens(t, q), sel, order, tv, boundary, answer)


def gen_example(rng: np.random.Generator, n: int, qtype: str | None = None) -> Example:
    t = gen_table(rng, n)
    return label(t, gen_query(rng, t, qtype))


def to_sql(t: Table, q: Query) -> str:
    """The same query as SQLite text (table ``t``, words as 'w017')."""
    def lit_sql(j, v):
        return str(v) if t.is_int[j] else f"'w{v:03d}'"

    where = ""
    if q.conds:
        parts = []
        for j, op, lit in q.conds:
            if op == "BETWEEN":
                parts.append(f'"{t.names[j]}" BETWEEN {lit_sql(j, lit[0])} AND {lit_sql(j, lit[1])}')
            else:
                parts.append(f'"{t.names[j]}" {op} {lit_sql(j, lit)}')
        where = " WHERE " + f" {q.conj} ".join(parts)
    col = f'"{t.names[q.target]}"' if q.target is not None else None
    if q.qtype == "page":
        order = f" ORDER BY {col} {'DESC' if q.desc else 'ASC'}, rowid" if col else " ORDER BY rowid"
        return f"SELECT rowid - 1 FROM t{where}{order} LIMIT {PAGE_SIZE + 1} OFFSET {q.page * PAGE_SIZE}"
    if q.qtype == "count":
        return f"SELECT COUNT(*) FROM t{where}"
    if q.qtype == "sum":
        return f"SELECT SUM({col}) FROM t{where}"
    if q.qtype == "avg":
        return f"SELECT (SUM({col}) * 100) / COUNT({col}) FROM t{where}"
    return f"SELECT rowid - 1 FROM t{where} ORDER BY {col} {'DESC' if q.desc else 'ASC'}, rowid LIMIT 1"
