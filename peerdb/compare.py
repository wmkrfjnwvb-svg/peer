"""Result comparison against a reference database."""

from __future__ import annotations

import math
import sqlite3
from collections import Counter


def _norm(v):
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, float):
        if math.isnan(v):
            return None
        if v.is_integer() and abs(v) < 2**53:
            return ("n", int(v))
        return ("n", float(f"{v:.12g}"))
    if isinstance(v, int):
        return ("n", v)
    if isinstance(v, (bytes, bytearray, memoryview)):
        return ("b", bytes(v))
    return v


def normalize(rows) -> list[tuple]:
    return [tuple(_norm(v) for v in r) for r in rows]


def same_result(got, want, ordered: bool) -> bool:
    """Rows equal (as a list if ``ordered``, else as a multiset).

    Floats are compared to 12 significant digits, since summation order may
    differ in the last bits; integers and text compare exactly.
    """
    g, w = normalize(got), normalize(want)
    if ordered:
        return g == w
    return Counter(g) == Counter(w)


def sqlite_rows(db_path: str, sql: str, timeout: float = 30.0):
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=timeout)
    conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
    try:
        return conn.execute(sql).fetchall()
    finally:
        conn.close()
