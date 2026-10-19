"""Data weights: a database held as tensors.

Each table is a list of columns; each column is a ``Vec`` (see vec.py). The
whole store can be saved to and loaded from a single file, so the data travels
with the model rather than living in an external database.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

import torch

from .vec import Strings, Unsupported, Vec, affinity_of


@dataclass
class Column:
    name: str
    decl: str
    affinity: str
    vec: Vec


@dataclass
class Table:
    name: str
    columns: list[Column]
    nrows: int

    def column(self, name: str) -> Column:
        lname = name.lower()
        for c in self.columns:
            if c.name.lower() == lname:
                return c
        raise KeyError(f"no column {name!r} in table {self.name!r}")


@dataclass
class DataWeights:
    tables: dict[str, Table] = field(default_factory=dict)
    strings: Strings = field(default_factory=Strings)

    def table(self, name: str) -> Table:
        t = self.tables.get(name.lower())
        if t is None:
            raise KeyError(f"no table {name!r}")
        return t

    # --- loading ------------------------------------------------------------

    @staticmethod
    def from_sqlite(path: str, max_rows: int | None = None) -> "DataWeights":
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
        dw = DataWeights()
        try:
            names = [
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                )
            ]
            for tname in names:
                info = conn.execute(f'PRAGMA table_info("{tname}")').fetchall()
                cols = [(r[1], r[2] or "") for r in info]
                limit = f" LIMIT {int(max_rows)}" if max_rows else ""
                rows = conn.execute(f'SELECT * FROM "{tname}"{limit}').fetchall()
                columns = []
                for j, (cname, decl) in enumerate(cols):
                    values = [r[j] for r in rows]
                    try:
                        vec = Vec.from_python(values, dw.strings)
                    except Unsupported:
                        # Keep the column visible but unusable (e.g. BLOB data).
                        vec = None
                    columns.append(Column(cname, decl, affinity_of(decl), vec))
                dw.tables[tname.lower()] = Table(tname, columns, len(rows))
        finally:
            conn.close()
        return dw

    # --- schema text for the model -------------------------------------------

    def schema_text(self) -> str:
        lines = []
        for t in self.tables.values():
            cols = ", ".join(f"{c.name} {c.decl or 'ANY'}".strip() for c in t.columns)
            lines.append(f"table {t.name}: {cols}")
        return "\n".join(lines)

    # --- persistence ----------------------------------------------------------

    def state_dict(self) -> dict:
        tensors, meta = {}, {"strings": self.strings.values, "tables": []}
        for t in self.tables.values():
            tmeta = {"name": t.name, "nrows": t.nrows, "columns": []}
            for c in t.columns:
                tmeta["columns"].append({"name": c.name, "decl": c.decl, "usable": c.vec is not None})
                if c.vec is not None:
                    for part in ("cls", "num", "isint", "ival", "code"):
                        tensors[f"{t.name}.{c.name}.{part}"] = getattr(c.vec, part)
            meta["tables"].append(tmeta)
        return {"tensors": tensors, "meta": meta}

    def save(self, path: str):
        torch.save(self.state_dict(), path)

    @staticmethod
    def load(path: str) -> "DataWeights":
        state = torch.load(path, weights_only=True)
        dw = DataWeights()
        for s in state["meta"]["strings"]:
            dw.strings.code(s)
        tensors = state["tensors"]
        for tmeta in state["meta"]["tables"]:
            cols = []
            for cm in tmeta["columns"]:
                vec = None
                if cm["usable"]:
                    key = f"{tmeta['name']}.{cm['name']}"
                    vec = Vec(*(tensors[f"{key}.{p}"] for p in ("cls", "num", "isint", "ival", "code")))
                cols.append(Column(cm["name"], cm["decl"], affinity_of(cm["decl"]), vec))
            dw.tables[tmeta["name"].lower()] = Table(tmeta["name"], cols, tmeta["nrows"])
        return dw
