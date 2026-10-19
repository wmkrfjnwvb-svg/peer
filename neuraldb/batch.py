"""Turn examples into padded tensors."""

from __future__ import annotations

import numpy as np
import torch

from .data import DIGITS, PAD, ROWCLS, TOK, TYPES, W0, Example

IGNORE = -100


def digits_of(v: np.ndarray) -> np.ndarray:
    """(N,) ints -> (N, DIGITS) most-significant digit first."""
    out = np.zeros((len(v), DIGITS), dtype=np.int64)
    x = v.copy()
    for p in range(DIGITS - 1, -1, -1):
        out[:, p] = x % 10
        x //= 10
    return out


def collate(examples: list[Example]) -> dict[str, torch.Tensor]:
    b = len(examples)
    n = max(e.table.values.shape[0] for e in examples)
    k = max(len(e.table.names) for e in examples)
    q = max(len(e.q_tokens) for e in examples)
    t_len = 1 + k * DIGITS

    tok = np.full((b, n, t_len), PAD, dtype=np.int64)
    col = np.full((b, n, t_len), PAD, dtype=np.int64)
    place = np.full((b, n, t_len), DIGITS, dtype=np.int64)
    q_tok = np.full((b, q), PAD, dtype=np.int64)
    row_mask = np.zeros((b, n), dtype=bool)
    sel = np.zeros((b, n), dtype=np.float32)
    boundary = np.zeros((b, n), dtype=np.float32)
    read = np.full((b, n, DIGITS), IGNORE, dtype=np.int64)
    qtype = np.zeros(b, dtype=np.int64)
    order = np.full(b, IGNORE, dtype=np.int64)
    page = np.full(b, IGNORE, dtype=np.int64)
    count = np.zeros(b, dtype=np.float32)
    total = np.zeros(b, dtype=np.float32)
    scale = np.ones(b, dtype=np.float32)

    for i, e in enumerate(examples):
        t = e.table
        rows = t.values.shape[0]
        tok[i, :rows, 0] = ROWCLS
        for j, name in enumerate(t.names):
            s = 1 + j * DIGITS
            col[i, :rows, s: s + DIGITS] = TOK[name]
            place[i, :rows, s: s + DIGITS] = np.arange(DIGITS)
            if t.is_int[j]:
                tok[i, :rows, s: s + DIGITS] = TOK["d0"] + digits_of(t.values[:, j])
            else:
                tok[i, :rows, s] = W0 + t.values[:, j]
                place[i, :rows, s] = 0
        q_tok[i, : len(e.q_tokens)] = e.q_tokens
        row_mask[i, :rows] = True
        sel[i, :rows] = e.sel
        boundary[i, :rows] = e.boundary
        if e.target_vals is not None:
            read[i, :rows] = digits_of(e.target_vals)
            scale[i] = float(max(t.hi[e.query.target], 1))
        qtype[i] = TYPES.index(e.query.qtype)
        if e.query.qtype in ("page", "min", "max"):
            order[i] = 0 if e.query.target is None else (2 if e.query.desc else 1)
        if e.query.qtype == "page":
            page[i] = e.query.page
        count[i] = e.sel.sum()
        if e.target_vals is not None:
            total[i] = float(e.target_vals[e.sel].sum())

    t = lambda a: torch.from_numpy(a)  # noqa: E731
    return {
        "tok": t(tok), "col": t(col), "place": t(place), "q_tok": t(q_tok),
        "row_mask": t(row_mask), "sel": t(sel), "boundary": t(boundary), "read": t(read),
        "qtype": t(qtype), "order": t(order), "page": t(page), "count": t(count), "total": t(total), "scale": t(scale),
    }
