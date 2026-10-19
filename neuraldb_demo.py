"""neuraldb_demo.py: a database computed entirely by a neural network.

The model reads every row of a table itself and makes every data decision.
We only take its outputs: row pointers (plus a has-more-pages flag) or
computed values (COUNT / SUM / AVG).

  per row    select logit   does the row satisfy WHERE
             read digits    value of the target column (6 digits x 10 classes)
  per query  answer type, ordering (row order / asc / desc), page number

Loss is built for exactness (every row must be right): boundary-row
weighting, a logit margin, hard-row mining, margin cross entropy on digits,
and a global COUNT / SUM consistency term. Training data are fresh random
tables each batch with a growing table size; labels are checked against
SQLite (``test``).

Usage
  python neuraldb_demo.py test
  # 1 GPU
  python neuraldb_demo.py train --out runs/nd --d 512 --r-layers 6 --grad-ckpt
  # 4 x RTX 4090
  torchrun --nproc_per_node 4 neuraldb_demo.py train --out runs/nd --d 512 --r-layers 6 \
      --grad-ckpt --n-max 1000 --steps 200000
  # accuracy by table size
  python neuraldb_demo.py eval --ckpt runs/nd/last.pt --sizes 10,100,1000,10000,100000

Requires: torch >= 2.1, numpy.
"""

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, IterableDataset, get_worker_info


# ============================================================================
# data
# ============================================================================

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
    q_places: list[int]  # digit place of each query token (DIGITS for non-digits)
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
    # numbers in queries use the same fixed-width, place-aligned digits as cells
    return [D0 + int(c) for c in str(v).zfill(DIGITS)]


def _cell_text(t: Table, j: int, v: int) -> list[int]:
    return _num_tokens(v) if t.is_int[j] else [W0 + v]


def query_places(toks: list[int]) -> list[int]:
    places, run = [], 0
    for tok in toks:
        if D0 <= tok < D0 + 10:
            places.append(run % DIGITS)
            run += 1
        else:
            places.append(DIGITS)
            run = 0
    return places


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
    toks = query_tokens(t, q)
    return Example(t, q, toks, query_places(toks), sel, order, tv, boundary, answer)


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


# ============================================================================
# batch
# ============================================================================

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
    q_place = np.full((b, q), DIGITS, dtype=np.int64)
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
        q_place[i, : len(e.q_places)] = e.q_places
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
        "tok": t(tok), "col": t(col), "place": t(place), "q_tok": t(q_tok), "q_place": t(q_place),
        "row_mask": t(row_mask), "sel": t(sel), "boundary": t(boundary), "read": t(read),
        "qtype": t(qtype), "order": t(order), "page": t(page), "count": t(count), "total": t(total), "scale": t(scale),
    }


# ============================================================================
# model
# ============================================================================

MAX_PAGES = 16


@dataclass
class Config:
    d: int = 256
    heads: int = 8
    q_layers: int = 4
    r_layers: int = 4
    max_q: int = 160
    dropout: float = 0.0


class Attention(nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        self.h = heads
        self.q = nn.Linear(d, d)
        self.kv = nn.Linear(d, 2 * d)
        self.o = nn.Linear(d, d)

    def forward(self, x, mem, key_pad=None):
        b, n, d = x.shape
        m = mem.shape[1]
        q = self.q(x).view(b, n, self.h, d // self.h).transpose(1, 2)
        k, v = self.kv(mem).view(b, m, 2, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        mask = None
        if key_pad is not None:
            mask = (~key_pad)[:, None, None, :]  # True = attend
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.o(y.transpose(1, 2).reshape(b, n, d))


class FFN(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        return self.net(x)


class QueryLayer(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.n1, self.att, self.n2, self.ffn = nn.LayerNorm(d), Attention(d, heads), nn.LayerNorm(d), FFN(d)

    def forward(self, x, pad):
        h = self.n1(x)
        x = x + self.att(h, h, pad)
        return x + self.ffn(self.n2(x))


class RowLayer(nn.Module):
    """Each row attends within itself, then to the query, then an FFN."""

    def __init__(self, d, heads):
        super().__init__()
        self.n1, self.self_att = nn.LayerNorm(d), Attention(d, heads)
        self.n2, self.cross = nn.LayerNorm(d), Attention(d, heads)
        self.n3, self.ffn = nn.LayerNorm(d), FFN(d)

    def forward(self, x, cell_pad, q, q_pad):
        b, n, t, d = x.shape
        h = self.n1(x).view(b * n, t, d)
        x = x + self.self_att(h, h, cell_pad.view(b * n, t)).view(b, n, t, d)
        h = self.n2(x).view(b, n * t, d)
        x = x + self.cross(h, q, q_pad).view(b, n, t, d)
        return x + self.ffn(self.n3(x))


class NeuralDB(nn.Module):
    def __init__(self, cfg: Config = Config()):
        super().__init__()
        self.cfg = cfg
        d = cfg.d
        self.emb = nn.Embedding(len(VOCAB), d, padding_idx=PAD)
        self.place = nn.Embedding(DIGITS + 1, d)
        self.qpos = nn.Embedding(cfg.max_q, d)
        self.q_layers = nn.ModuleList(QueryLayer(d, cfg.heads) for _ in range(cfg.q_layers))
        self.q_norm = nn.LayerNorm(d)
        self.r_layers = nn.ModuleList(RowLayer(d, cfg.heads) for _ in range(cfg.r_layers))
        self.r_norm = nn.LayerNorm(d)
        self.sel_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.read_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, DIGITS * 10))
        self.type_head = nn.Linear(d, len(TYPES))
        self.order_head = nn.Linear(d, 3)  # 0 = row order, 1 = ascending, 2 = descending
        self.page_head = nn.Linear(d, MAX_PAGES)
        self.grad_ckpt = False

    def encode_query(self, q_tok, q_place):
        pad = q_tok == PAD
        pos = torch.arange(q_tok.shape[1], device=q_tok.device)
        x = self.emb(q_tok) + self.qpos(pos)[None] + self.place(q_place)
        for layer in self.q_layers:
            x = layer(x, pad)
        return self.q_norm(x), pad

    def read_rows(self, tok, col, place, q, q_pad):
        cell_pad = tok == PAD
        cell_pad[..., 0] = False  # the row token is always attendable
        x = self.emb(tok) + self.emb(col) + self.place(place)
        for layer in self.r_layers:
            if self.grad_ckpt and self.training:
                x = checkpoint(layer, x, cell_pad, q, q_pad, use_reentrant=False)
            else:
                x = layer(x, cell_pad, q, q_pad)
        return self.r_norm(x[:, :, 0])  # (B, N, d) row states

    def forward(self, batch, row_chunk: int | None = None):
        q, q_pad = self.encode_query(batch["q_tok"], batch["q_place"])
        tok, col, place = batch["tok"], batch["col"], batch["place"]
        if row_chunk is None or tok.shape[1] <= row_chunk:
            rows = self.read_rows(tok, col, place, q, q_pad)
        else:  # large tables at inference: rows are independent, so read in chunks
            rows = torch.cat([
                self.read_rows(tok[:, s: s + row_chunk], col[:, s: s + row_chunk], place[:, s: s + row_chunk], q, q_pad)
                for s in range(0, tok.shape[1], row_chunk)
            ], dim=1)
        b, n, _ = rows.shape
        return {
            "sel": self.sel_head(rows).squeeze(-1),                      # (B, N)
            "read": self.read_head(rows).view(b, n, DIGITS, 10),          # (B, N, DIGITS, 10)
            "type": self.type_head(q[:, 0]),                              # (B, types)
            "order": self.order_head(q[:, 0]),                            # (B, 3)
            "page": self.page_head(q[:, 0]),                              # (B, MAX_PAGES)
        }


# ============================================================================
# losses
# ============================================================================

PLACE = torch.tensor([10 ** (DIGITS - 1 - p) for p in range(DIGITS)], dtype=torch.float64)


@dataclass
class LossConfig:
    boundary_w: float = 4.0
    margin: float = 5.0
    margin_w: float = 0.2
    hard_frac: float = 0.02
    hard_w: float = 1.0
    read_margin: float = 1.0
    selected_read_w: float = 3.0
    agg_w: float = 0.1


def _hard_mean(losses: torch.Tensor, frac: float) -> torch.Tensor:
    flat = losses.flatten()
    flat = flat[flat > 0]
    if flat.numel() == 0:
        return losses.sum() * 0
    k = max(1, int(flat.numel() * frac))
    return flat.topk(k).values.mean()


def compute_loss(out: dict, batch: dict, cfg: LossConfig = LossConfig()):
    rm = batch["row_mask"].float()
    y = batch["sel"]
    logit = out["sel"].float()

    # --- row selection ---------------------------------------------------------
    pos = (y * rm).sum()
    neg = ((1 - y) * rm).sum()
    pos_w = (neg / pos.clamp(min=1)).clamp(1.0, 10.0)
    w = rm * (1 + cfg.boundary_w * batch["boundary"]) * torch.where(y > 0, pos_w, torch.ones_like(y))
    bce = F.binary_cross_entropy_with_logits(logit, y, reduction="none")
    sign = 2 * y - 1
    hinge = F.relu(cfg.margin - sign * logit)
    row_loss = (bce + cfg.margin_w * hinge) * w
    l_sel = row_loss.sum() / w.sum().clamp(min=1)
    l_sel_hard = _hard_mean(row_loss * rm, cfg.hard_frac)

    # --- reading the target column -------------------------------------------
    read = out["read"].float()  # (B, N, D, 10)
    tgt = batch["read"]
    valid = tgt != IGNORE
    if valid.any():
        margin = torch.zeros_like(read)
        margin.scatter_(-1, tgt.clamp(min=0).unsqueeze(-1), cfg.read_margin)
        ce = F.cross_entropy((read - margin).flatten(0, 2), tgt.flatten(), ignore_index=IGNORE, reduction="none")
        ce = ce.view_as(tgt).float() * valid
        rw = rm * (1 + (cfg.selected_read_w - 1) * y)
        per_row = ce.sum(-1) * rw
        l_read = per_row.sum() / (valid.any(-1).float() * rw).sum().clamp(min=1)
        l_read_hard = _hard_mean(per_row, cfg.hard_frac)
    else:
        l_read = l_read_hard = logit.sum() * 0

    # --- global consistency (soft COUNT / SUM) ---------------------------------
    p = torch.sigmoid(logit) * rm
    soft_count = p.sum(-1)
    # log1p keeps the term bounded while the row decisions are still poor
    l_count = torch.log1p(((soft_count - batch["count"]) / (batch["count"] + 1).sqrt()).abs()).mean()
    has_total = batch["qtype"].unsqueeze(-1) == torch.tensor([TYPES.index("sum"), TYPES.index("avg")], device=logit.device)
    has_total = has_total.any(-1)
    if has_total.any():
        dig = torch.softmax(read, -1) @ torch.arange(10, device=read.device, dtype=read.dtype)  # (B, N, D)
        val = (dig * PLACE.to(read.device, read.dtype)).sum(-1)
        norm = batch["scale"] * (batch["count"] + 1).sqrt()
        soft_sum = (p * val).sum(-1)
        l_sum = torch.log1p(((soft_sum - batch["total"]) / norm)[has_total].abs()).mean()
    else:
        l_sum = logit.sum() * 0
    l_agg = l_count + l_sum

    # --- query-level decisions ---------------------------------------------------
    l_type = F.cross_entropy(out["type"].float(), batch["qtype"])
    l_order = F.cross_entropy(out["order"].float(), batch["order"], ignore_index=IGNORE) if (batch["order"] != IGNORE).any() else l_type * 0
    l_page = F.cross_entropy(out["page"].float(), batch["page"], ignore_index=IGNORE) if (batch["page"] != IGNORE).any() else l_type * 0

    total = (l_sel + cfg.hard_w * l_sel_hard + l_read + cfg.hard_w * l_read_hard
             + cfg.agg_w * l_agg + l_type + l_order + l_page)
    parts = {"sel": l_sel, "sel_hard": l_sel_hard, "read": l_read, "read_hard": l_read_hard,
             "agg": l_agg, "type": l_type, "order": l_order, "page": l_page}
    return total, {k: float(v.detach()) for k, v in parts.items()}


# --- decoding --------------------------------------------------------------------

@torch.no_grad()
def decode(out: dict, batch: dict) -> list[dict]:
    """Read answers off the model's decisions, with a confidence per answer."""
    rm = batch["row_mask"]
    sel_logit = out["sel"].float()
    sel = (sel_logit > 0) & rm
    read_p = torch.softmax(out["read"].float(), -1)
    digit_p, digits = read_p.max(-1)  # (B, N, D)
    values = (digits.double().cpu() * PLACE).sum(-1).round().long()
    qtype = out["type"].argmax(-1).tolist()
    order = out["order"].argmax(-1).tolist()
    page = out["page"].argmax(-1).tolist()
    q_conf = torch.minimum(torch.softmax(out["type"].float(), -1).max(-1).values,
                           torch.softmax(out["order"].float(), -1).max(-1).values)
    row_conf = torch.sigmoid(sel_logit.abs())
    results = []
    for i in range(sel.shape[0]):
        n = int(rm[i].sum())
        s = sel[i, :n].cpu()
        v = values[i, :n]
        rows = torch.arange(n)
        t = TYPES[qtype[i]]
        need_values = t in ("sum", "avg", "min", "max") or (t == "page" and order[i] != 0)
        conf = min(float(q_conf[i]), float(row_conf[i, :n].min()) if n else 1.0)
        if need_values and s.any():
            conf = min(conf, float(digit_p[i, :n][s.to(digit_p.device)].min()))
        if t in ("page", "min", "max"):
            if t == "page" and order[i] == 0:
                ordered = rows[s]
            else:
                desc = order[i] == 2 if t == "page" else t == "max"
                key = -v if desc else v
                idx = sorted(rows[s].tolist(), key=lambda r: (int(key[r]), r))
                ordered = torch.tensor(idx, dtype=torch.long)
            if t == "page":
                off = page[i] * PAGE_SIZE
                ans = (ordered[off: off + PAGE_SIZE].tolist(), bool(len(ordered) > off + PAGE_SIZE))
            else:
                ans = int(ordered[0]) if len(ordered) else None
        elif t == "count":
            ans = int(s.sum())
        elif t == "sum":
            ans = int(v[s].sum()) if s.any() else None
        else:
            ans = (int(v[s].sum()) * 100) // int(s.sum()) if s.any() else None
        results.append({"type": t, "answer": ans, "conf": conf})
    return results


# ============================================================================
# evaluate
# ============================================================================

THRESHOLDS = [0.0, 0.9, 0.99, 0.999]


def _same(a, b) -> bool:
    return a == b


@torch.no_grad()
def evaluate(model, sizes, n_queries: int, batch_size: int, device, seed: int = 1234, row_chunk: int = 2048,
             amp: bool = False) -> dict:
    model.eval()
    report = {}
    for n in sizes:
        rng = np.random.default_rng(seed + n)
        per_type = defaultdict(lambda: [0, 0])
        confs = []
        sel_err = sel_rows = read_err = read_rows = 0
        bs = max(1, min(batch_size, (batch_size * 1000) // max(n, 1)))
        done = 0
        while done < n_queries:
            k = min(bs, n_queries - done)
            exs = [gen_example(rng, n, TYPES[(done + i) % len(TYPES)]) for i in range(k)]
            batch = {key: v.to(device) for key, v in collate(exs).items()}
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                out = model(batch, row_chunk=row_chunk)
            preds = decode(out, batch)
            rm = batch["row_mask"]
            sel_hat = out["sel"].float() > 0
            sel_err += int(((sel_hat != (batch["sel"] > 0)) & rm).sum())
            sel_rows += int(rm.sum())
            tgt = batch["read"]
            has = (tgt != IGNORE).all(-1) & rm & (batch["sel"] > 0)
            wrong = (out["read"].argmax(-1) != tgt).any(-1) & has
            read_err += int(wrong.sum())
            read_rows += int(has.sum())
            for e, p in zip(exs, preds):
                ok = p["type"] == e.query.qtype and _same(p["answer"], e.answer)
                per_type[e.query.qtype][0] += ok
                per_type[e.query.qtype][1] += 1
                confs.append((p["conf"], ok))
            done += k
        total_ok = sum(v[0] for v in per_type.values())
        coverage = {}
        for t in THRESHOLDS:
            ans = [ok for c, ok in confs if c >= t]
            coverage[t] = (len(ans) / len(confs), (sum(ans) / len(ans)) if ans else None)
        report[n] = {
            "exact": total_ok / n_queries,
            "per_type": {t: v[0] / v[1] for t, v in per_type.items()},
            "row_select_error": sel_err / max(sel_rows, 1),
            "row_read_error": read_err / max(read_rows, 1),
            "coverage": coverage,
        }
    model.train()
    return report


def print_report(report: dict, flush: bool = False):
    types = TYPES
    print(f"{'rows':>8} {'exact':>7} " + " ".join(f"{t:>6}" for t in types) + f" {'sel_err/row':>12} {'read_err/row':>12}")
    for n, r in report.items():
        cells = " ".join(f"{r['per_type'].get(t, float('nan')):6.3f}" for t in types)
        print(f"{n:8d} {r['exact']:7.4f} {cells} {r['row_select_error']:12.2e} {r['row_read_error']:12.2e}")
    print("\nconfidence threshold -> (answered fraction, accuracy of answered)")
    for n, r in report.items():
        cov = "  ".join(f"{t}: ({a:.3f}, {('%.4f' % p) if p is not None else '-'})" for t, (a, p) in r["coverage"].items())
        print(f"{n:8d}  {cov}", flush=flush)


def eval_main(argv):
    ap = argparse.ArgumentParser(prog="neuraldb_demo.py eval")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--sizes", default="10,100,1000,10000")
    ap.add_argument("--queries", type=int, default=300)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args(argv)
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = NeuralDB(Config(**state["config"]))
    model.load_state_dict(state["model"])
    device = torch.device(args.device)
    model.to(device)
    sizes = [int(s) for s in args.sizes.split(",")]
    print_report(evaluate(model, sizes, args.queries, args.batch, device, amp=device.type == "cuda"))


# ============================================================================
# train
# ============================================================================

class RandomTables(IterableDataset):
    """Endless batches of fresh tables; table size follows the curriculum."""

    def __init__(self, batch: int, n_min: int, n_max: int, warm_steps: int, total_steps: int, seed: int,
                 max_rows_per_batch: int):
        self.batch, self.n_min, self.n_max = batch, n_min, n_max
        self.warm, self.total, self.seed = warm_steps, total_steps, seed
        self.max_rows = max_rows_per_batch

    def cur_max(self, step: int) -> int:
        frac = min(1.0, step / max(1, self.warm))
        return int(round(math.exp(math.log(self.n_min) + frac * (math.log(self.n_max) - math.log(self.n_min)))))

    def __iter__(self):
        info = get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        rng = np.random.default_rng(self.seed * 1000 + wid)
        step = wid
        while True:
            hi = max(self.n_min, self.cur_max(step))
            n = int(round(math.exp(rng.uniform(math.log(self.n_min), math.log(hi)))))
            # about max_rows rows per batch: many small tables early, few large ones later
            b = max(1, min(self.batch, self.max_rows // n))
            yield collate([gen_example(rng, n) for _ in range(b)])
            step += nw


def train_main(argv):
    ap = argparse.ArgumentParser(prog="neuraldb_demo.py train")
    ap.add_argument("--out", default="runs/neuraldb")
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--q-layers", type=int, default=4)
    ap.add_argument("--r-layers", type=int, default=4)
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--batch", type=int, default=256, help="max tables per batch per GPU")
    ap.add_argument("--max-rows-per-batch", type=int, default=8_000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n-min", type=int, default=4)
    ap.add_argument("--n-max", type=int, default=1000)
    ap.add_argument("--curriculum-frac", type=float, default=0.5, help="fraction of steps to reach n-max")
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--eval-sizes", default="10,100,1000")
    ap.add_argument("--eval-queries", type=int, default=240)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)

    ddp = "RANK" in os.environ
    if ddp:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        local = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local)
        device = torch.device("cuda", local)
    else:
        rank = 0
        device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(args.seed)
    amp = device.type == "cuda"

    cfg = Config(d=args.d, heads=args.heads, q_layers=args.q_layers, r_layers=args.r_layers)
    model = NeuralDB(cfg).to(device)
    model.grad_ckpt = args.grad_ckpt
    if rank == 0:
        print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M", flush=True)
    net = DDP(model, device_ids=[device.index]) if ddp else model
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, betas=(0.9, 0.98))
    warmup = min(2000, args.steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps))))

    data = RandomTables(args.batch, args.n_min, args.n_max, int(args.steps * args.curriculum_frac), args.steps,
                        seed=args.seed * 100 + rank, max_rows_per_batch=args.max_rows_per_batch)
    loader = DataLoader(data, batch_size=None, num_workers=args.workers, persistent_workers=args.workers > 0)
    lcfg = LossConfig()
    os.makedirs(args.out, exist_ok=True)
    eval_sizes = [int(s) for s in args.eval_sizes.split(",")]

    t0 = time.time()
    running = {}
    for step, batch in enumerate(loader):
        if step >= args.steps:
            break
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            out = net(batch)
        loss, parts = compute_loss(out, batch, lcfg)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        for k, v in parts.items():
            running[k] = 0.98 * running.get(k, v) + 0.02 * v
        if rank == 0 and step % 50 == 0:
            n = batch["tok"].shape[1]
            msg = " ".join(f"{k}={v:.4f}" for k, v in running.items())
            print(f"step {step} rows={n} lr={sched.get_last_lr()[0]:.2e} {msg} ({time.time() - t0:.0f}s)", flush=True)
        if rank == 0 and step and (step % args.eval_every == 0 or step == args.steps - 1):
            report = evaluate(model, eval_sizes, args.eval_queries, args.batch, device, amp=amp)
            print_report(report, flush=True)
            torch.save({"model": model.state_dict(), "config": asdict(cfg), "step": step}, os.path.join(args.out, "last.pt"))
            with open(os.path.join(args.out, "eval.jsonl"), "a") as f:
                f.write(json.dumps({"step": step, "report": {str(k): v for k, v in report.items()}}, default=str) + "\n")
    if rank == 0:
        torch.save({"model": model.state_dict(), "config": asdict(cfg), "step": args.steps}, os.path.join(args.out, "last.pt"))
    if ddp:
        dist.destroy_process_group()


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("train", "eval", "test"):
        print(__doc__)
        sys.exit(1)
    cmd, argv = sys.argv[1], sys.argv[2:]
    if cmd == "train":
        train_main(argv)
    elif cmd == "eval":
        eval_main(argv)
    else:
        self_test()


def self_test():
    """Check labels against SQLite and run one forward/backward pass."""
    import sqlite3

    for seed in range(200):
        rng = np.random.default_rng(seed)
        e = gen_example(rng, int(rng.choice([1, 5, 37, 200])), TYPES[seed % len(TYPES)])
        t, q = e.table, e.query
        conn = sqlite3.connect(":memory:")
        cols = ", ".join(f'"{n}" {"INTEGER" if i else "TEXT"}' for n, i in zip(t.names, t.is_int))
        conn.execute(f"CREATE TABLE t ({cols})")
        rows = [tuple(int(v) if t.is_int[j] else f"w{int(v):03d}" for j, v in enumerate(r)) for r in t.values]
        conn.executemany(f"INSERT INTO t VALUES ({', '.join('?' * len(t.names))})", rows)
        res = conn.execute(to_sql(t, q)).fetchall()
        if q.qtype == "page":
            ids = [r[0] for r in res]
            want = (ids[:PAGE_SIZE], len(ids) > PAGE_SIZE)
        elif q.qtype in ("min", "max"):
            want = res[0][0] if res else None
        else:
            want = res[0][0]
        assert e.answer == want, (seed, to_sql(t, q), e.answer, want)
    rng = np.random.default_rng(0)
    batch = collate([gen_example(rng, 13, t) for t in TYPES])
    model = NeuralDB(Config(d=32, heads=4, q_layers=1, r_layers=1))
    out = model(batch)
    loss, _ = compute_loss(out, batch)
    loss.backward()
    assert torch.isfinite(loss)
    print("self test passed: 200 labels match SQLite, forward/backward OK")


if __name__ == "__main__":
    main()
