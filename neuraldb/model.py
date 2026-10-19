"""End-to-end neural database model.

The model reads every row of the table itself and makes every data-dependent
decision:

  * per row: does it satisfy the WHERE clause (select logit), and what is the
    value of the target column (read logits, one digit per place);
  * per query: what kind of answer (type), how to order rows, which page.

Answers are read straight off those decisions (see ``decode``): page pointers
are the selected rows in order of their read values, COUNT is the number of
selected rows, SUM / AVG add up the read values of the selected rows, MIN / MAX
point at the first selected row in order. Rows are read independently of each
other, so a model trained on small tables applies unchanged to large ones;
what matters for exactness is the per-row error rate.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .data import DIGITS, PAD, TYPES, VOCAB

MAX_PAGES = 16


@dataclass
class Config:
    d: int = 256
    heads: int = 8
    q_layers: int = 4
    r_layers: int = 4
    max_q: int = 96
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

    def encode_query(self, q_tok):
        pad = q_tok == PAD
        pos = torch.arange(q_tok.shape[1], device=q_tok.device)
        x = self.emb(q_tok) + self.qpos(pos)[None]
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
        q, q_pad = self.encode_query(batch["q_tok"])
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
