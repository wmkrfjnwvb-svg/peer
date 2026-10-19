"""Loss and decoding for the neural database.

An exact answer needs every row decision to be right, so the loss is built
around the rows that are hardest to get right, not around the average row:

  select   weighted BCE, with
           - boundary weighting: rows whose value sits next to a condition's
             literal (age 30 vs 31 for ``age > 30``) count ``boundary_w`` times
           - a logit margin: a decision only stops costing once it is beyond
             +/- ``margin`` (prob ~0.99), so no row is left barely right
           - hard-row mining: the worst ``hard_frac`` of rows in the batch are
             added again, so rare failure modes keep receiving gradient
  read     per-digit cross entropy with an additive margin (CosFace style) on
           the correct digit; selected rows weigh more, since their values make
           up the answer; plus hard-row mining
  agg      a global consistency term: the soft COUNT / SUM computed from the
           row decisions must equal the true COUNT / SUM. It catches small
           systematic biases that per-row losses average away.
  query    cross entropy for answer type, ordering and page number.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .batch import IGNORE
from .data import DIGITS, PAGE_SIZE, TYPES

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
