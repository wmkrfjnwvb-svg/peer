"""Accuracy of the neural database by table size.

  python -m neuraldb.evaluate --ckpt runs/neuraldb/last.pt --sizes 10,100,1000,10000,100000
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
import torch

from .batch import IGNORE, collate
from .data import TYPES, gen_example
from .losses import decode
from .model import Config, NeuralDB

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


def print_report(report: dict):
    types = TYPES
    print(f"{'rows':>8} {'exact':>7} " + " ".join(f"{t:>6}" for t in types) + f" {'sel_err/row':>12} {'read_err/row':>12}")
    for n, r in report.items():
        cells = " ".join(f"{r['per_type'].get(t, float('nan')):6.3f}" for t in types)
        print(f"{n:8d} {r['exact']:7.4f} {cells} {r['row_select_error']:12.2e} {r['row_read_error']:12.2e}")
    print("\nconfidence threshold -> (answered fraction, accuracy of answered)")
    for n, r in report.items():
        cov = "  ".join(f"{t}: ({a:.3f}, {('%.4f' % p) if p is not None else '-'})" for t, (a, p) in r["coverage"].items())
        print(f"{n:8d}  {cov}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--sizes", default="10,100,1000,10000")
    ap.add_argument("--queries", type=int, default=300)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    state = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = NeuralDB(Config(**state["config"]))
    model.load_state_dict(state["model"])
    device = torch.device(args.device)
    model.to(device)
    sizes = [int(s) for s in args.sizes.split(",")]
    print_report(evaluate(model, sizes, args.queries, args.batch, device, amp=device.type == "cuda"))


if __name__ == "__main__":
    main()
