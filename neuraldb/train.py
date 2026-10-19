"""Train the end-to-end neural database on fresh random tables.

CPU / single GPU:
  python -m neuraldb.train --out runs/neuraldb

4 x RTX 4090:
  torchrun --nproc_per_node 4 -m neuraldb.train --out runs/neuraldb --d 512 --r-layers 6 \\
      --batch 16 --n-max 1000 --steps 200000
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from .batch import collate
from .data import gen_example
from .evaluate import evaluate, print_report
from .losses import LossConfig, compute_loss
from .model import Config, NeuralDB


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
            b = max(1, min(self.batch, self.max_rows // n))  # keep rows per batch bounded
            yield collate([gen_example(rng, n) for _ in range(b)])
            step += nw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/neuraldb")
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--q-layers", type=int, default=4)
    ap.add_argument("--r-layers", type=int, default=4)
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--batch", type=int, default=16, help="tables per batch per GPU")
    ap.add_argument("--max-rows-per-batch", type=int, default=16_000)
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
    args = ap.parse_args()

    ddp = "RANK" in os.environ
    if ddp:
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
        local = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local)
        device = torch.device("cuda", local)
    else:
        rank, world = 0, 1
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
            print_report(report)
            torch.save({"model": model.state_dict(), "config": asdict(cfg), "step": step}, os.path.join(args.out, "last.pt"))
            with open(os.path.join(args.out, "eval.jsonl"), "a") as f:
                f.write(json.dumps({"step": step, "report": {str(k): v for k, v in report.items()}}, default=str) + "\n")
    if rank == 0:
        torch.save({"model": model.state_dict(), "config": asdict(cfg), "step": args.steps}, os.path.join(args.out, "last.pt"))
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
