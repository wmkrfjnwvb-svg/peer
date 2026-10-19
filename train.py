"""Fine-tune a Qwen model to drive the data-weights executor.

Two tasks are trained together in one model:
  sql2plan  schema + SQL       -> operator plan   (the exact SQL path)
  nl2sql    schema + question  -> SQL             (natural-language front end)

Single GPU:
  python train.py --data data/spider --out runs/peer-1.5b

4 x RTX 4090 (24 GB), full fine-tune with FSDP sharding:
  torchrun --nproc_per_node 4 train.py --data data/spider --out runs/peer-1.5b --fsdp

Add more prepared datasets with repeated --data (e.g. SynSQL):
  torchrun --nproc_per_node 4 train.py --data data/spider --data data/synsql --fsdp
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import random

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from peerdb.prompts import examples_from_record


class SFTDataset(Dataset):
    def __init__(self, paths: list[str], split: str, tokenizer, max_len: int, tasks: set[str], limit: int | None):
        self.items = []
        skipped = 0
        records = []
        for p in paths:
            f = os.path.join(p, f"{split}.jsonl")
            if not os.path.exists(f):
                continue
            with open(f) as fh:
                for line in fh:
                    records.append(json.loads(line))
        random.Random(0).shuffle(records)
        eos = tokenizer.eos_token
        for rec in records:
            for task, prompt, target in examples_from_record(rec, tasks):
                p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
                t_ids = tokenizer(target + eos, add_special_tokens=False)["input_ids"]
                if len(p_ids) + len(t_ids) > max_len:
                    skipped += 1
                    continue
                self.items.append((p_ids, t_ids))
                if limit and len(self.items) >= limit:
                    break
            if limit and len(self.items) >= limit:
                break
        self.skipped = skipped

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, t = self.items[i]
        return {"input_ids": p + t, "labels": [-100] * len(p) + t}


class Collator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch):
        n = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), n), self.pad_id, dtype=torch.long)
        labels = torch.full((len(batch), n), -100, dtype=torch.long)
        mask = torch.zeros((len(batch), n), dtype=torch.long)
        for i, b in enumerate(batch):
            k = len(b["input_ids"])
            ids[i, :k] = torch.tensor(b["input_ids"])
            labels[i, :k] = torch.tensor(b["labels"])
            mask[i, :k] = 1
        return {"input_ids": ids, "labels": labels, "attention_mask": mask}


def dtype_kwarg(dtype) -> dict:
    """``dtype`` on recent transformers, ``torch_dtype`` on older ones."""
    params = inspect.signature(AutoModelForCausalLM.from_pretrained).parameters
    from transformers import __version__

    major, minor = (int(x) for x in __version__.split(".")[:2])
    return {"dtype": dtype} if (major, minor) >= (4, 56) or "dtype" in params else {"torch_dtype": dtype}


def training_args(warmup: float, **kw) -> TrainingArguments:
    """TrainingArguments across transformers 4.x and 5.x (renamed / removed fields)."""
    params = inspect.signature(TrainingArguments.__init__).parameters
    if "warmup_ratio" in params:
        kw["warmup_ratio"] = warmup
    else:
        kw["warmup_steps"] = warmup  # a float below 1 is a ratio in 5.x
    if "eval_strategy" not in params:
        kw["evaluation_strategy"] = kw.pop("eval_strategy")
    kw = {k: v for k, v in kw.items() if k in params}
    return TrainingArguments(**kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-1.5B")
    ap.add_argument("--data", action="append", required=True, help="prepared dataset dir (repeatable)")
    ap.add_argument("--out", default="runs/peer")
    ap.add_argument("--tasks", default="sql2plan,nl2sql")
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=4, help="per-device batch size")
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--fsdp", action="store_true", help="shard params/optimizer across GPUs (needed for 1.5B+ on 24 GB)")
    ap.add_argument("--limit", type=int, default=None, help="cap training examples (smoke tests)")
    ap.add_argument("--eval-limit", type=int, default=500)
    ap.add_argument("--save-steps", type=int, default=1000)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--no-bf16", action="store_true")
    args = ap.parse_args()

    tasks = set(args.tasks.split(","))
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    train = SFTDataset(args.data, "train", tok, args.max_len, tasks, args.limit)
    dev = SFTDataset(args.data, "dev", tok, args.max_len, tasks, args.eval_limit)
    if int(os.environ.get("RANK", "0")) == 0:
        print(f"train examples: {len(train)} (skipped {train.skipped} over max-len), dev: {len(dev)}")

    bf16 = not args.no_bf16 and torch.cuda.is_available()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, **dtype_kwarg(torch.bfloat16 if bf16 else torch.float32), attn_implementation="sdpa",
    )
    model.config.use_cache = False

    fsdp_kwargs = {}
    if args.fsdp:
        layer_cls = type(model.model.layers[0]).__name__
        fsdp_kwargs = {
            "fsdp": "full_shard auto_wrap",
            "fsdp_config": {"transformer_layer_cls_to_wrap": [layer_cls], "use_orig_params": True},
        }

    targs = training_args(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup=0.03,
        weight_decay=0.0,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=bf16,
        logging_steps=10,
        eval_strategy="steps" if len(dev) else "no",
        eval_steps=args.save_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        group_by_length=True,
        dataloader_num_workers=2,
        report_to="none",
        remove_unused_columns=False,
        **fsdp_kwargs,
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train,
        eval_dataset=dev if len(dev) else None,
        data_collator=Collator(tok.pad_token_id),
    )
    trainer.train()
    if args.fsdp:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
    trainer.save_model(os.path.join(args.out, "final"))
    if trainer.is_world_process_zero():
        tok.save_pretrained(os.path.join(args.out, "final"))


if __name__ == "__main__":
    main()
