"""Evaluate a trained model end to end on the data-weights executor.

For each dev example:
  sql mode: schema + gold SQL  -> model plan -> executor -> compare with SQLite
  nl  mode: schema + question  -> model SQL  -> model plan -> executor -> compare
            with SQLite on the gold SQL

Every answer carries a confidence (the lowest token probability over the
generated text). The report shows accuracy at several confidence thresholds:
above the threshold the system answers, below it refuses. That is the
precision / coverage trade-off ("recall control").

  python eval.py --model runs/peer-1.5b/final --data data/spider --db-root /data/spider/spider_data --mode sql
  python eval.py --model runs/peer-1.5b/final --data data/spider --db-root /data/spider/spider_data --mode nl
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from peerdb import DataWeights, Unsupported, execute, parse_plan
from peerdb.compare import same_result, sqlite_rows
from peerdb.prompts import nl2sql_prompt, sql2plan_prompt
from train import dtype_kwarg

THRESHOLDS = [0.0, 0.5, 0.8, 0.9, 0.95, 0.99, 0.999]


class Generator:
    def __init__(self, model_path: str, device: str, max_new_tokens: int = 512):
        self.tok = AutoTokenizer.from_pretrained(model_path)
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **dtype_kwarg(dtype)).to(device).eval()
        self.device = device
        self.max_new_tokens = max_new_tokens

    @torch.no_grad()
    def __call__(self, prompts: list[str]) -> list[tuple[str, float]]:
        enc = self.tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)
        out = self.model.generate(
            **enc, max_new_tokens=self.max_new_tokens, do_sample=False, num_beams=1,
            return_dict_in_generate=True, output_scores=True, pad_token_id=self.tok.pad_token_id,
        )
        seqs = out.sequences[:, enc["input_ids"].shape[1]:]
        logp = self.model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)
        results = []
        for i in range(len(prompts)):
            ids, lp = seqs[i].tolist(), logp[i].tolist()
            keep_lp = []
            for t, l in zip(ids, lp):
                if t == self.tok.eos_token_id or t == self.tok.pad_token_id:
                    break
                keep_lp.append(l)
            text = self.tok.decode(ids, skip_special_tokens=True).strip()
            conf = float(torch.tensor(keep_lp).exp().min()) if keep_lp else 0.0
            results.append((text, conf))
        return results


def db_path(root: str, rec: dict) -> str:
    sub = "test_database" if rec.get("split") == "test" else "database"
    if rec.get("source") == "synsql":
        sub = "databases"
    return os.path.join(root, sub, rec["db_id"], rec["db_id"] + ".sqlite")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True, help="prepared dataset dir")
    ap.add_argument("--db-root", required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--mode", choices=["sql", "nl"], default="sql")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None, help="write per-example results as JSONL")
    args = ap.parse_args()

    recs = []
    with open(os.path.join(args.data, f"{args.split}.jsonl")) as f:
        for line in f:
            r = json.loads(line)
            if r["status"] in ("sqlite_error", "db_load_error"):
                continue
            if args.mode == "sql" and not r.get("plan"):
                continue  # outside the supported feature set
            recs.append(r)
    if args.limit:
        recs = recs[: args.limit]
    print(f"{len(recs)} examples, mode={args.mode}")

    gen = Generator(args.model, args.device)
    dws: dict[str, DataWeights] = {}
    results = []
    for start in range(0, len(recs), args.batch):
        batch = recs[start: start + args.batch]
        if args.mode == "nl":
            sqls = gen([nl2sql_prompt(r["schema"], r["question"]) for r in batch])
        else:
            sqls = [(r["sql"], 1.0) for r in batch]
        plans = gen([sql2plan_prompt(r["schema"], s) for r, (s, _) in zip(batch, sqls)])
        for r, (sql, c1), (plan_text, c2) in zip(batch, sqls, plans):
            path = db_path(args.db_root, r)
            if path not in dws:
                dws[path] = DataWeights.from_sqlite(path)
            res = {"db_id": r["db_id"], "sql": r["sql"], "pred_sql": sql, "pred_plan": plan_text,
                   "conf": min(c1, c2), "plan_exact": plan_text == r.get("plan")}
            try:
                got = execute(dws[path], parse_plan(plan_text))
                want = sqlite_rows(path, r["sql"])
                ordered = r.get("ordered", False)
                res["correct"] = same_result(got, want, ordered) or (
                    ordered and r.get("status") == "ok_tie_order" and same_result(got, want, False))
                res["outcome"] = "correct" if res["correct"] else "wrong_result"
            except Unsupported as e:
                res["correct"], res["outcome"] = False, f"unsupported: {e}"
            except Exception as e:  # noqa: BLE001
                res["correct"], res["outcome"] = False, f"invalid_plan: {type(e).__name__}"
            results.append(res)
        done = len(results)
        acc = sum(r["correct"] for r in results) / done
        print(f"  {done}/{len(recs)}  exec acc {acc:.4f}", flush=True)

    n = len(results)
    print(f"\n=== {args.mode} mode, {n} examples ===")
    print(f"execution accuracy: {sum(r['correct'] for r in results) / n:.4f}")
    if args.mode == "sql":
        print(f"plan exact match:   {sum(r['plan_exact'] for r in results) / n:.4f}")
    print("outcomes:", dict(Counter(r["outcome"].split(":")[0] for r in results)))
    print("\nthreshold  answered  coverage  precision  wrong_answers")
    for t in THRESHOLDS:
        ans = [r for r in results if r["conf"] >= t]
        if not ans:
            print(f"  {t:7.3f}  {0:8d}")
            continue
        right = sum(r["correct"] for r in ans)
        print(f"  {t:7.3f}  {len(ans):8d}  {len(ans) / n:8.4f}  {right / len(ans):9.4f}  {len(ans) - right:6d}")
    if args.out:
        with open(args.out, "w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
