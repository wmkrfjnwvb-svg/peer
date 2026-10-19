"""Interactive demo: ask in SQL or natural language, answer from data weights.

  python demo.py --model runs/peer-1.5b/final --db /data/spider/spider_data/database/concert_singer/concert_singer.sqlite

The SQLite file is read once and converted into a data-weights file
(``--weights``); after that every answer comes from the tensors, never from
SQLite. Input starting with SELECT/WITH is treated as SQL, anything else as a
question that the model first turns into SQL.
"""

from __future__ import annotations

import argparse
import os

import torch

from eval import Generator
from peerdb import DataWeights, Unsupported, execute, parse_plan
from peerdb.prompts import nl2sql_prompt, sql2plan_prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--db", help="SQLite file to import into data weights")
    ap.add_argument("--weights", help="data weights file (.pt); created from --db if missing")
    ap.add_argument("--threshold", type=float, default=0.9, help="refuse below this confidence")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    weights = args.weights or (os.path.splitext(args.db)[0] + ".weights.pt")
    if os.path.exists(weights):
        dw = DataWeights.load(weights)
    else:
        if not args.db:
            raise SystemExit("--db is required to create the weights file")
        dw = DataWeights.from_sqlite(args.db)
        dw.save(weights)
        print(f"imported {args.db} -> {weights}")
    schema = dw.schema_text()
    print(schema, "\n")
    gen = Generator(args.model, args.device)

    while True:
        try:
            q = input("query> ").strip()
        except EOFError:
            break
        if not q:
            continue
        conf = 1.0
        if q.split()[0].upper() in ("SELECT", "WITH"):
            sql = q
        else:
            sql, conf = gen([nl2sql_prompt(schema, q)])[0]
            print(f"sql ({conf:.3f}): {sql}")
        plan_text, c2 = gen([sql2plan_prompt(schema, sql)])[0]
        conf = min(conf, c2)
        print(f"plan ({c2:.3f}):\n{plan_text}")
        if conf < args.threshold:
            print(f"refused: confidence {conf:.3f} below {args.threshold}\n")
            continue
        try:
            rows = execute(dw, parse_plan(plan_text))
        except Unsupported as e:
            print(f"error: not supported ({e})\n")
            continue
        except Exception as e:  # noqa: BLE001
            print(f"error: invalid plan ({type(e).__name__}: {e})\n")
            continue
        for r in rows[:50]:
            print("  ", r)
        print(f"({len(rows)} rows, confidence {conf:.3f})\n")


if __name__ == "__main__":
    main()
