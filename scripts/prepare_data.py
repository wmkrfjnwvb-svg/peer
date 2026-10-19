"""Build verified training data.

For every (database, question, SQL) example:
  1. load the SQLite database into data weights,
  2. compile the gold SQL into a plan with the teacher compiler,
  3. run the plan on the exact executor,
  4. keep the plan only if its result matches SQLite.

Output is JSONL with one record per example; ``plan`` is null when the
example is outside the supported feature set. A coverage report is printed
and saved next to the output.

Examples:
  python scripts/prepare_data.py spider --root /data/spider/spider_data --out data/spider
  python scripts/prepare_data.py synsql --root /data/synsql --out data/synsql --limit 200000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import sqlglot  # noqa: E402
from sqlglot import exp  # noqa: E402

from peerdb import DataWeights, Unsupported, execute, format_plan, parse_plan  # noqa: E402
from peerdb.compare import normalize, same_result, sqlite_rows  # noqa: E402
from peerdb.compiler import Compiler  # noqa: E402
from peerdb.plan import Call, Lit, Step, reg_index  # noqa: E402


def top_level_ordered(sql: str) -> bool:
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except sqlglot.errors.SqlglotError:
        return "order by" in sql.lower()
    return isinstance(tree, exp.Select) and tree.args.get("order") is not None


def without_top_limit(sql: str, plan):
    """The query and plan with the outermost LIMIT removed, or None."""
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except sqlglot.errors.SqlglotError:
        return None
    if not isinstance(tree, exp.Select) or tree.args.get("limit") is None or tree.args.get("offset"):
        return None
    tree.set("limit", None)
    by_dest = {s.dest: s for s in plan if s.dest is not None}
    reg = reg_index(plan[-1].op.args[0])
    while reg in by_dest:
        step = by_dest[reg]
        if step.op.fn == "limit":
            new = [s if s is not step else Step(s.dest, Call("limit", (s.op.args[0], Lit(-1)))) for s in plan]
            return tree.sql(dialect="sqlite"), new
        if step.op.fn not in ("project", "sort", "distinct"):
            return None
        reg = reg_index(step.op.args[0])
    return None


def process_db(db_path: str, examples: list[dict], max_rows: int | None) -> list[dict]:
    out = []
    try:
        dw = DataWeights.from_sqlite(db_path, max_rows=max_rows)
        schema = dw.schema_text()
    except Exception as e:  # noqa: BLE001
        for ex in examples:
            out.append({**ex, "plan": None, "status": "db_load_error", "detail": str(e)[:200]})
        return out
    for ex in examples:
        rec = {**ex, "schema": schema, "plan": None}
        sql = ex["sql"]
        try:
            want = sqlite_rows(db_path, sql)
        except Exception as e:  # noqa: BLE001
            out.append({**rec, "status": "sqlite_error", "detail": str(e)[:200]})
            continue
        try:
            compiler = Compiler(dw)
            plan = compiler.compile(sql)
            text = format_plan(plan)
            got = execute(dw, parse_plan(text))
        except Unsupported as e:
            out.append({**rec, "status": "unsupported", "detail": str(e)[:200]})
            continue
        except Exception as e:  # noqa: BLE001
            out.append({**rec, "status": "error", "detail": f"{type(e).__name__}: {str(e)[:200]}"})
            continue
        ordered = top_level_ordered(sql)
        rec["ordered"] = ordered
        if same_result(got, want, ordered):
            out.append({**rec, "plan": text, "status": "ok"})
        elif ordered and same_result(got, want, False):
            # Same rows; SQL leaves the order of tied sort keys undefined.
            out.append({**rec, "plan": text, "status": "ok_tie_order"})
        elif ordered and _ties_explain(db_path, dw, sql, plan):
            # Without the LIMIT both sides return the same rows, so the LIMIT
            # cut through a run of tied sort keys.
            out.append({**rec, "plan": text, "status": "ok_tie_order"})
        elif compiler.bare_columns:
            out.append({**rec, "plan_tried": text, "status": "ambiguous_bare_column"})
        else:
            out.append({
                **rec, "plan_tried": text, "status": "mismatch",
                "detail": f"got {normalize(got)[:5]} want {normalize(want)[:5]}"[:400],
            })
    return out


def _ties_explain(db_path, dw, sql, plan) -> bool:
    alt = without_top_limit(sql, plan)
    if alt is None:
        return False
    alt_sql, alt_plan = alt
    try:
        return same_result(execute(dw, alt_plan), sqlite_rows(db_path, alt_sql), False)
    except Exception:  # noqa: BLE001
        return False


def load_spider(root: str):
    db_dir = os.path.join(root, "database")
    splits = {"train": ["train_spider.json", "train_others.json"], "dev": ["dev.json"]}
    test_db_dir = os.path.join(root, "test_database")
    if os.path.exists(os.path.join(root, "test.json")):
        splits["test"] = ["test.json"]
    for split, files in splits.items():
        for f in files:
            for ex in json.load(open(os.path.join(root, f))):
                d = test_db_dir if split == "test" else db_dir
                yield split, os.path.join(d, ex["db_id"], ex["db_id"] + ".sqlite"), {
                    "db_id": ex["db_id"], "question": ex["question"], "sql": ex["query"], "source": "spider",
                }


def load_synsql(root: str, limit: int | None, dev_fraction: float):
    data = os.path.join(root, "data.json")
    try:
        import ijson
    except ImportError:
        sys.exit("pip install ijson  (needed to stream the 9 GB data.json)")
    with open(data, "rb") as f:
        for i, ex in enumerate(ijson.items(f, "item")):
            if limit and i >= limit:
                break
            split = "dev" if (i % 1000) < dev_fraction * 1000 else "train"
            db = os.path.join(root, "databases", ex["db_id"], ex["db_id"] + ".sqlite")
            q = ex["question"]
            if ex.get("external_knowledge"):
                q = f"{q}\nHint: {ex['external_knowledge']}"
            yield split, db, {"db_id": ex["db_id"], "question": q, "sql": ex["sql"], "source": "synsql"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["spider", "synsql"])
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--max-rows", type=int, default=None, help="cap rows loaded per table")
    ap.add_argument("--dev-fraction", type=float, default=0.01)
    args = ap.parse_args()

    if args.dataset == "spider":
        items = load_spider(args.root)
    else:
        items = load_synsql(args.root, args.limit, args.dev_fraction)

    groups: dict[tuple, list] = defaultdict(list)
    n = 0
    for split, db, ex in items:
        if args.dataset == "spider" and args.limit and n >= args.limit:
            break
        groups[(split, db)].append({**ex, "split": split})
        n += 1
    print(f"{n} examples over {len(groups)} (split, database) groups", flush=True)

    os.makedirs(args.out, exist_ok=True)
    writers = {}
    stats: dict[str, Counter] = defaultdict(Counter)
    details: dict[str, Counter] = defaultdict(Counter)
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(process_db, db, exs, args.max_rows) for (split, db), exs in groups.items()]
        for fut in as_completed(futs):
            for rec in fut.result():
                split = rec["split"]
                if split not in writers:
                    writers[split] = open(os.path.join(args.out, f"{split}.jsonl"), "w")
                writers[split].write(json.dumps(rec, ensure_ascii=False) + "\n")
                stats[split][rec["status"]] += 1
                if rec["status"] in ("unsupported", "error"):
                    details[rec["status"]][rec["detail"].split(":")[0][:60]] += 1
                done += 1
            if done and done % 2000 < 50:
                print(f"  {done}/{n}  {time.time() - t0:.0f}s", flush=True)
    for w in writers.values():
        w.close()

    report = {"dataset": args.dataset, "examples": n, "splits": {}, "top_unsupported": {}}
    print("\ncoverage (plan verified against SQLite):")
    for split, c in sorted(stats.items()):
        total = sum(c.values())
        ok = c["ok"] + c["ok_tie_order"]
        runnable = total - c["sqlite_error"] - c["db_load_error"]
        print(f"  {split:6s} total={total:7d} verified={ok:7d} ({100 * ok / max(runnable, 1):.1f}% of runnable)  {dict(c)}")
        report["splits"][split] = {"total": total, "verified": ok, "runnable": runnable, **c}
    for status, c in details.items():
        report["top_unsupported"][status] = c.most_common(25)
        print(f"\ntop {status} reasons:")
        for reason, k in c.most_common(15):
            print(f"  {k:6d}  {reason}")
    json.dump(report, open(os.path.join(args.out, "coverage.json"), "w"), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
