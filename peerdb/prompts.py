"""Prompt formats shared by training, evaluation and the demo."""

from __future__ import annotations


def sql2plan_prompt(schema: str, sql: str) -> str:
    return f"### schema\n{schema}\n### sql\n{sql.strip()}\n### plan\n"


def nl2sql_prompt(schema: str, question: str) -> str:
    return f"### schema\n{schema}\n### question\n{question.strip()}\n### sql\n"


def examples_from_record(rec: dict, tasks: set[str]) -> list[tuple[str, str, str]]:
    """(task, prompt, target) pairs for one prepared record."""
    out = []
    if rec.get("status") in ("sqlite_error", "db_load_error") or "schema" not in rec:
        return out
    if "sql2plan" in tasks and rec.get("plan"):
        out.append(("sql2plan", sql2plan_prompt(rec["schema"], rec["sql"]), rec["plan"]))
    if "nl2sql" in tasks and rec.get("question"):
        out.append(("nl2sql", nl2sql_prompt(rec["schema"], rec["question"]), " ".join(rec["sql"].split())))
    return out
