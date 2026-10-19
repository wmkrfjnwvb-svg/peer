"""Labels of the neural-database demo must match SQLite exactly."""

import os
import sqlite3
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from neuraldb.batch import collate  # noqa: E402
from neuraldb.data import TYPES, gen_example, to_sql  # noqa: E402
from neuraldb.losses import compute_loss, decode  # noqa: E402
from neuraldb.model import Config, NeuralDB  # noqa: E402


def sqlite_answer(e):
    t, q = e.table, e.query
    conn = sqlite3.connect(":memory:")
    cols = ", ".join(f'"{n}" {"INTEGER" if i else "TEXT"}' for n, i in zip(t.names, t.is_int))
    conn.execute(f"CREATE TABLE t ({cols})")
    rows = [tuple(int(v) if t.is_int[j] else f"w{int(v):03d}" for j, v in enumerate(r)) for r in t.values]
    conn.executemany(f"INSERT INTO t VALUES ({', '.join('?' * len(t.names))})", rows)
    res = conn.execute(to_sql(t, q)).fetchall()
    conn.close()
    if q.qtype == "page":
        ids = [r[0] for r in res]
        return ids[: 10], len(ids) > 10
    if q.qtype in ("min", "max"):
        return res[0][0] if res else None
    return res[0][0]


@pytest.mark.parametrize("seed", range(300))
def test_labels_match_sqlite(seed):
    rng = np.random.default_rng(seed)
    e = gen_example(rng, int(rng.choice([1, 5, 37, 200])), TYPES[seed % len(TYPES)])
    assert e.answer == sqlite_answer(e), e.query


def test_model_forward_loss_decode():
    rng = np.random.default_rng(0)
    exs = [gen_example(rng, 13, t) for t in TYPES]
    batch = collate(exs)
    model = NeuralDB(Config(d=32, heads=4, q_layers=1, r_layers=1))
    out = model(batch)
    loss, parts = compute_loss(out, batch)
    loss.backward()
    assert torch.isfinite(loss)
    out2 = model(batch, row_chunk=4)  # chunked reading gives the same row decisions
    assert torch.allclose(out["sel"], out2["sel"], atol=1e-5)
    assert len(decode(out, batch)) == len(exs)
