"""Differential tests: the executor must match SQLite exactly on its feature set."""

import os
import random
import sqlite3
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from peerdb import DataWeights, Unsupported, compile_sql, execute, format_plan, parse_plan  # noqa: E402
from peerdb.compare import same_result  # noqa: E402


@pytest.fixture(scope="module")
def db(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("db") / "t.sqlite")
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE emp (id INTEGER PRIMARY KEY, name TEXT, dept INT, salary REAL, boss INT, code TEXT);
        CREATE TABLE dept (id INT, title TEXT, city TEXT);
        """
    )
    rnd = random.Random(7)
    names = ["Ann", "bob", "Cara", "dan", "Eve", "Ann", None, "Zoë", "émile", "Bob"]
    for i in range(1, 61):
        conn.execute(
            "INSERT INTO emp VALUES (?,?,?,?,?,?)",
            (
                i,
                rnd.choice(names),
                rnd.choice([1, 2, 3, 4, None]),
                rnd.choice([1000.5, 2000, 3000.25, None, 1500, 2000]),
                rnd.choice([None, 1, 2, 3]),
                rnd.choice(["10", "9", "abc", "100", None]),
            ),
        )
    for d, t, c in [(1, "Eng", "SF"), (2, "Ops", "NY"), (3, "Ops", None), (5, "Lab", "SF")]:
        conn.execute("INSERT INTO dept VALUES (?,?,?)", (d, t, c))
    conn.commit()
    conn.close()
    return path


QUERIES = [
    "SELECT count(*) FROM emp",
    "SELECT name, salary FROM emp WHERE salary > 1500 ORDER BY salary DESC, id",
    "SELECT dept, count(*), sum(salary), avg(salary), min(name), max(name) FROM emp GROUP BY dept",
    "SELECT count(DISTINCT name), count(name), count(salary) FROM emp",
    "SELECT e.name, d.title FROM emp AS e JOIN dept AS d ON e.dept = d.id WHERE d.city = 'SF'",
    "SELECT e.name, d.title FROM emp AS e LEFT JOIN dept AS d ON e.dept = d.id ORDER BY e.id",
    "SELECT name FROM emp WHERE name LIKE 'a%' ORDER BY id",
    "SELECT name FROM emp WHERE name NOT LIKE '%b%' ORDER BY id",
    "SELECT id FROM emp WHERE dept IN (1, 3) AND salary IS NOT NULL ORDER BY id",
    "SELECT id FROM emp WHERE dept NOT IN (SELECT id FROM dept WHERE city = 'SF') ORDER BY id",
    "SELECT id FROM emp WHERE salary > (SELECT avg(salary) FROM emp) ORDER BY id",
    "SELECT dept FROM emp GROUP BY dept HAVING count(*) > 10 ORDER BY count(*) DESC, dept",
    "SELECT DISTINCT name FROM emp ORDER BY name",
    "SELECT name FROM emp WHERE dept = 1 UNION SELECT name FROM emp WHERE dept = 2",
    "SELECT name FROM emp WHERE dept = 1 INTERSECT SELECT name FROM emp WHERE dept = 2",
    "SELECT name FROM emp WHERE dept = 1 EXCEPT SELECT name FROM emp WHERE dept = 2",
    "SELECT id, salary * 2 + 1, id / 7, id % 7, -salary FROM emp ORDER BY id",
    "SELECT id FROM emp WHERE salary BETWEEN 1500 AND 3000 ORDER BY id LIMIT 5 OFFSET 2",
    "SELECT id FROM emp WHERE code > 50 ORDER BY id",
    "SELECT id FROM emp WHERE code = 10 ORDER BY id",
    "SELECT title, count(*) FROM dept GROUP BY title ORDER BY count(*) DESC LIMIT 1",
    "SELECT d.city, count(*) FROM emp AS e JOIN dept AS d ON e.dept = d.id GROUP BY d.city",
    "SELECT max(salary), name FROM emp",
    "WITH big AS (SELECT dept, count(*) AS n FROM emp GROUP BY dept) SELECT dept FROM big WHERE n > 12",
    "SELECT lower(name), upper(name), length(name) FROM emp ORDER BY id",
    "SELECT id FROM emp WHERE NOT (dept = 1 OR dept IS NULL) ORDER BY id",
    "SELECT count(*) FROM emp, dept",
    "SELECT coalesce(name, 'none') FROM emp ORDER BY id",
    "SELECT id, CASE WHEN salary > 2000 THEN 'high' WHEN salary > 1000 THEN 'mid' ELSE 'low' END FROM emp ORDER BY id",
    "SELECT id, CASE dept WHEN 1 THEN 'one' WHEN 2 THEN 'two' END FROM emp ORDER BY id",
    "SELECT dept, sum(CASE WHEN salary > 1500 THEN 1 ELSE 0 END) FROM emp GROUP BY dept",
    "SELECT name || '-' || dept FROM emp ORDER BY id",
]


@pytest.mark.parametrize("sql", QUERIES)
def test_matches_sqlite(db, sql):
    dw = DataWeights.from_sqlite(db)
    plan = compile_sql(dw, sql)
    text = format_plan(plan)
    assert format_plan(parse_plan(text)) == text  # plan text round-trips
    got = execute(dw, parse_plan(text))
    conn = sqlite3.connect(db)
    want = conn.execute(sql).fetchall()
    conn.close()
    ordered = "ORDER BY" in sql and "UNION" not in sql
    assert same_result(got, want, ordered), f"\n{text}\ngot  {got[:8]}\nwant {want[:8]}"


def test_weights_roundtrip(db, tmp_path):
    dw = DataWeights.from_sqlite(db)
    path = str(tmp_path / "w.pt")
    dw.save(path)
    dw2 = DataWeights.load(path)
    sql = "SELECT dept, count(*), avg(salary) FROM emp GROUP BY dept"
    assert execute(dw, compile_sql(dw, sql)) == execute(dw2, compile_sql(dw2, sql))


def test_unsupported_is_reported(db):
    dw = DataWeights.from_sqlite(db)
    with pytest.raises(Unsupported):
        compile_sql(dw, "SELECT name, row_number() OVER (ORDER BY id) FROM emp")
