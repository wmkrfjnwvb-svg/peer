"""Teacher compiler: SQL -> plan.

Used only to build training labels (and to measure executor coverage). At
inference time the model produces plans itself; this module is not needed.
"""

from __future__ import annotations

import sqlglot
from sqlglot import exp

from .plan import Call, Col, Lit, Name, Plan, Step
from .store import DataWeights
from .vec import Unsupported


class Scope:
    def __init__(self, parent: "Scope | None" = None):
        self.parent = parent
        self.tables: dict[str, list[str]] = {}  # alias -> column names (lowercase)

    def add(self, alias: str, columns: list[str]):
        a = alias.lower()
        if a in self.tables:
            raise Unsupported(f"duplicate alias {alias}")
        self.tables[a] = [c.lower() for c in columns]

    def resolve(self, name: str, table: str | None) -> Col:
        n = name.lower()
        if table:
            t = table.lower()
            if t in self.tables and n in self.tables[t]:
                return Col(t, n)
            if t in self.tables:
                raise KeyError(f"no column {table}.{name}")
            raise Unsupported(f"unknown table reference {table} (correlated subquery?)")
        hits = [a for a, cols in self.tables.items() if n in cols]
        if len(hits) == 1:
            return Col(hits[0], n)
        if len(hits) > 1:
            raise Unsupported(f"ambiguous column {name}")
        raise KeyError(f"no column {name}")


class Compiler:
    def __init__(self, dw: DataWeights):
        self.dw = dw

    def compile(self, sql: str) -> Plan:
        try:
            tree = sqlglot.parse_one(sql, read="sqlite")
        except sqlglot.errors.SqlglotError as e:
            raise Unsupported(f"parse error: {e}") from None
        self.steps: list[Step] = []
        self.ctes: dict[str, tuple[int, list[str]]] = {}
        # Set when an aggregate query selects a column that is neither grouped
        # nor aggregated; SQL leaves the chosen row unspecified.
        self.bare_columns = False
        out = self._query(tree)
        self.steps.append(Step(None, Call("return", (Name(f"r{out}"),))))
        return self.steps

    # --- helpers ------------------------------------------------------------------

    def _emit(self, fn: str, *args) -> int:
        r = len([s for s in self.steps if s.dest is not None])
        self.steps.append(Step(r, Call(fn, tuple(args))))
        return r

    @staticmethod
    def _reg(i: int) -> Name:
        return Name(f"r{i}")

    # --- queries ------------------------------------------------------------------

    def _query(self, node) -> int:
        if isinstance(node, exp.Subquery):
            return self._query(node.this)
        with_ = node.args.get("with_") or node.args.get("with")
        if with_ is not None:
            if with_.args.get("recursive"):
                raise Unsupported("recursive CTE")
            for cte in with_.expressions:
                reg = self._query(cte.this)
                cols = self._output_names(reg)
                self.ctes[cte.alias_or_name.lower()] = (reg, cols)
        if isinstance(node, (exp.Union, exp.Intersect, exp.Except)):
            for k in ("order", "limit", "offset"):
                if node.args.get(k):
                    raise Unsupported(f"{k} on a set operation")
            a = self._query(node.this)
            b = self._query(node.expression)
            if isinstance(node, exp.Union):
                fn = "union" if node.args.get("distinct", True) else "unionall"
            elif isinstance(node, exp.Intersect):
                fn = "intersect"
            else:
                fn = "except"
            return self._emit(fn, self._reg(a), self._reg(b))
        if not isinstance(node, exp.Select):
            raise Unsupported(f"statement {type(node).__name__}")
        return self._select(node)

    def _output_names(self, reg: int) -> list[str]:
        step = next(s for s in self.steps if s.dest == reg)
        op = step.op
        while op.fn not in ("project",):
            if op.fn in ("union", "unionall", "intersect", "except", "distinct"):
                src = int(op.args[0].text[1:])
                op = next(s for s in self.steps if s.dest == src).op
                continue
            raise Unsupported("cannot name CTE columns")
        names = []
        for i, e in enumerate(op.args[1:]):
            if isinstance(e, Call) and e.fn == "as":
                names.append(e.args[1].text.lower())
            elif isinstance(e, Col):
                names.append(e.name.lower())
            else:
                names.append(f"#{i}")
        return names

    def _source(self, table, scope: Scope) -> int:
        if isinstance(table, exp.Subquery):
            reg = self._query(table.this)
            alias = table.alias_or_name
            if not alias:
                raise Unsupported("subquery in FROM without alias")
            scope.add(alias, self._output_names(reg))
            return self._emit("scanreg", self._reg(reg), Name(alias.lower()))
        if not isinstance(table, exp.Table):
            raise Unsupported(f"FROM item {type(table).__name__}")
        name = table.name
        alias = table.alias_or_name
        if name.lower() in self.ctes:
            reg, cols = self.ctes[name.lower()]
            scope.add(alias, cols)
            return self._emit("scanreg", self._reg(reg), Name(alias.lower()))
        try:
            t = self.dw.table(name)
        except KeyError:
            raise KeyError(f"no table {name}") from None
        scope.add(alias, [c.name for c in t.columns])
        return self._emit("scan", Name(t.name), Name(alias.lower()))

    def _select(self, node: exp.Select) -> int:
        for k in ("windows", "qualify", "lateral", "laterals", "pivots", "sample", "locks"):
            if node.args.get(k):
                raise Unsupported(k)
        scope = Scope()
        from_ = node.args.get("from_") or node.args.get("from")
        if from_ is None:
            raise Unsupported("SELECT without FROM")
        cur = self._source(from_.this, scope)
        for j in node.args.get("joins") or []:
            right = self._source(j.this, scope)
            side = (j.args.get("side") or "").upper()
            kind = (j.args.get("kind") or "").upper()
            if j.args.get("using"):
                raise Unsupported("JOIN USING")
            on = j.args.get("on")
            if isinstance(on, exp.Boolean) and on.this:
                on = None  # sqlglot fills in ON TRUE for a bare JOIN
            if side == "LEFT":
                if on is None:
                    raise Unsupported("LEFT JOIN without ON")
                cur = self._emit("leftjoin", self._reg(cur), self._reg(right), self._expr(on, scope))
            elif side or kind not in ("", "INNER", "CROSS"):
                raise Unsupported(f"{side or kind} JOIN")
            else:
                args = [self._reg(cur), self._reg(right)]
                if on is not None:
                    args.append(self._expr(on, scope))
                cur = self._emit("join", *args)

        where = node.args.get("where")
        if where is not None:
            cur = self._emit("filter", self._reg(cur), self._expr(where.this, scope))

        # select list, with aliases
        items, alias_map = [], {}
        for e in node.expressions:
            if isinstance(e, exp.Star):
                for a, cols in scope.tables.items():
                    items.extend(Col(a, c) for c in cols)
                continue
            if isinstance(e, exp.Column) and isinstance(e.this, exp.Star):
                a = e.table.lower()
                items.extend(Col(a, c) for c in scope.tables[a])
                continue
            if isinstance(e, exp.Alias):
                pe = self._expr(e.this, scope)
                alias_map[e.alias.lower()] = pe
                items.append(Call("as", (pe, Name(e.alias.lower()))))
            else:
                items.append(self._expr(e, scope))
        plain_items = [i.args[0] if isinstance(i, Call) and i.fn == "as" else i for i in items]

        group = node.args.get("group")
        having = node.args.get("having")
        order = node.args.get("order")
        has_agg = any(_has_agg(i) for i in plain_items) or having is not None
        order_keys = []
        if order is not None:
            for o in order.expressions:
                key = self._order_expr(o.this, scope, alias_map, plain_items)
                order_keys.append(Call("desc" if o.args.get("desc") else "asc", (key,)))
                has_agg = has_agg or _has_agg(key)

        if group is not None:
            keys = [self._group_key(g, scope, alias_map) for g in group.expressions]
            cur = self._emit("group", self._reg(cur), *keys)
        elif has_agg:
            cur = self._emit("group", self._reg(cur))

        if group is not None or has_agg:
            grouped = set(keys) if group is not None else set()
            minmax = sum(_count_minmax(i) for i in plain_items)
            if minmax != 1 and any(_bare_cols(i, grouped) for i in plain_items):
                self.bare_columns = True

        if having is not None:
            cur = self._emit("filter", self._reg(cur), self._expr(having.this, scope, alias_map))

        limit = node.args.get("limit")
        offset = node.args.get("offset")
        lim_args = None
        if limit is not None:
            lim_args = [Lit(self._int_literal(limit.expression))]
            if offset is not None:
                lim_args.append(Lit(self._int_literal(offset.expression)))
        elif offset is not None:
            lim_args = [Lit(-1), Lit(self._int_literal(offset.expression))]

        distinct = node.args.get("distinct") is not None
        if distinct:
            if node.args["distinct"].args.get("on"):
                raise Unsupported("DISTINCT ON")
            cur = self._emit("project", self._reg(cur), *items)
            cur = self._emit("distinct", self._reg(cur))
            if order_keys:
                pos_keys = []
                for k in order_keys:
                    e = k.args[0]
                    if e not in plain_items:
                        raise Unsupported("ORDER BY expression not in DISTINCT select list")
                    pos_keys.append(Call(k.fn, (Call("pos", (Lit(plain_items.index(e)),)),)))
                cur = self._emit("sort", self._reg(cur), *pos_keys)
            if lim_args:
                cur = self._emit("limit", self._reg(cur), *lim_args)
            return cur

        if order_keys:
            cur = self._emit("sort", self._reg(cur), *order_keys)
        if lim_args:
            cur = self._emit("limit", self._reg(cur), *lim_args)
        return self._emit("project", self._reg(cur), *items)

    def _int_literal(self, e) -> int:
        if isinstance(e, exp.Literal) and not e.is_string:
            return int(e.this)
        if isinstance(e, exp.Neg) and isinstance(e.this, exp.Literal):
            return -int(e.this.this)
        raise Unsupported("non-literal LIMIT/OFFSET")

    def _order_expr(self, e, scope, alias_map, plain_items):
        if isinstance(e, exp.Literal) and not e.is_string:
            i = int(e.this) - 1
            if not 0 <= i < len(plain_items):
                raise Unsupported("ORDER BY position out of range")
            return plain_items[i]
        if isinstance(e, exp.Column) and not e.table and e.name.lower() in alias_map:
            try:
                return scope.resolve(e.name, None)
            except (KeyError, Unsupported):
                return alias_map[e.name.lower()]
        return self._expr(e, scope, alias_map)

    def _group_key(self, e, scope, alias_map):
        if isinstance(e, exp.Literal) and not e.is_string:
            raise Unsupported("GROUP BY position")
        if isinstance(e, exp.Column) and not e.table:
            try:
                return scope.resolve(e.name, None)
            except KeyError:
                if e.name.lower() in alias_map:
                    return alias_map[e.name.lower()]
                raise
        return self._expr(e, scope, alias_map)

    # --- expressions --------------------------------------------------------------

    def _expr(self, e, scope: Scope, alias_map: dict | None = None):
        X = lambda x: self._expr(x, scope, alias_map)  # noqa: E731
        if isinstance(e, exp.Paren):
            return X(e.this)
        if isinstance(e, exp.Column):
            if isinstance(e.this, exp.Star):
                raise Unsupported("table.* outside select list")
            try:
                return scope.resolve(e.name, e.table or None)
            except KeyError:
                if not e.table and alias_map and e.name.lower() in alias_map:
                    return alias_map[e.name.lower()]
                # SQLite treats an unresolvable "double quoted" name as a string
                if not e.table and e.this.args.get("quoted"):
                    return Lit(e.name)
                raise
        if isinstance(e, exp.Literal):
            if e.is_string:
                return Lit(e.this)
            text = e.this
            try:
                return Lit(int(text))
            except ValueError:
                return Lit(float(text))
        if isinstance(e, exp.Null):
            return Lit(None)
        if isinstance(e, exp.Boolean):
            return Lit(1 if e.this else 0)
        if isinstance(e, exp.Neg):
            inner = X(e.this)
            if isinstance(inner, Lit) and isinstance(inner.value, (int, float)):
                return Lit(-inner.value)
            return Call("neg", (inner,))
        cmp = {exp.EQ: "eq", exp.NEQ: "ne", exp.LT: "lt", exp.LTE: "le", exp.GT: "gt", exp.GTE: "ge"}
        for cls, fn in cmp.items():
            if type(e) is cls:
                return Call(fn, (X(e.this), X(e.expression)))
        arith = {exp.Add: "add", exp.Sub: "sub", exp.Mul: "mul", exp.Div: "div", exp.Mod: "mod"}
        for cls, fn in arith.items():
            if type(e) is cls:
                return Call(fn, (X(e.this), X(e.expression)))
        if isinstance(e, (exp.And, exp.Or)):
            fn = "and" if isinstance(e, exp.And) else "or"
            parts = []
            for side in (e.this, e.expression):
                sub = X(side)
                if isinstance(sub, Call) and sub.fn == fn:
                    parts.extend(sub.args)
                else:
                    parts.append(sub)
            return Call(fn, tuple(parts))
        if isinstance(e, exp.Not):
            inner = e.this
            if isinstance(inner, exp.Paren):
                inner = inner.this
            c = X(inner)
            flip = {"like": "notlike", "notlike": "like", "in": "notin", "notin": "in",
                    "insub": "notinsub", "notinsub": "insub", "isnull": "notnull", "notnull": "isnull"}
            if isinstance(c, Call) and c.fn in flip:
                return Call(flip[c.fn], c.args)
            if isinstance(c, Call) and c.fn == "not":
                return c.args[0]
            return Call("not", (c,))
        if isinstance(e, (exp.ILike, exp.Glob, exp.RegexpLike)):
            raise Unsupported(type(e).__name__)
        if isinstance(e, exp.Like):
            if e.args.get("escape"):
                raise Unsupported("LIKE ESCAPE")
            fn = "notlike" if e.args.get("negate") else "like"
            return Call(fn, (X(e.this), self._like_pattern(e.expression, scope)))
        if isinstance(e, exp.Is):
            if not isinstance(e.expression, exp.Null):
                raise Unsupported("IS with non-NULL")
            return Call("notnull" if e.args.get("negate") else "isnull", (X(e.this),))
        if isinstance(e, exp.Between):
            b = Call("between", (X(e.this), X(e.args["low"]), X(e.args["high"])))
            return Call("not", (b,)) if e.args.get("negate") else b
        if isinstance(e, exp.In):
            query = e.args.get("query")
            if query is not None:
                reg = self._query(query.this if isinstance(query, exp.Subquery) else query)
                fn = "notinsub" if e.args.get("negate") else "insub"
                return Call(fn, (X(e.this), self._reg(reg)))
            fn = "notin" if e.args.get("negate") else "in"
            return Call(fn, (X(e.this), *[X(v) for v in e.expressions]))
        if isinstance(e, exp.Subquery):
            reg = self._query(e.this)
            return Call("scalar", (self._reg(reg),))
        if isinstance(e, exp.Count):
            arg = e.this
            if isinstance(arg, exp.Star) or arg is None:
                return Call("count_star", ())
            if isinstance(arg, exp.Distinct):
                if len(arg.expressions) != 1:
                    raise Unsupported("COUNT(DISTINCT a, b)")
                return Call("count_distinct", (X(arg.expressions[0]),))
            return Call("count", (X(arg),))
        aggs = {exp.Sum: "sum", exp.Avg: "avg", exp.Min: "min", exp.Max: "max"}
        for cls, fn in aggs.items():
            if type(e) is cls:
                arg = e.this
                if isinstance(arg, exp.Distinct):
                    if fn in ("min", "max"):
                        return Call(fn, (X(arg.expressions[0]),))
                    return Call(fn + "_distinct", (X(arg.expressions[0]),))
                if e.expressions:
                    raise Unsupported(f"multi-argument {fn}()")
                return Call(fn, (X(arg),))
        simple = {exp.Lower: "lower", exp.Upper: "upper", exp.Length: "length", exp.Abs: "abs"}
        for cls, fn in simple.items():
            if type(e) is cls:
                return Call(fn, (X(e.this),))
        if isinstance(e, exp.Case):
            args = []
            subject = e.this
            for w in e.args.get("ifs") or []:
                cond = X(w.this) if subject is None else Call("eq", (X(subject), X(w.this)))
                args.extend([cond, X(w.args["true"])])
            default = e.args.get("default")
            args.append(X(default) if default is not None else Lit(None))
            return Call("case", tuple(args))
        if isinstance(e, exp.DPipe):
            return Call("concat", (X(e.this), X(e.expression)))
        if isinstance(e, exp.Coalesce):
            return Call("coalesce", (X(e.this), *[X(x) for x in e.expressions]))
        raise Unsupported(f"expression {type(e).__name__}")

    def _like_pattern(self, e, scope):
        if isinstance(e, exp.Literal) and e.is_string:
            return Lit(e.this)
        if isinstance(e, exp.Column) and not e.table and e.this.args.get("quoted"):
            try:
                scope.resolve(e.name, None)
            except (KeyError, Unsupported):
                return Lit(e.name)
        raise Unsupported("non-literal LIKE pattern")


def _has_agg(e) -> bool:
    from .executor import AGG_FNS

    if isinstance(e, Call):
        if e.fn in AGG_FNS:
            return True
        if e.fn in ("scalar", "insub", "notinsub"):
            return any(_has_agg(a) for a in e.args if not isinstance(a, Name))
        return any(_has_agg(a) for a in e.args)
    return False


def _count_minmax(e) -> int:
    if isinstance(e, Call):
        return (e.fn in ("min", "max")) + sum(_count_minmax(a) for a in e.args)
    return 0


def _bare_cols(e, grouped: set) -> bool:
    from .executor import AGG_FNS

    if e in grouped:
        return False
    if isinstance(e, Col):
        return True
    if isinstance(e, Call):
        if e.fn in AGG_FNS or e.fn in ("scalar", "insub", "notinsub"):
            return False
        return any(_bare_cols(a, grouped) for a in e.args)
    return False


def compile_sql(dw: DataWeights, sql: str) -> Plan:
    return Compiler(dw).compile(sql)
