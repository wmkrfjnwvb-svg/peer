"""Exact operator experts.

Each plan operator is a deterministic tensor routine over the data weights.
Nothing here is learned: the model only chooses which operators to run and
what they point at, and these routines compute the answer exactly. Features
that are not implemented raise ``Unsupported`` instead of returning a guess.
"""

from __future__ import annotations

import re

import torch

from .plan import Call, Col, Lit, Name, Plan, reg_index
from .store import DataWeights
from .vec import (
    NULL,
    NUM,
    TEXT,
    Bool,
    Unsupported,
    Vec,
    apply_numeric,
    apply_text,
    as_bool,
    bcast_len,
    coerce_pair,
    compare,
    row_keys,
)

MAX_CROSS_ROWS = 20_000_000

AGG_FNS = {
    "count_star", "count", "count_distinct", "sum", "sum_distinct",
    "avg", "avg_distinct", "min", "max",
}
CMP_FNS = {"eq", "ne", "lt", "le", "gt", "ge"}


class Rel:
    """A relation in working memory: named columns of equal length."""

    def __init__(self, names: list[tuple[str, str]], vecs: list[Vec | None], affs: list[str | None], n: int):
        self.names = names
        self.vecs = vecs
        self.affs = affs
        self.n = n

    def find(self, alias: str, name: str) -> int:
        key = (alias.lower(), name.lower())
        hits = [i for i, nm in enumerate(self.names) if nm == key]
        if len(hits) != 1:
            raise KeyError(f"column {alias}.{name} {'not found' if not hits else 'ambiguous'}")
        return hits[0]

    def take(self, idx: torch.Tensor) -> "Rel":
        return Rel(self.names, [v.take(idx) if v is not None else None for v in self.vecs], self.affs, int(idx.shape[0]))

    def rows(self, strings) -> list[tuple]:
        cols = [v.to_python(strings) for v in self.vecs]
        return list(zip(*cols)) if cols else [() for _ in range(self.n)]


class Grouped:
    """A relation partitioned into groups (output of ``group``)."""

    def __init__(self, rel: Rel, gid: torch.Tensor, ngroups: int, keys: tuple = ()):
        self.rel = rel
        self.keys = keys
        self.gid = gid
        self.ngroups = ngroups
        self.order = torch.arange(ngroups, dtype=torch.int64)
        self.minmax_calls: list[Call] = []
        idx = torch.arange(rel.n, dtype=torch.int64)
        last = torch.full((ngroups,), -1, dtype=torch.int64)
        if rel.n:
            last = last.scatter_reduce(0, gid, idx, reduce="amax", include_self=True)
        self.last_row = last


def _sort_rank(v: Vec, strings) -> torch.Tensor:
    """Dense rank consistent with SQLite ordering (NULL < numbers < text)."""
    n = len(v)
    if n == 0:
        return torch.zeros(0, dtype=torch.int64)
    value = torch.where(v.cls == NUM, v.num, torch.zeros_like(v.num))
    if (v.cls == TEXT).any():
        trank = strings.rank()[v.code.clamp(min=0)].to(torch.float64)
        value = torch.where(v.cls == TEXT, trank, value)
    order = torch.argsort(value, stable=True)
    order = order[torch.argsort(v.cls[order].to(torch.int64), stable=True)]
    sc, sv = v.cls[order].to(torch.int64), value[order]
    # integer ties beyond float precision are rare; compare exact ints too
    si = torch.where(v.isint[order], v.ival[order], torch.zeros_like(v.ival))
    change = torch.ones(n, dtype=torch.bool)
    change[1:] = (sc[1:] != sc[:-1]) | (sv[1:] != sv[:-1]) | (si[1:] != si[:-1])
    dense = torch.cumsum(change.to(torch.int64), 0)
    rank = torch.empty(n, dtype=torch.int64)
    rank[order] = dense
    return rank


def _dense_ids(*key_mats: torch.Tensor) -> list[torch.Tensor]:
    """Map rows of several key matrices to shared dense integer ids."""
    sizes = [k.shape[0] for k in key_mats]
    allk = torch.cat(key_mats, dim=0)
    if allk.shape[0] == 0:
        return [torch.zeros(0, dtype=torch.int64) for _ in key_mats]
    _, inv = torch.unique(allk, dim=0, return_inverse=True)
    return list(torch.split(inv, sizes))


def _first_occurrence(ids: torch.Tensor) -> torch.Tensor:
    """Indices of the first occurrence of each distinct id, in input order."""
    if ids.numel() == 0:
        return ids
    n = ids.shape[0]
    m = int(ids.max()) + 1
    first = torch.full((m,), n, dtype=torch.int64)
    first = first.scatter_reduce(0, ids, torch.arange(n, dtype=torch.int64), reduce="amin")
    first = first[first < n]
    return torch.sort(first).values


_ASCII_LOWER = {i: i + 32 for i in range(ord("A"), ord("Z") + 1)}


def _like_regex(pattern: str) -> re.Pattern:
    out = []
    for ch in pattern.translate(_ASCII_LOWER):
        if ch == "%":
            out.append(".*")
        elif ch == "_":
            out.append(".")
        else:
            out.append(re.escape(ch))
    return re.compile("".join(out), re.DOTALL)


class Executor:
    def __init__(self, dw: DataWeights):
        self.dw = dw
        self.strings = dw.strings

    # --- plan -------------------------------------------------------------------

    def run(self, plan: Plan) -> Rel:
        self.regs: dict[int, object] = {}
        for step in plan:
            if step.dest is None:
                out = self._reg(step.op.args[0])
                if not isinstance(out, Rel):
                    raise Unsupported("plan must return a projected relation")
                return out
            if step.dest in self.regs:
                raise ValueError(f"register r{step.dest} assigned twice")
            self.regs[step.dest] = self._op(step.op)
        raise ValueError("plan has no return")

    def _reg(self, x):
        i = reg_index(x)
        if i not in self.regs:
            raise ValueError(f"register r{i} used before assignment")
        return self.regs[i]

    def _op(self, c: Call):
        fn = getattr(self, f"op_{c.fn}", None)
        if fn is None:
            raise Unsupported(f"operator {c.fn}")
        return fn(*c.args)

    # --- operators --------------------------------------------------------------

    def op_scan(self, table: Name, alias: Name):
        t = self.dw.table(table.text)
        a = alias.text.lower()
        return Rel(
            [(a, c.name.lower()) for c in t.columns],
            [c.vec for c in t.columns],
            [c.affinity for c in t.columns],
            t.nrows,
        )

    def op_scanreg(self, reg: Name, alias: Name):
        src = self._reg(reg)
        if not isinstance(src, Rel):
            raise Unsupported("scanreg of a grouped relation")
        a = alias.text.lower()
        return Rel([(a, nm[1]) for nm in src.names], src.vecs, src.affs, src.n)

    def op_join(self, left: Name, right: Name, cond=None):
        return self._join(self._rel(left), self._rel(right), cond, outer=False)

    def op_leftjoin(self, left: Name, right: Name, cond=None):
        return self._join(self._rel(left), self._rel(right), cond, outer=True)

    def op_filter(self, src: Name, cond):
        r = self._reg(src)
        if isinstance(r, Grouped):
            mask = as_bool(self._eval_group(cond, r)).is_true().expand(r.ngroups)
            r.order = r.order[mask[r.order]]
            return r
        mask = as_bool(self._eval(cond, r)[0]).is_true().expand(r.n)
        return r.take(torch.nonzero(mask).flatten())

    def op_group(self, src: Name, *keys):
        r = self._rel(src)
        if keys:
            vecs = []
            for k in keys:
                v = self._eval(k, r)[0]
                vecs.append(v.to_vec() if isinstance(v, Bool) else v.expand(r.n))
            if r.n:
                _, gid = torch.unique(row_keys(vecs), dim=0, return_inverse=True)
                ngroups = int(gid.max()) + 1
            else:
                gid, ngroups = torch.zeros(0, dtype=torch.int64), 0
        else:
            gid, ngroups = torch.zeros(r.n, dtype=torch.int64), 1
        g = Grouped(r, gid, ngroups, keys)
        # SQLite assigns groups in key order; keep that so tie-breaks match.
        if keys and ngroups:
            g.order = self._sort_order([self._eval_group(k, g) for k in keys], [False] * len(keys), g.order)
        return g

    def op_sort(self, src: Name, *keys):
        r = self._reg(src)
        exprs, desc = [], []
        for k in keys:
            if not (isinstance(k, Call) and k.fn in ("asc", "desc") and len(k.args) == 1):
                raise Unsupported("sort keys must be asc(...) or desc(...)")
            exprs.append(k.args[0])
            desc.append(k.fn == "desc")
        if isinstance(r, Grouped):
            self._note_minmax(r, exprs)
            if r.keys and len(r.keys) == len(exprs):
                # SQLite emits groups in GROUP BY order, with each GROUP BY term
                # taking the direction of the matching ORDER BY term when both
                # have the same length; ties in the ORDER BY keep that order.
                kv = [self._eval_group(k, r) for k in r.keys]
                r.order = self._sort_order(kv, desc, r.order)
            vals = [self._eval_group(e, r) for e in exprs]
            r.order = self._sort_order(vals, desc, r.order)
            return r
        vals = [self._eval(e, r)[0] for e in exprs]
        idx = self._sort_order(vals, desc, torch.arange(r.n, dtype=torch.int64))
        return r.take(idx)

    def op_limit(self, src: Name, count: Lit, offset: Lit | None = None):
        r = self._reg(src)
        n, off = int(count.value), int(offset.value) if offset is not None else 0
        if isinstance(r, Grouped):
            r.order = r.order[off: off + n] if n >= 0 else r.order[off:]
            return r
        stop = r.n if n < 0 else min(r.n, off + n)
        return r.take(torch.arange(min(off, r.n), stop, dtype=torch.int64))

    def op_project(self, src: Name, *exprs):
        r = self._reg(src)
        names, items = [], []
        for i, e in enumerate(exprs):
            if isinstance(e, Call) and e.fn == "as":
                names.append(("", e.args[1].text.lower()))
                items.append(e.args[0])
            else:
                names.append(("", e.name.lower() if isinstance(e, Col) else f"#{i}"))
                items.append(e)
        if isinstance(r, Grouped):
            self._note_minmax(r, items)
            vecs, affs = [], []
            for e in items:
                v = self._eval_group(e, r)
                v = v.to_vec() if isinstance(v, Bool) else v
                vecs.append(v.expand(r.ngroups).take(r.order))
                affs.append(self._affinity(e, r.rel))
            return Rel(names, vecs, affs, int(r.order.shape[0]))
        vecs, affs = [], []
        for e in items:
            if self._has_agg(e):
                raise Unsupported("aggregate outside of group")
            v, aff = self._eval(e, r)
            v = v.to_vec() if isinstance(v, Bool) else v
            vecs.append(v.expand(r.n))
            affs.append(aff)
        return Rel(names, vecs, affs, r.n)

    def op_distinct(self, src: Name):
        r = self._rel(src)
        if r.n == 0:
            return r
        (ids,) = _dense_ids(row_keys(r.vecs))
        return r.take(_first_occurrence(ids))

    def op_unionall(self, a: Name, b: Name):
        ra, rb = self._setop_inputs(a, b)
        return Rel(ra.names, [Vec.cat([x, y]) for x, y in zip(ra.vecs, rb.vecs)], ra.affs, ra.n + rb.n)

    def op_union(self, a: Name, b: Name):
        r = self.op_unionall(a, b)
        if r.n == 0:
            return r
        (ids,) = _dense_ids(row_keys(r.vecs))
        return r.take(_first_occurrence(ids))

    def op_intersect(self, a: Name, b: Name):
        return self._member_setop(a, b, keep_members=True)

    def op_except(self, a: Name, b: Name):
        return self._member_setop(a, b, keep_members=False)

    # --- operator helpers -------------------------------------------------------

    def _rel(self, x) -> Rel:
        r = self._reg(x)
        if not isinstance(r, Rel):
            raise Unsupported("operator needs an ungrouped relation")
        return r

    def _setop_inputs(self, a, b):
        ra, rb = self._rel(a), self._rel(b)
        if len(ra.vecs) != len(rb.vecs):
            raise ValueError("set operation column count mismatch")
        return ra, rb

    def _member_setop(self, a, b, keep_members: bool):
        ra, rb = self._setop_inputs(a, b)
        ka, kb = row_keys(ra.vecs), row_keys(rb.vecs)
        ida, idb = _dense_ids(ka, kb)
        first = _first_occurrence(ida) if ra.n else ida
        member = torch.isin(ida[first], idb) if ra.n else torch.zeros(0, dtype=torch.bool)
        keep = first[member] if keep_members else first[~member]
        return ra.take(keep)

    def _sort_order(self, vals, desc, order: torch.Tensor) -> torch.Tensor:
        """Stable multi-key sort of ``order`` (row or group indices)."""
        ranks = []
        for v in vals:
            v = v.to_vec() if isinstance(v, Bool) else v
            ranks.append(_sort_rank(v, self.strings))
        for rank, d in reversed(list(zip(ranks, desc))):
            if rank.shape[0] == 1 and order.shape[0] != 1:
                continue  # constant key
            key = rank[order]
            perm = torch.sort(key, descending=d, stable=True).indices
            order = order[perm]
        return order

    def _join(self, left: Rel, right: Rel, cond, outer: bool) -> Rel:
        names = left.names + right.names
        affs = left.affs + right.affs

        equi, residual = [], []
        conjuncts = self._conjuncts(cond) if cond is not None else []
        for c in conjuncts:
            pair = self._equi_pair(c, left, right)
            (equi if pair else residual).append(pair or c)

        if outer and residual:
            raise Unsupported("LEFT JOIN with non-equality ON condition")

        nl, nr = left.n, right.n
        if equi:
            lvecs, rvecs = [], []
            for li, ri in equi:
                lv, rv = coerce_pair(
                    self._need(left.vecs[li]), left.affs[li], self._need(right.vecs[ri]), right.affs[ri], self.strings
                )
                lvecs.append(lv)
                rvecs.append(rv)
            lnull = torch.zeros(nl, dtype=torch.bool)
            for v in lvecs:
                lnull |= v.cls == NULL
            rnull = torch.zeros(nr, dtype=torch.bool)
            for v in rvecs:
                rnull |= v.cls == NULL
            lid, rid = _dense_ids(row_keys(lvecs), row_keys(rvecs))
            big = torch.iinfo(torch.int64).max
            rid = torch.where(rnull, torch.full_like(rid, big), rid)
            rs, rorder = torch.sort(rid, stable=True)
            start = torch.searchsorted(rs, lid, right=False)
            end = torch.searchsorted(rs, lid, right=True)
            counts = torch.where(lnull, torch.zeros_like(start), end - start)
        else:
            if nl * nr > MAX_CROSS_ROWS:
                raise Unsupported("cross join too large")
            start = torch.zeros(nl, dtype=torch.int64)
            counts = torch.full((nl,), nr, dtype=torch.int64)
            rorder = torch.arange(nr, dtype=torch.int64)

        if outer:
            out_counts = torch.clamp(counts, min=1)
        else:
            out_counts = counts
        total = int(out_counts.sum())
        li = torch.repeat_interleave(torch.arange(nl, dtype=torch.int64), out_counts)
        offs = torch.arange(total, dtype=torch.int64) - torch.repeat_interleave(
            torch.cumsum(out_counts, 0) - out_counts, out_counts
        )
        matched = counts[li] > 0
        ri = torch.where(matched, rorder[(start[li] + offs).clamp(max=max(nr - 1, 0))] if nr else offs, -1)

        out_vecs = [v.take(li) if v is not None else None for v in left.vecs]
        for v in right.vecs:
            if v is None:
                out_vecs.append(None)
                continue
            taken = v.take(ri.clamp(min=0)) if nr else Vec.empty(total)
            if outer:
                nullrow = ri < 0
                taken = Vec(
                    torch.where(nullrow, torch.tensor(NULL, dtype=torch.int8), taken.cls),
                    taken.num, taken.isint & ~nullrow, taken.ival, taken.code,
                )
            out_vecs.append(taken)
        out = Rel(names, out_vecs, affs, total)
        for c in residual:
            mask = as_bool(self._eval(c, out)[0]).is_true().expand(out.n)
            out = out.take(torch.nonzero(mask).flatten())
        return out

    def _conjuncts(self, cond):
        if isinstance(cond, Call) and cond.fn == "and":
            out = []
            for a in cond.args:
                out.extend(self._conjuncts(a))
            return out
        return [cond]

    def _equi_pair(self, c, left: Rel, right: Rel):
        if not (isinstance(c, Call) and c.fn == "eq" and all(isinstance(a, Col) for a in c.args)):
            return None
        a, b = c.args
        for x, y in ((a, b), (b, a)):
            try:
                li = left.find(x.alias, x.name)
                ri = right.find(y.alias, y.name)
                return (li, ri)
            except KeyError:
                continue
        return None

    def _need(self, v):
        if v is None:
            raise Unsupported("column with unsupported data (BLOB)")
        return v

    def _note_minmax(self, g: Grouped, exprs):
        for e in exprs:
            for call in self._agg_calls(e):
                if call.fn in ("min", "max") and call not in g.minmax_calls:
                    g.minmax_calls.append(call)

    # --- expressions ------------------------------------------------------------

    def _has_agg(self, e) -> bool:
        return bool(self._agg_calls(e))

    def _agg_calls(self, e) -> list[Call]:
        if isinstance(e, Call):
            if e.fn in AGG_FNS:
                return [e]
            out = []
            for a in e.args:
                out.extend(self._agg_calls(a))
            return out
        return []

    def _affinity(self, e, rel: Rel):
        if isinstance(e, Col):
            try:
                return rel.affs[rel.find(e.alias, e.name)]
            except KeyError:
                return None
        if isinstance(e, Call) and e.fn in ("min", "max") and len(e.args) == 1:
            return self._affinity(e.args[0], rel)
        return None

    def _eval(self, e, rel: Rel):
        """Row-wise evaluation. Returns (Vec | Bool, affinity)."""
        if isinstance(e, Col):
            i = rel.find(e.alias, e.name)
            return self._need(rel.vecs[i]), rel.affs[i]
        if isinstance(e, Lit):
            return Vec.scalar(e.value, self.strings), None
        if isinstance(e, Call):
            if e.fn == "pos":
                i = int(e.args[0].value)
                return rel.vecs[i], rel.affs[i]
            if e.fn in AGG_FNS:
                raise Unsupported("aggregate outside of group")
            return self._call(e, lambda x: self._eval(x, rel), rel.n)
        raise Unsupported(f"expression {e!r}")

    def _eval_group(self, e, g: Grouped):
        """Per-group evaluation (length ngroups, or 1 for constants)."""
        if isinstance(e, Call) and e.fn in AGG_FNS:
            return self._aggregate(e, g)
        if isinstance(e, Call) and e.fn not in ("pos",):
            return self._call(e, lambda x: (self._eval_group(x, g), self._affinity(x, g.rel)), g.ngroups)[0]
        if isinstance(e, Lit):
            return Vec.scalar(e.value, self.strings)
        # bare column: value from the representative row of each group
        v = self._eval(e, g.rel)[0]
        v = v.to_vec() if isinstance(v, Bool) else v
        rep = self._rep_rows(g)
        out = v.expand(max(g.rel.n, 1)).take(rep.clamp(min=0)) if g.rel.n else Vec.empty(g.ngroups)
        missing = rep < 0
        if missing.any():
            out = Vec(torch.where(missing, torch.tensor(NULL, dtype=torch.int8), out.cls),
                      out.num, out.isint & ~missing, out.ival, out.code)
        return out

    def _rep_rows(self, g: Grouped) -> torch.Tensor:
        # SQLite: with exactly one min()/max() aggregate, bare columns come from
        # the row holding the extreme value; otherwise from the last row.
        if len(g.minmax_calls) == 1:
            return self._extreme_rows(g.minmax_calls[0], g)
        return g.last_row

    def _extreme_rows(self, call: Call, g: Grouped) -> torch.Tensor:
        v = self._eval(call.args[0], g.rel)[0]
        v = (v.to_vec() if isinstance(v, Bool) else v).expand(g.rel.n)
        rank = _sort_rank(v, self.strings)
        n = g.rel.n
        rows = torch.arange(n, dtype=torch.int64)
        key = (rank if call.fn == "min" else -rank) * (n + 1) + rows
        big = torch.iinfo(torch.int64).max
        key = torch.where(v.cls == NULL, torch.full_like(key, big), key)
        best = torch.full((g.ngroups,), big, dtype=torch.int64)
        if n:
            best = best.scatter_reduce(0, g.gid, key, reduce="amin")
        found = best != big
        row = torch.where(found, torch.remainder(best, n + 1), torch.full_like(best, -1))
        return torch.where(found, row, g.last_row)

    def _aggregate(self, e: Call, g: Grouped) -> Vec:
        k, n = g.ngroups, g.rel.n
        if e.fn == "count_star":
            cnt = torch.zeros(k, dtype=torch.int64).scatter_add(0, g.gid, torch.ones(n, dtype=torch.int64))
            return _int_vec(cnt)
        v = self._eval(e.args[0], g.rel)[0]
        v = (v.to_vec() if isinstance(v, Bool) else v).expand(n)
        gid = g.gid
        valid = v.cls != NULL
        if e.fn.endswith("_distinct"):
            keys = torch.cat([gid.unsqueeze(1), row_keys([v])], dim=1)
            (ids,) = _dense_ids(keys)
            keep = _first_occurrence(ids) if n else ids
            v, gid, valid = v.take(keep), gid[keep], valid[keep]
        base = e.fn.replace("_distinct", "")
        cnt = torch.zeros(k, dtype=torch.int64).scatter_add(0, gid, valid.to(torch.int64))
        if base == "count":
            return _int_vec(cnt)
        if base in ("min", "max"):
            rank = _sort_rank(v, self.strings)
            m = v.cls.shape[0]
            rows = torch.arange(m, dtype=torch.int64)
            key = (rank if base == "min" else -rank) * (m + 1) + rows
            big = torch.iinfo(torch.int64).max
            key = torch.where(valid, key, torch.full_like(key, big))
            best = torch.full((k,), big, dtype=torch.int64)
            if m:
                best = best.scatter_reduce(0, gid, key, reduce="amin")
            found = best != big
            row = torch.remainder(best, m + 1).clamp(max=max(m - 1, 0))
            out = v.take(row) if m else Vec.empty(k)
            return Vec(torch.where(found, out.cls, torch.tensor(NULL, dtype=torch.int8)),
                       out.num, out.isint & found, out.ival, out.code)
        # sum / avg need numbers
        v = apply_numeric(v, self.strings)
        if ((v.cls == TEXT) & valid).any():
            raise Unsupported("sum/avg over non-numeric text")
        all_int = bool((v.isint | ~valid).all())
        if base == "sum" and all_int:
            s = torch.zeros(k, dtype=torch.int64).scatter_add(0, gid, torch.where(valid, v.ival, 0))
            out = _int_vec(s)
        else:
            s = torch.zeros(k, dtype=torch.float64).scatter_add(0, gid, torch.where(valid, v.num, 0.0))
            if base == "avg":
                s = s / cnt.clamp(min=1).to(torch.float64)
            out = _real_vec(s)
        empty = cnt == 0
        return Vec(torch.where(empty, torch.tensor(NULL, dtype=torch.int8), out.cls),
                   out.num, out.isint & ~empty, out.ival, out.code)

    def _call(self, e: Call, ev, n: int):
        fn, args = e.fn, e.args
        S = self.strings
        if fn in CMP_FNS:
            (a, aa), (b, ab) = ev(args[0]), ev(args[1])
            a, b = _vec(a), _vec(b)
            a, b = coerce_pair(a, aa, b, ab, S)
            sign, null = compare(a, b, S)
            val = {"eq": sign == 0, "ne": sign != 0, "lt": sign < 0,
                   "le": sign <= 0, "gt": sign > 0, "ge": sign >= 0}[fn]
            return Bool(val & ~null, null), None
        if fn in ("and", "or"):
            parts = [as_bool(ev(a)[0]) for a in args]
            m = bcast_len(*(len(p) for p in parts))
            parts = [Bool(p.val.expand(m), p.null.expand(m)) for p in parts]
            if fn == "and":
                false = torch.zeros(m, dtype=torch.bool)
                null = torch.zeros(m, dtype=torch.bool)
                for p in parts:
                    false |= ~p.val & ~p.null
                    null |= p.null
                return Bool(~false & ~null, null & ~false), None
            true = torch.zeros(m, dtype=torch.bool)
            null = torch.zeros(m, dtype=torch.bool)
            for p in parts:
                true |= p.is_true()
                null |= p.null
            return Bool(true, null & ~true), None
        if fn == "not":
            p = as_bool(ev(args[0])[0])
            return Bool(~p.val & ~p.null, p.null), None
        if fn in ("isnull", "notnull"):
            v = _vec(ev(args[0])[0])
            isn = v.cls == NULL
            return Bool(isn if fn == "isnull" else ~isn, torch.zeros_like(isn)), None
        if fn == "between":
            lo = self._call(Call("ge", (args[0], args[1])), ev, n)[0]
            hi = self._call(Call("le", (args[0], args[2])), ev, n)[0]
            return self._call(Call("and", (_Const(lo), _Const(hi))), lambda x: (x.value, None) if isinstance(x, _Const) else ev(x), n)
        if fn in ("in", "notin"):
            items = [Call("eq", (args[0], a)) for a in args[1:]]
            r = self._call(Call("or", tuple(_Const(self._call(i, ev, n)[0]) for i in items)),
                           lambda x: (x.value, None), n)[0]
            if fn == "notin":
                r = Bool(~r.val & ~r.null, r.null)
            return r, None
        if fn in ("insub", "notinsub"):
            return self._in_subquery(args, ev, fn == "notinsub"), None
        if fn == "scalar":
            sub = self._reg(args[0])
            if not isinstance(sub, Rel) or len(sub.vecs) != 1:
                raise Unsupported("scalar subquery must return one column")
            if sub.n == 0:
                return Vec.empty(1), sub.affs[0]
            return sub.vecs[0].take(torch.tensor([0])), sub.affs[0]
        if fn in ("like", "notlike"):
            v = apply_text(_vec(ev(args[0])[0]), S)
            pat = args[1]
            if not (isinstance(pat, Lit) and isinstance(pat.value, str)):
                raise Unsupported("LIKE pattern must be a string literal")
            rx = _like_regex(pat.value)
            codes = torch.unique(v.code[v.cls == TEXT])
            table = torch.zeros(len(S.values) + 1, dtype=torch.bool)
            for c in codes.tolist():
                table[c] = rx.fullmatch(S.values[c].translate(_ASCII_LOWER)) is not None
            match = table[v.code.clamp(min=0)] & (v.cls == TEXT)
            null = v.cls == NULL
            val = match if fn == "like" else ~match
            return Bool(val & ~null, null), None
        if fn in ("add", "sub", "mul", "div", "mod", "neg"):
            return self._arith(fn, [_vec(ev(a)[0]) for a in args]), None
        if fn in ("coalesce", "ifnull"):
            vals = [_vec(ev(a)[0]) for a in args]
            m = bcast_len(*(len(v) for v in vals))
            out = vals[-1].expand(m)
            for v in reversed(vals[:-1]):
                v = v.expand(m)
                keep = v.cls != NULL
                out = Vec(torch.where(keep, v.cls, out.cls), torch.where(keep, v.num, out.num),
                          torch.where(keep, v.isint, out.isint), torch.where(keep, v.ival, out.ival),
                          torch.where(keep, v.code, out.code))
            return out, None
        if fn == "case":
            conds = [as_bool(ev(c)[0]) for c in args[:-1:2]]
            vals = [_vec(ev(v)[0]) for v in args[1::2]] + [_vec(ev(args[-1])[0])]
            m = bcast_len(*(len(x) for x in conds), *(len(v) for v in vals))
            out = vals[-1].expand(m)
            # apply branches from last to first so the first true WHEN wins
            for c, v in reversed(list(zip(conds, vals[:-1]))):
                hit = c.is_true().expand(m)
                v = v.expand(m)
                out = Vec(torch.where(hit, v.cls, out.cls), torch.where(hit, v.num, out.num),
                          torch.where(hit, v.isint, out.isint), torch.where(hit, v.ival, out.ival),
                          torch.where(hit, v.code, out.code))
            return out, None
        if fn == "concat":
            a, b = (apply_text(_vec(ev(x)[0]), S) for x in args)
            m = bcast_len(len(a), len(b))
            a, b = a.expand(m), b.expand(m)
            null = (a.cls == NULL) | (b.cls == NULL)
            code = torch.full((m,), -1, dtype=torch.int64)
            ac, bc, nl = a.code.tolist(), b.code.tolist(), null.tolist()
            for i in range(m):
                if not nl[i]:
                    code[i] = S.code(S.values[ac[i]] + S.values[bc[i]])
            cls = torch.where(null, torch.tensor(NULL, dtype=torch.int8), torch.tensor(TEXT, dtype=torch.int8))
            return Vec(cls, torch.zeros(m, dtype=torch.float64), torch.zeros(m, dtype=torch.bool),
                       torch.zeros(m, dtype=torch.int64), code), None
        if fn in ("lower", "upper", "length", "abs"):
            return self._scalar_fn(fn, _vec(ev(args[0])[0])), None
        raise Unsupported(f"function {fn}")

    def _in_subquery(self, args, ev, negate: bool) -> Bool:
        x, aff = ev(args[0])
        x = _vec(x)
        sub = self._reg(args[1])
        if not isinstance(sub, Rel) or len(sub.vecs) != 1:
            raise Unsupported("IN subquery must return one column")
        s, saff = sub.vecs[0], sub.affs[0]
        x, s = coerce_pair(x, aff, s, saff, self.strings)
        m = len(x)
        snull = s.cls == NULL
        set_has_null = bool(snull.any())
        sv = s.take(torch.nonzero(~snull).flatten())
        xid, sid = _dense_ids(row_keys([x]), row_keys([sv]))
        hit = torch.isin(xid, sid) & (x.cls != NULL)
        if sub.n == 0:
            val = torch.zeros(m, dtype=torch.bool)
            null = torch.zeros(m, dtype=torch.bool)
        else:
            null = ~hit & ((x.cls == NULL) | set_has_null)
            val = hit
        if negate:
            return Bool(~val & ~null, null)
        return Bool(val & ~null, null)

    def _arith(self, fn: str, vals: list[Vec]) -> Vec:
        vals = [apply_numeric(v, self.strings) for v in vals]
        for v in vals:
            if (v.cls == TEXT).any():
                raise Unsupported("arithmetic on non-numeric text")
        m = bcast_len(*(len(v) for v in vals))
        vals = [v.expand(m) for v in vals]
        if fn == "neg":
            (a,) = vals
            return Vec(a.cls, -a.num, a.isint, -a.ival, a.code)
        a, b = vals
        null = (a.cls == NULL) | (b.cls == NULL)
        both_int = a.isint & b.isint
        if fn in ("div", "mod"):
            zero = torch.where(b.isint, b.ival == 0, b.num == 0)
            null = null | zero
        safe_bi = torch.where(b.ival == 0, torch.ones_like(b.ival), b.ival)
        safe_bn = torch.where(b.num == 0, torch.ones_like(b.num), b.num)
        if fn == "add":
            iv, fv = a.ival + b.ival, a.num + b.num
        elif fn == "sub":
            iv, fv = a.ival - b.ival, a.num - b.num
        elif fn == "mul":
            iv, fv = a.ival * b.ival, a.num * b.num
        elif fn == "div":
            iv, fv = torch.div(a.ival, safe_bi, rounding_mode="trunc"), a.num / safe_bn
        else:
            iv = a.ival - torch.div(a.ival, safe_bi, rounding_mode="trunc") * safe_bi
            fv = torch.fmod(a.num, safe_bn)
            if not bool((both_int | null).all()):
                raise Unsupported("modulo on real numbers")
        num = torch.where(both_int, iv.to(torch.float64), fv)
        cls = torch.where(null, torch.tensor(NULL, dtype=torch.int8), torch.tensor(NUM, dtype=torch.int8))
        return Vec(cls, num, both_int & ~null, torch.where(both_int, iv, 0), torch.full((m,), -1, dtype=torch.int64))

    def _scalar_fn(self, fn: str, v: Vec) -> Vec:
        S = self.strings
        if fn == "abs":
            v = apply_numeric(v, S)
            if (v.cls == TEXT).any():
                raise Unsupported("abs of text")
            return Vec(v.cls, v.num.abs(), v.isint, v.ival.abs(), v.code)
        t = apply_text(v, S)
        out_code = t.code.clone()
        lens = torch.zeros(len(t), dtype=torch.int64)
        cache = {}
        codes = t.code.tolist()
        cls = t.cls.tolist()
        for i, c in enumerate(codes):
            if cls[i] != TEXT:
                continue
            if c not in cache:
                s = S.values[c]
                if fn == "lower":
                    cache[c] = S.code(s.translate(_ASCII_LOWER))
                elif fn == "upper":
                    cache[c] = S.code(s.translate({v_: k for k, v_ in _ASCII_LOWER.items()}))
                else:
                    cache[c] = len(s)
            if fn == "length":
                lens[i] = cache[c]
            else:
                out_code[i] = cache[c]
        if fn == "length":
            # length() of a number is the length of its text rendering
            return Vec(t.cls.where(t.cls == NULL, torch.tensor(NUM, dtype=torch.int8)), lens.to(torch.float64),
                       t.cls != NULL, lens, torch.full_like(lens, -1))
        return Vec(t.cls, t.num, t.isint, t.ival, out_code)


class _Const:
    """Wraps an already-evaluated value so it can be passed back through _call."""

    def __init__(self, value):
        self.value = value


def _vec(x) -> Vec:
    return x.to_vec() if isinstance(x, Bool) else x


def _int_vec(t: torch.Tensor) -> Vec:
    n = t.shape[0]
    return Vec(torch.full((n,), NUM, dtype=torch.int8), t.to(torch.float64), torch.ones(n, dtype=torch.bool), t,
               torch.full((n,), -1, dtype=torch.int64))


def _real_vec(t: torch.Tensor) -> Vec:
    n = t.shape[0]
    return Vec(torch.full((n,), NUM, dtype=torch.int8), t, torch.zeros(n, dtype=torch.bool),
               torch.zeros(n, dtype=torch.int64), torch.full((n,), -1, dtype=torch.int64))


def execute(dw: DataWeights, plan: Plan) -> list[tuple]:
    rel = Executor(dw).run(plan)
    return rel.rows(dw.strings)
