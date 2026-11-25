"""Dashboard engine: KPIs and charts computed in code, safe chat edits, themes, integrity checks, HTML export.
The language model only proposes edits as JSON. This code validates and runs them. No model-written code is executed."""
import re, json, html, copy
import numpy as np
import pandas as pd
import altair as alt
from llm import llm, llm_json
from pipeline import tidy, critique

THEMES = {
    "default": {"label": "Default", "accent": "#0072B2", "positive": "#2E7D32", "negative": "#C62828",
                "sign_colors": True, "finance": False},
    "finance": {"label": "Finance", "accent": "#1F4E79", "positive": "#2E7D32", "negative": "#C62828",
                "sign_colors": True, "finance": True},
    "mono": {"label": "Print friendly", "accent": "#404040", "positive": "#404040", "negative": "#8C8C8C",
             "sign_colors": False, "finance": False},
}
CHART_TYPES = {"bar", "hbar", "line", "hist"}
AGGS = {"sum", "mean", "median", "count"}
KPI_METRICS = {"sum", "mean", "median", "min", "max", "count", "nunique", "ratio", "top_share",
               "mom_change", "yoy_change"}
FILTER_OPS = {"equals", "not_equals", "gte", "lte"}
SIGNED_HINTS = ("profit", "loss", "pnl", "net_income", "net", "variance", "delta", "growth", "change",
                "margin", "balance", "savings")
GOOD_DOWN_HINTS = ("cost", "expense", "churn", "loss", "return", "refund", "defect", "complaint", "discount")
MEASURE_HINTS = ("revenue", "sales", "amount", "value", "total", "price", "profit", "cost", "expense")
PII_HINTS = ("email", "phone", "mobile", "ssn", "aadhaar", "aadhar", "passport", "address", "dob", "birth",
             "password", "card_number", "iban", "pan_number")
FINANCE_TYPES = ("financ", "bank", "invest", "account", "insur", "trading", "budget", "expense")
CURRENCIES = {"₹": "₹", "$": "$", "€": "€", "£": "£", "": "", "inr": "₹", "usd": "$", "eur": "€", "gbp": "£"}
HEX = re.compile(r"^#[0-9a-fA-F]{6}$")
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


# ---------- small helpers ----------
def h(col):
    return str(col).replace("_", " ")


def kind_of(s):
    if pd.api.types.is_datetime64_any_dtype(s):
        return "date"
    if pd.api.types.is_numeric_dtype(s):
        return "num"
    return "cat"


def is_pii(name):
    return any(p in str(name).lower() for p in PII_HINTS)


def match_col(df, name):
    if name is None or str(name).strip() == "":
        return None
    n = str(name).strip()
    if n in df.columns:
        return n
    norm = lambda s: re.sub(r"[\W_]+", "", str(s).lower())
    for c in df.columns:
        if norm(c) == norm(n):
            return c
    return None


def compact(a, style):
    if style == "indian":
        if a >= 1e7:
            return f"{a / 1e7:.2f} Cr"
        if a >= 1e5:
            return f"{a / 1e5:.2f} L"
    else:
        if a >= 1e9:
            return f"{a / 1e9:.2f}B"
        if a >= 1e6:
            return f"{a / 1e6:.2f}M"
    return f"{a:,.0f}" if (abs(a - round(a)) < 1e-9 or a >= 1000) else f"{a:,.2f}"


def fmt_value(v, kind, cur, style):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    if kind == "percent":
        return f"{v * 100:.1f}%"
    pre = cur if kind == "currency" else ""
    s = compact(abs(float(v)), style)
    return f"-{pre}{s}" if v < 0 else f"{pre}{s}"


def theme_ctx(spec):
    t = dict(THEMES.get(spec.get("theme", "default"), THEMES["default"]))
    t.update({k: v for k, v in (spec.get("colors") or {}).items() if v})
    sem = spec.get("sign_semantics")
    t["sign_semantics"] = t["finance"] if sem is None else bool(sem)
    t["cur"] = spec.get("currency", "")
    t["style"] = spec.get("number_style", "intl")
    return t


def is_signed(col, values, ctx):
    if not ctx["sign_colors"]:
        return False
    neg = len(values) > 0 and float(np.nanmin(values)) < 0
    hinted = ctx["sign_semantics"] and any(x in str(col or "").lower() for x in SIGNED_HINTS)
    return bool(neg or hinted)


def apply_filters(df, filters):
    d = df
    for f in filters or []:
        c, op, v = f.get("column"), f.get("op", "equals"), f.get("value")
        if c not in d.columns:
            continue
        s, k = d[c], kind_of(d[c])
        if k == "date":
            v = pd.to_datetime(v, errors="coerce")
        elif k == "num":
            v = pd.to_numeric(v, errors="coerce")
        if v is None or (not isinstance(v, str) and pd.isna(v)):
            continue
        if op in ("equals", "not_equals"):
            mask = (s == v) if k != "cat" else (s.astype(str).str.strip().str.lower() == str(v).strip().lower())
            d = d[mask] if op == "equals" else d[~mask]
        elif op == "gte":
            d = d[s >= v]
        elif op == "lte":
            d = d[s <= v]
    return d


def filter_text(filters):
    sym = {"equals": "=", "not_equals": "is not", "gte": ">=", "lte": "<="}
    return "; ".join(f"{h(f['column'])} {sym.get(f.get('op', 'equals'), '=')} {f['value']}" for f in filters or [])


def month_series(d, date_col, col, agg="sum"):
    """Monthly values with incomplete first or last months removed. Returns (series, dropped month labels)."""
    sub = d.dropna(subset=[date_col])
    if sub.empty:
        return pd.Series(dtype=float), []
    idx = sub.set_index(date_col)
    t = idx.resample("MS").size() if agg == "count" else getattr(idx[col].resample("MS"), agg)()
    lo, hi = sub[date_col].min(), sub[date_col].max()
    dropped = []
    for m in {t.index[0], t.index[-1]}:
        end = min(hi, m + pd.offsets.MonthEnd(0))
        covered = (end.normalize() - max(lo, m).normalize()).days + 1
        if covered < m.days_in_month * 0.9:
            dropped.append(m)
    return t.drop(dropped), [m.strftime("%b %Y") for m in sorted(dropped)]


# ---------- validation of chart and KPI specs ----------
def clean_filters(filters, df):
    out = []
    for f in filters or []:
        c = match_col(df, f.get("column"))
        if c is None:
            raise ValueError(f"filter column '{f.get('column')}' does not exist")
        if is_pii(c):
            raise ValueError(f"'{c}' looks like personal data, so I will not filter on it")
        op = str(f.get("op", "equals")).lower()
        if op not in FILTER_OPS or f.get("value") in (None, ""):
            raise ValueError("a filter needs an operator (equals, not_equals, gte, lte) and a value")
        out.append({"column": c, "op": op, "value": f["value"]})
    return out


def clean_chart(c, df):
    t = str(c.get("type", "bar")).lower()
    if t not in CHART_TYPES:
        raise ValueError(f"chart type '{t}' is not supported (use bar, hbar, line or hist)")
    x = match_col(df, c.get("x"))
    if x is None:
        raise ValueError(f"there is no column called '{c.get('x')}'. Columns: {', '.join(map(str, df.columns))}")
    y = match_col(df, c.get("y")) if c.get("y") else None
    if c.get("y") and y is None:
        raise ValueError(f"there is no column called '{c.get('y')}'. Columns: {', '.join(map(str, df.columns))}")
    for col in (x, y):
        if col and is_pii(col):
            raise ValueError(f"'{col}' looks like personal data, so I will not chart it. Pick another column")
    agg = str(c.get("agg") or "sum").lower()
    if agg not in AGGS:
        raise ValueError(f"'{agg}' is not a supported aggregation (sum, mean, median, count)")
    kx = kind_of(df[x])
    if t == "hist":
        if kx != "num":
            raise ValueError(f"a histogram needs a numeric column, and '{x}' is not numeric")
        y, agg = None, "count"
    elif t == "line" and kx != "date":
        raise ValueError(f"a line chart needs a date column on the x axis, and '{x}' is not a date. Try a bar chart")
    elif kx == "num" and df[x].nunique() > 20:
        raise ValueError(f"'{x}' is numeric with many values. Try a histogram, or choose a category column")
    if t != "hist":
        if y is None:
            agg = "count"
        elif agg != "count" and kind_of(df[y]) != "num":
            raise ValueError(f"'{y}' is not numeric, so it cannot be summed or averaged")
    freq = c.get("freq") or ("month" if kx == "date" else None)
    if freq not in (None, "month", "weekday"):
        raise ValueError("freq must be month or weekday")
    try:
        top_n = max(3, min(20, int(c.get("top_n", 8))))
    except (TypeError, ValueError):
        top_n = 8
    sort = str(c.get("sort", "desc")).lower()
    return {"type": t, "x": x, "y": y, "agg": agg, "title": str(c.get("title") or "")[:90], "top_n": top_n,
            "sort": sort if sort in ("desc", "asc", "none") else "desc", "freq": freq,
            "zero_baseline": bool(c.get("zero_baseline", True)),
            "filters": clean_filters(c.get("filters"), df)}


def default_format(metric, col, cur):
    if metric in ("ratio", "top_share"):
        return "percent"
    if cur and col and any(x in str(col).lower() for x in MEASURE_HINTS):
        return "currency"
    return "number"


def clean_kpi(k, df, cur=""):
    m = str(k.get("metric", "")).lower()
    if m not in KPI_METRICS:
        raise ValueError(f"KPI type '{m}' is not supported")
    col = match_col(df, k.get("column"))
    col2 = match_col(df, k.get("column2"))
    dim = match_col(df, k.get("dim"))
    if m != "count":
        if col is None:
            raise ValueError(f"there is no column called '{k.get('column')}'. Columns: {', '.join(map(str, df.columns))}")
        if is_pii(col):
            raise ValueError(f"'{col}' looks like personal data, so I will not use it")
        if m != "nunique" and kind_of(df[col]) != "num":
            raise ValueError(f"'{col}' is not numeric")
    if m == "ratio" and (col2 is None or kind_of(df[col2]) != "num"):
        raise ValueError("a ratio needs two numeric columns (column and column2)")
    if m == "top_share" and (dim is None or kind_of(df[dim]) != "cat"):
        raise ValueError("top_share needs a category column in 'dim'")
    if m in ("mom_change", "yoy_change") and not any(kind_of(df[c]) == "date" for c in df.columns):
        raise ValueError("this KPI needs a date column")
    fmt = str(k.get("format") or default_format(m, col, cur)).lower()
    if m in ("ratio", "top_share"):
        fmt = "percent"
    good = k.get("good") or ("down" if col and any(x in str(col).lower() for x in GOOD_DOWN_HINTS) else "up")
    label = str(k.get("label") or f"{m.replace('_', ' ').title()} {h(col) if col else 'records'}")[:70]
    return {"label": label, "metric": m, "column": col, "column2": col2, "dim": dim,
            "format": fmt if fmt in ("number", "currency", "percent") else "number",
            "good": good if good in ("up", "down", "neutral") else "up",
            "filters": clean_filters(k.get("filters"), df)}


# ---------- chart data and rendering ----------
def auto_title(c):
    x, y, agg = h(c["x"]), c.get("y"), c.get("agg", "sum")
    if c["type"] == "hist":
        return f"Distribution of {x}"
    by = "weekday" if c.get("freq") == "weekday" else ("month" if c.get("freq") == "month" else x)
    if not y or agg == "count":
        return f"Number of records by {by}"
    return f"{agg.title()} of {h(y)} by {by}"


def chart_data(df, c):
    d = apply_filters(df, c.get("filters"))
    notes, x, y, agg, t = [], c["x"], c.get("y"), c.get("agg", "sum"), c["type"]
    empty = pd.DataFrame({"x": [], "y": []})
    if d.empty:
        return empty, ["No rows match the filters."]
    if t == "hist":
        s = pd.to_numeric(d[x], errors="coerce").dropna()
        if s.empty:
            return empty, ["No numeric values to show."]
        counts, edges = np.histogram(s, bins=min(20, max(5, int(np.sqrt(len(s))))))
        labels = [f"{edges[i]:,.4g} to {edges[i + 1]:,.4g}" for i in range(len(counts))]
        return pd.DataFrame({"x": labels, "y": counts}), notes
    val = pd.to_numeric(d[y], errors="coerce") if y else None
    if kind_of(d[x]) == "date":
        sub = d.dropna(subset=[x])
        if c.get("freq") == "weekday":
            wk = sub[x].dt.day_name()
            s = wk.value_counts() if (y is None or agg == "count") else val.loc[sub.index].groupby(wk).agg(agg)
            s = s.reindex([w for w in WEEKDAYS if w in s.index])
            return pd.DataFrame({"x": s.index.astype(str), "y": s.values}), notes
        tmp = sub.assign(_v=val.loc[sub.index] if y else 1)
        s, dropped = month_series(tmp, x, "_v", "count" if (y is None or agg == "count") else agg)
        if dropped:
            notes.append("Excluded incomplete month(s): " + ", ".join(dropped))
        return pd.DataFrame({"x": s.index, "y": s.values}), notes
    if y is None or agg == "count":
        s = d.groupby(x).size()
    else:
        s = val.groupby(d[x]).agg(agg)
    if c.get("sort", "desc") != "none":
        s = s.sort_values(ascending=(c["sort"] == "asc"))
    total_groups = len(s)
    if total_groups > c.get("top_n", 8):
        s = s.head(c["top_n"])
        notes.append(f"Showing the top {c['top_n']} of {total_groups} groups.")
    if y and agg in ("mean", "median"):
        n = d.groupby(x).size().reindex(s.index)
        low = int((n < 30).sum())
        if low:
            notes.append(f"{low} group(s) have fewer than 30 records, so their {agg} is less reliable.")
    return pd.DataFrame({"x": s.index.astype(str), "y": s.values}), notes


def build_chart(df, c, ctx):
    data, notes = chart_data(df, c)
    if data.empty:
        return None, notes
    x_title = "weekday" if c.get("freq") == "weekday" else ("month" if c.get("freq") == "month" else h(c["x"]))
    y_title = "Number of records" if (not c.get("y") or c["agg"] == "count") else f"{c['agg'].title()} of {h(c['y'])}"
    signed = is_signed(c.get("y"), data["y"].values, ctx)
    color = (alt.condition(alt.datum.y >= 0, alt.value(ctx["positive"]), alt.value(ctx["negative"]))
             if signed else alt.value(ctx["accent"]))
    zero = c.get("zero_baseline", True)
    sub = []
    if c.get("filters"):
        sub.append("Filtered: " + filter_text(c["filters"]))
    if not zero:
        sub.append("Vertical axis does not start at zero")
        notes.append("This chart's axis does not start at zero, so differences look larger than they are.")
    title = alt.TitleParams(text=c.get("title") or auto_title(c), subtitle=sub or alt.Undefined)
    lab_fmt = ",.2f" if float(data["y"].abs().max()) < 100 else ",.0f"
    tips = [alt.Tooltip("x", title=x_title), alt.Tooltip("y:Q", title=y_title, format=",.2f")]
    t = c["type"]
    if t == "line":
        data = data.assign(x=pd.to_datetime(data["x"]))
        base = alt.Chart(data).encode(
            x=alt.X("x:T", title=x_title, axis=alt.Axis(format="%b %Y")),
            y=alt.Y("y:Q", title=y_title, scale=alt.Scale(zero=zero)))
        parts = [base.mark_line(color=ctx["accent"]),
                 base.mark_point(filled=True, size=55).encode(color=color, tooltip=[alt.Tooltip("x:T", title=x_title, format="%b %Y"), tips[1]])]
        height = 300
    elif t == "hbar":
        enc = dict(x=alt.X("y:Q", title=y_title, scale=alt.Scale(zero=zero)), y=alt.Y("x:N", sort=None, title=x_title))
        base = alt.Chart(data)
        parts = [base.mark_bar().encode(color=color, tooltip=tips, **enc)]
        if len(data) <= 12:
            parts.append(base.mark_text(align="left", dx=4, fontSize=11).encode(text=alt.Text("y:Q", format=lab_fmt), **enc))
        height = max(200, 30 * len(data))
    else:
        enc = dict(x=alt.X("x:N", sort=None, title=x_title, axis=alt.Axis(labelAngle=-30 if len(data) > 6 else 0)),
                   y=alt.Y("y:Q", title=y_title, scale=alt.Scale(zero=zero)))
        base = alt.Chart(data)
        parts = [base.mark_bar().encode(color=color, tooltip=tips, **enc)]
        if len(data) <= 12:
            parts.append(base.mark_text(dy=-6, fontSize=11).encode(text=alt.Text("y:Q", format=lab_fmt), **enc))
        height = 300
    if float(data["y"].min()) < 0 and t != "hbar":
        parts.append(alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#888888", strokeDash=[4, 3]).encode(y="y:Q"))
    return alt.layer(*parts).properties(width="container", height=height, title=title), notes


# ---------- KPIs ----------
def compute_kpi(df, k, spec):
    ctx = theme_ctx(spec)
    cur, style = ctx["cur"], ctx["style"]
    d = apply_filters(df, k.get("filters"))
    m, col = k["metric"], k.get("column")
    out = {"id": k["id"], "label": k["label"], "value": "n/a", "delta": None, "delta_color": "normal",
           "note": "", "status": None}
    fmt = lambda v: fmt_value(v, k.get("format", "number"), cur, style)
    try:
        v = None
        if m == "count":
            v = float(len(d))
            out["value"] = f"{len(d):,}"
        elif m in ("sum", "mean", "median", "min", "max"):
            v = float(getattr(pd.to_numeric(d[col], errors="coerce"), m)())
            out["value"] = fmt(v)
        elif m == "nunique":
            v = float(d[col].nunique())
            out["value"] = f"{int(v):,}"
        elif m == "ratio":
            den = float(pd.to_numeric(d[k["column2"]], errors="coerce").sum())
            v = float(pd.to_numeric(d[col], errors="coerce").sum()) / den if den else None
            out["value"] = fmt(v)
            out["note"] = f"{h(col)} divided by {h(k['column2'])}"
        elif m == "top_share":
            g = pd.to_numeric(d[col], errors="coerce").groupby(d[k["dim"]]).sum().sort_values(ascending=False)
            tot = float(g.sum())
            v = float(g.iloc[0]) / tot if tot else None
            out["value"] = fmt(v)
            out["note"] = f"Largest: {g.index[0]} (of {len(g)} {h(k['dim'])} groups)"
        else:
            dc = next(c for c in df.columns if kind_of(df[c]) == "date")
            s, dropped = month_series(d, dc, col, "sum")
            if m == "mom_change" and len(s) >= 2:
                last, prev = float(s.iloc[-1]), float(s.iloc[-2])
                v = last
                out["value"] = fmt(last)
                if prev:
                    out["delta"] = f"{(last - prev) / abs(prev) * 100:+.1f}% vs {s.index[-2]:%b %Y}"
                out["note"] = f"{s.index[-1]:%b %Y}, latest complete month"
            elif m == "yoy_change" and len(s) >= 24:
                last, prev = float(s.iloc[-12:].sum()), float(s.iloc[-24:-12].sum())
                v = last
                out["value"] = fmt(last)
                if prev:
                    out["delta"] = f"{(last - prev) / abs(prev) * 100:+.1f}% vs previous 12 months"
                out["note"] = f"12 months to {s.index[-1]:%b %Y}"
            else:
                out["note"] = "Not enough complete months for this KPI"
            if dropped:
                out["note"] += f". Excluded incomplete month(s): {', '.join(dropped)}"
        if out["delta"]:
            out["delta_color"] = {"up": "normal", "down": "inverse", "neutral": "off"}[k.get("good", "up")]
        if v is not None and m in ("sum", "mean", "median", "min", "max") and is_signed(col, [v], ctx):
            money = any(x in str(col).lower() for x in ("profit", "pnl", "net_income"))
            out["status"] = ("Profit", "Loss") if money else ("Positive", "Negative")
            out["status"] = out["status"][0] if v >= 0 else out["status"][1]
            out["status_positive"] = v >= 0
    except Exception as e:
        out["note"] = f"Could not compute ({type(e).__name__})"
    return out


# ---------- default spec and suggestions ----------
def _new_id(spec, prefix):
    n = spec["next_id"]
    spec["next_id"] = n + 1
    return f"{prefix}{n}"


def _add_chart(spec, df, **c):
    try:
        cc = clean_chart(c, df)
    except ValueError:
        return
    cc["id"] = _new_id(spec, "c")
    spec["charts"].append(cc)


def _add_kpi(spec, df, **k):
    try:
        kk = clean_kpi(k, df, spec.get("currency", ""))
    except ValueError:
        return
    kk["id"] = _new_id(spec, "k")
    spec["kpis"].append(kk)


def default_spec(df, stats, btype=""):
    cur = df.attrs.get("currency", "unknown")
    cur = "" if cur == "unknown" else cur
    measure = stats.get("main_measure")
    pre = f"{measure}_total_by_"
    dims = [k[len(pre):] for k in stats if measure and k.startswith(pre)]
    dates = [c for c in df.columns if kind_of(df[c]) == "date"]
    finance = (any(x in (btype or "") for x in FINANCE_TYPES)
               or any(any(x in c for x in ("profit", "pnl", "net_income", "variance")) for c in df.columns))
    spec = {"theme": "finance" if finance else "default", "currency": cur, "colors": {}, "kpis": [], "charts": [],
            "number_style": "indian" if cur == "₹" else "intl", "next_id": 1}
    if not measure:
        _add_kpi(spec, df, label="Records", metric="count")
        return spec
    _add_kpi(spec, df, label=f"Total {h(measure)}", metric="sum", column=measure)
    _add_kpi(spec, df, label=f"Average {h(measure)} per record", metric="mean", column=measure)
    _add_kpi(spec, df, label="Records", metric="count")
    if dims:
        _add_kpi(spec, df, label=f"Top {h(dims[0])} share of {h(measure)}", metric="top_share", column=measure, dim=dims[0])
    profit = next((c for c in df.columns if kind_of(df[c]) == "num" and any(x in c for x in ("profit", "net_income", "pnl"))), None)
    if profit:
        _add_kpi(spec, df, label=f"Total {h(profit)}", metric="sum", column=profit)
        if profit != measure:
            _add_kpi(spec, df, label=f"{h(profit).title()} margin", metric="ratio", column=profit, column2=measure)
    if dates:
        s, _ = month_series(df, dates[0], measure, "sum")
        if len(s) >= 3:
            _add_kpi(spec, df, label=f"Latest month {h(measure)}", metric="mom_change", column=measure)
        if len(s) >= 24:
            _add_kpi(spec, df, label=f"12-month {h(measure)}", metric="yoy_change", column=measure)
    spec["kpis"] = spec["kpis"][:6]
    for d in dims[:3]:
        _add_chart(spec, df, type="hbar" if df[d].nunique() > 6 else "bar", x=d, y=measure, agg="sum")
    if dates:
        _add_chart(spec, df, type="line", x=dates[0], y=measure, agg="sum")
    if profit and dims:
        _add_chart(spec, df, type="bar" if df[dims[0]].nunique() <= 6 else "hbar", x=dims[0], y=profit, agg="sum")
    return spec


def chart_key(c):
    return (c["type"], c["x"], c.get("y"), c.get("agg"), c.get("freq"), json.dumps(c.get("filters") or [], sort_keys=True))


def suggest(df, spec, stats):
    """Extra charts and KPIs worth adding, found by code. Returns [{label, op}]."""
    out, have = [], {chart_key(c) for c in spec["charts"]}
    measure = stats.get("main_measure")
    if not measure:
        return out

    def offer(label, op):
        try:
            if op["op"] == "add_chart" and chart_key(clean_chart(op, df)) in have:
                return
        except ValueError:
            return
        out.append({"label": label, "op": op})
    offer(f"Distribution of {h(measure)} per record", {"op": "add_chart", "type": "hist", "x": measure})
    dates = [c for c in df.columns if kind_of(df[c]) == "date"]
    if dates:
        offer(f"{h(measure).title()} by weekday", {"op": "add_chart", "type": "bar", "x": dates[0], "y": measure, "freq": "weekday", "sort": "none"})
    used = {c["x"] for c in spec["charts"]}
    cats = [c for c in df.columns if kind_of(df[c]) == "cat" and 2 <= df[c].nunique() <= 30 and not is_pii(c)]
    for c in [c for c in cats if c not in used][:2]:
        offer(f"{h(measure).title()} by {h(c)}", {"op": "add_chart", "type": "bar" if df[c].nunique() <= 6 else "hbar", "x": c, "y": measure})
    if cats:
        offer(f"Average {h(measure)} by {h(cats[0])}", {"op": "add_chart", "type": "bar" if df[cats[0]].nunique() <= 6 else "hbar", "x": cats[0], "y": measure, "agg": "mean"})
    if dates:
        s, _ = month_series(df, dates[0], measure, "sum")
        if len(s) >= 24 and not any(k["metric"] == "yoy_change" for k in spec["kpis"]):
            out.append({"label": f"12-month {h(measure)} versus the previous 12 months", "op": {"op": "add_kpi", "label": f"12-month {h(measure)}", "metric": "yoy_change", "column": measure}})
    if not any(k["metric"] == "median" for k in spec["kpis"]):
        out.append({"label": f"Median {h(measure)} per record", "op": {"op": "add_kpi", "label": f"Median {h(measure)}", "metric": "median", "column": measure}})
    return out[:6]


# ---------- integrity review ----------
def ethics_review(df, spec):
    ctx, w = theme_ctx(spec), []
    if not ctx["cur"]:
        w.append("No currency is known for this data, so amounts are shown as plain numbers. You can state one in the chat.")
    if ctx["sign_colors"]:
        w.append("Green and red are used only for gains and losses (positive and negative values), with arrows or signs as a second cue.")
    for c in spec["charts"]:
        name = c.get("title") or auto_title(c)
        if not c.get("zero_baseline", True):
            w.append(f"'{name}': the vertical axis does not start at zero. This is printed on the chart.")
        if c.get("filters"):
            w.append(f"'{name}' shows filtered data ({filter_text(c['filters'])}). The filter is printed on the chart.")
        if c.get("y") and df[c["y"]].isna().mean() > 0.2:
            w.append(f"'{name}': column '{c['y']}' is {df[c['y']].isna().mean():.0%} empty.")
        w += [f"'{name}': {n}" for n in chart_data(df, c)[1]]
    if len(spec["charts"]) > 8 or len(spec["kpis"]) > 8:
        w.append("A crowded dashboard can hide the message. Consider removing items that do not support a decision.")
    return w


# ---------- applying edits ----------
def _name(it):
    return str((it.get("title") or auto_title(it)) if "type" in it else it.get("label", "")).lower()


def find_item(items, ref):
    r = str(ref or "").strip().lower()
    for it in items:
        if it["id"].lower() == r:
            return it
    for it in items:
        if r and _name(it) == r:
            return it
    for it in items:
        if r and r in _name(it):
            return it
    return None


def _apply_one(spec, op, df, report_reqs):
    name = str(op.get("op", ""))
    if name == "add_chart":
        if len(spec["charts"]) >= 12:
            raise ValueError("the dashboard already has 12 charts. Remove one first")
        c = clean_chart(op, df)
        if chart_key(c) in {chart_key(x) for x in spec["charts"]}:
            raise ValueError("that chart is already on the dashboard")
        c["id"] = _new_id(spec, "c")
        spec["charts"].append(c)
        return f"Added chart '{c['title'] or auto_title(c)}'"
    if name == "update_chart":
        old = find_item(spec["charts"], op.get("ref"))
        if old is None:
            raise ValueError(f"I could not find a chart matching '{op.get('ref')}'")
        merged = {**old, **{k: v for k, v in op.items() if k not in ("op", "ref", "id")}}
        new = clean_chart(merged, df)
        new["id"] = old["id"]
        spec["charts"][spec["charts"].index(old)] = new
        return f"Updated chart '{new['title'] or auto_title(new)}'"
    if name == "remove_chart":
        old = find_item(spec["charts"], op.get("ref"))
        if old is None:
            raise ValueError(f"I could not find a chart matching '{op.get('ref')}'")
        spec["charts"].remove(old)
        return f"Removed chart '{old['title'] or auto_title(old)}'"
    if name == "add_kpi":
        if len(spec["kpis"]) >= 10:
            raise ValueError("the dashboard already has 10 KPIs. Remove one first")
        k = clean_kpi(op, df, spec.get("currency", ""))
        k["id"] = _new_id(spec, "k")
        spec["kpis"].append(k)
        return f"Added KPI '{k['label']}'"
    if name == "update_kpi":
        old = find_item(spec["kpis"], op.get("ref"))
        if old is None:
            raise ValueError(f"I could not find a KPI matching '{op.get('ref')}'")
        new = clean_kpi({**old, **{k: v for k, v in op.items() if k not in ("op", "ref", "id")}}, df, spec.get("currency", ""))
        new["id"] = old["id"]
        spec["kpis"][spec["kpis"].index(old)] = new
        return f"Updated KPI '{new['label']}'"
    if name == "remove_kpi":
        old = find_item(spec["kpis"], op.get("ref"))
        if old is None:
            raise ValueError(f"I could not find a KPI matching '{op.get('ref')}'")
        spec["kpis"].remove(old)
        return f"Removed KPI '{old['label']}'"
    if name == "set_theme":
        t = str(op.get("theme", "")).lower()
        if t not in THEMES:
            raise ValueError(f"theme must be one of {', '.join(THEMES)}")
        spec["theme"] = t
        return f"Switched to the {THEMES[t]['label']} theme"
    if name == "set_colors":
        bad = [k for k, v in op.items() if k in ("positive", "negative", "accent") and not HEX.match(str(v))]
        if bad or not any(k in op for k in ("positive", "negative", "accent")):
            raise ValueError("colors must be given as hex codes like #2E7D32")
        spec["colors"] = {**spec.get("colors", {}), **{k: op[k] for k in ("positive", "negative", "accent") if k in op}}
        return "Updated colors"
    if name == "set_sign_colors":
        spec["sign_semantics"] = bool(op.get("enabled", True))
        return "Green for gains and red for losses is " + ("on" if spec["sign_semantics"] else "off")
    if name == "set_currency":
        sym = str(op.get("symbol", "")).strip().lower() if str(op.get("symbol", "")).strip().lower() in CURRENCIES else str(op.get("symbol", "")).strip()
        if sym not in CURRENCIES:
            raise ValueError("currency must be one of ₹, $, €, £ or empty")
        spec["currency"] = CURRENCIES[sym]
        if CURRENCIES[sym] == "₹":
            spec["number_style"] = "indian"
        return f"Currency set to {CURRENCIES[sym] or 'none'}"
    if name == "set_number_style":
        s = str(op.get("style", "")).lower()
        if s not in ("intl", "indian"):
            raise ValueError("number style must be intl or indian")
        spec["number_style"] = s
        return f"Number style set to {s} (" + ("lakh and crore" if s == "indian" else "thousand, million, billion") + ")"
    if name == "rewrite_report":
        ins = str(op.get("instruction", "")).strip()
        if not ins:
            raise ValueError("the report edit had no instruction")
        report_reqs.append(ins)
        return "Queued a report edit"
    raise ValueError(f"I do not know how to do '{name}'")


def apply_ops(spec, ops, df):
    new, done, refused, report_reqs = copy.deepcopy(spec), [], [], []
    ops = ops or []
    if len(ops) > 12:
        refused.append(f"I applied the first 12 of {len(ops)} requested changes. Ask again for the rest.")
    for op in ops[:12]:
        if not isinstance(op, dict):
            continue
        try:
            done.append(_apply_one(new, op, df, report_reqs))
        except (ValueError, KeyError, TypeError) as e:
            refused.append(str(e))
    done = [d for d in done if d != "Queued a report edit"]
    return new, done, refused, report_reqs


# ---------- chat ----------
CHAT_PROMPT = """You help a person edit a business dashboard and written report. Reply with JSON only:
{"reply": "1 to 3 plain sentences", "actions": [ ... ]}
Do not claim that changes are done. The system runs your actions and reports the results itself.

ACTIONS (each is an object with "op"):
- add_chart: type (bar|hbar|line|hist), x, y (numeric column, optional for counts), agg (sum|mean|median|count), title, top_n, sort (desc|asc|none), freq (month|weekday, date x only), zero_baseline (true|false), filters [{"column","op":equals|not_equals|gte|lte,"value"}]
- update_chart: ref (chart id or title) plus any add_chart fields to change. remove_chart: ref
- add_kpi: label, metric (sum|mean|median|min|max|count|nunique|ratio|top_share|mom_change|yoy_change), column, column2 (ratio denominator), dim (for top_share), format (number|currency|percent), good (up|down|neutral), filters
- update_kpi: ref plus fields to change. remove_kpi: ref
- set_theme: theme (default|finance|mono). set_colors: positive, negative, accent as #RRGGBB. set_sign_colors: enabled (true|false) for green gains and red losses
- set_currency: symbol (one of ₹ $ € £ or empty). set_number_style: style (intl|indian)
- rewrite_report: instruction (what to change in the written report, for example shorter, for executives, remove a section)

RULES (professional ethics, always follow):
1. Never invent numbers. Numbers come only from charts and KPIs the system computes. If the person asks for a figure, add the chart or KPI that computes it, or quote it from KPI VALUES or STATS below.
2. Do not help mislead. Refuse and explain, then offer an honest alternative, if asked to hide or delete complete unfavorable data, cherry-pick a date range to flatter results, exaggerate with a truncated axis, or remove outliers silently. Filters and axis choices are always disclosed on the chart.
3. Use green and red only for gains and losses (positive and negative values), never to rank ordinary categories.
4. Do not chart personal identifiers (names, emails, phones, addresses, ID numbers).
5. Do not assume a currency. Use set_currency only if the person states one.
6. Use only column names from COLUMNS. If a request is unclear, ask one short question and return no actions.

COLUMNS: __COLS__
CURRENT CHARTS: __CHARTS__
CURRENT KPIs: __KPIS__
THEME: __THEME__   CURRENCY: __CUR__
KPI VALUES: __KPIVALS__
STATS: __STATS__
RECENT CHAT: __HIST__
PERSON'S MESSAGE: __MSG__"""


def chat_turn(message, spec, df, stats, history):
    cols = {c: kind_of(df[c]) for c in df.columns}
    kv = [f"{r['label']}: {r['value']} {r['delta'] or ''}".strip() for r in (compute_kpi(df, k, spec) for k in spec["kpis"])]
    p = (CHAT_PROMPT.replace("__COLS__", json.dumps(cols))
         .replace("__CHARTS__", json.dumps([{k: c.get(k) for k in ("id", "title", "type", "x", "y", "agg")} for c in spec["charts"]]))
         .replace("__KPIS__", json.dumps([{k: x.get(k) for k in ("id", "label", "metric", "column")} for x in spec["kpis"]]))
         .replace("__THEME__", spec.get("theme", "default")).replace("__CUR__", spec.get("currency") or "unknown")
         .replace("__KPIVALS__", json.dumps(kv)[:1500]).replace("__STATS__", json.dumps(stats, default=str)[:3000])
         .replace("__HIST__", json.dumps([{"role": m["role"], "text": m["content"][:300]} for m in history[-6:]]))
         .replace("__MSG__", json.dumps(message)))
    out = llm_json(p, role="smart")
    return out if isinstance(out, dict) else {"reply": "", "actions": []}


def rewrite_report(report, instruction, stats, facts, kpi_text):
    new = llm(f"""Edit the business report below exactly as instructed.
INSTRUCTION: {instruction}
Rules: use only numbers that already appear in the report, STATS or KPI VALUES. Do not add or change any figure.
Keep the [Your data] and [Web #id] evidence labels. Keep every Confidence line honest and do not raise a
confidence level. Keep the External data and Notes and limits sections unless told to remove them.
If the instruction asks you to hide complete unfavorable results or to overstate findings, do not do that.
Do not use em dashes. Return the full edited report in markdown only.

STATS: {json.dumps(stats, default=str)[:3500]}
KPI VALUES: {kpi_text}
REPORT:
{report}""", role="smart", max_tokens=3500)
    new = tidy(new.strip())
    if len(new) < 200:
        return report, ["The edited report came back too short, so I kept the original."]
    verdict = critique(new, stats, facts)
    warns = []
    if not verdict.get("ok") and verdict.get("issues"):
        warns = ["The checker flagged: " + "; ".join(map(str, verdict["issues"][:3])) + ". Use Undo if you prefer the previous version."]
    return new, warns


def handle_message(message, spec, report, df, res, history):
    try:
        plan = chat_turn(message, spec, df, res["stats"], history)
    except Exception as e:
        return {"reply": f"I could not reach the AI service ({str(e)[:90]}). Please try again in a minute.",
                "changed": False, "spec": spec, "report": report}
    new_spec, done, refused, reqs = apply_ops(spec, plan.get("actions", []), df)
    new_report, warns = report, []
    if reqs:
        kv = "; ".join(f"{r['label']}: {r['value']} {r['delta'] or ''}".strip() for r in (compute_kpi(df, k, new_spec) for k in new_spec["kpis"]))
        for ins in reqs:
            try:
                new_report, w = rewrite_report(new_report, ins, res["stats"], res.get("facts", []), kv)
                warns += w
                if new_report != report:
                    done.append(f"Edited the report: {ins}")
            except Exception as e:
                refused.append(f"report edit failed ({str(e)[:80]})")
    parts = [str(plan.get("reply", "")).strip()]
    if done:
        parts.append("**Changes made**\n" + "\n".join(f"- {d}" for d in done))
    if refused:
        parts.append("**Could not do**\n" + "\n".join(f"- {r}" for r in refused))
    if warns:
        parts.append("\n".join(f"Note: {w}" for w in warns))
    changed = new_spec != spec or new_report != report
    return {"reply": "\n\n".join(p for p in parts if p) or "I did not change anything.",
            "changed": changed, "spec": new_spec, "report": new_report}


# ---------- export ----------
def _vega_scripts():
    page = alt.Chart(pd.DataFrame({"a": [1]})).mark_point().encode(x="a").to_html()
    return re.findall(r'<script[^>]+src="([^"]+)"', page)


def export_html(df, spec, title, report_md=""):
    ctx, esc = theme_ctx(spec), html.escape
    cards = ""
    for k in spec["kpis"]:
        r = compute_kpi(df, k, spec)
        cls = ""
        if r["delta"] and r["delta_color"] != "off":
            neg = r["delta"].startswith("-")
            cls = ctx["negative"] if (neg != (r["delta_color"] == "inverse")) else ctx["positive"]
        delta = f'<div class="kd" style="color:{cls}">{"&#9660;" if r["delta"] and r["delta"].startswith("-") else "&#9650;"} {esc(r["delta"])}</div>' if r["delta"] else ""
        stat = ""
        if r.get("status"):
            col = ctx["positive"] if r.get("status_positive") else ctx["negative"]
            stat = f'<div class="kd" style="color:{col}">{"&#9650;" if r.get("status_positive") else "&#9660;"} {esc(r["status"])}</div>'
        cards += f'<div class="kpi"><div class="kl">{esc(r["label"])}</div><div class="kv">{esc(r["value"])}</div>{delta}{stat}<div class="kn">{esc(r["note"])}</div></div>'
    blocks, embeds = "", ""
    for i, c in enumerate(spec["charts"]):
        chart, notes = build_chart(df, c, ctx)
        if chart is None:
            continue
        blocks += f'<div class="card"><div id="v{i}"></div>' + "".join(f'<p class="note">{esc(n)}</p>' for n in notes) + "</div>"
        js = json.dumps(chart.to_dict()).replace("</", "<\\/")
        embeds += f'vegaEmbed("#v{i}", {js}, {{actions:false}});\n'
    srcs = "".join(f'<script src="{esc(s)}"></script>' for s in _vega_scripts())
    review = "".join(f"<li>{esc(w)}</li>" for w in ethics_review(df, spec))
    return f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(title)}</title>{srcs}
<style>body{{font-family:Segoe UI,Arial,sans-serif;margin:0;background:#f5f6f8;color:#1b1f23}}
main{{max-width:1180px;margin:auto;padding:24px}}h1{{margin:0 0 4px}}.sub{{color:#57606a;margin-bottom:20px}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px;margin-bottom:20px}}
.kpi,.card{{background:#fff;border:1px solid #d8dee4;border-radius:10px;padding:14px}}
.kl{{color:#57606a;font-size:13px}}.kv{{font-size:28px;font-weight:600;margin:4px 0}}.kd{{font-size:13px;font-weight:600}}
.kn{{color:#6e7781;font-size:12px;margin-top:4px}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(460px,1fr));gap:14px}}
.note{{color:#6e7781;font-size:12px;margin:6px 0 0}}.foot{{margin-top:22px;color:#57606a;font-size:13px}}</style></head>
<body><main><h1>{esc(title)}</h1><div class="sub">Source: your uploaded data, {len(df):,} rows after cleaning. Web facts are not charted here.</div>
<div class="kpis">{cards}</div><div class="grid">{blocks}</div>
<div class="foot"><b>Integrity notes</b><ul>{review}</ul></div></main>
<script>{embeds}</script></body></html>"""
