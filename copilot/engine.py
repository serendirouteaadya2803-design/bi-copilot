"""
AI Business Intelligence Copilot - core engine.

Pipeline (see report for the architecture diagram):
  1. UNDERSTAND   question -> intent + filters + time period   (LLM router if API key, else rules)
  2. ANALYSE      pandas "tools" compute every number           (never the LLM)
  3. RETRIEVE     TF-IDF search over company knowledge base     (the RAG step)
  4. EXPLAIN      narrative written from the verified facts      (LLM if API key, else template)
  5. AUDIT        every figure in the narrative is checked against the verified facts
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE = Path(__file__).resolve().parent.parent
REGIONS = ["North", "South", "East", "West"]
CATEGORIES = ["Electronics", "Appliances", "Apparel", "Grocery", "Furniture"]
CHANNELS = ["Online", "Retail Store"]
CITY_REGION = {
    "Delhi": "North", "Chandigarh": "North", "Jaipur": "North", "Lucknow": "North",
    "Chennai": "South", "Bengaluru": "South", "Hyderabad": "South", "Kochi": "South",
    "Kolkata": "East", "Bhubaneswar": "East", "Patna": "East", "Guwahati": "East",
    "Mumbai": "West", "Pune": "West", "Ahmedabad": "West", "Surat": "West",
}
MONTH_NAMES = {m: i + 1 for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"])}
DEFAULT_MODEL = "claude-sonnet-5-5"


# ----------------------------------------------------------------------------- data
def load_data(path: Path | str | None = None) -> pd.DataFrame:
    df = pd.read_csv(path or BASE / "sales_data.csv", parse_dates=["month"])
    return df


# ----------------------------------------------------------------------------- formatting
def fmt_money(x: float) -> str:
    sign = "-" if x < 0 else ""
    x = abs(x)
    if x >= 1e7:
        return f"{sign}₹{x / 1e7:,.2f} crore"
    return f"{sign}₹{x / 1e5:,.1f} lakh"


def fmt_pct(x: float, signed: bool = False) -> str:
    return f"{x:+.1f}%" if signed else f"{x:.1f}%"


def fmt_units(x: float) -> str:
    return f"{x:,.0f} units"


def fmt_metric(metric: str, x: float) -> str:
    return {"sales": fmt_money, "profit": fmt_money, "units": fmt_units}[metric](x)


def month_label(ts: pd.Timestamp) -> str:
    return ts.strftime("%b %Y")


# ----------------------------------------------------------------------------- periods
@dataclass
class Period:
    cur: list          # list of month Timestamps (current period)
    prev: list         # list of month Timestamps (comparison period)
    label_cur: str
    label_prev: str
    kind: str          # quarter | month | yoy


def _range_label(months: list) -> str:
    if len(months) == 1:
        return month_label(months[0])
    return f"{months[0].strftime('%b')}-{months[-1].strftime('%b %Y')}"


def resolve_period(df: pd.DataFrame, text: str) -> Period:
    months = sorted(df["month"].unique())
    months = [pd.Timestamp(m) for m in months]
    q = text.lower()

    m = re.search(r"\b(?:in|for|during|of)\s+(" + "|".join(MONTH_NAMES) + r")\b", q)
    if m:
        num = MONTH_NAMES[m.group(1)]
        match = [x for x in months if x.month == num]
        if match:
            cur_m = match[-1]
            idx = months.index(cur_m)
            if idx >= 1:
                return Period([cur_m], [months[idx - 1]], month_label(cur_m), month_label(months[idx - 1]), "month")

    if re.search(r"last month|latest month|this month|month on month|\bmom\b", q):
        return Period([months[-1]], [months[-2]], month_label(months[-1]), month_label(months[-2]), "month")

    if re.search(r"year on year|year-on-year|\byoy\b|last year|previous year", q):
        cur, prev = months[-3:], months[-15:-12]
        return Period(cur, prev, _range_label(cur), _range_label(prev), "yoy")

    cur, prev = months[-3:], months[-6:-3]
    return Period(cur, prev, _range_label(cur), _range_label(prev), "quarter")


# ----------------------------------------------------------------------------- understanding
@dataclass
class Query:
    intent: str                     # diagnose | top | compare | trend | summary | definition | unsupported
    metric: str = "sales"
    region: str | None = None
    city: str | None = None
    category: str | None = None
    channel: str | None = None
    dimension: str | None = None    # for 'top'
    rank_by: str = "value"          # value | change
    ascending: bool = False
    raw: str = ""


UNSUPPORTED = r"forecast|predict|projection|next (?:month|quarter|year)|will (?:sales|we)|salary|employee|stock price|weather|share price"
DIAGNOSE = r"\bwhy\b|reason|cause|declin|drop|fall|fell|decreas|dip|slump|down\b|weaken|lower"
TOP = r"\btop\b|best|highest|biggest|largest|worst|lowest|weakest|leading|fastest|\bwhich (?:region|city|cities|categor|channel|product)"
COMPARE = r"compare|versus|\bvs\.?\b|across regions|between regions|region.?wise"
TREND = r"trend|over time|monthly|month by month|month-by-month|trajectory|history"
DEFINE = r"what is|what does|define|meaning of|explain the term|stands for"
DATA_TERMS = r"sales|revenue|profit|units|discount|margin|price|qoq|mom|yoy|quarter on quarter|region|city|categor|channel|product|customer|south|north|east|west|decline|performance|growth|trend"


def parse_question_rules(question: str) -> Query:
    q = question.lower()
    qy = Query(intent="summary", raw=question)

    # filters
    for r in REGIONS:
        if re.search(rf"\b{r.lower()}\b", q):
            qy.region = r
    for c in CITY_REGION:
        if re.search(rf"\b{c.lower()}\b", q):
            qy.city = c
            qy.region = qy.region or CITY_REGION[c]
    for c in CATEGORIES:
        if re.search(rf"\b{c.lower()}\b", q):
            qy.category = c
    if re.search(r"\bonline\b", q):
        qy.channel = "Online"
    elif re.search(r"\bretail\b|\bstores?\b|offline", q):
        qy.channel = "Retail Store"

    # metric
    if "profit" in q:
        qy.metric = "profit"
    elif re.search(r"\bunits\b|volume|quantity|sold", q):
        qy.metric = "units"

    # intent (order matters)
    no_data_terms = not re.search(DATA_TERMS, q) and not (qy.region or qy.city or qy.category)
    if re.search(UNSUPPORTED, q):
        qy.intent = "unsupported"
    elif re.search(DEFINE, q) and not (qy.region or qy.city) and re.search(DATA_TERMS, q):
        qy.intent = "definition"
    elif no_data_terms:
        qy.intent = "unsupported"
    elif re.search(r"\bwhy\b|reason|cause", q):
        qy.intent = "diagnose"
    elif re.search(r"which (?:region|city|cities|categor|channel|product)|\btop\b|best|highest|biggest|largest|worst|lowest|weakest|fastest", q):
        qy.intent = "top"
        # dimension: explicit words win; "North region" as a FILTER must not be mistaken for "rank the regions"
        if re.search(r"categor|product", q):
            qy.dimension = "category"
        elif re.search(r"\bcity|cities", q):
            qy.dimension = "city"
        elif "channel" in q:
            qy.dimension = "channel"
        elif re.search(r"which region|regions|(?:top|best|worst|highest|lowest|weakest|biggest|largest)\s+region", q):
            qy.dimension = "region"
        else:
            qy.dimension = "category"
        if re.search(r"declin|drop|fell|fall|grew|growth|fastest|increase", q):
            qy.rank_by = "change"
        qy.ascending = bool(re.search(r"worst|lowest|weakest|declin|drop|fell|fall|least", q))
    elif re.search(COMPARE, q):
        qy.intent = "compare"
    elif re.search(TREND, q):
        qy.intent = "trend"
    elif re.search(DIAGNOSE, q):
        qy.intent = "diagnose"
    return qy


ROUTER_SYSTEM = (
    "You are the query router of a business-intelligence copilot. Convert the manager's question into JSON ONLY "
    "(no prose, no code fences) with keys: intent (diagnose|top|compare|trend|summary|definition|unsupported), "
    "metric (sales|profit|units), region (North|South|East|West|null), city (string|null), "
    "category (Electronics|Appliances|Apparel|Grocery|Furniture|null), channel (Online|Retail Store|null), "
    "dimension (region|city|category|channel|null), rank_by (value|change), ascending (true|false). "
    "Use 'unsupported' for forecasts, predictions or anything not answerable from historical sales data."
)


def parse_question_llm(question: str, api_key: str, model: str = DEFAULT_MODEL) -> Query | None:
    """Optional LLM router. Output is VALIDATED against allowed values; anything invalid -> None (rules take over)."""
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        msg = client.messages.create(model=model, max_tokens=300, system=ROUTER_SYSTEM,
                                     messages=[{"role": "user", "content": question}])
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        data = json.loads(re.sub(r"```json|```", "", text).strip())
        return validate_router_output(data, question)
    except Exception:
        return None


def validate_router_output(data: dict, question: str) -> Query | None:
    ok_intents = {"diagnose", "top", "compare", "trend", "summary", "definition", "unsupported"}
    if data.get("intent") not in ok_intents:
        return None
    def pick(key, allowed):
        v = data.get(key)
        return v if v in allowed else None
    q = Query(intent=data["intent"], raw=question)
    q.metric = pick("metric", {"sales", "profit", "units"}) or "sales"
    q.region = pick("region", set(REGIONS))
    q.city = pick("city", set(CITY_REGION))
    q.category = pick("category", set(CATEGORIES))
    q.channel = pick("channel", set(CHANNELS))
    q.dimension = pick("dimension", {"region", "city", "category", "channel"})
    q.rank_by = pick("rank_by", {"value", "change"}) or "value"
    q.ascending = bool(data.get("ascending", False))
    if q.city and not q.region:
        q.region = CITY_REGION[q.city]
    if q.intent == "top" and not q.dimension:
        q.dimension = "category"
    return q


# ----------------------------------------------------------------------------- analysis tools
def apply_filters(df: pd.DataFrame, qy: Query) -> pd.DataFrame:
    out = df
    if qy.region:
        out = out[out["region"] == qy.region]
    if qy.city:
        out = out[out["city"] == qy.city]
    if qy.category:
        out = out[out["category"] == qy.category]
    if qy.channel:
        out = out[out["channel"] == qy.channel]
    return out


def slice_months(df: pd.DataFrame, months: list) -> pd.DataFrame:
    return df[df["month"].isin(months)]


def kpis(d: pd.DataFrame) -> dict:
    sales, profit, units = d["sales"].sum(), d["profit"].sum(), d["units"].sum()
    gross = (d["units"] * d["list_price"]).sum()
    return {
        "sales": float(sales), "profit": float(profit), "units": float(units),
        "margin_pct": float(profit / sales * 100) if sales else 0.0,
        "net_price": float(sales / units) if units else 0.0,
        "discount_pct": float((1 - sales / gross) * 100) if gross else 0.0,
    }


def pct_change(cur: float, prev: float) -> float:
    return float((cur - prev) / prev * 100) if prev else 0.0


def driver_table(cur: pd.DataFrame, prev: pd.DataFrame, dim: str, metric: str) -> pd.DataFrame:
    a = cur.groupby(dim)[metric].sum().rename("current")
    b = prev.groupby(dim)[metric].sum().rename("previous")
    t = pd.concat([a, b], axis=1).fillna(0.0)
    t["change"] = t["current"] - t["previous"]
    t["change_pct"] = np.where(t["previous"] != 0, t["change"] / t["previous"] * 100, 0.0)
    total = t["change"].sum()
    t["share_of_change_pct"] = np.where(total != 0, t["change"] / total * 100, 0.0)
    asc = total < 0  # biggest contributors to the move first
    return t.sort_values("change", ascending=asc).reset_index()


def price_volume(cur: pd.DataFrame, prev: pd.DataFrame) -> dict:
    """Split the sales change into a VOLUME effect and a PRICE effect.
    Done at the finest level (city x category x channel) so that a shift in product mix
    (e.g. more low-priced grocery units) is not mistaken for a price change.
        volume effect = sum over segments of (units_now - units_before) x net_price_before
        price effect  = sum over segments of (net_price_now - net_price_before) x units_now
    The two add up exactly to the total change in sales."""
    keys = ["city", "category", "channel"]
    a = cur.groupby(keys).agg(u2=("units", "sum"), s2=("sales", "sum"))
    b = prev.groupby(keys).agg(u1=("units", "sum"), s1=("sales", "sum"))
    t = a.join(b, how="outer").fillna(0.0)
    p1 = np.where(t["u1"] > 0, t["s1"] / t["u1"].replace(0, np.nan), 0.0)
    p2 = np.where(t["u2"] > 0, t["s2"] / t["u2"].replace(0, np.nan), 0.0)
    p1 = np.nan_to_num(p1)
    p2 = np.nan_to_num(p2)
    vol = float(((t["u2"] - t["u1"]) * p1).sum())
    price = float(((p2 - p1) * t["u2"]).sum())
    return {"volume_effect": vol, "price_effect": price, "check": float(t["s2"].sum() - t["s1"].sum())}


def hotspots(cur: pd.DataFrame, prev: pd.DataFrame, metric: str, n: int = 3) -> pd.DataFrame:
    keys = ["city", "category", "channel"]
    a = cur.groupby(keys)[metric].sum().rename("current")
    b = prev.groupby(keys)[metric].sum().rename("previous")
    t = pd.concat([a, b], axis=1).fillna(0.0)
    t["change"] = t["current"] - t["previous"]
    total = t["change"].sum()
    t["share_of_change_pct"] = np.where(total != 0, t["change"] / total * 100, 0.0)
    asc = total < 0
    return t.sort_values("change", ascending=asc).head(n).reset_index()


def monthly_series(d: pd.DataFrame, metric: str) -> pd.DataFrame:
    return d.groupby("month")[metric].sum().reset_index()


# ----------------------------------------------------------------------------- retrieval (RAG)
def _load_chunks() -> list[dict]:
    chunks = []
    for line in (BASE / "knowledge" / "business_events.md").read_text(encoding="utf-8").splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) == 4 and re.match(r"\d{4}-\d{2}-\d{2}", parts[0]):
            chunks.append({"type": "event", "date": pd.Timestamp(parts[0]), "region": parts[1],
                           "area": parts[2], "text": parts[3],
                           "search": f"{parts[1]} {parts[2]} {parts[3]}"})
    for line in (BASE / "knowledge" / "metric_glossary.md").read_text(encoding="utf-8").splitlines():
        if ":" in line and not line.startswith("#"):
            term, desc = line.split(":", 1)
            chunks.append({"type": "glossary", "term": term.strip(), "text": desc.strip(),
                           "search": f"{term} {desc}"})
    return chunks


class KnowledgeBase:
    def __init__(self):
        self.chunks = _load_chunks()
        self.vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.matrix = self.vec.fit_transform([c["search"] for c in self.chunks])

    def search(self, query: str, kind: str, k: int = 3, region: str | None = None,
               window: tuple | None = None, min_score: float = 0.05) -> list[dict]:
        sims = cosine_similarity(self.vec.transform([query]), self.matrix)[0]
        scored = []
        for chunk, s in zip(self.chunks, sims):
            if chunk["type"] != kind:
                continue
            score = float(s)
            if kind == "event":
                if region and chunk["region"] in (region, "All"):
                    score += 0.15
                if window and window[0] <= chunk["date"] <= window[1]:
                    score += 0.15
                elif window:
                    score -= 0.10
            scored.append((score, chunk))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [dict(c, score=round(s, 3)) for s, c in scored[:k] if s >= min_score]


# ----------------------------------------------------------------------------- answer object
@dataclass
class Answer:
    question: str
    query: Query
    period: Period | None
    facts: dict
    tables: dict = field(default_factory=dict)
    text: str = ""
    mode: str = "template"        # template | llm
    audit: dict = field(default_factory=dict)
    router: str = "rules"


def _scope_label(qy: Query) -> str:
    parts = [p for p in [qy.city, qy.region if not qy.city else None, qy.category, qy.channel] if p]
    return " / ".join(parts) if parts else "All regions"


def _rows(t: pd.DataFrame, dim: str, metric: str, n: int = 5) -> list[dict]:
    out = []
    for _, r in t.head(n).iterrows():
        out.append({
            dim: r[dim],
            "previous": fmt_metric(metric, r["previous"]),
            "current": fmt_metric(metric, r["current"]),
            "change": fmt_metric(metric, r["change"]),
            "change_pct": fmt_pct(r["change_pct"], signed=True),
            "share_of_total_change": fmt_pct(r["share_of_change_pct"]),
        })
    return out


# ----------------------------------------------------------------------------- intent handlers
def analyse(df: pd.DataFrame, qy: Query, kb: KnowledgeBase) -> Answer:
    period = resolve_period(df, qy.raw)
    scope = apply_filters(df, qy)
    facts: dict = {"intent": qy.intent, "metric": qy.metric, "scope": _scope_label(qy)}
    tables: dict = {}

    if qy.intent == "unsupported":
        facts["note"] = ("This question is outside what the copilot can answer from historical sales data "
                         "(e.g. forecasts, predictions, or topics unrelated to the sales dataset).")
        return Answer(qy.raw, qy, None, facts, tables)

    if qy.intent == "definition":
        hits = kb.search(qy.raw, "glossary", k=2, min_score=0.03)
        facts["definitions"] = [{"term": h["term"], "meaning": h["text"]} for h in hits]
        return Answer(qy.raw, qy, None, facts, tables)

    cur, prev = slice_months(scope, period.cur), slice_months(scope, period.prev)
    k_cur, k_prev = kpis(cur), kpis(prev)
    m = qy.metric
    facts["period"] = {"current": period.label_cur, "previous": period.label_prev, "kind": period.kind}
    facts["kpi"] = {
        "current": fmt_metric(m, k_cur[m]), "previous": fmt_metric(m, k_prev[m]),
        "change": fmt_metric(m, k_cur[m] - k_prev[m]),
        "change_pct": fmt_pct(pct_change(k_cur[m], k_prev[m]), signed=True),
        "direction": "fell" if k_cur[m] < k_prev[m] else "rose",
        "margin_current": fmt_pct(k_cur["margin_pct"]), "margin_previous": fmt_pct(k_prev["margin_pct"]),
        "discount_current": fmt_pct(k_cur["discount_pct"]), "discount_previous": fmt_pct(k_prev["discount_pct"]),
    }
    facts["_raw"] = {"cur": k_cur, "prev": k_prev}
    said_down = bool(re.search(r"declin|drop|fall|fell|decreas|dip|slump|lower|weaken", qy.raw.lower()))
    said_up = bool(re.search(r"\brise|rose|increas|grew|growth|jump|surge|higher", qy.raw.lower()))
    went_down = k_cur[m] < k_prev[m]
    if (said_down and not went_down) or (said_up and went_down):
        facts["premise_check"] = (f"Note: the question assumes {m} {'declined' if said_down else 'increased'}, "
                                  f"but the data shows {m} {facts['kpi']['direction']} {facts['kpi']['change_pct'].lstrip('+-')} "
                                  f"in this scope and period. The analysis below explains what actually happened.")

    if qy.intent == "diagnose":
        dims = ["category", "city", "channel"]
        if not qy.region and not qy.city:
            dims = ["region"] + dims
        facts["drivers"] = {}
        for d in dims:
            if d == "category" and qy.category:
                continue
            if d == "channel" and qy.channel:
                continue
            if d == "city" and qy.city:
                continue
            t = driver_table(cur, prev, d, m)
            tables[f"by_{d}"] = t
            facts["drivers"][d] = _rows(t, d, m, 4)
        pv = price_volume(cur, prev)
        facts["price_volume"] = {
            "volume_effect": fmt_money(pv["volume_effect"]), "price_effect": fmt_money(pv["price_effect"]),
            "volume_dominates": abs(pv["volume_effect"]) >= abs(pv["price_effect"]),
        }
        hs = hotspots(cur, prev, m, 3)
        tables["hotspots"] = hs
        facts["hotspots"] = [{
            "city": r.city, "category": r.category, "channel": r.channel,
            "change": fmt_metric(m, r.change), "share_of_total_change": fmt_pct(r.share_of_change_pct),
        } for r in hs.itertuples()]
        tables["trend"] = monthly_series(scope, m)
        if not qy.region:
            tables["trend_by_region"] = df.groupby(["month", "region"])[m].sum().reset_index()

        # RAG: retrieve company context for the likely causes
        top_terms = " ".join([facts["hotspots"][0]["city"], facts["hotspots"][0]["category"],
                              facts["hotspots"][0]["channel"]] if hs.shape[0] else [])
        region_for_ctx = qy.region or (CITY_REGION.get(hs.iloc[0]["city"]) if hs.shape[0] else None)
        win = (min(period.prev), max(period.cur) + pd.offsets.MonthEnd(0))
        ctx = kb.search(f"{qy.raw} {region_for_ctx or ''} {top_terms} discount price competitor decline",
                        "event", k=3, region=region_for_ctx, window=win)
        facts["context"] = [{"date": c["date"].strftime("%d %b %Y"), "region": c["region"],
                             "area": c["area"], "note": c["text"]} for c in ctx]

    elif qy.intent == "top":
        dim = qy.dimension or "category"
        base = df if dim == "region" else scope
        t = driver_table(slice_months(base, period.cur), slice_months(base, period.prev), dim, m)
        key = "change" if qy.rank_by == "change" else "current"
        t = t.sort_values(key, ascending=qy.ascending).reset_index(drop=True)
        tables["ranking"] = t
        facts["ranking"] = {"dimension": dim, "ranked_by": "change" if qy.rank_by == "change" else "current value",
                            "order": "lowest first" if qy.ascending else "highest first",
                            "rows": _rows(t, dim, m, 5)}

    elif qy.intent == "compare":
        t = driver_table(slice_months(df, period.cur), slice_months(df, period.prev), "region", m)
        t = t.sort_values("current", ascending=False).reset_index(drop=True)
        tables["regions"] = t
        facts["regions"] = _rows(t, "region", m, 4)

    elif qy.intent == "trend":
        s = monthly_series(scope, m).tail(12).reset_index(drop=True)
        tables["trend"] = s
        hi, lo = s.loc[s[m].idxmax()], s.loc[s[m].idxmin()]
        facts["trend"] = {
            "months_shown": f"{month_label(s['month'].iloc[0])} to {month_label(s['month'].iloc[-1])}",
            "first": fmt_metric(m, s[m].iloc[0]), "last": fmt_metric(m, s[m].iloc[-1]),
            "change_pct": fmt_pct(pct_change(s[m].iloc[-1], s[m].iloc[0]), signed=True),
            "peak_month": month_label(hi["month"]), "peak_value": fmt_metric(m, hi[m]),
            "low_month": month_label(lo["month"]), "low_value": fmt_metric(m, lo[m]),
        }
    else:  # summary
        t = driver_table(cur, prev, "category" if not qy.category else "city", m)
        tables["breakdown"] = t
        facts["breakdown"] = _rows(t, t.columns[0], m, 5)
        tables["trend"] = monthly_series(scope, m)

    return Answer(qy.raw, qy, period, facts, tables)


# ----------------------------------------------------------------------------- recommendations (rule-based, grounded in facts)
def recommend(ans: Answer) -> list[str]:
    f = ans.facts
    if ans.query.intent != "diagnose" or "kpi" not in f:
        return []
    recs = []
    raw = f["_raw"]
    if f["kpi"]["direction"] == "rose":
        k = f["kpi"]
        recs.append(f"Check that the gain is sustainable: margin moved from {k['margin_previous']} to {k['margin_current']} "
                    f"while the average discount moved from {k['discount_previous']} to {k['discount_current']}.")
        if f.get("hotspots"):
            h = f["hotspots"][0]
            recs.append(f"Study what is working in {h['category']} ({h['channel']}) in {h['city']}, the largest positive contributor, "
                        f"and see whether it can be repeated elsewhere.")
        recs.append("Watch units sold closely: growth that comes from price or margin rather than volume can reverse quickly.")
        recs.append("Re-run this analysis weekly on fresh data to confirm the trend.")
        return recs
    disc_drop = raw["prev"]["discount_pct"] - raw["cur"]["discount_pct"]
    top = f["hotspots"][0] if f.get("hotspots") else None
    ch = f["drivers"].get("channel", [])
    if disc_drop >= 3:
        cats = ", ".join(r["category"] for r in f["drivers"].get("category", [])[:2]) or "affected categories"
        recs.append(f"Review the discount cut: the average discount fell by {disc_drop:.1f} percentage points. "
                    f"Test targeted, time-limited offers on {cats} instead of removing promotions entirely.")
    if ch and ch[0]["channel"] == "Online" and float(ch[0]["share_of_total_change"].rstrip('%')) >= 50:
        recs.append("The online channel carries most of the decline: price-match the top-selling items, add free-delivery "
                    "thresholds, and track competitor online prices weekly.")
    if top:
        recs.append(f"Prioritise a recovery plan for {top['category']} ({top['channel']}) in {top['city']}, "
                    f"the single biggest hotspot ({top['share_of_total_change']} of the total change).")
    if f["price_volume"]["volume_dominates"]:
        recs.append("Volume (fewer units sold) explains more of the fall than price, so focus on demand recovery "
                    "(promotions, availability, local marketing) before changing list prices.")
    recs.append("Re-run this analysis weekly on fresh data to confirm whether the actions are working.")
    return recs[:5]


# ----------------------------------------------------------------------------- explanation
def public_facts(facts: dict) -> dict:
    return {k: v for k, v in facts.items() if not k.startswith("_")}


def narrate_template(ans: Answer, recs: list[str]) -> str:
    f, qy = ans.facts, ans.query
    m = qy.metric
    if qy.intent == "unsupported":
        return f"**Out of scope.** {f['note']} Try asking, for example: *\"Why did sales decline in the South region?\"*"
    if qy.intent == "definition":
        if not f["definitions"]:
            return "I could not find that term in the company glossary."
        return "\n\n".join(f"**{d['term']}**: {d['meaning']}" for d in f["definitions"])

    k, p = f["kpi"], f["period"]
    head = (f"**{f['scope']} {m} {k['direction']} {k['change_pct'].lstrip('+-')}** "
            f"({p['current']} vs {p['previous']}): from {k['previous']} to {k['current']}, a change of {k['change']}.")

    if qy.intent == "diagnose":
        lines = [head] + (["", f["premise_check"]] if f.get("premise_check") else []) + ["", "**What drove it**"]
        for dim, rows in f["drivers"].items():
            top = rows[0]
            lines.append(f"- By {dim}: **{top[dim]}** contributed most ({top['change']}, "
                         f"{top['share_of_total_change']} of the total change).")
        pv = f["price_volume"]
        dom = "volume (units sold)" if pv["volume_dominates"] else "price (net price per unit)"
        lines.append(f"- Price vs volume: volume effect {pv['volume_effect']}, price effect {pv['price_effect']}; "
                     f"{dom} is the larger driver. The average discount moved from {k['discount_previous']} to {k['discount_current']}.")
        if f.get("hotspots"):
            h = f["hotspots"][0]
            lines.append(f"- Biggest hotspot: **{h['category']} ({h['channel']}) in {h['city']}**, "
                         f"{h['change']} ({h['share_of_total_change']} of the total change).")
        if f.get("context"):
            lines += ["", "**Possible context from company records** (retrieved background, not proof of cause)"]
            for c in f["context"]:
                lines.append(f"- {c['date']} ({c['region']}): {c['note']}")
        if recs:
            lines += ["", "**Suggested actions**"] + [f"{i}. {r}" for i, r in enumerate(recs, 1)]
        return "\n".join(lines)

    if qy.intent == "top":
        r = f["ranking"]
        lines = [head, "", f"**{r['dimension'].title()} ranking** (by {r['ranked_by']}, {r['order']}):"]
        for row in r["rows"]:
            lines.append(f"- {row[r['dimension']]}: {row['current']} ({row['change_pct']} vs {p['previous']})")
        return "\n".join(lines)

    if qy.intent == "compare":
        lines = [head, "", "**Regions compared**"]
        for row in f["regions"]:
            lines.append(f"- {row['region']}: {row['current']} ({row['change_pct']} vs {p['previous']})")
        return "\n".join(lines)

    if qy.intent == "trend":
        t = f["trend"]
        return (f"**{f['scope']} {m} trend, {t['months_shown']}**: it moved from {t['first']} to {t['last']} "
                f"({t['change_pct']}). The peak was {t['peak_month']} at {t['peak_value']} and the lowest point "
                f"was {t['low_month']} at {t['low_value']}.")

    lines = [head, "", "**Breakdown**"]
    for row in f["breakdown"]:
        name = [v for kx, v in row.items() if kx not in ("previous", "current", "change", "change_pct", "share_of_total_change")][0]
        lines.append(f"- {name}: {row['current']} ({row['change_pct']} vs {p['previous']})")
    return "\n".join(lines)


EXPLAIN_SYSTEM = """You are a business-intelligence copilot writing for a busy manager.
STRICT RULES:
1. Use ONLY the facts in the JSON provided. Never invent or recalculate numbers; copy figures exactly as written.
2. If the facts contain premise_check, state it right after the headline. Retrieved company context is background, not proof. Use wording like "may have contributed" or "is consistent with".
3. Structure: one-line headline, "What drove it" (bullets), "Possible context" (if provided), "Suggested actions" (2-4 numbered, specific, grounded in the facts).
4. If the facts do not answer the question, say so plainly. Be concise (under 250 words). Plain business English."""


def narrate_llm(ans: Answer, recs: list[str], api_key: str, model: str = DEFAULT_MODEL) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    payload = {"question": ans.question, "facts": public_facts(ans.facts), "draft_recommendations": recs}
    msg = client.messages.create(
        model=model, max_tokens=900, system=EXPLAIN_SYSTEM,
        messages=[{"role": "user", "content": json.dumps(payload, ensure_ascii=False, default=str)}])
    return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()


# ----------------------------------------------------------------------------- audit (hallucination guard)
FIG = re.compile(r"(₹\s?)?(\d[\d,]*(?:\.\d+)?)\s?(%|lakh|crore|units|percentage points)?", re.I)


def _figures(text: str) -> set[float]:
    out = set()
    for m in FIG.finditer(text):
        if m.group(1) or m.group(3):  # only tokens that look like business figures
            out.add(round(float(m.group(2).replace(",", "")), 2))
    return out


def audit_numbers(text: str, facts: dict, extra_allowed: str = "") -> dict:
    """Every money/percent/unit figure in the narrative must appear in the verified facts."""
    allowed_text = json.dumps(public_facts(facts), ensure_ascii=False, default=str) + " " + extra_allowed
    allowed = _figures(allowed_text)
    # also accept figures re-derived from the raw KPI block expressed in the same units (discount drop etc.)
    found = _figures(text)
    unverified = sorted(x for x in found if x not in allowed)
    return {"figures_found": len(found), "unverified": unverified, "passed": len(unverified) == 0}


# ----------------------------------------------------------------------------- orchestrator
class Copilot:
    def __init__(self, df: pd.DataFrame | None = None, api_key: str | None = None, model: str = DEFAULT_MODEL):
        self.df = df if df is not None else load_data()
        self.kb = KnowledgeBase()
        self.api_key = api_key or None
        self.model = model

    def ask(self, question: str) -> Answer:
        qy, router = None, "rules"
        if self.api_key:
            qy = parse_question_llm(question, self.api_key, self.model)
            router = "llm" if qy else "rules"
        if qy is None:
            qy = parse_question_rules(question)
        ans = analyse(self.df, qy, self.kb)
        ans.router = router
        recs = recommend(ans)
        ans.facts["recommendations"] = recs

        text, mode = None, "template"
        if self.api_key and qy.intent not in ("unsupported", "definition"):
            try:
                text, mode = narrate_llm(ans, recs, self.api_key, self.model), "llm"
            except Exception:
                text = None
        if text is None:
            text, mode = narrate_template(ans, recs), "template"
        ans.text, ans.mode = text, mode
        extra = " ".join(c["text"] for c in self.kb.chunks if c["type"] == "event")
        ans.audit = audit_numbers(text, ans.facts, extra)
        return ans
