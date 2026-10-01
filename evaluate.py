"""
Evaluation harness for the BI Copilot.  Run:  python evaluate.py
Ground truth is computed INDEPENDENTLY with plain pandas (not with the engine's functions),
then compared with what the copilot says.

Metrics
  intent_ok        copilot understood the question type correctly
  figure_ok        the key figure (e.g. % change) in the answer equals the independently computed value
  driver_ok        for 'why' questions: the true top drivers (category, city, channel) are named
  premise_ok       direction is right (does NOT accept a false premise such as "sales declined" when they rose)
  audit_ok         number audit passed (no unverified figure in the narrative)
  scope_ok         out-of-scope questions are declined; in-scope ones are answered
"""
import pandas as pd
from copilot.engine import Copilot, load_data, validate_router_output, audit_numbers

df = load_data()
months = sorted(df["month"].unique())
CUR, PREV = months[-3:], months[-6:-3]


def tot(d, metric, ms):
    return d[d["month"].isin(ms)][metric].sum()


def pct(d, metric, cur=CUR, prev=PREV):
    return (tot(d, metric, cur) - tot(d, metric, prev)) / tot(d, metric, prev) * 100


def top_driver(d, dim, metric):
    g = (d[d["month"].isin(CUR)].groupby(dim)[metric].sum() - d[d["month"].isin(PREV)].groupby(dim)[metric].sum())
    return g.idxmin() if g.sum() < 0 else g.idxmax()


def case(q, intent, **kw):
    return dict(q=q, intent=intent, **kw)


R = lambda r: df[df.region == r]
C = lambda c: df[df.city == c]
cases = []
# --- diagnose
d = R("South"); cases.append(case("Why did sales decline in the South region?", "diagnose", pct=pct(d, "sales"),
    drivers=[top_driver(d, "category", "sales"), top_driver(d, "city", "sales"), top_driver(d, "channel", "sales")], direction="fell"))
d = C("Bengaluru"); cases.append(case("Why did profit fall in Bengaluru?", "diagnose", pct=pct(d, "profit"),
    drivers=[top_driver(d, "category", "profit"), top_driver(d, "channel", "profit")], direction="rose", flag=True))  # FALSE PREMISE: profit rose (discount cut lifted margin)
d = C("Chennai"); cases.append(case("Why did sales drop in Chennai?", "diagnose", pct=pct(d, "sales"),
    drivers=[top_driver(d, "category", "sales"), top_driver(d, "channel", "sales")], direction="fell"))
d = R("South")[R("South").category == "Electronics"]; cases.append(case("Why did Electronics sales decline in the South region?", "diagnose", pct=pct(d, "sales"),
    drivers=[top_driver(d, "city", "sales"), top_driver(d, "channel", "sales")], direction="fell"))
d = R("South")[R("South").channel == "Online"]; cases.append(case("Why did online sales fall in the South region?", "diagnose", pct=pct(d, "sales"),
    drivers=[top_driver(d, "category", "sales"), top_driver(d, "city", "sales")], direction="fell"))
d = R("North"); cases.append(case("Why did sales rise in the North region?", "diagnose", pct=pct(d, "sales"), direction="rose"))
d = R("West"); cases.append(case("Why did sales decline in the West region?", "diagnose", pct=pct(d, "sales"), direction="rose", flag=True))  # FALSE PREMISE
d = R("South"); cases.append(case("Why did sales decline in the South region year on year?", "diagnose",
    pct=pct(d, "sales", CUR, months[-15:-12]), direction=None))
# --- top / compare / trend / summary / definition / unsupported
reg_chg = {r: pct(R(r), "sales") for r in ["North", "South", "East", "West"]}
cases.append(case("Which region declined the most?", "top", first=min(reg_chg, key=reg_chg.get)))
reg_val = {r: tot(R(r), "sales", CUR) for r in reg_chg}
cases.append(case("Which region has the highest sales?", "top", first=max(reg_val, key=reg_val.get)))
d = R("North"); cat_val = d[d.month.isin(CUR)].groupby("category")["sales"].sum()
cases.append(case("Top categories in the North region", "top", first=cat_val.idxmax()))
cases.append(case("Compare regions by sales", "compare", first=max(reg_val, key=reg_val.get)))
d = R("West"); s = d.groupby("month")["sales"].sum().tail(12)
cases.append(case("Show the monthly sales trend for the West region", "trend", peak=pd.Timestamp(s.idxmax()).strftime("%b %Y")))
d = R("East"); s = d.groupby("month")["profit"].sum().tail(12)
cases.append(case("Monthly profit trend in the East region", "trend", peak=pd.Timestamp(s.idxmax()).strftime("%b %Y")))
cases.append(case("How did the North region perform?", "summary", pct=pct(R("North"), "sales")))
d = R("South")[R("South").channel == "Online"]; cases.append(case("How did online sales perform in the South?", "summary", pct=pct(d, "sales")))
cases.append(case("What is net price per unit?", "definition", contains="average price customers actually paid"))
cases.append(case("What does QoQ mean?", "definition", contains="quarter on quarter"))
cases.append(case("Forecast sales for next quarter", "unsupported"))
cases.append(case("What will profit be next year?", "unsupported"))
cases.append(case("Who is the CEO of the company?", "unsupported"))

cp = Copilot()  # offline template mode (no API key) - deterministic and reproducible
rows = []
for c in cases:
    a = cp.ask(c["q"])
    t = a.text
    row = {"question": c["q"], "expected_intent": c["intent"], "got_intent": a.query.intent}
    row["intent_ok"] = a.query.intent == c["intent"]
    row["figure_ok"] = row["driver_ok"] = row["premise_ok"] = None
    if "pct" in c:
        tok = f"{abs(c['pct']):.1f}%"
        row["figure_ok"] = tok in t
    if c.get("drivers"):
        row["driver_ok"] = all(x in t for x in c["drivers"])
    if c.get("direction"):
        row["premise_ok"] = f"{c['direction']}" in t.split("(")[0]
    if c.get("flag"):  # a false premise must be called out explicitly
        row["premise_ok"] = bool(row["premise_ok"]) and "the question assumes" in t
    if "first" in c:
        body = t.split("\n", 2)[-1]
        names = ["North", "South", "East", "West", "Electronics", "Appliances", "Apparel", "Grocery", "Furniture"]
        pos = {n: body.find(n) for n in names if n in body}
        row["figure_ok"] = bool(pos) and min(pos, key=pos.get) == c["first"]
    if "peak" in c:
        row["figure_ok"] = f"peak was {c['peak']}" in t
    if "contains" in c:
        row["figure_ok"] = c["contains"] in t.lower()
    row["audit_ok"] = a.audit["passed"]
    row["scope_ok"] = (c["intent"] == "unsupported") == (a.query.intent == "unsupported")
    rows.append(row)

res = pd.DataFrame(rows)
res.to_csv("eval_results.csv", index=False)

summary = {}
for col in ["intent_ok", "figure_ok", "driver_ok", "premise_ok", "audit_ok", "scope_ok"]:
    s = res[col].dropna()
    summary[col] = f"{int(s.sum())}/{len(s)} ({s.mean() * 100:.0f}%)"
print(res[["question", "got_intent", "intent_ok", "figure_ok", "driver_ok", "premise_ok", "audit_ok"]].to_string(index=False))
print("\nSUMMARY");  [print(f"  {k:10s} {v}") for k, v in summary.items()]
pd.Series(summary).to_csv("eval_summary.csv", header=["result"])

# ---- guardrail tests -------------------------------------------------------------------------
print("\nGUARDRAIL TESTS")
a = cp.ask("Why did sales decline in the South region?")
fake = a.text + "\nSales also fell by 42.3% in Hyderabad, costing ₹99.9 crore."
r = audit_numbers(fake, a.facts)
print(f"  Audit catches fabricated figures: {not r['passed']} (flagged {r['unverified']})")
bad = [{"intent": "drop table"}, {"intent": "diagnose", "region": "Mars", "metric": "sales"},
       {"intent": "top", "dimension": "dept; DROP", "metric": "profit"}]
outs = [validate_router_output(b, "x") for b in bad]
print(f"  Router validation rejects bad intent: {outs[0] is None}; sanitises invalid fields: "
      f"{outs[1].region is None and outs[2].dimension == 'category'}")
