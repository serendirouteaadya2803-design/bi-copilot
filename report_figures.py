"""Creates the figures used in the project report (matplotlib)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from copilot.engine import Copilot

NAVY, TEAL, GREY, RED, GREEN = "#1F3A5F", "#2A9D8F", "#6B7280", "#C0392B", "#2E8B57"

# ---------- 1. architecture diagram
fig, ax = plt.subplots(figsize=(11, 5.6)); ax.set_xlim(0, 110); ax.set_ylim(0, 56); ax.axis("off")
def box(x, y, w, h, title, sub, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.5", fc=color, ec="none"))
    ax.text(x + w / 2, y + h - 3.2, title, ha="center", va="top", color="white", fontsize=10.5, fontweight="bold")
    ax.text(x + w / 2, y + h - 8, sub, ha="center", va="top", color="white", fontsize=8.2, linespacing=1.35)
def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14, color=GREY, lw=1.6))
box(1, 36, 20, 16, "Manager", "asks a question in\nplain English\n\"Why did sales decline\nin the South region?\"", GREY)
box(27, 36, 24, 16, "1  Understand", "LLM router (optional)\nor rule-based parser\n-> intent, metric, filters,\ntime period (validated)", NAVY)
box(57, 36, 24, 16, "2  Analyse (pandas)", "KPIs, driver breakdown,\nprice vs volume effect,\nhotspot drill-down\n(ALL numbers computed here)", TEAL)
box(87, 36, 22, 16, "3  Retrieve (RAG)", "TF-IDF search over\nbusiness events log +\nmetric glossary\n(region & time boost)", TEAL)
box(57, 14, 24, 16, "4  Explain", "LLM writer (optional) or\ntemplate writer, using ONLY\nthe verified facts.\nSuggests actions.", NAVY)
box(27, 14, 24, 16, "5  Audit", "every figure in the text is\nchecked against the facts;\nunverified numbers are\nflagged", RED)
box(1, 14, 20, 16, "Output", "answer + dashboard +\nevidence tables +\nretrieved context\n(human decides)", GREEN)
ax.add_patch(FancyBboxPatch((1, 0.5), 108, 10.5, boxstyle="round,pad=0.4,rounding_size=1.5", fc="#374151", ec="none"))
ax.text(55, 9.3, "Data, knowledge & governance layer", ha="center", va="top", color="white", fontsize=10.5, fontweight="bold")
ax.text(55, 5.2, "sales_data.csv (synthetic, 3,360 rows)  |  business_events.md  |  metric_glossary.md\nNo personal data  |  offline fallback if no API key  |  human makes the final decision", ha="center", va="top", color="white", fontsize=8.2, linespacing=1.4)
arrow(21, 44, 27, 44); arrow(51, 44, 57, 44); arrow(81, 44, 87, 44)
arrow(69, 36, 69, 30); arrow(92, 36, 81, 29)
arrow(57, 22, 51, 22); arrow(27, 22, 21, 22)
plt.tight_layout(); plt.savefig("figures/architecture.png", dpi=170); plt.close()

# ---------- 2/3. South decline evidence from the prototype's own tools
cp = Copilot(); a = cp.ask("Why did sales decline in the South region?")
d = cp.df.groupby(["month", "region"])["sales"].sum().reset_index()
fig, ax = plt.subplots(figsize=(8.5, 3.8))
for r, g in d.groupby("region"):
    ax.plot(g["month"], g["sales"] / 1e7, label=r, lw=2.6 if r == "South" else 1.4, color=RED if r == "South" else None, alpha=1 if r == "South" else .7)
ax.axvline(__import__("pandas").Timestamp("2026-07-01"), color=GREY, ls="--", lw=1); ax.text(__import__("pandas").Timestamp("2026-07-05"), ax.get_ylim()[1] * .985, "Jul 2026:\ndiscount cut +\nVoltMart price war", fontsize=8, va="top", color=GREY)
ax.set_ylabel("Monthly sales (Rs crore)"); ax.set_title("Monthly sales by region: the South breaks away from Jul 2026", fontsize=11, fontweight="bold")
ax.legend(frameon=False, ncol=4, loc="lower left"); ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout(); plt.savefig("figures/trend_by_region.png", dpi=170); plt.close()

fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3))
for ax, dim in zip(axes, ["category", "city", "channel"]):
    t = a.tables[f"by_{dim}"].sort_values("change")
    ax.barh(t[dim], t["change"] / 1e5, color=[RED if v < 0 else GREEN for v in t["change"]])
    ax.set_title(f"By {dim}", fontsize=10, fontweight="bold"); ax.set_xlabel("Change in sales (Rs lakh)", fontsize=8.5)
    ax.spines[["top", "right"]].set_visible(False); ax.tick_params(labelsize=8.5)
fig.suptitle("South region, Jul-Sep 2026 vs Apr-Jun 2026: where the sales change came from", fontsize=11, fontweight="bold")
plt.tight_layout(); plt.savefig("figures/south_drivers.png", dpi=170); plt.close()
print("figures done")
