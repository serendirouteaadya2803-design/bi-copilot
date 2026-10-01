"""
Generates a SYNTHETIC retail sales dataset (Jan 2025 - Sep 2026) for the BI Copilot demo.

A deliberate "story" is planted so the copilot has a real answer to discover:
  From Jul 2026, South-region Electronics & Appliances sales fall because
  (1) promotional discounts were cut from ~10% to ~2%, and
  (2) the ONLINE channel in Chennai and Bengaluru lost volume (competitor price war).
Other regions keep growing normally, with a few smaller, unrelated events.
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

CITIES = {
    "North": ["Delhi", "Chandigarh", "Jaipur", "Lucknow"],
    "South": ["Chennai", "Bengaluru", "Hyderabad", "Kochi"],
    "East":  ["Kolkata", "Bhubaneswar", "Patna", "Guwahati"],
    "West":  ["Mumbai", "Pune", "Ahmedabad", "Surat"],
}
CITY_SIZE = {  # relative market size
    "Delhi": 1.5, "Chandigarh": 0.7, "Jaipur": 0.8, "Lucknow": 0.8,
    "Chennai": 1.3, "Bengaluru": 1.5, "Hyderabad": 1.2, "Kochi": 0.7,
    "Kolkata": 1.2, "Bhubaneswar": 0.6, "Patna": 0.6, "Guwahati": 0.5,
    "Mumbai": 1.7, "Pune": 1.1, "Ahmedabad": 1.0, "Surat": 0.8,
}
# category: (base monthly units per city-channel at size 1.0, base list price in Rs, base discount %)
CATEGORIES = {
    "Electronics": (60, 18000, 0.10),
    "Appliances":  (45, 22000, 0.10),
    "Apparel":     (260, 1800, 0.12),
    "Grocery":     (900, 450, 0.03),
    "Furniture":   (30, 14000, 0.08),
}
CHANNEL_SHARE = {"Online": 0.45, "Retail Store": 0.55}
MARGIN = {"Electronics": 0.14, "Appliances": 0.16, "Apparel": 0.30, "Grocery": 0.08, "Furniture": 0.24}
SEASONAL = {1: 0.95, 2: 0.92, 3: 1.00, 4: 0.98, 5: 1.00, 6: 0.97, 7: 0.98, 8: 1.02, 9: 1.05, 10: 1.25, 11: 1.20, 12: 1.10}

months = pd.period_range("2025-01", "2026-09", freq="M")
rows = []
for i, m in enumerate(months):
    growth = 1 + 0.008 * i  # steady ~0.8% monthly organic growth
    for region, cities in CITIES.items():
        for city in cities:
            for cat, (base_u, price, base_disc) in CATEGORIES.items():
                for ch, share in CHANNEL_SHARE.items():
                    units = base_u * CITY_SIZE[city] * share * 2 * SEASONAL[m.month] * growth
                    disc = base_disc
                    list_price = price * (1 + 0.002 * i)

                    # ---- planted story: South Electronics & Appliances from Jul 2026 ----
                    if region == "South" and cat in ("Electronics", "Appliances") and m >= pd.Period("2026-07"):
                        disc = 0.02                       # discount policy cut
                        units *= 0.82                      # general price-sensitivity loss
                        if ch == "Online" and city in ("Chennai", "Bengaluru"):
                            units *= 0.55                  # competitor price war hits online hardest
                    # ---- smaller unrelated events (so not everything is South) ----
                    if region == "North" and m in (pd.Period("2025-10"), pd.Period("2025-11")) and cat in ("Apparel", "Electronics"):
                        units *= 1.18                      # Diwali campaign
                    if region == "West" and cat == "Apparel" and m == pd.Period("2026-03"):
                        units *= 1.25                      # West apparel promo
                    if city == "Bhubaneswar" and m >= pd.Period("2026-01"):
                        units *= 1.30                      # new store opened

                    units = max(1, int(round(units * rng.normal(1, 0.04))))
                    net_price = list_price * (1 - disc) * rng.normal(1, 0.01)
                    sales = units * net_price
                    # discount cut protects margin slightly (more margin per unit)
                    profit = sales * (MARGIN[cat] + (base_disc - disc) * 0.6) * rng.normal(1, 0.03)
                    rows.append((m.to_timestamp().strftime("%Y-%m-01"), region, city, cat, ch,
                                 units, round(list_price, 2), round(disc * 100, 1), round(sales, 2), round(profit, 2)))

df = pd.DataFrame(rows, columns=["month", "region", "city", "category", "channel",
                                 "units", "list_price", "discount_pct", "sales", "profit"])
df.to_csv("sales_data.csv", index=False)
print(df.shape)
print(df.groupby("region")["sales"].sum().round(0))
