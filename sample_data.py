"""Creates a deliberately messy sample file so you can test normalization: python sample_data.py"""
import numpy as np, pandas as pd
rng = np.random.default_rng(7)
n = 600
cities = ["Mumbai", "Delhi", "Bengaluru", "Pune", "Hyderabad"]
rests = ["Spice Hub", "Burger Barn", "Pizza Point", "Green Bowl", "Biryani House"]
df = pd.DataFrame({
    "Ord Dt": pd.date_range("2025-01-01", periods=n, freq="14h").strftime("%d/%m/%Y"),
    "Restaurant Name": rng.choice(rests + [r.upper() for r in rests[:2]], n),
    "City ": rng.choice(cities, n),
    "Order Value (INR)": ["₹" + f"{v:,}" for v in rng.integers(150, 1800, n)],
    "Delivery Fee": rng.choice([0, 20, 35, 49], n),
    "Customer Rating": np.where(rng.random(n) < 0.15, np.nan, rng.integers(1, 6, n)),
})
df = pd.concat([df, df.sample(15, random_state=1)])
df.to_csv("sample_orders.csv", index=False)
print("Wrote sample_orders.csv with", len(df), "rows")
