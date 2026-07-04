"""
Generates a deterministic synthetic OHLCV dataset (10,000 rows) for local
testing of run.py. The Google Sheet linked in the task spec is private and
could not be fetched, so this script produces a stand-in dataset with the
same shape/columns. Replace data.csv with the real file at any time --
run.py does not depend on how data.csv was produced.
"""
import numpy as np
import pandas as pd
import datetime as dt

np.random.seed(42)

N = 10000
start = dt.datetime(2020, 1, 1)
timestamps = [start + dt.timedelta(minutes=i) for i in range(N)]

# Random-walk close price, realistic-ish OHLCV around it
price = 100.0
closes = []
for _ in range(N):
    price += np.random.normal(0, 0.15)
    price = max(price, 1.0)
    closes.append(price)
closes = np.array(closes)

opens = closes + np.random.normal(0, 0.05, N)
highs = np.maximum(opens, closes) + np.abs(np.random.normal(0, 0.08, N))
lows = np.minimum(opens, closes) - np.abs(np.random.normal(0, 0.08, N))
volumes = np.random.randint(100, 10000, N)

df = pd.DataFrame({
    "timestamp": timestamps,
    "open": opens.round(4),
    "high": highs.round(4),
    "low": lows.round(4),
    "close": closes.round(4),
    "volume": volumes,
})

df.to_csv("data.csv", index=False)
print(f"Wrote data.csv with {len(df)} rows")
