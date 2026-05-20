"""
V35 Diagnostic: Analyze VAS.AX signals at exact BULL entry months (2016, 2019)
vs successful years (2017, 2021). Goal: find ONE filter that blocks 2016/2019
without blocking 2017/2021.
"""
import pickle, warnings
import pandas as pd
import numpy as np
warnings.filterwarnings("ignore")

# Load cached data from V32
CACHE = "sim_v32_data_cache.pkl"
print(f"Loading {CACHE}...")
with open(CACHE, "rb") as f:
    cache = pickle.load(f)

vas = cache.get("vas_df")
if vas is None:
    print("vas_df not in cache!", list(cache.keys())); exit()

v = vas.copy()
v["EMA50"]   = v["Close"].ewm(span=50, adjust=False).mean()
v["EMA200"]  = v["Close"].ewm(span=200, adjust=False).mean()
v["SMA200"]  = v["Close"].rolling(200).mean()
v["SMA50"]   = v["Close"].rolling(50).mean()
v["MOM20"]   = v["Close"].pct_change(20)
v["MOM5"]    = v["Close"].pct_change(5)
v["HIGH52W"] = v["Close"].rolling(252).max()
v["DIST_52W"]= (v["Close"] - v["HIGH52W"]) / v["HIGH52W"]  # negative = below peak
v["HIGH20d"] = v["Close"].rolling(20).max()
v["DIST_20d"]= (v["Close"] - v["HIGH20d"]) / v["HIGH20d"]  # drawdown from 20d high
v["ROC3M"]   = v["Close"].pct_change(63)  # 3-month rate of change
v["SMA20"]   = v["Close"].rolling(20).mean()
v["SLOPE"]   = v["EMA50"].diff()  # EMA50 slope

# VAS monthly returns (for gate_3m / gate_6m)
vm = v["Close"].resample("ME").last().pct_change()

def monthly_ret(date, n):
    """Last n monthly returns before date"""
    dt = pd.Timestamp(date)
    hist = vm[vm.index < dt].tail(n)
    if len(hist) < n:
        return None
    return list(hist)

# Test dates: first trading day of each BULL month
# From V32 output we know: 2016 had trades (Jan/Feb 2016 entries)
# 2019 had 2 trades, 2017 had 2 trades, 2021 had 6 trades

test_dates = {
    # Failed entries
    "2016-Jan [FAIL]":   "2016-01-04",
    "2016-Feb [FAIL]":   "2016-02-01",
    "2019-Mar [FAIL?]":  "2019-03-01",
    "2019-May [FAIL?]":  "2019-05-01",
    # Successful entries
    "2017-Oct [WIN]":    "2017-10-02",
    "2017-Nov [WIN]":    "2017-11-01",
    "2021-Mar [WIN]":    "2021-03-01",
    "2021-Apr [WIN]":    "2021-04-01",
    "2021-May [WIN]":    "2021-05-03",
    "2021-Jun [WIN]":    "2021-06-01",
    "2021-Jul [WIN]":    "2021-07-01",
    "2021-Aug [WIN]":    "2021-08-02",
    # V32 extra failures
    "2012-Apr [WIN->blocked_v33]": "2012-04-02",
    "2024-Mar [sub10]":  "2024-03-01",
    "2024-Apr [sub10]":  "2024-04-01",
}

print(f"\n{'Label':<30} {'Close':>7} {'MOM20':>7} {'MOM5':>6} {'DIST52W':>8} {'DIST20d':>8} {'ROC3M':>7} {'gate3m':>8} {'gate6m':>8}")
print("-"*105)

for label, date_str in test_dates.items():
    try:
        dt = pd.Timestamp(date_str)
        row = v.loc[v.index <= dt].iloc[-1]
        close = row["Close"]
        mom20 = row["MOM20"]
        mom5  = row["MOM5"]
        d52   = row["DIST_52W"]
        d20   = row["DIST_20d"]
        roc3  = row["ROC3M"]
        m3    = monthly_ret(dt, 3)
        m6    = monthly_ret(dt, 6)
        g3    = "PASS" if m3 and all(r > 0 for r in m3) else "FAIL"
        g6    = "PASS" if m6 and all(r > 0 for r in m6) else "FAIL"
        print(f"{label:<30} {close:>7.2f} {mom20*100:>6.1f}% {mom5*100:>5.1f}% {d52*100:>7.1f}% {d20*100:>7.1f}% {roc3*100:>6.1f}% {g3:>8} {g6:>8}")

        print(f"  {'':>30} months3={[f'{r*100:+.1f}%' for r in m3] if m3 else 'N/A'}")
        if m6:
            print(f"  {'':>30} months6={[f'{r*100:+.1f}%' for r in m6]}")
    except Exception as e:
        print(f"{label:<30} ERROR: {e}")

# Summary: what would block 2016/2019 but NOT 2017/2021?
print("\n\n=== DECISION MATRIX ===")
print("We need a filter that is FAIL for 2016/2019 and PASS for 2017/2021\n")

filters = {
    "MOM20 > 0": lambda r: r["MOM20"] > 0,
    "MOM5  > 0": lambda r: r["MOM5"] > 0,
    "DIST52W > -10%": lambda r: r["DIST_52W"] > -0.10,
    "DIST52W > -8%":  lambda r: r["DIST_52W"] > -0.08,
    "DIST52W > -12%": lambda r: r["DIST_52W"] > -0.12,
    "DIST20d > -3%":  lambda r: r["DIST_20d"] > -0.03,
    "ROC3M > 0":      lambda r: r["ROC3M"] > 0,
    "ROC3M > -5%":    lambda r: r["ROC3M"] > -0.05,
}

focus = {
    "2016-Jan": "2016-01-04", "2016-Feb": "2016-02-01",
    "2019-Mar": "2019-03-01", "2019-May": "2019-05-01",
    "2017-Oct": "2017-10-02", "2017-Nov": "2017-11-01",
    "2021-Mar": "2021-03-01", "2021-Jun": "2021-06-01",
}

print(f"{'Filter':<22}", end="")
for lbl in focus: print(f" {lbl:>10}", end="")
print()
print("-"*120)

for fname, ffunc in filters.items():
    print(f"{fname:<22}", end="")
    for lbl, dstr in focus.items():
        try:
            dt = pd.Timestamp(dstr)
            row = v.loc[v.index <= dt].iloc[-1]
            result = "PASS" if ffunc(row) else "FAIL"
        except:
            result = "ERR"
        print(f" {result:>10}", end="")
    print()
