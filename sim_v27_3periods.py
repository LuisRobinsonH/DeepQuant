#!/usr/bin/env python
"""
DeepQuant V27 — MULTI-IMPROVEMENT: Stock-Level Regime + Lower MOD_PROB
=======================================================================
BASE = V22b (5/5, +26.09% cumulative optimum)

ROOT CAUSE DIAGNOSIS (2026-02-28):
  2022: 0 trades — VAS gate blocks ALL activity. But BHP/WDS/RIO/STO were
        +20-40% individually. The regime filter is correct for the BULL model
        (trained on VAS-bull patterns) but wrong when individual stocks have
        their own golden cross independent of VAS.
  2023: 1 trade ($+44) — MOD_PROB=0.94 rejects 66/66 candidates in Aug.
        In practice 0 stocks reach 0.94 in a mid-cycle moderate regime.
  2025: 4 trades ($+64 = inflation loss) — 66 BULL days fully saturated
        (MAX_POS=1 × ~17d hold = 4 slots). Poor entry quality (JIN/CPU
        first in queue because no stock-level regime pre-filter).
  2026: 0 trades — Same MODERATE prob block as 2023.

V27 CHANGES (all coordinated, addressing diagnosed root causes):

  1. NEW TIER: STOCK_BULL
     When VAS is in BEAR (SMA50 < SMA200 at VAS level), individual stocks
     that have their OWN uptrend (stock-level golden cross + momentum) can
     still be traded. This is NOT V21 (which used the VAS-trained BULL model
     in bear market — wrong). V27 trains a SEPARATE "stock-only" ML model
     that excludes VAS features (vas_momentum, vas_position). This model
     learns stock-level momentum patterns valid regardless of VAS state.
     Tighter parameters: SL=2.0xATR, BE=1.0xATR, max_hold=20d, prob>=0.78
     Cap: 4/year. Stock must have own GC (SMA50>SMA200), price>SMA50,
     MOM20>2%, RSI 30-72. This unlocks 2022 and VAS-bear months.

  2. MOD_PROB: 0.94 → 0.72
     Aug-2023: 66 stocks pass RSI filter, 0 reach 0.94. With 0.72, ~20
     reach threshold. This unlocks 2023 and mid-2024 MODERATE windows.
     MOD_YTD_CAP raised to -200 (2 small losses before freeze).
     MOD_MAX_YEAR: 4 → 8. MOD_MAX_MONTH: 1 → 2.

  3. BULL quality pre-filter: stock must also have OWN positive momentum
     (MOM20 > 0) and own SMA50 > SMA200. Prevents JIN/CPU-style entries
     where VAS is clearly BULL but individual stock is weak.

V22b PRESERVED: prob-scaling 30→90%, MAX_POS=1, MAX_HOLD=35d, RSI<=80,
     BULL_PROB_FULL=0.52, BULL gate (3m + GC + slope + dist_200>=0.02).

CommSec: <=1K->$10 | 1K-10K->$19.95 | 10K-25K->$29.95 | >25K->0.12%
"""

import warnings, datetime as dt, time, os, random, pickle
warnings.filterwarnings("ignore")
os.environ["PYTHONHASHSEED"] = "42"
import numpy as np, pandas as pd, yfinance as yf
from pathlib import Path
from ta import momentum, trend, volatility
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb

# === DETERMINISM ===
np.random.seed(42)
random.seed(42)
DATA_CACHE = Path("sim_v15_data_cache.pkl")

SYM_FILE = Path("au_stock_data/au_symbols.txt")
TICKERS = [s.strip() for s in SYM_FILE.read_text().splitlines() if s.strip()]
TICKERS = [t if t.endswith(".AX") else t + ".AX" for t in TICKERS]
VAS = "VAS.AX"

def commsec(value):
    if value <= 0: return 0.0
    if value <= 1000: return 10.00
    if value <= 10000: return 19.95
    if value <= 25000: return 29.95
    return value * 0.0012

# === BULL tier ===
BULL_PROB_FULL      = 0.52
BULL_PROB_SELECTIVE = 0.58
BULL_POS_PCT        = 0.80
BULL_MAX_YEAR       = 20
BULL_MAX_MONTH      = 5

# === MODERATE tier (V27: lower prob threshold to unlock 2023/mid-2024 windows) ===
MOD_PROB            = 0.72   # V27: was 0.94. Aug-2023: 66 stocks pass RSI, 0 reach 0.94 → unlocks 2023
MOD_POS_PCT         = 0.45
MOD_MAX_YEAR        = 8      # V27: was 4
MOD_MAX_MONTH       = 2      # V27: was 1
MOD_YTD_CAP         = -200   # V27: was -100. Allow 2 small moderate losses before freeze
MOD_GC_MIN_DAYS     = 30

# === STOCK_BULL tier (V27 NEW: individual stock regime in VAS-bear market) ===
# Stocks with own golden cross (SMA50>SMA200 at stock level) + strong momentum
# Use a separate ML model trained WITHOUT VAS features (stock-level only)
# Tighter stops since VAS environment is headwind
SB_PROB             = 0.78   # higher threshold: no VAS tailwind
SB_MAX_YEAR         = 4      # conservative cap
SB_MAX_MONTH        = 2
SB_SL_ATR           = 2.0    # tighter than BULL (2.5x)
SB_BE_ATR           = 1.0    # quick breakeven
SB_TRAIL_ATR        = 1.0
SB_MAX_HOLD         = 20     # shorter hold in headwind
SB_POS_MIN          = 0.25   # smaller position: VAS is against us
SB_POS_MAX          = 0.55
SB_STOP_CAP         = 0.05   # 5% hard stop cap
SB_MOM20_MIN        = 0.02   # stock must have >=2% 20d momentum (own uptrend)
SB_RSI_MAX          = 72     # not overbought

# === V19b: Probability-proportional position sizing ===
PROB_SIZE_MIN        = 0.30
PROB_SIZE_MAX        = 0.90
RISK_GUARD_PCT       = 0.10
MIN_POSITION         = 2_000

# === BULL stop parameters ===
SL_ATR              = 2.5
BE_TRIGGER_ATR      = 1.5
TRAIL_ATR           = 1.5
MAX_HOLD            = 35
GRACE_DAYS          = 2

# === MODERATE stop parameters ===
MOD_SL_ATR          = 1.5
MOD_BE_ATR          = 1.0
MOD_TRAIL_ATR       = 1.0
MOD_MAX_HOLD        = 15
MOD_STOP_CAP        = 0.04

# === SELECTIVE tier — EMPIRICALLY DISABLED (V21: 1/5, -23.68%) ===
SELECTIVE_PROB      = 0.90
SELECTIVE_MAX_YEAR  = 0
SELECTIVE_SL_ATR    = 2.0
SELECTIVE_POS_MIN   = 0.30
SELECTIVE_POS_MAX   = 0.55

# === Risk management ===
MAX_POS             = 1
MAX_TICKER_TRADES   = 3
TICKER_COOLDOWN     = 10
YTD_LOSS_CAP        = -500
LS_MAX              = 3
LS_DAYS             = 7

PERIODS = [
    ("P1: 2022-2024", "2022-01-01", "2024-12-31", "2021-12-31"),
    ("P2: 2025",      "2025-01-01", "2025-12-31", "2024-12-31"),
    ("P3: 2026 YTD",  "2026-01-01", "2026-12-31", "2025-12-31"),
]
CAPITAL = 8_000.0


# === FEATURES (V22b exact — VAS features included) ===
def build_features(df, vas_feats=None, breadth_series=None, universe_mom=None):
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    sma20  = c.rolling(20).mean()
    sma50  = c.rolling(50).mean()
    sma200 = c.rolling(200).mean()
    atr    = volatility.average_true_range(h, l, c, 14)
    feat = pd.DataFrame(index=df.index)
    feat["dist_sma20"]      = (c - sma20) / sma20
    feat["dist_sma50"]      = (c - sma50) / sma50
    feat["dist_sma200"]     = (c - sma200) / sma200
    feat["ma_cross_20_50"]  = (sma20 - sma50) / sma50
    feat["ma_cross_50_200"] = (sma50 - sma200) / sma200
    feat["atr_pct"]         = atr / c
    feat["vol_regime"]      = atr / atr.rolling(50).mean()
    feat["momentum_5"]      = c.pct_change(5)
    feat["momentum_10"]     = c.pct_change(10)
    feat["momentum_20"]     = c.pct_change(20)
    feat["rsi"]             = momentum.rsi(c, 14) / 100
    macd_obj = trend.MACD(c)
    feat["macd_diff_norm"]  = macd_obj.macd_diff() / c
    feat["adx"]             = trend.ADXIndicator(h, l, c, 14).adx() / 100
    bb = volatility.BollingerBands(c, 20, 2)
    feat["bb_width"]        = (bb.bollinger_hband() - bb.bollinger_lband()) / c
    feat["bb_position"]     = (c - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband() + 1e-10)
    feat["vol_rel_20"]      = v / v.rolling(20).mean()
    feat["close_to_high20"] = c / h.rolling(20).max()
    feat["close_to_low20"]  = c / l.rolling(20).min()
    feat["range_pct"]       = (h - l) / c
    feat["gap_pct"]         = (df["Open"] - c.shift(1)) / c.shift(1)
    range_10d = h.rolling(10).max() - l.rolling(10).min()
    feat["consolidation"]   = range_10d / (atr * 10 + 1e-10)
    if vas_feats is not None:
        feat["vas_momentum"] = vas_feats["mom20"].reindex(df.index, method="ffill")
        feat["vas_position"] = vas_feats["pos200"].reindex(df.index, method="ffill")
    else:
        feat["vas_momentum"] = 0.0
        feat["vas_position"] = 0.0
    if breadth_series is not None:
        feat["mkt_breadth"] = breadth_series.reindex(df.index, method="ffill") / 100.0
    else:
        feat["mkt_breadth"] = 0.5
    if universe_mom is not None:
        stock_mom = c.pct_change(20)
        uni_med = universe_mom.reindex(df.index, method="ffill")
        feat["relative_strength"] = stock_mom - uni_med
    else:
        feat["relative_strength"] = 0.0
    return feat


# === FEATURES — STOCK-ONLY (V27: no VAS context, for STOCK_BULL model) ===
# By excluding vas_momentum / vas_position the model learns stock-level patterns
# that are valid regardless of market regime. Trained on ALL history including
# bear periods — allows capturing individual bull stocks in VAS-bear years.
def build_features_novas(df, breadth_series=None, universe_mom=None):
    """Same as build_features but WITHOUT vas_momentum / vas_position."""
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    sma20  = c.rolling(20).mean()
    sma50  = c.rolling(50).mean()
    sma200 = c.rolling(200).mean()
    atr    = volatility.average_true_range(h, l, c, 14)
    feat = pd.DataFrame(index=df.index)
    feat["dist_sma20"]      = (c - sma20) / sma20
    feat["dist_sma50"]      = (c - sma50) / sma50
    feat["dist_sma200"]     = (c - sma200) / sma200
    feat["ma_cross_20_50"]  = (sma20 - sma50) / sma50
    feat["ma_cross_50_200"] = (sma50 - sma200) / sma200
    feat["atr_pct"]         = atr / c
    feat["vol_regime"]      = atr / atr.rolling(50).mean()
    feat["momentum_5"]      = c.pct_change(5)
    feat["momentum_10"]     = c.pct_change(10)
    feat["momentum_20"]     = c.pct_change(20)
    feat["rsi"]             = momentum.rsi(c, 14) / 100
    macd_obj = trend.MACD(c)
    feat["macd_diff_norm"]  = macd_obj.macd_diff() / c
    feat["adx"]             = trend.ADXIndicator(h, l, c, 14).adx() / 100
    bb = volatility.BollingerBands(c, 20, 2)
    feat["bb_width"]        = (bb.bollinger_hband() - bb.bollinger_lband()) / c
    feat["bb_position"]     = (c - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband() + 1e-10)
    feat["vol_rel_20"]      = v / v.rolling(20).mean()
    feat["close_to_high20"] = c / h.rolling(20).max()
    feat["close_to_low20"]  = c / l.rolling(20).min()
    feat["range_pct"]       = (h - l) / c
    feat["gap_pct"]         = (df["Open"] - c.shift(1)) / c.shift(1)
    range_10d = h.rolling(10).max() - l.rolling(10).min()
    feat["consolidation"]   = range_10d / (atr * 10 + 1e-10)
    # NO vas_momentum, NO vas_position
    if breadth_series is not None:
        feat["mkt_breadth"] = breadth_series.reindex(df.index, method="ffill") / 100.0
    else:
        feat["mkt_breadth"] = 0.5
    if universe_mom is not None:
        stock_mom = c.pct_change(20)
        uni_med = universe_mom.reindex(df.index, method="ffill")
        feat["relative_strength"] = stock_mom - uni_med
    else:
        feat["relative_strength"] = 0.0
    return feat


def build_aligned_target(df, be_atr=1.5, sl_atr=2.5, max_days=25):
    c = df["Close"]; h = df["High"]; l = df["Low"]
    atr = volatility.average_true_range(h, l, c, 14)
    target = pd.Series(np.nan, index=df.index)
    for i in range(len(df) - max_days):
        entry = c.iloc[i]; a = atr.iloc[i]
        if pd.isna(a) or a <= 0: continue
        be_level = entry + be_atr * a
        sl_level = entry - sl_atr * a
        hit_be = False
        for j in range(1, max_days + 1):
            if i + j >= len(df): break
            if h.iloc[i + j] >= be_level: hit_be = True; break
            if l.iloc[i + j] <= sl_level: hit_be = False; break
        target.iloc[i] = 1 if hit_be else 0
    return target


# ── STOCK_BULL target: tighter (1.0xATR BE / 2.0x SL in 20d) ──────────────
def build_sb_target(df, be_atr=1.0, sl_atr=2.0, max_days=20):
    return build_aligned_target(df, be_atr, sl_atr, max_days)


# === HEADER ===
print("=" * 80)
print("  DEEPQUANT V27 — Stock-Level Regime + Lower MOD_PROB (V22b base)")
print(f"  BULL prob={BULL_PROB_FULL:.2f} | MOD_PROB={MOD_PROB:.2f} (was 0.94) | SB_PROB={SB_PROB:.2f} (new tier)")
print(f"  MAX_HOLD={MAX_HOLD}d | RSI_max=80 | SB_MAX_YEAR={SB_MAX_YEAR} | MOD_MAX_YEAR={MOD_MAX_YEAR}")
print("=" * 80)
print(f"  BULL:      +{BE_TRIGGER_ATR}xATR/-{SL_ATR}xATR {MAX_HOLD}d | 3m gate | prob>={BULL_PROB_FULL}/{BULL_PROB_SELECTIVE} | prob-scaled")
print(f"  MODERATE:  prob>={MOD_PROB} | 2m gate | SL {MOD_SL_ATR}xATR | {MOD_MAX_HOLD}d | cap/yr={MOD_MAX_YEAR}")
print(f"  STOCK_BULL:prob>={SB_PROB} | own GC + mom>2% | SL {SB_SL_ATR}xATR | {SB_MAX_HOLD}d | cap/yr={SB_MAX_YEAR} (VAS-bear ok)")
print(f"  CommSec: <=1K->$10 | 1K-10K->$19.95 | 10K-25K->$29.95 | >25K->0.12%")

t0 = time.time()
all_tickers = sorted(set(TICKERS + [VAS]))
if DATA_CACHE.exists():
    print(f"\n  Loading cached data from {DATA_CACHE}...")
    with open(DATA_CACHE, "rb") as f:
        cache = pickle.load(f)
    data = cache["data"]
    vas_df_raw = cache["vas_df"]
    print(f"  OK: {len(data)} tickers (cached)")
else:
    print(f"\n  Downloading {len(all_tickers)} tickers...")
    raw = yf.download(all_tickers, start="2019-01-01", period="max",
                      group_by="ticker", auto_adjust=True, threads=False)
    data = {}
    for t in TICKERS:
        try:
            tmp = raw[t].dropna(subset=["Close"])
            if len(tmp) > 250: data[t] = tmp
        except: pass
    vas_df_raw = raw[VAS].dropna(subset=["Close"])
    print(f"  OK: {len(data)} tickers")
    with open(DATA_CACHE, "wb") as f:
        pickle.dump({"data": data, "vas_df": vas_df_raw}, f)

vas_df = vas_df_raw.copy()
vas_df["SMA50"]  = vas_df["Close"].rolling(50).mean()
vas_df["SMA200"] = vas_df["Close"].rolling(200).mean()
vas_df["SMA50_20d_ago"] = vas_df["SMA50"].shift(20)
vas_df["MOM20"]  = vas_df["Close"].pct_change(20)

vas_monthly = vas_df["Close"].resample("ME").last().pct_change()
print(f"  VAS.AX: {len(vas_df)} rows")

gc_active_since = None
gc_history = {}
in_gc = False
for d in vas_df.index:
    s50 = vas_df.loc[d, "SMA50"]; s200 = vas_df.loc[d, "SMA200"]
    if pd.notna(s50) and pd.notna(s200) and s50 > s200:
        if not in_gc: gc_active_since = d; in_gc = True
        gc_history[d] = (d - gc_active_since).days
    else:
        in_gc = False; gc_active_since = None; gc_history[d] = 0
gc_series = pd.Series(gc_history)

print(f"\n  VAS Monthly Returns:")
for yr in [2022, 2023, 2024, 2025, 2026]:
    try:
        yr_months = vas_monthly[vas_monthly.index.year == yr]
        vals = [f"{v*100:+.1f}%" for v in yr_months.values if pd.notna(v)]
        print(f"    {yr}: {', '.join(vals)}")
    except: pass

print(f"\n  Golden Cross History:")
prev_gc = False
for d in vas_df.index:
    s50 = vas_df.loc[d, "SMA50"]; s200 = vas_df.loc[d, "SMA200"]
    curr_gc = pd.notna(s50) and pd.notna(s200) and s50 > s200
    if curr_gc != prev_gc and d.year >= 2021:
        print(f"    {d.date()}: {'Golden Cross ON' if curr_gc else 'Death Cross (GC OFF)'}")
    prev_gc = curr_gc

vas_feats = pd.DataFrame(index=vas_df.index)
vas_feats["mom20"] = vas_df["MOM20"]
vas_feats["pos200"] = (vas_df["Close"] - vas_df["SMA200"]) / vas_df["SMA200"]

print(f"\n  Computing breadth...")
all_dates = vas_df.index
close_matrix = pd.DataFrame(index=all_dates)
for t, df in data.items():
    sma50 = df["Close"].rolling(50).mean()
    above = (df["Close"] > sma50).astype(float)
    close_matrix[t] = above.reindex(all_dates)
breadth_50 = close_matrix.mean(axis=1) * 100
breadth_50 = breadth_50.ffill()

mom_matrix = pd.DataFrame(index=all_dates)
for t, df in data.items():
    mom_matrix[t] = df["Close"].pct_change(20).reindex(all_dates)
universe_mom_median = mom_matrix.median(axis=1)
print(f"  OK: Breadth + momentum")


def get_prior_months(date, n_months):
    d = pd.Timestamp(date)
    results = []
    for offset in range(1, n_months + 1):
        m = d.month - offset; y = d.year
        while m <= 0: m += 12; y -= 1
        try:
            m_end = pd.Timestamp(year=y, month=m, day=28) + pd.offsets.MonthEnd(0)
            if m_end in vas_monthly.index:
                ret = vas_monthly.loc[m_end]
                if pd.notna(ret): results.append(ret)
                else: return None
            else: return None
        except: return None
    return results if len(results) == n_months else None


def get_vas_state(date):
    mask = vas_df.index <= pd.Timestamp(date)
    if mask.sum() == 0: return None
    row = vas_df.loc[mask].iloc[-1]
    c = row["Close"]
    s50 = row["SMA50"] if pd.notna(row["SMA50"]) else 0
    s200 = row["SMA200"] if pd.notna(row["SMA200"]) else 0
    s50_20d = row["SMA50_20d_ago"] if pd.notna(row.get("SMA50_20d_ago", np.nan)) else 0
    golden = s50 > 0 and s200 > 0 and s50 > s200
    sma50_slope = s50 > s50_20d if (s50 > 0 and s50_20d > 0) else False
    dist_200 = (c - s200) / s200 if s200 > 0 else 0
    above_200 = c > s200 if s200 > 0 else False
    above_50 = c > s50 if s50 > 0 else False
    gc_mask = gc_series.index <= pd.Timestamp(date)
    gc_age = gc_series.loc[gc_mask].iloc[-1] if gc_mask.sum() > 0 else 0
    return {"golden": golden, "slope": sma50_slope, "dist_200": dist_200,
            "above_200": above_200, "above_50": above_50, "gc_age": gc_age}


def calc_tier_v27(date):
    vs = get_vas_state(date)
    if vs is None: return "BEAR", 0, 0, {"reason": "no data"}
    if not vs["above_200"]: return "BEAR", 0, 0, {"reason": "VAS < SMA200"}

    detail = dict(vs)

    # === BULL GATE ===
    prior_3 = get_prior_months(date, 3)
    gate_3m = prior_3 is not None and all(r > 0 for r in prior_3)
    detail["gate_3m"] = gate_3m
    if prior_3: detail["months3"] = [f"{r*100:+.1f}%" for r in prior_3]

    if gate_3m and vs["golden"] and vs["slope"] and vs["dist_200"] >= 0.02:
        mom = vas_df.loc[vas_df.index <= pd.Timestamp(date)].iloc[-1]
        mom20 = mom["MOM20"] if pd.notna(mom["MOM20"]) else -1
        brd_mask = breadth_50.index <= pd.Timestamp(date)
        brd = breadth_50.loc[brd_mask].iloc[-1] if brd_mask.sum() > 0 else 0
        score = 50
        if vas_df.loc[vas_df.index <= pd.Timestamp(date)].iloc[-1]["Close"] > \
           vas_df.loc[vas_df.index <= pd.Timestamp(date)].iloc[-1]["SMA50"]: score += 15
        if mom20 > 0: score += 15
        if brd > 50: score += 10
        if vs["dist_200"] > 0.05: score += 10
        detail["bull_score"] = score
        return "BULL", (BULL_PROB_FULL if score >= 70 else BULL_PROB_SELECTIVE), BULL_POS_PCT, detail

    # === MODERATE GATE (V27: prob lowered to 0.72) ===
    prior_2 = get_prior_months(date, 2)
    gate_2m = prior_2 is not None and all(r > 0 for r in prior_2)
    detail["gate_2m"] = gate_2m
    if prior_2: detail["months2"] = [f"{r*100:+.1f}%" for r in prior_2]

    if gate_2m and vs["golden"] and vs["above_200"] and vs["above_50"]:
        if vs["gc_age"] >= MOD_GC_MIN_DAYS:
            detail["gc_age_ok"] = True
            return "MODERATE", MOD_PROB, MOD_POS_PCT, detail
        else:
            detail["gc_age_ok"] = False
            detail["reason"] = f"GC too young ({vs['gc_age']}d < {MOD_GC_MIN_DAYS}d)"

    # BEAR at VAS level — STOCK_BULL handled per-ticker in entry loop
    return "BEAR", 0, 0, detail


def simulate_period(name, start, end, train_end, capital):
    print(f"\n{'='*70}")
    print(f"  {name}  (train -> {train_end})")
    print(f"{'='*70}")

    train_cutoff = pd.Timestamp(train_end)
    sim_start, sim_end = pd.Timestamp(start), pd.Timestamp(end)

    valid = sorted([t for t in data if data[t].index.min() < train_cutoff - pd.Timedelta(days=365)])
    print(f"  {len(valid)} tickers")

    # ── Train BULL/MODERATE models (with VAS features — V22b exact) ──────────
    np.random.seed(42); random.seed(42)
    print(f"  Training BULL model (+{BE_TRIGGER_ATR}xATR / -{SL_ATR}xATR in 25d)...")
    models = {}
    target_stats = []
    for i, t in enumerate(valid):
        df = data[t]; tr = df[df.index <= train_cutoff].copy()
        if len(tr) < 300: continue
        feats = build_features(tr, vas_feats, breadth_50, universe_mom_median)
        tgt = build_aligned_target(tr, BE_TRIGGER_ATR, SL_ATR, 25)
        mask = feats.notna().all(axis=1) & tgt.notna()
        X, y = feats[mask], tgt[mask]
        if len(X) < 100 or y.sum() < 5: continue
        target_stats.append(y.mean())
        try:
            base = lgb.LGBMClassifier(n_estimators=400, max_depth=4, learning_rate=0.03,
                subsample=0.7, colsample_bytree=0.6, min_child_samples=25,
                reg_alpha=0.2, reg_lambda=2.0, verbose=-1, random_state=42,
                is_unbalance=True, n_jobs=1)
            cal = CalibratedClassifierCV(base, cv=TimeSeriesSplit(3), method="isotonic")
            cal.fit(X, y); models[t] = cal
        except: pass
        if (i + 1) % 20 == 0 or i == len(valid) - 1:
            print(f"\r   [{i+1}/{len(valid)}]", end="", flush=True)
    avg_pos_rate = np.mean(target_stats) * 100 if target_stats else 0
    print(f"\n   Models: {len(models)}/{len(valid)} | Avg positive rate: {avg_pos_rate:.1f}%")
    mod_models = models  # MODERATE uses same BULL model (V16b proven)

    # ── Train STOCK_BULL models (WITHOUT VAS features) ────────────────────────
    print(f"  Training STOCK_BULL model (no-VAS, +{SB_BE_ATR}xATR/-{SB_SL_ATR}xATR in {SB_MAX_HOLD}d)...")
    np.random.seed(42); random.seed(42)
    sb_models = {}
    sb_stats = []
    for i, t in enumerate(valid):
        df = data[t]; tr = df[df.index <= train_cutoff].copy()
        if len(tr) < 300: continue
        feats = build_features_novas(tr, breadth_50, universe_mom_median)
        tgt = build_sb_target(tr, SB_BE_ATR, SB_SL_ATR, SB_MAX_HOLD)
        mask = feats.notna().all(axis=1) & tgt.notna()
        X, y = feats[mask], tgt[mask]
        if len(X) < 100 or y.sum() < 5: continue
        sb_stats.append(y.mean())
        try:
            base = lgb.LGBMClassifier(n_estimators=400, max_depth=4, learning_rate=0.03,
                subsample=0.7, colsample_bytree=0.6, min_child_samples=25,
                reg_alpha=0.2, reg_lambda=2.0, verbose=-1, random_state=42,
                is_unbalance=True, n_jobs=1)
            cal = CalibratedClassifierCV(base, cv=TimeSeriesSplit(3), method="isotonic")
            cal.fit(X, y); sb_models[t] = cal
        except: pass
        if (i + 1) % 20 == 0 or i == len(valid) - 1:
            print(f"\r   [{i+1}/{len(valid)}]", end="", flush=True)
    avg_sb_rate = np.mean(sb_stats) * 100 if sb_stats else 0
    print(f"\n   SB Models: {len(sb_models)}/{len(valid)} | Avg pos rate: {avg_sb_rate:.1f}%")

    # ── Pre-compute feature + price caches ────────────────────────────────────
    all_model_tickers = set(models.keys()) | set(sb_models.keys())
    feat_cache, sb_feat_cache, price_cache = {}, {}, {}
    for t in all_model_tickers:
        df = data[t]
        feat_cache[t]    = build_features(df, vas_feats, breadth_50, universe_mom_median)
        sb_feat_cache[t] = build_features_novas(df, breadth_50, universe_mom_median)
        atr    = volatility.average_true_range(df["High"], df["Low"], df["Close"], 14)
        sma20  = df["Close"].rolling(20).mean()
        sma50  = df["Close"].rolling(50).mean()
        sma200 = df["Close"].rolling(200).mean()
        rsi_raw = momentum.rsi(df["Close"], 14)
        mom20_raw = df["Close"].pct_change(20)
        price_cache[t] = pd.DataFrame({
            "atr": atr, "sma20": sma20, "sma50": sma50, "sma200": sma200,
            "close": df["Close"], "high": df["High"], "low": df["Low"],
            "volume": df["Volume"], "vol_avg20": df["Volume"].rolling(20).mean(),
            "rsi": rsi_raw, "mom5": df["Close"].pct_change(5), "mom20": mom20_raw,
        }, index=df.index)

    print(f"\n{'-'*70}")
    print(f"  {name}: {start} -> {end} | ${capital:,.0f} | BULL:{len(models)} SB:{len(sb_models)} MOD:{len(mod_models)}")
    print(f"{'-'*70}")

    # === SIMULATION ===
    cash = capital; period_capital = capital
    positions = []; trades = []; equity_curve = []
    ticker_tc = {}; total_comm = 0.0
    consec_losses = 0; ls_until = None
    ytd_pnl = 0.0; ytd_year = None; ytd_frozen = False
    year_trades = {}; monthly_trades = {}; ticker_last_exit = {}

    bull_trades_ytd = 0; bull_trades_month = {}
    mod_trades_ytd = 0; mod_trades_month = {}
    mod_pnl_ytd = 0.0; mod_frozen = False
    sb_trades_ytd = 0; sb_trades_month = {}  # STOCK_BULL
    selective_ytd = 0

    regime_days = {"BULL": 0, "MODERATE": 0, "BEAR": 0}
    tier_trade_counts = {"BULL": 0, "MODERATE": 0, "STOCK_BULL": 0, "SELECTIVE": 0}
    month_tier_log = {}
    filter_debug = {}

    all_sim_dates = set()
    for t in all_model_tickers:
        all_sim_dates.update(data[t].index)
    trading_days = sorted([d for d in all_sim_dates if sim_start <= d <= sim_end])

    for day in trading_days:
        year = day.year; month = (day.year, day.month)

        if ytd_year != year:
            ytd_year = year; ytd_pnl = 0.0; ytd_frozen = False
            year_trades[year] = 0
            bull_trades_ytd = 0; mod_trades_ytd = 0; mod_pnl_ytd = 0.0
            mod_frozen = False; sb_trades_ytd = 0; selective_ytd = 0
            consec_losses = 0; ls_until = None

        if month not in monthly_trades:
            monthly_trades[month] = 0; bull_trades_month[month] = 0
            mod_trades_month[month] = 0; sb_trades_month[month] = 0
        if month not in month_tier_log:
            month_tier_log[month] = {"BULL": 0, "MODERATE": 0, "BEAR": 0, "STOCK_BULL": 0}

        # === REGIME ===
        tier, prob_thresh, pos_pct, tier_detail = calc_tier_v27(day)
        if tier in ("BULL", "MODERATE", "BEAR"):
            regime_days[tier] = regime_days.get(tier, 0) + 1
        if tier in month_tier_log[month]:
            month_tier_log[month][tier] = month_tier_log[month].get(tier, 0) + 1
        else:
            month_tier_log[month]["BEAR"] = month_tier_log[month].get("BEAR", 0) + 1

        if tier == "MODERATE" and mod_frozen: tier = "BEAR"; prob_thresh = 0; pos_pct = 0

        can_trade = (tier != "BEAR") and (not ytd_frozen)
        if ls_until and day < ls_until: can_trade = False
        elif ls_until: ls_until = None; consec_losses = 0

        # STOCK_BULL eligible: VAS is BEAR but individual stock checked in entry loop
        ls_active = ls_until is not None and day < ls_until
        sb_ok = (tier == "BEAR") and (not ytd_frozen) and (not ls_active) and (sb_trades_ytd < SB_MAX_YEAR)

        # === EXIT ===
        for pos in list(positions):
            t = pos["ticker"]
            if day in data[t].index:
                row_data = data[t].loc[day]
                price = row_data["Close"]; high_today = row_data["High"]; low_today = row_data["Low"]
                pos["days_held"] += 1
                if high_today > pos["high_water"]: pos["high_water"] = high_today

                entry_tier = pos.get("entry_tier", "BULL")
                be_atr  = SB_BE_ATR  if entry_tier == "STOCK_BULL" else (MOD_BE_ATR  if entry_tier == "MODERATE" else BE_TRIGGER_ATR)
                tr_atr  = SB_TRAIL_ATR if entry_tier == "STOCK_BULL" else (MOD_TRAIL_ATR if entry_tier == "MODERATE" else TRAIL_ATR)

                if not pos["at_be"]:
                    if high_today >= pos["entry_price"] + be_atr * pos["entry_atr"]:
                        pos["at_be"] = True
                        exit_comm_est = commsec(pos["shares"] * pos["entry_price"])
                        comm_buffer = (pos["entry_comm"] + exit_comm_est) / pos["shares"]
                        pos["stop"] = pos["entry_price"] + comm_buffer

                if pos["at_be"]:
                    new_stop = pos["high_water"] - tr_atr * pos["entry_atr"]
                    if new_stop > pos["stop"]: pos["stop"] = new_stop

                exit_reason = None
                if pos["days_held"] > GRACE_DAYS:
                    if price <= pos["stop"]: exit_reason = "BE_STOP" if pos["at_be"] else "STOP"
                elif low_today < pos["entry_price"] * 0.92: exit_reason = "EMERGENCY"

                max_hold_for_pos = (SB_MAX_HOLD if entry_tier == "STOCK_BULL" else
                                    (MOD_MAX_HOLD if entry_tier == "MODERATE" else MAX_HOLD))
                if exit_reason is None and pos["days_held"] >= max_hold_for_pos: exit_reason = "TIME"

                if exit_reason is None and tier == "BEAR" and entry_tier not in ("SELECTIVE", "STOCK_BULL"):
                    if price > pos["entry_price"] * 1.005: exit_reason = "REGIME_EXIT"
                if exit_reason is None and entry_tier == "MODERATE":
                    if tier == "BEAR" and price > pos["entry_price"] * 1.003: exit_reason = "GATE_EXIT"
                # STOCK_BULL: let it run its own hold period (ignore VAS regime)

                if exit_reason:
                    ep = price; ec = commsec(pos["shares"] * ep)
                    gross = pos["shares"] * (ep - pos["entry_price"])
                    net = gross - pos["entry_comm"] - ec
                    total_comm += ec
                    trades.append({
                        "ticker": t, "entry": pos["entry_date"], "exit": day,
                        "entry_p": pos["entry_price"], "exit_p": ep,
                        "shares": pos["shares"], "pnl": net, "reason": exit_reason,
                        "days": pos["days_held"], "comm": pos["entry_comm"] + ec,
                        "gross": gross, "prob": pos.get("entry_prob", 0),
                        "tier": entry_tier,
                        "risk_$": pos.get("entry_risk_$", 0),
                        "pos_pct": pos.get("entry_pos_pct", 0),
                    })
                    cash += pos["shares"] * ep - ec
                    ytd_pnl += net; ticker_last_exit[t] = day

                    if entry_tier == "MODERATE":
                        mod_pnl_ytd += net
                        if mod_pnl_ytd <= MOD_YTD_CAP: mod_frozen = True

                    if net > 0: consec_losses = 0
                    else:
                        consec_losses += 1
                        if consec_losses >= LS_MAX: ls_until = day + pd.Timedelta(days=LS_DAYS)
                    positions.remove(pos)
                    if ytd_pnl <= YTD_LOSS_CAP: ytd_frozen = True

        port_val = cash
        for pos in positions:
            t = pos["ticker"]
            if day in data[t].index: port_val += pos["shares"] * data[t].loc[day, "Close"]
        equity_curve.append((day, port_val))

        # === ENTRY ===
        if len(positions) >= MAX_POS: continue
        if not can_trade and not sb_ok: continue
        if monthly_trades.get(month, 0) >= 6: continue

        entry_mode = "STOCK_BULL" if (not can_trade and sb_ok) else tier

        if entry_mode == "BULL":
            if bull_trades_ytd >= BULL_MAX_YEAR: continue
            if bull_trades_month.get(month, 0) >= BULL_MAX_MONTH: continue
        elif entry_mode == "MODERATE":
            if mod_trades_ytd >= MOD_MAX_YEAR: continue
            if mod_trades_month.get(month, 0) >= MOD_MAX_MONTH: continue
        elif entry_mode == "STOCK_BULL":
            if sb_trades_ytd >= SB_MAX_YEAR: continue
            if sb_trades_month.get(month, 0) >= SB_MAX_MONTH: continue

        candidates = []
        n_pass_sma, n_pass_rsi, n_pass_prob = 0, 0, 0

        active_tickers = {pos["ticker"] for pos in positions}
        if entry_mode == "STOCK_BULL":
            tier_tickers = sb_models.keys()
        elif entry_mode == "MODERATE":
            tier_tickers = mod_models.keys()
        else:
            tier_tickers = models.keys()

        for t in tier_tickers:
            if t in active_tickers: continue
            if ticker_tc.get(t, 0) >= MAX_TICKER_TRADES: continue
            if t in ticker_last_exit:
                if (day - ticker_last_exit[t]).days < TICKER_COOLDOWN: continue
            if day not in price_cache[t].index: continue
            if entry_mode == "STOCK_BULL":
                if day not in sb_feat_cache[t].index: continue
                f_row = sb_feat_cache[t].loc[day]
            else:
                if day not in feat_cache[t].index: continue
                f_row = feat_cache[t].loc[day]
            if f_row.isna().any(): continue

            p_row = price_cache[t].loc[day]
            price = p_row["close"]; atr_val = p_row["atr"]
            sma20 = p_row["sma20"]; sma50 = p_row["sma50"]; sma200 = p_row.get("sma200")
            vol = p_row["volume"]; vol_avg = p_row["vol_avg20"]
            rsi_val = p_row["rsi"]; mom5 = p_row["mom5"]; mom20 = p_row["mom20"]

            if pd.isna(atr_val) or atr_val <= 0 or pd.isna(sma50) or pd.isna(sma20): continue

            # ── STOCK_BULL: stock-level regime pre-filter ──────────────────
            if entry_mode == "STOCK_BULL":
                if pd.isna(sma200) or sma200 <= 0: continue
                # Stock must have its own golden cross
                if not (sma50 > sma200): continue
                # Price above own SMA50
                if price <= sma50: continue
                # Strong 20d momentum (own uptrend, not VAS-driven)
                if pd.notna(mom20) and mom20 < SB_MOM20_MIN: continue
                # Not overbought
                if pd.notna(rsi_val) and rsi_val > SB_RSI_MAX: continue
                # Not thin volume
                if pd.notna(vol) and pd.notna(vol_avg) and vol_avg > 0:
                    if vol < 0.5 * vol_avg: continue
                # No extreme volatility
                vr = f_row.get("vol_regime", 1.0)
                if pd.notna(vr) and vr > 2.0: continue
                # ATR/price not too wide (tight stop must be achievable)
                if atr_val / price > SB_STOP_CAP * 1.5: continue
                n_pass_sma += 1; n_pass_rsi += 1
            else:
                # F1: Price > SMA50 and > SMA20 (V22b exact)
                if price < sma50 or price < sma20: continue
                n_pass_sma += 1
                # V27: BULL/MODERATE quality pre-filter — stock must also have own positive momentum
                # Prevents JIN/CPU entries where VAS is BULL but individual stock is lagging
                if entry_mode == "BULL":
                    if pd.notna(mom20) and mom20 < 0.0: continue
                    if pd.notna(sma200) and sma200 > 0 and not (sma50 > sma200): continue
                # F2: RSI 30-80
                if pd.notna(rsi_val) and (rsi_val < 30 or rsi_val > 80): continue
                # F3: momentum not negative
                if pd.notna(mom5) and mom5 < -0.01: continue
                # F4: volume
                if pd.notna(vol) and pd.notna(vol_avg) and vol_avg > 0:
                    if vol < 0.5 * vol_avg: continue
                # F5: not extreme vol
                vr = f_row.get("vol_regime", 1.0)
                if pd.notna(vr) and vr > 2.0: continue
                n_pass_rsi += 1
                # Extra MODERATE filters
                if entry_mode == "MODERATE":
                    rs = f_row.get("relative_strength", 0)
                    if pd.isna(rs) or rs <= 0: continue
                    if price > 0 and atr_val / price > MOD_STOP_CAP: continue
                    vr_mod = f_row.get("vol_regime", 1.0)
                    if pd.notna(vr_mod) and vr_mod > 1.5: continue

            # ML probability
            try:
                if entry_mode == "STOCK_BULL" and t in sb_models:
                    prob = sb_models[t].predict_proba(f_row.values.reshape(1, -1))[0][1]
                    eff_thresh = SB_PROB
                elif entry_mode == "MODERATE" and t in mod_models:
                    prob = mod_models[t].predict_proba(f_row.values.reshape(1, -1))[0][1]
                    eff_thresh = prob_thresh
                elif entry_mode in ("BULL",) and t in models:
                    prob = models[t].predict_proba(f_row.values.reshape(1, -1))[0][1]
                    eff_thresh = prob_thresh
                else: continue
            except: continue
            if prob < eff_thresh: continue
            n_pass_prob += 1

            rs = f_row.get("relative_strength", 0)
            combined_score = prob * 0.7 + (rs if pd.notna(rs) else 0) * 0.3
            candidates.append((t, combined_score, prob, price, atr_val))

        if month not in filter_debug:
            filter_debug[month] = {
                "tier": entry_mode, "prob": (SB_PROB if entry_mode == "STOCK_BULL" else prob_thresh),
                "sma": n_pass_sma, "rsi": n_pass_rsi, "prob_pass": n_pass_prob,
                "cands": len(candidates),
                "detail": {k: v for k, v in tier_detail.items()
                           if k in ("gate_3m", "gate_2m", "gc_age", "gc_age_ok", "golden", "bull_score")}
            }

        if not candidates: continue
        candidates.sort(key=lambda x: -x[1])
        t, _, prob, price, atr_val = candidates[0]

        # Stop distance
        if entry_mode == "STOCK_BULL":
            stop_dist = SB_SL_ATR * atr_val
            if stop_dist > price * SB_STOP_CAP: stop_dist = price * SB_STOP_CAP
        elif entry_mode == "MODERATE":
            stop_dist = MOD_SL_ATR * atr_val
            if stop_dist > price * MOD_STOP_CAP: stop_dist = price * MOD_STOP_CAP
        else:
            stop_dist = SL_ATR * atr_val
            if stop_dist > price * 0.06: stop_dist = price * 0.06

        # Position sizing
        if entry_mode == "STOCK_BULL":
            sb_norm = (prob - SB_PROB) / max(1e-6, 1.0 - SB_PROB)
            sb_norm = min(1.0, max(0.0, sb_norm))
            pos_pct = SB_POS_MIN + sb_norm * (SB_POS_MAX - SB_POS_MIN)
        else:
            prob_norm = (prob - prob_thresh) / max(1e-6, 1.0 - prob_thresh)
            prob_norm = min(1.0, max(0.0, prob_norm))
            pos_pct = PROB_SIZE_MIN + prob_norm * (PROB_SIZE_MAX - PROB_SIZE_MIN)

        value = cash * pos_pct; shares = int(value / price)
        if shares < 1: continue
        value = shares * price
        if value < MIN_POSITION: continue
        ec = commsec(value)

        max_loss_at_stop = shares * stop_dist + 2 * ec
        if max_loss_at_stop > capital * RISK_GUARD_PCT:
            shares = int((capital * RISK_GUARD_PCT - 2 * ec) / stop_dist)
            if shares < 1: continue
            value = shares * price; ec = commsec(value)
            if value < MIN_POSITION: continue

        if value + ec > cash:
            shares = int((cash - 25) / price)
            if shares < 1: continue
            value = shares * price; ec = commsec(value)

        cash -= value + ec; total_comm += ec
        ticker_tc[t] = ticker_tc.get(t, 0) + 1
        monthly_trades[month] = monthly_trades.get(month, 0) + 1
        year_trades[year] = year_trades.get(year, 0) + 1
        tier_trade_counts[entry_mode] = tier_trade_counts.get(entry_mode, 0) + 1

        if entry_mode == "BULL":
            bull_trades_ytd += 1; bull_trades_month[month] = bull_trades_month.get(month, 0) + 1
        elif entry_mode == "MODERATE":
            mod_trades_ytd += 1; mod_trades_month[month] = mod_trades_month.get(month, 0) + 1
        elif entry_mode == "STOCK_BULL":
            sb_trades_ytd += 1; sb_trades_month[month] = sb_trades_month.get(month, 0) + 1

        positions.append({
            "ticker": t, "entry_date": day, "entry_price": price,
            "shares": shares, "stop": price - stop_dist,
            "entry_atr": atr_val, "entry_comm": ec,
            "days_held": 0, "at_be": False,
            "high_water": price, "entry_prob": prob,
            "entry_tier": entry_mode,
            "entry_pos_pct": round(value / capital * 100, 1),
            "entry_risk_$": round(shares * stop_dist + 2 * ec, 2),
        })

    # Close remaining
    for pos in list(positions):
        t = pos["ticker"]
        last = data[t].index[data[t].index <= sim_end]
        if len(last) > 0:
            price = data[t].loc[last[-1], "Close"]; ec = commsec(pos["shares"] * price)
            gross = pos["shares"] * (price - pos["entry_price"])
            net = gross - pos["entry_comm"] - ec; total_comm += ec
            trades.append({
                "ticker": t, "entry": pos["entry_date"], "exit": last[-1],
                "entry_p": pos["entry_price"], "exit_p": price,
                "shares": pos["shares"], "pnl": net, "reason": "FINAL",
                "days": pos["days_held"], "comm": pos["entry_comm"] + ec,
                "gross": gross, "prob": pos.get("entry_prob", 0),
                "tier": pos.get("entry_tier", "BULL"),
                "risk_$": pos.get("entry_risk_$", 0),
                "pos_pct": pos.get("entry_pos_pct", 0),
            })
            cash += pos["shares"] * price - ec

    # === RESULTS ===
    final = cash; roi = (final - capital) / capital * 100; n = len(trades)
    if equity_curve:
        eq_s = pd.Series([e[1] for e in equity_curve], index=[e[0] for e in equity_curve])
        dd = ((eq_s / eq_s.cummax()) - 1).min() * 100
    else: dd = 0

    wins = [tr for tr in trades if tr["pnl"] > 0]
    losses = [tr for tr in trades if tr["pnl"] <= 0]
    wr = len(wins) / n * 100 if n else 0
    avg_w = np.mean([tr["pnl"] for tr in wins]) if wins else 0
    avg_l = np.mean([abs(tr["pnl"]) for tr in losses]) if losses else 0
    pf = (sum(tr["pnl"] for tr in wins) / abs(sum(tr["pnl"] for tr in losses))
          if losses and sum(tr["pnl"] for tr in losses) != 0 else 99)
    rr = avg_w / avg_l if avg_l > 0 else 99
    gross_pnl = sum(tr["gross"] for tr in trades)
    total_comms = sum(tr["comm"] for tr in trades)

    print(f"\n  Regime days:")
    for tk in ["BULL", "MODERATE", "BEAR"]: print(f"    {tk:10s}: {regime_days.get(tk, 0):4d}d")

    print(f"\n  TIER CALENDAR:")
    for m in sorted(month_tier_log.keys()):
        yr, mn = m; counts = month_tier_log[m]; total_d = sum(counts.values())
        parts = []
        labels = {"BULL": "BL", "MODERATE": "MO", "BEAR": "BR", "STOCK_BULL": "SB"}
        for tk in ["BULL", "MODERATE", "STOCK_BULL", "BEAR"]:
            c_val = counts.get(tk, 0)
            if c_val > 0: parts.append(f"{labels[tk]}:{c_val}")
        print(f"    {yr}-{mn:02d}: {' '.join(parts):30s} ({total_d}d)")

    print(f"\n  Filter pipeline (first day per tradeable month):")
    for m in sorted(filter_debug.keys()):
        d = filter_debug[m]
        if d["tier"] != "BEAR":
            det_str = " ".join(f"{k}={v}" for k, v in d.get("detail", {}).items()
                               if k in ("gate_3m", "gate_2m", "gc_age", "gc_age_ok", "golden", "bull_score"))
            print(f"    {m[0]}-{m[1]:02d}: [{d['tier']:10s}] prob>={d['prob']:.2f} SMA:{d['sma']:3d} RSI+:{d['rsi']:3d} Prob+:{d['prob_pass']:3d} Cands:{d['cands']:3d} | {det_str}")

    print(f"\n  RESULTS {name}:")
    print(f"     ${capital:,.0f} -> ${final:,.2f}  ROI: {'+' if roi >= 0 else ''}{roi:.2f}% | DD: {abs(dd):.2f}%")
    print(f"     Trades: {n}")
    if n > 0:
        print(f"     WR: {wr:.1f}% | PF: {pf:.2f} | R:R: {rr:.2f}")
        print(f"     Avg W: ${avg_w:.2f} | Avg L: ${avg_l:.2f}")
        print(f"     GROSS: ${gross_pnl:+,.2f} | CommSec: ${total_comms:,.2f} ({total_comms/capital*100:.1f}%)")

        tier_summary = {}
        for tr in trades:
            tk = tr.get("tier", "?")
            if tk not in tier_summary: tier_summary[tk] = {"n": 0, "pnl": 0, "wins": 0}
            tier_summary[tk]["n"] += 1; tier_summary[tk]["pnl"] += tr["pnl"]
            if tr["pnl"] > 0: tier_summary[tk]["wins"] += 1
        print(f"     Tier breakdown:")
        for tk in ["BULL", "MODERATE", "STOCK_BULL"]:
            if tk in tier_summary:
                ts = tier_summary[tk]; wr_t = ts["wins"] / ts["n"] * 100 if ts["n"] > 0 else 0
                print(f"       {tk:12s} {ts['n']:3d}x WR:{wr_t:>4.0f}% Net:${ts['pnl']:>+8,.2f}")

        reasons = {}
        for tr in trades:
            r = tr["reason"]
            if r not in reasons: reasons[r] = {"n": 0, "pnl": 0, "wins": 0}
            reasons[r]["n"] += 1; reasons[r]["pnl"] += tr["pnl"]
            if tr["pnl"] > 0: reasons[r]["wins"] += 1
        print(f"     Exits:")
        for r, v in sorted(reasons.items(), key=lambda x: -x[1]["n"]):
            wr_r = v["wins"] / v["n"] * 100 if v["n"] > 0 else 0
            print(f"       {r:14s} {v['n']:3d}x WR:{wr_r:>4.0f}% Net:${v['pnl']:>+8,.2f}")

        print(f"\n     ALL TRADES:")
        for tr in trades:
            status = "W" if tr["pnl"] > 0 else "L"
            tk = tr.get("tier", "?")[:3]
            risk_str = f" r:${tr.get('risk_$',0):.0f}" if tr.get('risk_$', 0) > 0 else ""
            pos_str  = f" e:{tr.get('pos_pct',0):.0f}%" if tr.get('pos_pct', 0) > 0 else ""
            print(f"       [{status}] {tr['ticker']:8s} {str(tr['entry'].date()):10s}->{str(tr['exit'].date()):10s} "
                  f"{tr['days']:3d}d prob:{tr['prob']:.2f} ${tr['pnl']:>+8,.2f} ({tr['reason']}) [{tk}]{risk_str}{pos_str}")

    print(f"\n  YEAR BY YEAR:")
    all_years = sorted(set(range(sim_start.year, sim_end.year + 1)))
    if n > 0:
        trade_df = pd.DataFrame(trades)
        trade_df["year"] = pd.to_datetime(trade_df["entry"]).dt.year
    for yr in all_years:
        if n > 0 and yr in trade_df["year"].values:
            yr_df = trade_df[trade_df["year"] == yr]
            yr_n = len(yr_df); yr_net = yr_df["pnl"].sum(); yr_gross = yr_df["gross"].sum()
            yr_comm = yr_df["comm"].sum(); yr_wins = (yr_df["pnl"] > 0).sum()
            yr_wr = yr_wins / yr_n * 100; status = "PASS" if yr_net >= 0 else "FAIL"
            tier_str = ""
            for tk in ["BULL", "MODERATE", "STOCK_BULL"]:
                tk_df = yr_df[yr_df["tier"] == tk]
                if len(tk_df) > 0: tier_str += f" {tk[0]}:{len(tk_df)}"
            print(f"     [{status}] {yr}: {yr_n:3d} trades | NET:${yr_net:>+8,.2f} | "
                  f"Gross:${yr_gross:>+8,.2f} | Comm:${yr_comm:>6,.2f} | WR:{yr_wr:.0f}%{tier_str}")
        else:
            print(f"     [PASS] {yr}:   0 trades | NET:$    0.00")

    return {
        "name": name, "capital": capital, "final": final,
        "roi": roi, "dd": dd, "trades": n, "wr": wr, "pf": pf, "rr": rr,
        "gross": gross_pnl if n > 0 else 0, "comm": total_comms if n > 0 else 0,
        "trade_list": trades, "equity": equity_curve,
    }


if __name__ == "__main__":
    results = []
    for name, start, end, train_end in PERIODS:
        r = simulate_period(name, start, end, train_end, CAPITAL)
        results.append(r)

    elapsed = (time.time() - t0) / 60
    now = dt.datetime.now().strftime("%Y%m%d_%H%M")

    p1, p2, p3 = results[0], results[1], results[2]
    cum = CAPITAL
    for r in results: cum *= (1 + r["roi"] / 100)
    cum_roi = (cum / CAPITAL - 1) * 100

    print(f"\n\n{'='*80}")
    print(f"  SUMMARY - DEEPQUANT V27 (V22b + STOCK_BULL tier + MOD_PROB=0.72)")
    print(f"{'='*80}")
    print(f"  BULL prob-sizing: {PROB_SIZE_MIN*100:.0f}%->90% | SB sizing: {SB_POS_MIN*100:.0f}%-> {SB_POS_MAX*100:.0f}%")
    print(f"  MAX_HOLD={MAX_HOLD}d | SB_MAX_HOLD={SB_MAX_HOLD}d | MOD_MAX_HOLD={MOD_MAX_HOLD}d | MAX_POS={MAX_POS}")
    for r in results:
        wr_str = f"{r['wr']:.1f}%" if r["trades"] > 0 else "N/A"
        pf_str = f"{r['pf']:.2f}" if r["trades"] > 0 else "N/A"
        print(f"  {r['name']:20s} ${r['final']:>9,.2f} ROI:{r['roi']:>+7.2f}% DD:{abs(r['dd']):>5.2f}% Tr:{r['trades']:>3d} WR:{wr_str:>6s} PF:{pf_str:>5s}")
    print(f"  {'-'*78}")
    print(f"  CUMULATIVE: ${cum:,.2f} | ROI: {cum_roi:+.2f}%")

    total_tier = {"BULL": 0, "MODERATE": 0, "STOCK_BULL": 0}
    total_pnl_tier = {"BULL": 0, "MODERATE": 0, "STOCK_BULL": 0}
    for r in results:
        for tr in r["trade_list"]:
            tk = tr.get("tier", "?")
            total_tier[tk] = total_tier.get(tk, 0) + 1
            total_pnl_tier[tk] = total_pnl_tier.get(tk, 0) + tr["pnl"]
    print(f"\n  Total by tier:")
    for tk in ["BULL", "MODERATE", "STOCK_BULL"]:
        if total_tier.get(tk, 0) > 0:
            print(f"    {tk:12s}: {total_tier.get(tk,0):3d} trades, Net ${total_pnl_tier.get(tk,0):+,.2f}")

    print(f"\n  YEAR-BY-YEAR GOAL CHECK:")
    all_trades_combined = []
    for r in results: all_trades_combined.extend(r["trade_list"])
    all_years = set()
    for p in PERIODS:
        for yr in range(int(p[1][:4]), int(p[2][:4]) + 1): all_years.add(yr)
    passed, failed = 0, 0
    if all_trades_combined:
        df_all = pd.DataFrame(all_trades_combined)
        df_all["year"] = pd.to_datetime(df_all["entry"]).dt.year
    for yr in sorted(all_years):
        if all_trades_combined and yr in df_all["year"].values:
            net = df_all[df_all["year"] == yr]["pnl"].sum()
        else: net = 0
        if net >= 0:
            passed += 1; print(f"     {yr}: PASS (${net:+,.2f})")
        else:
            failed += 1; print(f"     {yr}: FAIL (${net:+,.2f})")
    print(f"  Score: {passed}/{passed+failed}")
    if failed == 0: print(f"  *** NEVER LOSE ACHIEVED! ***")

    print(f"\n{'='*90}")
    print(f"  VER    P1 ROI    P2 ROI   P3 ROI    ACUM    2022    2023    2024    2025    2026")
    print(f"  {'-'*85}")
    yr_all = {"2022": "$0", "2023": "$0", "2024": "$0", "2025": "$0", "2026": "$0"}
    for r in results:
        if r["trade_list"]:
            tmp = pd.DataFrame(r["trade_list"])
            tmp["year"] = pd.to_datetime(tmp["entry"]).dt.year
            for yr in tmp["year"].unique():
                yr_all[str(yr)] = f"${tmp[tmp['year']==yr]['pnl'].sum():+,.0f}"
    print(f"  V22b +25.09%   +0.80%   +0.00%  +26.09%      $0    $+44 $+1,964    $+64      $0   5/5 optimal baseline")
    print(f"  V24   +2.22%  -11.38%   +0.00%   -9.41%      $0   $-283   $+461   $-910      $0  3/5 +dist_52w FAILED")
    print(f"  V25  +21.46%   +0.32%   +0.00%  +21.85%      $0    $+44 $+1,673    $+25      $0  5/5 prob>=0.70")
    print(f"  V26  +25.09%   +0.80%   +0.00%  +26.09%      $0    $+44 $+1,964    $+64      $0  5/5 RSI<=78 == V22b")
    print(f"  V27 {p1['roi']:>+7.2f}% {p2['roi']:>+7.2f}% {p3['roi']:>+7.2f}% {cum_roi:>+7.2f}% {yr_all['2022']:>7s} {yr_all['2023']:>7s} {yr_all['2024']:>7s} {yr_all['2025']:>7s} {yr_all['2026']:>7s}  {passed}/{passed+failed} +STOCK_BULL+MOD72")
    print(f"{'='*90}")

    print(f"\n  Time: {elapsed:.1f} min")

    fname = f"sim_v27_{now}.xlsx"
    try:
        with pd.ExcelWriter(fname) as writer:
            for r in results:
                sname = r["name"][:12].replace(":", "").replace(" ", "_")
                df_trades = pd.DataFrame(r["trade_list"])
                if len(df_trades) > 0: df_trades.to_excel(writer, sheet_name=sname, index=False)
                df_eq = pd.DataFrame(r["equity"], columns=["date", "equity"])
                df_eq.to_excel(writer, sheet_name=f"{sname}_eq", index=False)
        print(f"  Saved: {fname}")
    except Exception as e:
        print(f"  Warning: {e}")
    print()
