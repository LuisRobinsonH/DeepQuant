#!/usr/bin/env python
"""
DeepQuant V24 — SINGLE EXPERIMENT: dist_52w_high feature only
=============================================================
BASE = V22b (= V20b confirmed optimum, 5/5, cumulative +26.09%)

CHANGE: ONE new ML feature added to build_features():
  feat["dist_52w_high"] = (close - 52w_high) / 52w_high
  Captures how far price is below its 52-week high.
  → Hypothesis: stocks further from 52w-high may have more upside room.
  → Alternative: overbought stocks near 52w-high may continue (momentum).

ALL OTHER PARAMETERS IDENTICAL TO V22b:
  BULL_PROB_FULL = 0.52 | RSI max 80 | LGB defaults (min_child=25, lambda=2.0)
  No new entry filters added.

QUESTION: Does dist_52w_high improve or harm 2024 P&L?
  2024 baseline (V22b): +$1,964 = +24.5%
  CHC entry Aug-2024: RSI ~78-80, well below 52w-high -> dist_52w_high = negative

NOTE: ML models retrain from scratch with this new feature. Probability
rankings for all stocks will shift. This is a clean single-variable test.

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
BULL_POS_PCT        = 0.80    # legacy reference only — sizing now via Kelly
BULL_MAX_YEAR       = 20
BULL_MAX_MONTH      = 5    # V20b: 5/month is sufficient; raising it exposed to lower-quality signals (V22)

# === MODERATE tier ===
MOD_PROB            = 0.94
MOD_POS_PCT         = 0.45    # legacy reference only — sizing now via Kelly
MOD_MAX_YEAR        = 4
MOD_MAX_MONTH       = 1
MOD_YTD_CAP         = -100      # VERY tight: 1 small loss = done for the year
MOD_GC_MIN_DAYS     = 30        # golden cross must be active >= 30 days

# === V19b: Probability-proportional position sizing ===
# pos_pct = PROB_SIZE_MIN + (prob - prob_thresh)/(1 - prob_thresh) * range
# As prob rises: uncertainty drops -> risk drops -> we invest proportionally more
PROB_SIZE_MIN        = 0.30   # 30% of cash at minimum threshold prob (e.g. 0.52)
PROB_SIZE_MAX        = 0.90   # 90% of cash at prob = 1.00 (larger than V19b's 80%)
RISK_GUARD_PCT       = 0.10   # hard ceiling: actual loss at stop <= 10% of capital
MIN_POSITION         = 2_000  # $2k floor: CommSec $19.95 = 1.00% drag

# === BULL stop parameters (aligned with BULL target) ===
SL_ATR              = 2.5
BE_TRIGGER_ATR      = 1.5
TRAIL_ATR           = 1.5
MAX_HOLD            = 35   # V22 test confirmed: shorter hold (20d) exposed to lower-quality sequential signals
GRACE_DAYS          = 2

# === MODERATE stop parameters (aligned with MODERATE target) ===
MOD_SL_ATR          = 1.5
MOD_BE_ATR          = 1.0
MOD_TRAIL_ATR       = 1.0
MOD_MAX_HOLD        = 15
MOD_STOP_CAP        = 0.04      # max 4% stop distance for MODERATE

# === SELECTIVE tier (individual stock golden cross in VAS-BEAR market) ===
# EMPIRICALLY DISABLED (V21 test confirmed: Score 1/5, -23.68%)
# Root cause: ML model trained on 2017-2021 bull patterns. In bear markets,
# even prob=1.00 signals (WES Jan 2022) STOP-out in 4 days.
# The VAS regime filter is a MODEL VALIDITY guard, not just market timing.
# Individual stock GC does not overcome systematic model invalidity in bear market.
# SELECTIVE_MAX_YEAR=0 disables the tier entirely (preserves V20b behavior).
SELECTIVE_PROB      = 0.90   # threshold (inactive)
SELECTIVE_MAX_YEAR  = 0      # DISABLED: set to 0 to prevent bear-market model misfires
SELECTIVE_SL_ATR    = 2.0
SELECTIVE_POS_MIN   = 0.30
SELECTIVE_POS_MAX   = 0.55

# === Risk management ===
MAX_POS             = 1   # ONE position at a time — avoids correlated losses
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


# === FEATURES (V13 exact) ===
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
    hw52 = h.rolling(252).max()
    feat["dist_52w_high"]   = (c - hw52) / (hw52 + 1e-10)  # V24: distance below 52-week high
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


def build_aligned_target(df, be_atr=1.5, sl_atr=2.5, max_days=25):
    """ALIGNED TARGET: Reach +be_atr*ATR BEFORE -sl_atr*ATR in max_days."""
    c = df["Close"]
    h = df["High"]
    l = df["Low"]
    atr = volatility.average_true_range(h, l, c, 14)
    target = pd.Series(np.nan, index=df.index)
    for i in range(len(df) - max_days):
        entry = c.iloc[i]
        a = atr.iloc[i]
        if pd.isna(a) or a <= 0:
            continue
        be_level = entry + be_atr * a
        sl_level = entry - sl_atr * a
        hit_be = False
        for j in range(1, max_days + 1):
            if i + j >= len(df):
                break
            if h.iloc[i + j] >= be_level:
                hit_be = True
                break
            if l.iloc[i + j] <= sl_level:
                hit_be = False
                break
        target.iloc[i] = 1 if hit_be else 0
    return target


# === HEADER ===
print("=" * 80)
print("  DEEPQUANT V24 — SINGLE EXP: dist_52w_high feature only (V22b base)")
print(f"  prob={BULL_PROB_FULL:.2f}→{PROB_SIZE_MIN*100:.0f}% | prob=1.00→{PROB_SIZE_MAX*100:.0f}% | risk_guard={RISK_GUARD_PCT*100:.0f}% | MAX_POS={MAX_POS}")
print(f"  MAX_HOLD={MAX_HOLD}d | RSI_max=80 | NEW_FEAT=dist_52w_high | all other params=V22b")
print("=" * 80)
print(f"  BULL:     +{BE_TRIGGER_ATR}xATR/-{SL_ATR}xATR {MAX_HOLD}d | 3m+ | prob>=0.52/0.58 | prob-scaled")
print(f"  BULL limits: max {BULL_MAX_MONTH}/mo {BULL_MAX_YEAR}/yr")
print(f"  MODERATE: prob>={MOD_PROB} | 2m gate | SL {MOD_SL_ATR}xATR | {MOD_MAX_HOLD}d max")
print(f"  CommSec: <=1K->$10 | 1K-10K->$19.95 | 10K-25K->$29.95 | >25K->0.12%")

t0 = time.time()
all_tickers = sorted(set(TICKERS + [VAS]))  # sorted for determinism
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
                      group_by="ticker", auto_adjust=True, threads=False)  # threads=False for determinism
    data = {}
    for t in TICKERS:
        try:
            tmp = raw[t].dropna(subset=["Close"])
            if len(tmp) > 250:
                data[t] = tmp
        except:
            pass
    vas_df_raw = raw[VAS].dropna(subset=["Close"])
    print(f"  OK: {len(data)} tickers")
    # Cache to disk
    with open(DATA_CACHE, "wb") as f:
        pickle.dump({"data": data, "vas_df": vas_df_raw}, f)
    print(f"  Data cached to {DATA_CACHE}")

vas_df = vas_df_raw.copy()
vas_df["SMA50"]  = vas_df["Close"].rolling(50).mean()
vas_df["SMA200"] = vas_df["Close"].rolling(200).mean()
vas_df["SMA50_20d_ago"] = vas_df["SMA50"].shift(20)
vas_df["MOM20"]  = vas_df["Close"].pct_change(20)

vas_monthly = vas_df["Close"].resample("ME").last().pct_change()
print(f"  VAS.AX: {len(vas_df)} rows")

# Track golden cross history for the GC-age requirement
gc_active_since = None
gc_history = {}  # date -> days_since_gc_start
in_gc = False
for d in vas_df.index:
    s50 = vas_df.loc[d, "SMA50"]
    s200 = vas_df.loc[d, "SMA200"]
    if pd.notna(s50) and pd.notna(s200) and s50 > s200:
        if not in_gc:
            gc_active_since = d
            in_gc = True
        gc_history[d] = (d - gc_active_since).days
    else:
        in_gc = False
        gc_active_since = None
        gc_history[d] = 0
gc_series = pd.Series(gc_history)

print(f"\n  VAS Monthly Returns:")
for yr in [2022, 2023, 2024, 2025, 2026]:
    try:
        yr_months = vas_monthly[vas_monthly.index.year == yr]
        vals = [f"{v*100:+.1f}%" for v in yr_months.values if pd.notna(v)]
        print(f"    {yr}: {', '.join(vals)}")
    except:
        pass

# Golden cross transitions
print(f"\n  Golden Cross History:")
prev_gc = False
for d in vas_df.index:
    s50 = vas_df.loc[d, "SMA50"]
    s200 = vas_df.loc[d, "SMA200"]
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
    """Get VAS monthly returns for the n months prior to `date`."""
    d = pd.Timestamp(date)
    results = []
    for offset in range(1, n_months + 1):
        m = d.month - offset
        y = d.year
        while m <= 0:
            m += 12
            y -= 1
        try:
            m_end = pd.Timestamp(year=y, month=m, day=28) + pd.offsets.MonthEnd(0)
            if m_end in vas_monthly.index:
                ret = vas_monthly.loc[m_end]
                if pd.notna(ret):
                    results.append(ret)
                else:
                    return None
            else:
                return None
        except:
            return None
    return results if len(results) == n_months else None


def get_vas_state(date):
    """Get VAS indicator values at a given date."""
    mask = vas_df.index <= pd.Timestamp(date)
    if mask.sum() == 0:
        return None
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

    # GC age
    gc_mask = gc_series.index <= pd.Timestamp(date)
    gc_age = gc_series.loc[gc_mask].iloc[-1] if gc_mask.sum() > 0 else 0

    return {
        "golden": golden, "slope": sma50_slope,
        "dist_200": dist_200, "above_200": above_200,
        "above_50": above_50, "gc_age": gc_age
    }


def calc_tier_v15(date):
    """
    EXACT GATE classification (no scoring!).
    Returns: (tier_name, prob_threshold, pos_pct, details)
    """
    vs = get_vas_state(date)
    if vs is None:
        return "BEAR", 0, 0, {"reason": "no data"}

    # Mandatory: VAS > SMA200
    if not vs["above_200"]:
        return "BEAR", 0, 0, {"reason": "VAS < SMA200"}

    detail = dict(vs)

    # === BULL GATE (V13 exact) ===
    prior_3 = get_prior_months(date, 3)
    gate_3m = prior_3 is not None and all(r > 0 for r in prior_3)
    detail["gate_3m"] = gate_3m
    if prior_3:
        detail["months3"] = [f"{r*100:+.1f}%" for r in prior_3]

    if gate_3m and vs["golden"] and vs["slope"] and vs["dist_200"] >= 0.02:
        # Compute regime score for BULL sub-tier (prob selection)
        mom = vas_df.loc[vas_df.index <= pd.Timestamp(date)].iloc[-1]
        mom20 = mom["MOM20"] if pd.notna(mom["MOM20"]) else -1
        brd_mask = breadth_50.index <= pd.Timestamp(date)
        brd = breadth_50.loc[brd_mask].iloc[-1] if brd_mask.sum() > 0 else 0

        score = 50
        if vas_df.loc[vas_df.index <= pd.Timestamp(date)].iloc[-1]["Close"] > \
           vas_df.loc[vas_df.index <= pd.Timestamp(date)].iloc[-1]["SMA50"]:
            score += 15
        if mom20 > 0:
            score += 15
        if brd > 50:
            score += 10
        if vs["dist_200"] > 0.05:
            score += 10
        detail["bull_score"] = score
        return "BULL", (BULL_PROB_FULL if score >= 70 else BULL_PROB_SELECTIVE), BULL_POS_PCT, detail

    # === MODERATE GATE (V16c, 2-month) ===
    prior_2 = get_prior_months(date, 2)
    gate_2m = prior_2 is not None and all(r > 0 for r in prior_2)
    detail["gate_2m"] = gate_2m
    if prior_2:
        detail["months2"] = [f"{r*100:+.1f}%" for r in prior_2]

    if gate_2m and vs["golden"] and vs["above_200"] and vs["above_50"]:
        # Extra: Golden Cross must be established (>= MOD_GC_MIN_DAYS)
        if vs["gc_age"] >= MOD_GC_MIN_DAYS:
            detail["gc_age_ok"] = True
            return "MODERATE", MOD_PROB, MOD_POS_PCT, detail
        else:
            detail["gc_age_ok"] = False
            detail["reason"] = f"GC too young ({vs['gc_age']}d < {MOD_GC_MIN_DAYS}d)"

    return "BEAR", 0, 0, detail


def simulate_period(name, start, end, train_end, capital):
    print(f"\n{'='*70}")
    print(f"  {name}  (train -> {train_end})")
    print(f"{'='*70}")

    train_cutoff = pd.Timestamp(train_end)
    sim_start, sim_end = pd.Timestamp(start), pd.Timestamp(end)

    valid = sorted([t for t in data if data[t].index.min() < train_cutoff - pd.Timedelta(days=365)])
    print(f"  {len(valid)} tickers")

    # Training BULL models (V13 exact: +1.5xATR / -2.5xATR in 25d)
    np.random.seed(42)  # Reset seed for reproducibility per period
    random.seed(42)
    print(f"  Training BULL model (+{BE_TRIGGER_ATR}xATR / -{SL_ATR}xATR in 25d)...")
    models = {}
    target_stats = []
    for i, t in enumerate(valid):
        df = data[t]
        tr = df[df.index <= train_cutoff].copy()
        if len(tr) < 300:
            continue
        feats = build_features(tr, vas_feats, breadth_50, universe_mom_median)
        tgt = build_aligned_target(tr, BE_TRIGGER_ATR, SL_ATR, 25)
        mask = feats.notna().all(axis=1) & tgt.notna()
        X, y = feats[mask], tgt[mask]
        if len(X) < 100 or y.sum() < 5:
            continue
        target_stats.append(y.mean())
        try:
            base = lgb.LGBMClassifier(
                n_estimators=400, max_depth=4, learning_rate=0.03,
                subsample=0.7, colsample_bytree=0.6, min_child_samples=25,
                reg_alpha=0.2, reg_lambda=2.0, verbose=-1, random_state=42,
                is_unbalance=True, n_jobs=1
            )
            cal = CalibratedClassifierCV(base, cv=TimeSeriesSplit(3), method="isotonic")
            cal.fit(X, y)
            models[t] = cal
        except:
            pass
        if (i + 1) % 20 == 0 or i == len(valid) - 1:
            print(f"\r   [{i+1}/{len(valid)}]", end="", flush=True)

    avg_pos_rate = np.mean(target_stats) * 100 if target_stats else 0
    print(f"\n   Models: {len(models)}/{len(valid)} | Avg positive rate: {avg_pos_rate:.1f}%")
    # V16b: MODERATE uses same BULL model (proven), just higher prob threshold
    mod_models = models

    # Pre-compute (include tickers from both models)
    all_model_tickers = set(models.keys()) | set(mod_models.keys())
    feat_cache, price_cache = {}, {}
    for t in all_model_tickers:
        df = data[t]
        feats = build_features(df, vas_feats, breadth_50, universe_mom_median)
        atr = volatility.average_true_range(df["High"], df["Low"], df["Close"], 14)
        sma20 = df["Close"].rolling(20).mean()
        sma50 = df["Close"].rolling(50).mean()
        sma200 = df["Close"].rolling(200).mean()
        rsi_raw = momentum.rsi(df["Close"], 14)
        feat_cache[t] = feats
        price_cache[t] = pd.DataFrame({
            "atr": atr, "sma20": sma20, "sma50": sma50, "sma200": sma200,
            "close": df["Close"], "high": df["High"], "low": df["Low"],
            "volume": df["Volume"], "vol_avg20": df["Volume"].rolling(20).mean(),
            "rsi": rsi_raw, "mom5": df["Close"].pct_change(5)
        }, index=df.index)

    print(f"\n{'-'*70}")
    print(f"  {name}: {start} -> {end} | ${capital:,.0f} | {len(models)} models")
    print(f"{'-'*70}")

    # === SIMULATION ===
    cash = capital
    period_capital = capital          # starting capital for per-slot sizing
    positions = []                    # active position dicts (up to MAX_POS concurrent)
    trades = []
    equity_curve = []
    ticker_tc = {}
    total_comm = 0.0
    consec_losses = 0
    ls_until = None
    ytd_pnl = 0.0
    ytd_year = None
    ytd_frozen = False
    year_trades = {}
    monthly_trades = {}
    ticker_last_exit = {}

    # Per-tier tracking
    bull_trades_ytd = 0
    bull_trades_month = {}
    mod_trades_ytd = 0
    mod_trades_month = {}
    mod_pnl_ytd = 0.0
    mod_frozen = False
    selective_ytd = 0   # V21: SELECTIVE tier counter

    # Stats
    regime_days = {"BULL": 0, "MODERATE": 0, "BEAR": 0}
    tier_trade_counts = {"BULL": 0, "MODERATE": 0, "SELECTIVE": 0}
    month_tier_log = {}  # (yr,mn) -> {BULL:n, MOD:n, BEAR:n}
    filter_debug = {}

    all_sim_dates = set()
    for t in all_model_tickers:
        all_sim_dates.update(data[t].index)
    trading_days = sorted([d for d in all_sim_dates if sim_start <= d <= sim_end])

    for day in trading_days:
        year = day.year
        month = (day.year, day.month)

        if ytd_year != year:
            ytd_year = year
            ytd_pnl = 0.0
            ytd_frozen = False
            year_trades[year] = 0
            bull_trades_ytd = 0
            mod_trades_ytd = 0
            mod_pnl_ytd = 0.0
            mod_frozen = False
            selective_ytd = 0   # V21: reset SELECTIVE counter per year
            consec_losses = 0
            ls_until = None

        if month not in monthly_trades:
            monthly_trades[month] = 0
            bull_trades_month[month] = 0
            mod_trades_month[month] = 0
        if month not in month_tier_log:
            month_tier_log[month] = {"BULL": 0, "MODERATE": 0, "BEAR": 0}

        # === REGIME ===
        tier, prob_thresh, pos_pct, tier_detail = calc_tier_v15(day)
        regime_days[tier] = regime_days.get(tier, 0) + 1
        month_tier_log[month][tier] = month_tier_log[month].get(tier, 0) + 1

        # Check if MODERATE is frozen
        if tier == "MODERATE" and mod_frozen:
            tier = "BEAR"
            prob_thresh = 0
            pos_pct = 0

        # Trading eligibility
        can_trade = (tier != "BEAR") and (not ytd_frozen)
        if ls_until and day < ls_until:
            can_trade = False
        elif ls_until:
            ls_until = None
            consec_losses = 0

        # === EXIT (iterate all active positions) ===
        for pos in list(positions):
            t = pos["ticker"]
            if day in data[t].index:
                row_data = data[t].loc[day]
                price = row_data["Close"]
                high_today = row_data["High"]
                low_today = row_data["Low"]
                pos["days_held"] += 1

                if high_today > pos["high_water"]:
                    pos["high_water"] = high_today

                # Breakeven trigger (tier-specific)
                be_atr = MOD_BE_ATR if pos.get("entry_tier") == "MODERATE" else BE_TRIGGER_ATR
                trail_atr = MOD_TRAIL_ATR if pos.get("entry_tier") == "MODERATE" else TRAIL_ATR
                if not pos["at_be"]:
                    if high_today >= pos["entry_price"] + be_atr * pos["entry_atr"]:
                        pos["at_be"] = True
                        # Commission-aware BE stop: move to entry + comm_roundtrip/shares
                        exit_comm_est = commsec(pos["shares"] * pos["entry_price"])
                        comm_buffer = (pos["entry_comm"] + exit_comm_est) / pos["shares"]
                        pos["stop"] = pos["entry_price"] + comm_buffer

                # Trailing stop update
                if pos["at_be"]:
                    new_stop = pos["high_water"] - trail_atr * pos["entry_atr"]
                    if new_stop > pos["stop"]:
                        pos["stop"] = new_stop

                exit_reason = None

                # Stop check after grace
                if pos["days_held"] > GRACE_DAYS:
                    if price <= pos["stop"]:
                        exit_reason = "BE_STOP" if pos["at_be"] else "STOP"
                elif low_today < pos["entry_price"] * 0.92:
                    exit_reason = "EMERGENCY"

                # Time exit (tier-specific max hold)
                max_hold_for_pos = MOD_MAX_HOLD if pos.get("entry_tier") == "MODERATE" else MAX_HOLD
                if exit_reason is None and pos["days_held"] >= max_hold_for_pos:
                    exit_reason = "TIME"

                # Regime protection: exit on BEAR if profitable
                # SELECTIVE entries: skip this rule (they were opened in BEAR; don't force-exit)
                if exit_reason is None and tier == "BEAR" and pos.get("entry_tier") != "SELECTIVE":
                    if price > pos["entry_price"] * 1.005:
                        exit_reason = "REGIME_EXIT"

                # MODERATE position: exit if regime flips to BEAR and profitable
                if exit_reason is None and pos.get("entry_tier") == "MODERATE":
                    if tier == "BEAR" and price > pos["entry_price"] * 1.003:
                        exit_reason = "GATE_EXIT"

                if exit_reason:
                    ep = price
                    ec = commsec(pos["shares"] * ep)
                    gross = pos["shares"] * (ep - pos["entry_price"])
                    net = gross - pos["entry_comm"] - ec
                    total_comm += ec
                    entry_tier = pos.get("entry_tier", "BULL")

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
                    ytd_pnl += net
                    ticker_last_exit[t] = day

                    if entry_tier == "MODERATE":
                        mod_pnl_ytd += net
                        if mod_pnl_ytd <= MOD_YTD_CAP:
                            mod_frozen = True

                    if net > 0:
                        consec_losses = 0
                    else:
                        consec_losses += 1
                        if consec_losses >= LS_MAX:
                            ls_until = day + pd.Timedelta(days=LS_DAYS)

                    positions.remove(pos)

                    if ytd_pnl <= YTD_LOSS_CAP:
                        ytd_frozen = True

        # Portfolio value (sum all active positions)
        port_val = cash
        for pos in positions:
            t = pos["ticker"]
            if day in data[t].index:
                port_val += pos["shares"] * data[t].loc[day, "Close"]
        equity_curve.append((day, port_val))

        # === ENTRY ===
        if len(positions) >= MAX_POS:
            continue

        # V21: SELECTIVE can trade even in BEAR regime (individual stock conviction)
        ls_active = ls_until is not None and day < ls_until
        selective_ok = (tier == "BEAR" and not ytd_frozen and not ls_active
                        and selective_ytd < SELECTIVE_MAX_YEAR)
        if not can_trade and not selective_ok:
            continue

        if monthly_trades.get(month, 0) >= 6:
            continue

        # entry_mode: what trading tier we're actually using this bar
        entry_mode = "SELECTIVE" if (not can_trade and selective_ok) else tier

        # Check tier-specific limits
        if entry_mode == "BULL":
            if bull_trades_ytd >= BULL_MAX_YEAR:
                continue
            if bull_trades_month.get(month, 0) >= BULL_MAX_MONTH:
                continue
        elif entry_mode == "MODERATE":
            if mod_trades_ytd >= MOD_MAX_YEAR:
                continue
            if mod_trades_month.get(month, 0) >= MOD_MAX_MONTH:
                continue
        # SELECTIVE: capped by selective_ytd < SELECTIVE_MAX_YEAR (checked above)

        candidates = []
        n_pass_sma, n_pass_rsi, n_pass_prob = 0, 0, 0

        # Use correct ticker pool; skip tickers already held
        active_tickers = {pos["ticker"] for pos in positions}
        tier_tickers = mod_models.keys() if entry_mode == "MODERATE" else models.keys()
        for t in tier_tickers:
            if t in active_tickers:
                continue
            if ticker_tc.get(t, 0) >= MAX_TICKER_TRADES:
                continue
            if t in ticker_last_exit:
                if (day - ticker_last_exit[t]).days < TICKER_COOLDOWN:
                    continue
            if day not in feat_cache[t].index or day not in price_cache[t].index:
                continue

            f_row = feat_cache[t].loc[day]
            p_row = price_cache[t].loc[day]
            if f_row.isna().any():
                continue

            price = p_row["close"]
            atr_val = p_row["atr"]
            sma20 = p_row["sma20"]
            sma50 = p_row["sma50"]
            vol = p_row["volume"]
            vol_avg = p_row["vol_avg20"]
            rsi_val = p_row["rsi"]
            mom5 = p_row["mom5"]

            if pd.isna(atr_val) or atr_val <= 0 or pd.isna(sma50) or pd.isna(sma20):
                continue

            # F1: Price > SMA50 and > SMA20
            if price < sma50 or price < sma20:
                continue
            n_pass_sma += 1

            # V21 SELECTIVE: individual stock must have its own golden cross
            if entry_mode == "SELECTIVE":
                sma200_v = p_row.get("sma200")
                if pd.isna(sma200_v) or sma200_v <= 0:
                    continue
                # stock must be above own SMA200 AND SMA50 > SMA200
                if price <= sma200_v or sma50 <= sma200_v:
                    continue

            # F2: RSI 30-80
            if pd.notna(rsi_val) and (rsi_val < 30 or rsi_val > 80):
                continue

            # F3: momentum not negative
            if pd.notna(mom5) and mom5 < -0.01:
                continue

            # F4: volume
            if pd.notna(vol) and pd.notna(vol_avg) and vol_avg > 0:
                if vol < 0.5 * vol_avg:
                    continue

            # F5: not extreme vol
            vr = f_row.get("vol_regime", 1.0)
            if pd.notna(vr) and vr > 2.0:
                continue
            n_pass_rsi += 1

            # Extra filter for MODERATE: relative strength > 0 + low volatility
            if entry_mode == "MODERATE":
                rs = f_row.get("relative_strength", 0)
                if pd.isna(rs) or rs <= 0:
                    continue
                # V16: ATR/price filter - avoid volatile stocks
                if price > 0 and atr_val / price > MOD_STOP_CAP:
                    continue
                # V16: vol_regime must be calm
                vr_mod = f_row.get("vol_regime", 1.0)
                if pd.notna(vr_mod) and vr_mod > 1.5:
                    continue

            # ML probability (use tier-specific model)
            try:
                if entry_mode == "MODERATE" and t in mod_models:
                    prob = mod_models[t].predict_proba(f_row.values.reshape(1, -1))[0][1]
                elif entry_mode in ("BULL", "SELECTIVE") and t in models:
                    prob = models[t].predict_proba(f_row.values.reshape(1, -1))[0][1]
                else:
                    continue
            except:
                continue
            # Effective probability threshold depends on entry mode
            eff_thresh = SELECTIVE_PROB if entry_mode == "SELECTIVE" else prob_thresh
            if prob < eff_thresh:
                continue
            n_pass_prob += 1

            rs = f_row.get("relative_strength", 0)
            combined_score = prob * 0.7 + (rs if pd.notna(rs) else 0) * 0.3
            candidates.append((t, combined_score, prob, price, atr_val))

        # Debug
        if month not in filter_debug:
            filter_debug[month] = {
                "tier": entry_mode, "prob": eff_thresh if entry_mode == "SELECTIVE" else prob_thresh,
                "sma": n_pass_sma, "rsi": n_pass_rsi, "prob_pass": n_pass_prob,
                "cands": len(candidates),
                "detail": {k: v for k, v in tier_detail.items() if k in ("gate_3m", "gate_2m", "gc_age", "gc_age_ok", "golden", "bull_score")}
            }

        if not candidates:
            continue

        candidates.sort(key=lambda x: -x[1])
        t, _, prob, price, atr_val = candidates[0]

        # Tier-specific stop distance
        if entry_mode == "MODERATE":
            stop_dist = MOD_SL_ATR * atr_val
            if stop_dist > price * MOD_STOP_CAP:
                stop_dist = price * MOD_STOP_CAP
        elif entry_mode == "SELECTIVE":
            stop_dist = SELECTIVE_SL_ATR * atr_val
            if stop_dist > price * 0.05:   # cap at 5% for SELECTIVE
                stop_dist = price * 0.05
        else:
            stop_dist = SL_ATR * atr_val
            if stop_dist > price * 0.06:
                stop_dist = price * 0.06

        # === V21: Probability-proportional position sizing (entry_mode aware) ===
        if entry_mode == "SELECTIVE":
            # Smaller, conservative sizing since market has headwind
            sel_norm = (prob - SELECTIVE_PROB) / max(1e-6, 1.0 - SELECTIVE_PROB)
            sel_norm = min(1.0, max(0.0, sel_norm))
            pos_pct = SELECTIVE_POS_MIN + sel_norm * (SELECTIVE_POS_MAX - SELECTIVE_POS_MIN)
        else:
            # Standard prob-proportional: 30% at threshold → 90% at prob=1.00
            prob_norm = (prob - prob_thresh) / max(1e-6, 1.0 - prob_thresh)
            prob_norm = min(1.0, max(0.0, prob_norm))
            pos_pct = PROB_SIZE_MIN + prob_norm * (PROB_SIZE_MAX - PROB_SIZE_MIN)

        # Position value and shares
        value = cash * pos_pct
        shares = int(value / price)
        if shares < 1:
            continue
        value = shares * price

        # Commission-efficiency floor: skip if too small ($19.95 drag would be excessive)
        if value < MIN_POSITION:
            continue
        ec = commsec(value)

        # Risk governor: max actual dollar loss if stop is hit <= RISK_GUARD_PCT x capital
        # Protects against high-ATR stocks where vol is extreme (MIN, FMG, PLS)
        max_loss_at_stop = shares * stop_dist + 2 * ec
        if max_loss_at_stop > capital * RISK_GUARD_PCT:
            # Scale shares down to respect the guard
            shares = int((capital * RISK_GUARD_PCT - 2 * ec) / stop_dist)
            if shares < 1:
                continue
            value = shares * price
            ec = commsec(value)
            if value < MIN_POSITION:
                continue

        if value + ec > cash:
            shares = int((cash - 25) / price)
            if shares < 1:
                continue
            value = shares * price
            ec = commsec(value)

        cash -= value + ec
        total_comm += ec
        ticker_tc[t] = ticker_tc.get(t, 0) + 1
        monthly_trades[month] = monthly_trades.get(month, 0) + 1
        year_trades[year] = year_trades.get(year, 0) + 1
        tier_trade_counts[entry_mode] = tier_trade_counts.get(entry_mode, 0) + 1

        if entry_mode == "BULL":
            bull_trades_ytd += 1
            bull_trades_month[month] = bull_trades_month.get(month, 0) + 1
        elif entry_mode == "MODERATE":
            mod_trades_ytd += 1
            mod_trades_month[month] = mod_trades_month.get(month, 0) + 1
        elif entry_mode == "SELECTIVE":
            selective_ytd += 1

        positions.append({
            "ticker": t, "entry_date": day, "entry_price": price,
            "shares": shares, "stop": price - stop_dist,
            "entry_atr": atr_val, "entry_comm": ec,
            "days_held": 0, "at_be": False,
            "high_water": price, "entry_prob": prob,
            "entry_tier": entry_mode,
            "entry_pos_pct": round(value / capital * 100, 1),  # exposure % of capital
            "entry_risk_$": round(shares * stop_dist + 2 * ec, 2),  # $ at risk at stop
        })

    # Close all remaining positions
    for pos in list(positions):
        t = pos["ticker"]
        last = data[t].index[data[t].index <= sim_end]
        if len(last) > 0:
            price = data[t].loc[last[-1], "Close"]
            ec = commsec(pos["shares"] * price)
            gross = pos["shares"] * (price - pos["entry_price"])
            net = gross - pos["entry_comm"] - ec
            total_comm += ec
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
    final = cash
    roi = (final - capital) / capital * 100
    n = len(trades)
    if len(equity_curve) > 0:
        eq_s = pd.Series([e[1] for e in equity_curve], index=[e[0] for e in equity_curve])
        dd = ((eq_s / eq_s.cummax()) - 1).min() * 100
    else:
        dd = 0

    wins = [tr for tr in trades if tr["pnl"] > 0]
    losses = [tr for tr in trades if tr["pnl"] <= 0]
    wr = len(wins) / n * 100 if n else 0
    avg_w = np.mean([tr["pnl"] for tr in wins]) if wins else 0
    avg_l = np.mean([abs(tr["pnl"]) for tr in losses]) if losses else 0
    pf = sum(tr["pnl"] for tr in wins) / abs(sum(tr["pnl"] for tr in losses)) if losses and sum(tr["pnl"] for tr in losses) != 0 else 99
    rr = avg_w / avg_l if avg_l > 0 else 99
    gross_pnl = sum(tr["gross"] for tr in trades)
    total_comms = sum(tr["comm"] for tr in trades)

    # Regime summary
    print(f"\n  Regime days:")
    for tk in ["BULL", "MODERATE", "BEAR"]:
        print(f"    {tk:10s}: {regime_days.get(tk, 0):4d}d")

    # Tier calendar
    print(f"\n  TIER CALENDAR:")
    for m in sorted(month_tier_log.keys()):
        yr, mn = m
        counts = month_tier_log[m]
        total_d = sum(counts.values())
        parts = []
        labels = {"BULL": "BL", "MODERATE": "MO", "BEAR": "BR"}
        for tk in ["BULL", "MODERATE", "BEAR"]:
            c_val = counts.get(tk, 0)
            if c_val > 0:
                parts.append(f"{labels[tk]}:{c_val}")
        print(f"    {yr}-{mn:02d}: {' '.join(parts):25s} ({total_d}d)")

    # Filter pipeline
    print(f"\n  Filter pipeline (first day per tradeable month):")
    for m in sorted(filter_debug.keys()):
        d = filter_debug[m]
        if d["tier"] != "BEAR":
            det_str = " ".join(f"{k}={v}" for k, v in d.get("detail", {}).items() if k in ("gate_3m", "gate_2m", "gc_age", "gc_age_ok", "golden", "bull_score"))
            print(f"    {m[0]}-{m[1]:02d}: [{d['tier']:8s}] prob>={d['prob']:.2f} SMA:{d['sma']:3d} RSI+:{d['rsi']:3d} Prob+:{d['prob_pass']:3d} Cands:{d['cands']:3d} | {det_str}")

    print(f"\n  RESULTS {name}:")
    print(f"     ${capital:,.0f} -> ${final:,.2f}")
    print(f"     ROI: {'+' if roi >= 0 else ''}{roi:.2f}% | DD: {abs(dd):.2f}%")
    print(f"     Trades: {n}")
    if n > 0:
        print(f"     WR: {wr:.1f}% | PF: {pf:.2f} | R:R: {rr:.2f}")
        print(f"     Avg W: ${avg_w:.2f} | Avg L: ${avg_l:.2f}")
        print(f"     GROSS: ${gross_pnl:+,.2f}")
        print(f"     CommSec: ${total_comms:,.2f} ({total_comms/capital*100:.1f}% cap)")

        # Tier breakdown
        tier_summary = {}
        for tr in trades:
            tk = tr.get("tier", "?")
            if tk not in tier_summary:
                tier_summary[tk] = {"n": 0, "pnl": 0, "wins": 0}
            tier_summary[tk]["n"] += 1
            tier_summary[tk]["pnl"] += tr["pnl"]
            if tr["pnl"] > 0:
                tier_summary[tk]["wins"] += 1
        print(f"     Tier breakdown:")
        for tk in ["BULL", "MODERATE"]:
            if tk in tier_summary:
                ts = tier_summary[tk]
                wr_t = ts["wins"] / ts["n"] * 100 if ts["n"] > 0 else 0
                print(f"       {tk:10s} {ts['n']:3d}x WR:{wr_t:>4.0f}% Net:${ts['pnl']:>+8,.2f}")

        # Exit reasons
        reasons = {}
        for tr in trades:
            r = tr["reason"]
            if r not in reasons:
                reasons[r] = {"n": 0, "pnl": 0, "wins": 0}
            reasons[r]["n"] += 1
            reasons[r]["pnl"] += tr["pnl"]
            if tr["pnl"] > 0:
                reasons[r]["wins"] += 1
        print(f"     Exits:")
        for r, v in sorted(reasons.items(), key=lambda x: -x[1]["n"]):
            wr_r = v["wins"] / v["n"] * 100 if v["n"] > 0 else 0
            print(f"       {r:14s} {v['n']:3d}x WR:{wr_r:>4.0f}% Net:${v['pnl']:>+8,.2f}")

        print(f"\n     ALL TRADES:")
        for tr in trades:
            status = "W" if tr["pnl"] > 0 else "L"
            tk = tr.get("tier", "?")[:3]
            risk_str  = f" r:${tr.get('risk_$',0):.0f}" if tr.get('risk_$', 0) > 0 else ""
            pos_str   = f" e:{tr.get('pos_pct',0):.0f}%" if tr.get('pos_pct', 0) > 0 else ""
            print(f"       [{status}] {tr['ticker']:8s} {str(tr['entry'].date()):10s}->{str(tr['exit'].date()):10s} {tr['days']:3d}d prob:{tr['prob']:.2f} ${tr['pnl']:>+8,.2f} ({tr['reason']}) [{tk}]{risk_str}{pos_str}")

    print(f"\n  YEAR BY YEAR:")
    all_years = sorted(set(range(sim_start.year, sim_end.year + 1)))
    if n > 0:
        trade_df = pd.DataFrame(trades)
        trade_df["year"] = pd.to_datetime(trade_df["entry"]).dt.year
    for yr in all_years:
        if n > 0 and yr in trade_df["year"].values:
            yr_trades_df = trade_df[trade_df["year"] == yr]
            yr_n = len(yr_trades_df)
            yr_net = yr_trades_df["pnl"].sum()
            yr_gross = yr_trades_df["gross"].sum()
            yr_comm = yr_trades_df["comm"].sum()
            yr_wins = (yr_trades_df["pnl"] > 0).sum()
            yr_wr = yr_wins / yr_n * 100
            status = "PASS" if yr_net >= 0 else "FAIL"
            # Tier detail
            tier_detail_str = ""
            for tk in ["BULL", "MODERATE"]:
                tk_df = yr_trades_df[yr_trades_df["tier"] == tk]
                if len(tk_df) > 0:
                    tier_detail_str += f" {tk[0]}:{len(tk_df)}"
            print(f"     [{status}] {yr}: {yr_n:3d} trades | NET:${yr_net:>+8,.2f} | Gross:${yr_gross:>+8,.2f} | Comm:${yr_comm:>6,.2f} | WR:{yr_wr:.0f}%{tier_detail_str}")
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
    for r in results:
        cum *= (1 + r["roi"] / 100)
    cum_roi = (cum / CAPITAL - 1) * 100

    print(f"\n\n{'='*80}")
    print(f"  SUMMARY - DEEPQUANT V24 (V22b + dist_52w_high feature)")
    print(f"{'='*80}")
    print(f"  prob-sizing: {PROB_SIZE_MIN*100:.0f}%→{PROB_SIZE_MAX*100:.0f}% linear | risk_guard={RISK_GUARD_PCT*100:.0f}% | floor=${MIN_POSITION:,} | MAX_POS={MAX_POS}")
    print(f"  MAX_HOLD={MAX_HOLD}d | BULL_MAX_MONTH={BULL_MAX_MONTH} | MOD_MAX_HOLD={MOD_MAX_HOLD}d")
    for r in results:
        wr_str = f"{r['wr']:.1f}%" if r["trades"] > 0 else "N/A"
        pf_str = f"{r['pf']:.2f}" if r["trades"] > 0 else "N/A"
        print(f"  {r['name']:20s} ${r['final']:>8,.2f} ROI:{r['roi']:>+7.2f}% DD:{abs(r['dd']):>5.2f}% Tr:{r['trades']:>3d} WR:{wr_str:>6s} PF:{pf_str:>5s}")
    print(f"  {'-'*78}")
    print(f"  CUMULATIVE: ${cum:,.2f} | ROI: {cum_roi:+.2f}%")

    # Total tier breakdown
    total_tier = {"BULL": 0, "MODERATE": 0, "SELECTIVE": 0}
    total_pnl_tier = {"BULL": 0, "MODERATE": 0, "SELECTIVE": 0}
    for r in results:
        for tr in r["trade_list"]:
            tk = tr.get("tier", "?")
            total_tier[tk] = total_tier.get(tk, 0) + 1
            total_pnl_tier[tk] = total_pnl_tier.get(tk, 0) + tr["pnl"]
    print(f"\n  Total by tier:")
    for tk in ["BULL", "MODERATE", "SELECTIVE"]:
        if total_tier.get(tk, 0) > 0:
            print(f"    {tk}: {total_tier.get(tk,0)} trades, Net ${total_pnl_tier.get(tk,0):+,.2f}")

    print(f"\n  YEAR-BY-YEAR GOAL CHECK:")
    all_trades_combined = []
    for r in results:
        all_trades_combined.extend(r["trade_list"])
    all_years = set()
    for p in PERIODS:
        for yr in range(int(p[1][:4]), int(p[2][:4]) + 1):
            all_years.add(yr)
    passed, failed = 0, 0
    if all_trades_combined:
        df_all = pd.DataFrame(all_trades_combined)
        df_all["year"] = pd.to_datetime(df_all["entry"]).dt.year
    for yr in sorted(all_years):
        if all_trades_combined and yr in df_all["year"].values:
            yr_tr = df_all[df_all["year"] == yr]
            net = yr_tr["pnl"].sum()
        else:
            net = 0
        if net >= 0:
            passed += 1
            print(f"     {yr}: PASS (${net:+,.2f})")
        else:
            failed += 1
            print(f"     {yr}: FAIL (${net:+,.2f})")
    print(f"  Score: {passed}/{passed+failed}")
    if failed == 0:
        print(f"  *** NEVER LOSE ACHIEVED! ***")

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
    print(f"  V13  +24.69%   +6.44%   +0.00%  +32.73%      $0      $0 $+1,975   $+515      $0   5/5 BULL-only")
    print(f"  V16c +25.57%   +1.23%   +0.00%  +27.12%      $0      $2 $+2,044    $+98      $0   5/5 flat-80%")
    print(f"  V19b +21.78%   +0.54%   +0.00%  +22.44%      $0    $+35 $+1,707    $+44      $0   5/5 prob 30->80")
    print(f"  V20  +19.70%   -1.67%   +0.00%  +17.70%      $0    $+35 $+1,542   $-134      $0   4/5 prob x2")
    print(f"  V20b +25.09%   +0.80%   +0.00%  +26.09%      $0    $+44 $+1,964    $+64      $0   5/5 prob 30->90")
    print(f"  V21  disabled  -1.58%   +0.00%   -23.7%   $-535  $-176 $-1,086   $-126      $0   1/5 SELECTIVE")
    print(f"  V22  +16.07%   -5.94%   +0.00%   +9.17%      $0    $+44 $+1,242   $-475      $0   4/5 hold20")
    print(f"  V22b +25.09%   +0.80%   +0.00%  +26.09%      $0    $+44 $+1,964    $+64      $0   5/5 optimal")
    print(f"  V23   -5.93%   -5.99%   -5.15%  -16.13%      $0    $+33   $-507   $-479   $-412  2/5 anti-FP FAILED")
    print(f"  V24 {p1['roi']:>+7.2f}% {p2['roi']:>+7.2f}% {p3['roi']:>+7.2f}% {cum_roi:>+7.2f}% {yr_all['2022']:>7s} {yr_all['2023']:>7s} {yr_all['2024']:>7s} {yr_all['2025']:>7s} {yr_all['2026']:>7s}  {passed}/{passed+failed} +dist_52w_high")
    print(f"{'='*90}")

    print(f"\n  Time: {elapsed:.1f} min")

    fname = f"sim_v24_{now}.xlsx"
    try:
        with pd.ExcelWriter(fname) as writer:
            for r in results:
                sname = r["name"][:12].replace(":", "").replace(" ", "_")
                df_trades = pd.DataFrame(r["trade_list"])
                if len(df_trades) > 0:
                    df_trades.to_excel(writer, sheet_name=sname, index=False)
                df_eq = pd.DataFrame(r["equity"], columns=["date", "equity"])
                df_eq.to_excel(writer, sheet_name=f"{sname}_eq", index=False)
        print(f"  Saved: {fname}")
    except Exception as e:
        print(f"  Warning: {e}")
    print()
