#!/usr/bin/env python
"""
DeepQuant V14 — Multi-Tier Adaptive Regime System
===================================================
V13 Problem: 0 trades in 2022, 2023, and 2026  too conservative.
  The 3-month-positive gate blocks ALL months in choppy markets.

V14 Solution: Continuous regime SCORING (0-100) with 3 tiers of risk management.
  Instead of binary ON/OFF, the system adapts its risk parameters
  based on how favorable the market environment is.

SCORING (max 100):
  Golden Cross (SMA50>SMA200):     +25
  SMA50 slope positive (20d):      +10
  VAS dist from SMA200: >2% -> +5, >5% -> +10
  Monthly: 3 pos -> +25 | 2 pos (mild) -> +15 | 2 pos (harsh) -> +8 | deep neg penalty -10
  VAS > SMA50:                     +10
  Breadth >50% -> +10, >60% -> +15
  VAS 20d momentum >0:            +5

TIERS:
  BULL     (>= 75): Like V13. prob>=0.52, 80% pos, SL 2.5xATR, BE 1.5xATR, 35d
  MODERATE (>= 55): prob>=0.60, 55% pos, SL 2.0xATR, BE 1.2xATR, 25d
  CAUTIOUS (>= 40): prob>=0.68, 40% pos, SL 1.5xATR, BE 1.0xATR, 20d
  BEAR     (< 40):  No trading

MANDATORY: VAS > SMA200 for ANY trading (blocks deep bear markets).

RISK CONTROLS:
  Global YTD cap: -$500
  Per-tier YTD caps: BULL -$500, MODERATE -$250, CAUTIOUS -$150
  Consecutive losses: BULL 3->7d pause | MOD/CAUT 2->14d pause
  Max 20 trades/year, 5/month, 4 per ticker
  Extra filters for lower tiers: relative strength > 0 required
  Ticker cooldown: 10 days

ML: Same as V13 (aligned target: +1.5xATR before -2.5xATR in 25d)
CommSec: <=1K->$10 | 1K-10K->$19.95 | 10K-25K->$29.95 | >25K->0.12%
"""

import warnings, datetime as dt, time
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, yfinance as yf
from pathlib import Path
from ta import momentum, trend, volatility
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb

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

# === TIER CONFIGURATION ===
TIER_PARAMS = {
    "BULL": {
        "prob": 0.52,
        "pos_pct": 0.80,
        "sl_atr": 2.5,
        "be_atr": 1.5,
        "trail_atr": 1.5,
        "max_hold": 35,
        "tier_ytd_cap": -500,
    },
    "MODERATE": {
        "prob": 0.60,
        "pos_pct": 0.55,
        "sl_atr": 2.0,
        "be_atr": 1.2,
        "trail_atr": 1.2,
        "max_hold": 25,
        "tier_ytd_cap": -250,
    },
    "CAUTIOUS": {
        "prob": 0.68,
        "pos_pct": 0.40,
        "sl_atr": 1.5,
        "be_atr": 1.0,
        "trail_atr": 1.0,
        "max_hold": 20,
        "tier_ytd_cap": -150,
    },
}

MAX_POS            = 1
MAX_TRADES_YEAR    = 20
MAX_TRADES_MONTH   = 5
MAX_TICKER_TRADES  = 4
TICKER_COOLDOWN    = 10
YTD_LOSS_CAP       = -500       # global
GRACE_DAYS         = 2
LS_MAX_BULL        = 3          # 3 consecutive losses -> pause
LS_PAUSE_BULL      = 7
LS_MAX_LOWER       = 2          # 2 consecutive losses in MOD/CAUT -> pause
LS_PAUSE_LOWER     = 14

PERIODS = [
    ("P1: 2022-2024", "2022-01-01", "2024-12-31", "2021-12-31"),
    ("P2: 2025",      "2025-01-01", "2025-12-31", "2024-12-31"),
    ("P3: 2026 YTD",  "2026-01-01", "2026-12-31", "2025-12-31"),
]
CAPITAL = 8_000.0


# === FEATURES (same as V13) ===
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
print("=" * 70)
print("  DEEPQUANT V14 - Multi-Tier Adaptive Regime System")
print("  BULL | MODERATE | CAUTIOUS tiers based on regime score 0-100")
print("=" * 70)
print(f"  ML: Aligned target +1.5xATR before -2.5xATR in 25d")
print(f"  BULL (>=75):     prob>=0.52 | 80% pos | SL 2.5xATR | BE 1.5xATR | 35d")
print(f"  MODERATE (>=55): prob>=0.60 | 55% pos | SL 2.0xATR | BE 1.2xATR | 25d")
print(f"  CAUTIOUS (>=40): prob>=0.68 | 40% pos | SL 1.5xATR | BE 1.0xATR | 20d")
print(f"  BEAR (<40):      No trading")
print(f"  VAS > SMA200 MANDATORY for all tiers")
print(f"  CommSec: <=1K->$10 | 1K-10K->$19.95 | 10K-25K->$29.95 | >25K->0.12%")

t0 = time.time()
all_tickers = list(set(TICKERS + [VAS]))
print(f"\n  Downloading {len(all_tickers)} tickers...")
raw = yf.download(all_tickers, start="2019-01-01", period="max",
                  group_by="ticker", auto_adjust=True, threads=True)

data = {}
for t in TICKERS:
    try:
        tmp = raw[t].dropna(subset=["Close"])
        if len(tmp) > 250:
            data[t] = tmp
    except:
        pass
print(f"  OK: {len(data)} tickers")

vas_df = raw[VAS].dropna(subset=["Close"])
vas_df["SMA50"]  = vas_df["Close"].rolling(50).mean()
vas_df["SMA200"] = vas_df["Close"].rolling(200).mean()
vas_df["SMA50_20d_ago"] = vas_df["SMA50"].shift(20)
vas_df["MOM20"]  = vas_df["Close"].pct_change(20)

vas_monthly = vas_df["Close"].resample("ME").last().pct_change()
print(f"  VAS.AX: {len(vas_df)} rows")

print(f"\n  VAS Monthly Returns:")
for yr in [2022, 2023, 2024, 2025, 2026]:
    try:
        yr_months = vas_monthly[vas_monthly.index.year == yr]
        vals = [f"{v*100:+.1f}%" for v in yr_months.values if pd.notna(v)]
        print(f"    {yr}: {', '.join(vals)}")
    except:
        pass

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


def get_prior_3_months(date):
    """Get VAS monthly returns for the 3 months prior to `date`."""
    d = pd.Timestamp(date)
    results = []
    for offset in [1, 2, 3]:
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
    return results if len(results) == 3 else None


def calc_regime_v14(date):
    """
    Continuous regime scoring (0-100).
    Returns: (score, tier_name, details_dict)
    MANDATORY: VAS > SMA200 for any trading.
    """
    mask = vas_df.index <= pd.Timestamp(date)
    if mask.sum() == 0:
        return 0, "BEAR", {"reason": "no data"}

    row = vas_df.loc[mask].iloc[-1]
    c = row["Close"]
    s50 = row["SMA50"] if pd.notna(row["SMA50"]) else 0
    s200 = row["SMA200"] if pd.notna(row["SMA200"]) else 0

    # MANDATORY: VAS must be above SMA200
    if s200 <= 0 or c <= s200:
        return 0, "BEAR", {"reason": "VAS <= SMA200"}

    score = 0
    det = {}

    # 1. Golden Cross (SMA50 > SMA200): +25
    golden = s50 > 0 and s50 > s200
    if golden:
        score += 25
    det["golden"] = golden

    # 2. SMA50 slope positive (vs 20 days ago): +10
    s50_prev = row.get("SMA50_20d_ago", np.nan) if "SMA50_20d_ago" in vas_df.columns else np.nan
    slope = False
    if pd.notna(s50_prev) and s50 > 0:
        slope = s50 > s50_prev
    if slope:
        score += 10
    det["slope"] = slope

    # 3. VAS distance from SMA200: +5 or +10
    dist = (c - s200) / s200
    if dist > 0.05:
        score += 10
    elif dist > 0.02:
        score += 5
    det["dist200"] = f"{dist*100:.1f}%"

    # 4. Monthly returns analysis: up to +25, penalty up to -10
    prior_3 = get_prior_3_months(date)
    if prior_3:
        n_pos = sum(1 for r in prior_3 if r > 0)
        avg_3m = np.mean(prior_3)
        worst = min(prior_3)

        if n_pos == 3:
            score += 25
        elif n_pos == 2 and avg_3m > 0:
            if worst > -0.02:
                score += 15      # mild negative month
            else:
                score += 8       # one month quite negative
        elif n_pos >= 1 and avg_3m > 0:
            score += 3

        # Penalty for very bad month in the window
        if worst < -0.05:
            score -= 10

        det["months"] = f"{n_pos}/3 pos, avg={avg_3m*100:.1f}%, worst={worst*100:.1f}%"
    else:
        det["months"] = "N/A"

    # 5. VAS above SMA50: +10
    above50 = c > s50 if s50 > 0 else False
    if above50:
        score += 10
    det["above50"] = above50

    # 6. Breadth: +10 or +15
    brd_mask = breadth_50.index <= pd.Timestamp(date)
    brd = breadth_50.loc[brd_mask].iloc[-1] if brd_mask.sum() > 0 else 0
    if brd > 60:
        score += 15
    elif brd > 50:
        score += 10
    det["breadth"] = f"{brd:.0f}%"

    # 7. VAS 20d momentum positive: +5
    mom = row["MOM20"] if pd.notna(row["MOM20"]) else -1
    if mom > 0:
        score += 5
    det["mom20"] = f"{mom*100:.1f}%" if mom != -1 else "N/A"

    det["score"] = score

    # Tier determination
    if score >= 75:
        tier = "BULL"
    elif score >= 55:
        tier = "MODERATE"
    elif score >= 40:
        tier = "CAUTIOUS"
    else:
        tier = "BEAR"

    return score, tier, det


def simulate_period(name, start, end, train_end, capital):
    print(f"\n{'='*70}")
    print(f"  {name}  (train -> {train_end})")
    print(f"{'='*70}")

    train_cutoff = pd.Timestamp(train_end)
    sim_start, sim_end = pd.Timestamp(start), pd.Timestamp(end)

    valid = [t for t in data if data[t].index.min() < train_cutoff - pd.Timedelta(days=365)]
    print(f"  {len(valid)} tickers")

    # Training with ALIGNED target (same as V13)
    print(f"  Training (ALIGNED target: +1.5xATR before -2.5xATR in 25d)...")
    models = {}
    target_stats = []
    for i, t in enumerate(valid):
        df = data[t]
        tr = df[df.index <= train_cutoff].copy()
        if len(tr) < 300:
            continue
        feats = build_features(tr, vas_feats, breadth_50, universe_mom_median)
        tgt = build_aligned_target(tr, 1.5, 2.5, 25)
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
                is_unbalance=True
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

    # Pre-compute features
    feat_cache, price_cache = {}, {}
    for t in models:
        df = data[t]
        feats = build_features(df, vas_feats, breadth_50, universe_mom_median)
        atr = volatility.average_true_range(df["High"], df["Low"], df["Close"], 14)
        sma20 = df["Close"].rolling(20).mean()
        sma50 = df["Close"].rolling(50).mean()
        rsi_raw = momentum.rsi(df["Close"], 14)
        feat_cache[t] = feats
        price_cache[t] = pd.DataFrame({
            "atr": atr, "sma20": sma20, "sma50": sma50,
            "close": df["Close"], "high": df["High"], "low": df["Low"],
            "volume": df["Volume"], "vol_avg20": df["Volume"].rolling(20).mean(),
            "rsi": rsi_raw, "mom5": df["Close"].pct_change(5)
        }, index=df.index)

    print(f"\n{'-'*70}")
    print(f"  {name}: {start} -> {end} | ${capital:,.0f} | {len(models)} models")
    print(f"{'-'*70}")

    # === SIMULATION STATE ===
    cash = capital
    position = None     # single position dict
    trades = []
    equity_curve = []
    ticker_tc = {}
    total_comm = 0.0
    ytd_pnl = 0.0
    ytd_year = None
    ytd_frozen = False
    year_trades = {}
    monthly_trades = {}
    ticker_last_exit = {}

    # Per-tier tracking
    tier_pnl_ytd = {"BULL": 0, "MODERATE": 0, "CAUTIOUS": 0}
    tier_frozen = {"MODERATE": False, "CAUTIOUS": False}  # BULL can't freeze per-tier
    tier_trades_ytd = {"BULL": 0, "MODERATE": 0, "CAUTIOUS": 0}

    # Loss streak
    consec_losses_bull = 0
    consec_losses_lower = 0
    ls_bull_until = None
    ls_lower_until = None

    # Stats
    regime_day_counts = {"BULL": 0, "MODERATE": 0, "CAUTIOUS": 0, "BEAR": 0}
    tier_trade_counts = {"BULL": 0, "MODERATE": 0, "CAUTIOUS": 0}

    # Monthly tier calendar for display
    month_tier_log = {}  # (year,month) -> {"BULL": count, "MODERATE": count, ...}
    filter_debug = {}

    all_sim_dates = set()
    for t in models:
        all_sim_dates.update(data[t].index)
    trading_days = sorted([d for d in all_sim_dates if sim_start <= d <= sim_end])

    for day in trading_days:
        year = day.year
        month = (day.year, day.month)

        # Year reset
        if ytd_year != year:
            ytd_year = year
            ytd_pnl = 0.0
            ytd_frozen = False
            year_trades[year] = 0
            tier_pnl_ytd = {"BULL": 0, "MODERATE": 0, "CAUTIOUS": 0}
            tier_frozen = {"MODERATE": False, "CAUTIOUS": False}
            tier_trades_ytd = {"BULL": 0, "MODERATE": 0, "CAUTIOUS": 0}
            consec_losses_bull = 0
            consec_losses_lower = 0
            ls_bull_until = None
            ls_lower_until = None

        if month not in monthly_trades:
            monthly_trades[month] = 0
        if month not in month_tier_log:
            month_tier_log[month] = {"BULL": 0, "MODERATE": 0, "CAUTIOUS": 0, "BEAR": 0}

        # === REGIME ===
        score, tier, details = calc_regime_v14(day)
        tier_cfg = TIER_PARAMS.get(tier)
        regime_day_counts[tier] = regime_day_counts.get(tier, 0) + 1
        month_tier_log[month][tier] = month_tier_log[month].get(tier, 0) + 1

        # Trading allowed?
        can_trade = not ytd_frozen and tier_cfg is not None

        # Loss streak pause
        if tier == "BULL":
            if ls_bull_until and day < ls_bull_until:
                can_trade = False
            elif ls_bull_until:
                ls_bull_until = None
                consec_losses_bull = 0
        elif tier in ("MODERATE", "CAUTIOUS"):
            if ls_lower_until and day < ls_lower_until:
                can_trade = False
            elif ls_lower_until:
                ls_lower_until = None
                consec_losses_lower = 0
            if tier_frozen.get(tier, False):
                can_trade = False

        # === EXIT ===
        if position is not None:
            t = position["ticker"]
            if day in data[t].index:
                row_data = data[t].loc[day]
                price = row_data["Close"]
                high_today = row_data["High"]
                low_today = row_data["Low"]
                position["days_held"] += 1

                if high_today > position["high_water"]:
                    position["high_water"] = high_today

                # Breakeven trigger
                if not position["at_be"]:
                    if high_today >= position["entry_price"] + position["be_atr"] * position["entry_atr"]:
                        position["at_be"] = True
                        position["stop"] = position["entry_price"]

                # Trailing stop update
                if position["at_be"]:
                    new_stop = position["high_water"] - position["trail_atr"] * position["entry_atr"]
                    if new_stop > position["stop"]:
                        position["stop"] = new_stop

                exit_reason = None

                # Stop check (after grace period)
                if position["days_held"] > GRACE_DAYS:
                    if price <= position["stop"]:
                        exit_reason = "BE_STOP" if position["at_be"] else "STOP"
                elif low_today < position["entry_price"] * 0.92:
                    exit_reason = "EMERGENCY"

                # Time exit
                if exit_reason is None and position["days_held"] >= position["max_hold"]:
                    exit_reason = "TIME"

                # Regime protection: exit on BEAR if profitable
                if exit_reason is None and tier == "BEAR" and price > position["entry_price"] * 1.003:
                    exit_reason = "REGIME_EXIT"

                # Tier downgrade protection: entered BULL, now CAUTIOUS/BEAR
                if exit_reason is None and position.get("entry_tier") == "BULL":
                    if tier in ("CAUTIOUS", "BEAR") and price > position["entry_price"] * 1.003:
                        exit_reason = "TIER_EXIT"

                # Entered MODERATE, now BEAR
                if exit_reason is None and position.get("entry_tier") == "MODERATE":
                    if tier == "BEAR" and price > position["entry_price"] * 1.003:
                        exit_reason = "TIER_EXIT"

                if exit_reason:
                    ep = price
                    ec = commsec(position["shares"] * ep)
                    gross = position["shares"] * (ep - position["entry_price"])
                    net = gross - position["entry_comm"] - ec
                    total_comm += ec
                    entry_tier = position.get("entry_tier", "BULL")

                    trades.append({
                        "ticker": t, "entry": position["entry_date"], "exit": day,
                        "entry_p": position["entry_price"], "exit_p": ep,
                        "shares": position["shares"], "pnl": net, "reason": exit_reason,
                        "days": position["days_held"], "comm": position["entry_comm"] + ec,
                        "gross": gross, "prob": position.get("entry_prob", 0),
                        "tier": entry_tier
                    })
                    cash += position["shares"] * ep - ec
                    ytd_pnl += net
                    tier_pnl_ytd[entry_tier] = tier_pnl_ytd.get(entry_tier, 0) + net
                    ticker_last_exit[t] = day

                    # Loss tracking
                    if net > 0:
                        consec_losses_bull = 0
                        consec_losses_lower = 0
                    else:
                        if entry_tier == "BULL":
                            consec_losses_bull += 1
                            if consec_losses_bull >= LS_MAX_BULL:
                                ls_bull_until = day + pd.Timedelta(days=LS_PAUSE_BULL)
                        else:
                            consec_losses_lower += 1
                            if consec_losses_lower >= LS_MAX_LOWER:
                                ls_lower_until = day + pd.Timedelta(days=LS_PAUSE_LOWER)

                    position = None

                    # Check caps
                    if ytd_pnl <= YTD_LOSS_CAP:
                        ytd_frozen = True
                    for tk in ["MODERATE", "CAUTIOUS"]:
                        if tier_pnl_ytd.get(tk, 0) <= TIER_PARAMS[tk]["tier_ytd_cap"]:
                            tier_frozen[tk] = True

        # Portfolio value
        port_val = cash
        if position is not None:
            t = position["ticker"]
            if day in data[t].index:
                port_val += position["shares"] * data[t].loc[day, "Close"]
        equity_curve.append((day, port_val))

        # === ENTRY ===
        if position is not None or not can_trade or tier_cfg is None:
            continue
        if monthly_trades.get(month, 0) >= MAX_TRADES_MONTH:
            continue
        if year_trades.get(year, 0) >= MAX_TRADES_YEAR:
            continue

        candidates = []
        n_pass_sma, n_pass_rsi, n_pass_prob = 0, 0, 0

        for t in models:
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

            # F2: RSI 30-80
            if pd.notna(rsi_val) and (rsi_val < 30 or rsi_val > 80):
                continue

            # F3: 5-day momentum not deeply negative
            if pd.notna(mom5) and mom5 < -0.01:
                continue

            # F4: Some volume
            if pd.notna(vol) and pd.notna(vol_avg) and vol_avg > 0:
                if vol < 0.5 * vol_avg:
                    continue

            # F5: Not extreme volatility
            vr = f_row.get("vol_regime", 1.0)
            if pd.notna(vr) and vr > 2.0:
                continue
            n_pass_rsi += 1

            # Extra filter for MODERATE/CAUTIOUS: relative strength > 0
            if tier in ("MODERATE", "CAUTIOUS"):
                rs = f_row.get("relative_strength", 0)
                if pd.isna(rs) or rs <= 0:
                    continue

            # ML probability
            try:
                prob = models[t].predict_proba(f_row.values.reshape(1, -1))[0][1]
            except:
                continue
            if prob < tier_cfg["prob"]:
                continue
            n_pass_prob += 1

            rs = f_row.get("relative_strength", 0)
            combined_score = prob * 0.7 + (rs if pd.notna(rs) else 0) * 0.3
            candidates.append((t, combined_score, prob, price, atr_val))

        # Debug first day per tradeable month
        if month not in filter_debug:
            filter_debug[month] = {
                "tier": tier, "score": score,
                "sma": n_pass_sma, "rsi": n_pass_rsi, "prob": n_pass_prob,
                "cands": len(candidates)
            }

        if not candidates:
            continue

        candidates.sort(key=lambda x: -x[1])
        t, _, prob, price, atr_val = candidates[0]

        sl_dist = tier_cfg["sl_atr"] * atr_val
        if sl_dist > price * 0.06:
            sl_dist = price * 0.06

        value = cash * tier_cfg["pos_pct"]
        shares = int(value / price)
        if shares < 1:
            continue
        value = shares * price
        if value < 1500:
            continue
        ec = commsec(value)
        if value + ec > cash:
            shares = int((cash - 25) * 0.85 / price)
            if shares < 1:
                continue
            value = shares * price
            ec = commsec(value)

        cash -= value + ec
        total_comm += ec
        ticker_tc[t] = ticker_tc.get(t, 0) + 1
        monthly_trades[month] = monthly_trades.get(month, 0) + 1
        year_trades[year] = year_trades.get(year, 0) + 1
        tier_trades_ytd[tier] = tier_trades_ytd.get(tier, 0) + 1
        tier_trade_counts[tier] = tier_trade_counts.get(tier, 0) + 1

        position = {
            "ticker": t, "entry_date": day, "entry_price": price,
            "shares": shares, "stop": price - sl_dist,
            "entry_atr": atr_val, "entry_comm": ec,
            "days_held": 0, "at_be": False,
            "high_water": price, "entry_prob": prob,
            "entry_tier": tier,
            "be_atr": tier_cfg["be_atr"],
            "trail_atr": tier_cfg["trail_atr"],
            "max_hold": tier_cfg["max_hold"],
        }

    # Close remaining
    if position is not None:
        t = position["ticker"]
        last = data[t].index[data[t].index <= sim_end]
        if len(last) > 0:
            price = data[t].loc[last[-1], "Close"]
            ec = commsec(position["shares"] * price)
            gross = position["shares"] * (price - position["entry_price"])
            net = gross - position["entry_comm"] - ec
            total_comm += ec
            trades.append({
                "ticker": t, "entry": position["entry_date"], "exit": last[-1],
                "entry_p": position["entry_price"], "exit_p": price,
                "shares": position["shares"], "pnl": net, "reason": "FINAL",
                "days": position["days_held"], "comm": position["entry_comm"] + ec,
                "gross": gross, "prob": position.get("entry_prob", 0),
                "tier": position.get("entry_tier", "BULL")
            })
            cash += position["shares"] * price - ec

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

    # Regime day summary
    print(f"\n  Regime days:")
    for tk in ["BULL", "MODERATE", "CAUTIOUS", "BEAR"]:
        print(f"    {tk:10s}: {regime_day_counts.get(tk, 0):4d}d")

    # Monthly tier calendar
    print(f"\n  TIER CALENDAR:")
    for m in sorted(month_tier_log.keys()):
        yr, mn = m
        counts = month_tier_log[m]
        total_d = sum(counts.values())
        parts = []
        for tk in ["BULL", "MODERATE", "CAUTIOUS", "BEAR"]:
            c_val = counts.get(tk, 0)
            if c_val > 0:
                parts.append(f"{tk[0]}:{c_val}")
        tstr = " ".join(parts)
        print(f"    {yr}-{mn:02d}: {tstr:30s} ({total_d}d)")

    # Filter pipeline
    print(f"\n  Filter pipeline (first day per tradeable month):")
    for m in sorted(filter_debug.keys()):
        d = filter_debug[m]
        if d["tier"] != "BEAR":
            print(f"    {m[0]}-{m[1]:02d}: [{d['tier']:8s}] score={d['score']:3d} SMA:{d['sma']:3d} RSI+:{d['rsi']:3d} Prob+:{d['prob']:3d} Cands:{d['cands']:3d}")

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
        for tk in ["BULL", "MODERATE", "CAUTIOUS"]:
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
            print(f"       [{status}] {tr['ticker']:8s} {str(tr['entry'].date()):10s}->{str(tr['exit'].date()):10s} {tr['days']:3d}d prob:{tr['prob']:.2f} ${tr['pnl']:>+8,.2f} ({tr['reason']}) [{tk}]")

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
            tier_detail = ""
            for tk in ["BULL", "MODERATE", "CAUTIOUS"]:
                tk_df = yr_trades_df[yr_trades_df["tier"] == tk]
                if len(tk_df) > 0:
                    tier_detail += f" {tk[0]}:{len(tk_df)}"
            print(f"     [{status}] {yr}: {yr_n:3d} trades | NET:${yr_net:>+8,.2f} | Gross:${yr_gross:>+8,.2f} | Comm:${yr_comm:>6,.2f} | WR:{yr_wr:.0f}%{tier_detail}")
        else:
            print(f"     [PASS] {yr}:   0 trades | NET:$    0.00")

    return {
        "name": name, "capital": capital, "final": final,
        "roi": roi, "dd": dd, "trades": n, "wr": wr, "pf": pf, "rr": rr,
        "gross": gross_pnl if n > 0 else 0, "comm": total_comms if n > 0 else 0,
        "trade_list": trades, "equity": equity_curve,
        "tier_counts": tier_trade_counts
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
    print(f"  SUMMARY - DEEPQUANT V14")
    print(f"{'='*80}")
    for r in results:
        wr_str = f"{r['wr']:.1f}%" if r["trades"] > 0 else "N/A"
        pf_str = f"{r['pf']:.2f}" if r["trades"] > 0 else "N/A"
        print(f"  {r['name']:20s} ${r['final']:>8,.2f} ROI:{r['roi']:>+7.2f}% DD:{abs(r['dd']):>5.2f}% Tr:{r['trades']:>3d} WR:{wr_str:>6s} PF:{pf_str:>5s}")
    print(f"  {'-'*78}")
    print(f"  CUMULATIVE: ${cum:,.2f} | ROI: {cum_roi:+.2f}%")

    # Total tier breakdown
    total_tier = {"BULL": 0, "MODERATE": 0, "CAUTIOUS": 0}
    for r in results:
        for tr in r["trade_list"]:
            tk = tr.get("tier", "?")
            total_tier[tk] = total_tier.get(tk, 0) + 1
    print(f"\n  Total trades by tier: BULL={total_tier.get('BULL',0)} MODERATE={total_tier.get('MODERATE',0)} CAUTIOUS={total_tier.get('CAUTIOUS',0)}")

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
    print(f"  V13  +24.69%   +6.44%   +0.00%  +32.73%      $0      $0 $+1,975   $+515      $0")
    print(f"  V14 {p1['roi']:>+7.2f}% {p2['roi']:>+7.2f}% {p3['roi']:>+7.2f}% {cum_roi:>+7.2f}% {yr_all['2022']:>7s} {yr_all['2023']:>7s} {yr_all['2024']:>7s} {yr_all['2025']:>7s} {yr_all['2026']:>7s}")
    print(f"{'='*90}")

    print(f"\n  Time: {elapsed:.1f} min")

    fname = f"sim_v14_{now}.xlsx"
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
