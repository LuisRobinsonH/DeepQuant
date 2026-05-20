#!/usr/bin/env python
"""
DeepQuant V9 — Never-Lose Year-by-Year Model
==============================================
DESIGN PHILOSOPHY: It is better to miss gains than to take losses.
With $8K capital and CommSec fees, every trade costs ~1% round-trip.
The model must be RIGHT or NOT TRADE.

ARCHITECTURE:
├── Layer 1: REGIME SCORE (0-100) — calculated daily from VAS.AX + breadth
│   ├── VAS > SMA200                         (+25)
│   ├── VAS > SMA50                          (+15)
│   ├── SMA50 > SMA200 (golden cross)        (+20)
│   ├── VAS 20d momentum > 0                 (+15)
│   ├── Market breadth (% > SMA50) > 50%     (+15)
│   └── VAS volatility below average         (+10)
│
├── Layer 2: ML STOCK SELECTION — LightGBM with 25 features
│   ├── 20 standard technical features
│   ├── VAS momentum (market context)
│   ├── VAS position (distance from SMA200)
│   ├── Market breadth (% above SMA50)
│   ├── Relative strength vs universe
│   └── Consolidation score (10d range / ATR — breakout potential)
│
├── Layer 3: ENTRY FILTERS
│   ├── Regime score ≥ 80 → prob ≥ 0.55
│   ├── Regime score 65-80 → prob ≥ 0.68
│   ├── Regime score < 65 → NO TRADE
│   ├── Stock > SMA50 (confirmed uptrend)
│   ├── Pullback near SMA20 (max 3.5%)
│   ├── RSI 35-68 (not overbought)
│   ├── ADX > 0.15 (trend strength)
│   └── Volume > 0.8× 20d average (participation)
│
├── Layer 4: POSITION MANAGEMENT
│   ├── Max 2 positions, min $2,500 each
│   ├── Risk 3.5% of equity per trade
│   ├── Initial SL: 2.5 × ATR
│   ├── Breakeven stop at +2.0 × ATR
│   ├── Trail 2.0 × ATR after breakeven
│   ├── NO partial TP (saves commission)
│   └── Max hold 40 days
│
└── Layer 5: SAFETY NETS
    ├── Circuit breaker: -7% from peak → cash 20 days
    ├── Loss streak: 3 consecutive → pause 10 days
    ├── Max 3 trades per ticker per period
    └── CommSec real tiered commissions

TARGET: 5% forward return in 20 days, max 4% drawdown
(High hurdle → model learns BIG moves that cover commissions)
"""

import warnings, datetime as dt, time
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, yfinance as yf
from pathlib import Path
from ta import momentum, trend, volatility
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb

# ── Config ──────────────────────────────────────────────
SYM_FILE = Path("au_stock_data/au_symbols.txt")
TICKERS = [s.strip() for s in SYM_FILE.read_text().splitlines() if s.strip()]
TICKERS = [t if t.endswith(".AX") else t + ".AX" for t in TICKERS]
VAS = "VAS.AX"

# CommSec tiered brokerage
def commsec(value):
    if value <= 0: return 0.0
    if value <= 1000: return 10.00
    if value <= 10000: return 19.95
    if value <= 25000: return 29.95
    return value * 0.0012

# ── Strategy ────────────────────────────────────────────
MAX_POS         = 2
RISK_PCT        = 0.035
SL_ATR          = 2.5
BE_TRIGGER_ATR  = 2.0     # move to breakeven when price reaches entry + 2×ATR
TRAIL_ATR       = 2.0     # trail width after breakeven
MAX_HOLD        = 40
PULLBACK_PCT    = 0.035
MIN_POS_VALUE   = 2500
MAX_TICKER_TRADES = 3

# Regime thresholds
REGIME_FULL     = 80      # full trading
REGIME_SELECTIVE = 65     # selective trading
PROB_FULL       = 0.55    # probability threshold in full regime
PROB_SELECTIVE  = 0.68    # probability threshold in selective regime

# Safety nets
CB_PCT = -0.07
CB_DAYS = 20
LS_MAX = 3
LS_DAYS = 10

PERIODS = [
    ("P1: 2022-2024", "2022-01-01", "2024-12-31", "2021-12-31"),
    ("P2: 2025",      "2025-01-01", "2025-12-31", "2024-12-31"),
    ("P3: 2026 YTD",  "2026-01-01", "2026-12-31", "2025-12-31"),
]
CAPITAL = 8_000.0

# ╔══════════════════════════════════════════════════════════╗
# ║  FEATURES                                                ║
# ╚══════════════════════════════════════════════════════════╝

def build_features(df, vas_feats=None, breadth_series=None, universe_mom=None):
    """25 features: 20 standard + 5 market context."""
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    sma20  = c.rolling(20).mean()
    sma50  = c.rolling(50).mean()
    sma200 = c.rolling(200).mean()
    atr    = volatility.average_true_range(h, l, c, 14)

    feat = pd.DataFrame(index=df.index)
    # ── Standard 20 features ──
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

    # ── 5 Market Context features ──
    # Consolidation score: 10d range vs ATR (low = coiling for breakout)
    range_10d = h.rolling(10).max() - l.rolling(10).min()
    feat["consolidation"] = range_10d / (atr * 10 + 1e-10)

    if vas_feats is not None:
        # VAS momentum
        feat["vas_momentum"] = vas_feats["mom20"].reindex(df.index, method="ffill")
        # VAS position relative to SMA200
        feat["vas_position"] = vas_feats["pos200"].reindex(df.index, method="ffill")
    else:
        feat["vas_momentum"] = 0.0
        feat["vas_position"] = 0.0

    if breadth_series is not None:
        feat["mkt_breadth"] = breadth_series.reindex(df.index, method="ffill") / 100.0
    else:
        feat["mkt_breadth"] = 0.5

    if universe_mom is not None:
        # Relative strength: stock's 20d return vs universe median
        stock_mom = c.pct_change(20)
        uni_med = universe_mom.reindex(df.index, method="ffill")
        feat["relative_strength"] = stock_mom - uni_med
    else:
        feat["relative_strength"] = 0.0

    return feat

FEATURE_NAMES = None  # Will be set from first training


def build_target(df, fwd=20, min_ret=0.05, max_dd=0.04):
    """Target: 5% return in 20 days, max 4% drawdown.
    High hurdle → model learns BIG moves that justify CommSec fees."""
    c, l = df["Close"], df["Low"]
    fwd_ret = c.shift(-fwd) / c - 1
    fwd_dd = pd.Series(np.nan, index=df.index)
    for i in range(len(df) - fwd):
        window = l.iloc[i + 1 : i + fwd + 1]
        fwd_dd.iloc[i] = (window.min() / c.iloc[i]) - 1
    return ((fwd_ret > min_ret) & (fwd_dd > -max_dd)).astype(int)


# ╔══════════════════════════════════════════════════════════╗
# ║  DOWNLOAD & PREPARE                                     ║
# ╚══════════════════════════════════════════════════════════╝

print("╔══════════════════════════════════════════════════════════════╗")
print("║  DEEPQUANT V9 — NEVER-LOSE YEAR-BY-YEAR MODEL              ║")
print("║  5-Layer Architecture: Regime + ML + Filters + Mgmt + Safety║")
print("╚══════════════════════════════════════════════════════════════╝")
print(f"\n  CommSec: ≤$1K→$10 | $1K-$10K→$19.95 | $10K-$25K→$29.95 | >$25K→0.12%")
print(f"  Target: 20d fwd >5% return, <4% drawdown")
print(f"  Regime: score ≥{REGIME_FULL}→prob≥{PROB_FULL} | score {REGIME_SELECTIVE}-{REGIME_FULL}→prob≥{PROB_SELECTIVE} | <{REGIME_SELECTIVE}→NO TRADE")

t0 = time.time()
all_tickers = list(set(TICKERS + [VAS]))
print(f"\n  📡 Descargando {len(all_tickers)} tickers...")
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
print(f"  ✅ {len(data)} tickers")

# VAS features
vas_df = raw[VAS].dropna(subset=["Close"])
vas_df["SMA50"]  = vas_df["Close"].rolling(50).mean()
vas_df["SMA200"] = vas_df["Close"].rolling(200).mean()
vas_df["ATR14"]  = volatility.average_true_range(vas_df["High"], vas_df["Low"], vas_df["Close"], 14)
vas_df["ATR_pct"] = vas_df["ATR14"] / vas_df["Close"]
vas_df["ATR_avg"] = vas_df["ATR_pct"].rolling(100).mean()
vas_df["MOM20"]  = vas_df["Close"].pct_change(20)
print(f"  📊 VAS.AX: {len(vas_df)} rows")

# VAS features for ML
vas_feats = pd.DataFrame(index=vas_df.index)
vas_feats["mom20"] = vas_df["MOM20"]
vas_feats["pos200"] = (vas_df["Close"] - vas_df["SMA200"]) / vas_df["SMA200"]


# ╔══════════════════════════════════════════════════════════╗
# ║  MARKET BREADTH (vectorized)                             ║
# ╚══════════════════════════════════════════════════════════╝

print("  📊 Calculando breadth (% stocks > SMA50)...")
# Build matrix of close prices
all_dates = vas_df.index
close_matrix = pd.DataFrame(index=all_dates)
for t, df in data.items():
    sma50 = df["Close"].rolling(50).mean()
    above = (df["Close"] > sma50).astype(float)
    close_matrix[t] = above.reindex(all_dates)

breadth_50 = close_matrix.mean(axis=1) * 100  # % above SMA50
breadth_50 = breadth_50.ffill()
print(f"  ✅ Breadth calculado: {len(breadth_50)} días")

# Universe median momentum (for relative strength)
mom_matrix = pd.DataFrame(index=all_dates)
for t, df in data.items():
    mom_matrix[t] = df["Close"].pct_change(20).reindex(all_dates)
universe_mom_median = mom_matrix.median(axis=1)


# ╔══════════════════════════════════════════════════════════╗
# ║  REGIME SCORING                                          ║
# ╚══════════════════════════════════════════════════════════╝

def calc_regime_score(date):
    """Calculate regime score (0-100) for a given date."""
    mask = vas_df.index <= pd.Timestamp(date)
    if mask.sum() == 0:
        return 0, {}

    row = vas_df.loc[mask].iloc[-1]
    c = row["Close"]
    s50 = row["SMA50"] if pd.notna(row["SMA50"]) else 0
    s200 = row["SMA200"] if pd.notna(row["SMA200"]) else 0
    mom = row["MOM20"] if pd.notna(row["MOM20"]) else -1
    atr_pct = row["ATR_pct"] if pd.notna(row["ATR_pct"]) else 99
    atr_avg = row["ATR_avg"] if pd.notna(row["ATR_avg"]) else 99

    # Get breadth
    brd_mask = breadth_50.index <= pd.Timestamp(date)
    brd = breadth_50.loc[brd_mask].iloc[-1] if brd_mask.sum() > 0 else 0

    score = 0
    details = {}

    # VAS > SMA200 (+25)
    if s200 > 0 and c > s200:
        score += 25
        details["above_200"] = True
    else:
        details["above_200"] = False

    # VAS > SMA50 (+15)
    if s50 > 0 and c > s50:
        score += 15
        details["above_50"] = True
    else:
        details["above_50"] = False

    # Golden cross: SMA50 > SMA200 (+20)
    if s50 > 0 and s200 > 0 and s50 > s200:
        score += 20
        details["golden"] = True
    else:
        details["golden"] = False

    # VAS 20d momentum > 0 (+15)
    if mom > 0:
        score += 15
        details["mom_pos"] = True
    else:
        details["mom_pos"] = False

    # Breadth > 50% (+15)
    if brd > 50:
        score += 15
        details["breadth_ok"] = True
    else:
        details["breadth_ok"] = False

    # Low volatility: ATR% < average (+10)
    if atr_pct < atr_avg:
        score += 10
        details["low_vol"] = True
    else:
        details["low_vol"] = False

    details["score"] = score
    details["breadth"] = brd
    details["mom"] = mom * 100 if pd.notna(mom) else 0
    return score, details


# ╔══════════════════════════════════════════════════════════╗
# ║  SIMULATION                                              ║
# ╚══════════════════════════════════════════════════════════╝

def simulate_period(name, start, end, train_end, capital):
    global FEATURE_NAMES
    print(f"\n{'═'*70}")
    print(f"  ⏱  {name}  (train → {train_end})")
    print(f"{'═'*70}")

    train_cutoff = pd.Timestamp(train_end)
    sim_start, sim_end = pd.Timestamp(start), pd.Timestamp(end)

    # ── Train ──
    valid = [t for t in data if data[t].index.min() < train_cutoff - pd.Timedelta(days=365)]
    print(f"  📦 {len(valid)} tickers")
    models = {}

    print(f"  🧠 Training (target: 20d >5%, DD <4%)...")
    for i, t in enumerate(valid):
        df = data[t]
        tr = df[df.index <= train_cutoff].copy()
        if len(tr) < 300:
            continue
        feats = build_features(tr, vas_feats, breadth_50, universe_mom_median)
        tgt = build_target(tr)
        mask = feats.notna().all(axis=1) & tgt.notna()
        X, y = feats[mask], tgt[mask]
        if FEATURE_NAMES is None:
            FEATURE_NAMES = list(X.columns)
        if len(X) < 100 or y.sum() < 5:
            continue
        try:
            base = lgb.LGBMClassifier(
                n_estimators=500, max_depth=4, learning_rate=0.03,
                subsample=0.7, colsample_bytree=0.6, min_child_samples=30,
                reg_alpha=0.2, reg_lambda=2.0, verbose=-1, random_state=42,
                is_unbalance=True  # handle class imbalance
            )
            cal = CalibratedClassifierCV(base, cv=TimeSeriesSplit(3), method="isotonic")
            cal.fit(X, y)
            models[t] = cal
        except:
            pass
        print(f"\r   [{i+1}/{len(valid)}] {t:12s}", end="", flush=True)
    print(f"\n   ✅ Modelos: {len(models)}/{len(valid)}")

    # ── Pre-compute ──
    feat_cache, price_cache = {}, {}
    for t in models:
        df = data[t]
        feats = build_features(df, vas_feats, breadth_50, universe_mom_median)
        atr = volatility.average_true_range(df["High"], df["Low"], df["Close"], 14)
        sma20 = df["Close"].rolling(20).mean()
        sma50 = df["Close"].rolling(50).mean()
        feat_cache[t] = feats
        price_cache[t] = pd.DataFrame({
            "atr": atr, "sma20": sma20, "sma50": sma50,
            "close": df["Close"], "high": df["High"], "low": df["Low"],
            "volume": df["Volume"], "vol_avg20": df["Volume"].rolling(20).mean()
        }, index=df.index)

    # ── Sim ──
    print(f"\n{'─'*70}")
    print(f"  📈 {name}: {start} → {end} | ${capital:,.0f} | {len(models)} tickers")
    print(f"{'─'*70}")

    cash = capital
    positions = []
    trades = []
    equity_curve = []
    ticker_tc = {}
    total_comm = 0.0

    # Safety state
    eq_peak = capital
    cb_until = None
    consec_losses = 0
    ls_until = None

    # Regime counters
    regime_days = {"full": 0, "selective": 0, "skip": 0, "cb": 0, "ls": 0}
    # Year-by-year tracking
    yearly = {}

    # Trading days
    all_dates = set()
    for t in models:
        all_dates.update(data[t].index)
    trading_days = sorted([d for d in all_dates if sim_start <= d <= sim_end])

    for day in trading_days:
        year = day.year
        if year not in yearly:
            yearly[year] = {"start_eq": cash + sum(
                pos["shares"] * data[pos["ticker"]].loc[day, "Close"]
                for pos in positions if day in data[pos["ticker"]].index
            ), "trades": 0}

        score, regime = calc_regime_score(day)

        if score >= REGIME_FULL:
            regime_days["full"] += 1
            prob_thresh = PROB_FULL
        elif score >= REGIME_SELECTIVE:
            regime_days["selective"] += 1
            prob_thresh = PROB_SELECTIVE
        else:
            regime_days["skip"] += 1
            prob_thresh = None  # no trading

        # Safety overrides
        trading_ok = True
        if cb_until and day < cb_until:
            regime_days["cb"] += 1
            trading_ok = False
        elif ls_until and day < ls_until:
            regime_days["ls"] += 1
            trading_ok = False
        else:
            cb_until = None
            ls_until = None

        # ── EXIT ──
        closed = []
        for pos in positions:
            t = pos["ticker"]
            if day not in data[t].index:
                continue
            row = data[t].loc[day]
            price, low, high = row["Close"], row["Low"], row["High"]
            pos["days_held"] += 1
            if high > pos["high_water"]:
                pos["high_water"] = high

            # Breakeven trigger
            if not pos["at_be"]:
                if high >= pos["entry_price"] + BE_TRIGGER_ATR * pos["entry_atr"]:
                    pos["at_be"] = True
                    pos["stop"] = pos["entry_price"]  # breakeven
                    pos["trail_on"] = True

            # Trail after breakeven
            if pos["trail_on"]:
                new_stop = pos["high_water"] - TRAIL_ATR * pos["entry_atr"]
                if new_stop > pos["stop"]:
                    pos["stop"] = new_stop

            # Check stop hit
            if low <= pos["stop"]:
                ep = max(pos["stop"], low)
                ec = commsec(pos["shares"] * ep)
                gross = pos["shares"] * (ep - pos["entry_price"])
                net = gross - pos["entry_comm"] - ec
                total_comm += ec
                reason = "BE_STOP" if pos["at_be"] else "STOP"
                trades.append({
                    "ticker": t, "entry": pos["entry_date"], "exit": day,
                    "entry_p": pos["entry_price"], "exit_p": ep,
                    "shares": pos["shares"], "pnl": net, "reason": reason,
                    "days": pos["days_held"], "comm": pos["entry_comm"] + ec,
                    "gross": gross
                })
                cash += pos["shares"] * ep - ec
                consec_losses = consec_losses + 1 if net <= 0 else 0
                closed.append(pos)
                continue

            # Regime exit: if score drops below 50 and profitable
            if score < 50 and price > pos["entry_price"] * 1.005:
                ec = commsec(pos["shares"] * price)
                gross = pos["shares"] * (price - pos["entry_price"])
                net = gross - pos["entry_comm"] - ec
                total_comm += ec
                trades.append({
                    "ticker": t, "entry": pos["entry_date"], "exit": day,
                    "entry_p": pos["entry_price"], "exit_p": price,
                    "shares": pos["shares"], "pnl": net, "reason": "REGIME_EXIT",
                    "days": pos["days_held"], "comm": pos["entry_comm"] + ec,
                    "gross": gross
                })
                cash += pos["shares"] * price - ec
                consec_losses = 0 if net > 0 else consec_losses + 1
                closed.append(pos)
                continue

            # Max hold
            if pos["days_held"] >= MAX_HOLD:
                ec = commsec(pos["shares"] * price)
                gross = pos["shares"] * (price - pos["entry_price"])
                net = gross - pos["entry_comm"] - ec
                total_comm += ec
                trades.append({
                    "ticker": t, "entry": pos["entry_date"], "exit": day,
                    "entry_p": pos["entry_price"], "exit_p": price,
                    "shares": pos["shares"], "pnl": net, "reason": "TIME",
                    "days": pos["days_held"], "comm": pos["entry_comm"] + ec,
                    "gross": gross
                })
                cash += pos["shares"] * price - ec
                consec_losses = 0 if net > 0 else consec_losses + 1
                closed.append(pos)
                continue

        positions = [p for p in positions if p not in closed]

        # ── EQUITY ──
        port_val = cash + sum(
            pos["shares"] * data[pos["ticker"]].loc[day, "Close"]
            for pos in positions if day in data[pos["ticker"]].index
        )
        equity_curve.append((day, port_val))
        if port_val > eq_peak:
            eq_peak = port_val

        # Circuit breaker
        dd = (port_val - eq_peak) / eq_peak
        if dd <= CB_PCT and cb_until is None:
            cb_until = day + pd.Timedelta(days=CB_DAYS)
            for pos in positions:
                t = pos["ticker"]
                if day in data[t].index:
                    price = data[t].loc[day, "Close"]
                    ec = commsec(pos["shares"] * price)
                    gross = pos["shares"] * (price - pos["entry_price"])
                    net = gross - pos["entry_comm"] - ec
                    total_comm += ec
                    trades.append({
                        "ticker": t, "entry": pos["entry_date"], "exit": day,
                        "entry_p": pos["entry_price"], "exit_p": price,
                        "shares": pos["shares"], "pnl": net, "reason": "CIRCUIT_BRK",
                        "days": pos["days_held"], "comm": pos["entry_comm"] + ec,
                        "gross": gross
                    })
                    cash += pos["shares"] * price - ec
            positions = []
            continue

        # Loss streak
        if consec_losses >= LS_MAX and ls_until is None:
            ls_until = day + pd.Timedelta(days=LS_DAYS)

        # ── ENTRY ──
        if not trading_ok or prob_thresh is None or len(positions) >= MAX_POS:
            continue

        candidates = []
        for t in models:
            if any(p["ticker"] == t for p in positions):
                continue
            if ticker_tc.get(t, 0) >= MAX_TICKER_TRADES:
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

            if pd.isna(atr_val) or atr_val <= 0 or pd.isna(sma20) or pd.isna(sma50):
                continue

            # FILTER: Stock > SMA50 (uptrend)
            if price < sma50:
                continue

            # FILTER: Pullback near SMA20
            dist = abs(price - sma20) / sma20
            if dist > PULLBACK_PCT:
                continue
            if price < sma20 * 0.99:
                continue

            # FILTER: RSI 35-68
            rsi = f_row.get("rsi", 0.5)
            if pd.notna(rsi) and (rsi < 0.35 or rsi > 0.68):
                continue

            # FILTER: ADX > 0.15 (meaningful trend)
            adx = f_row.get("adx", 0.0)
            if pd.notna(adx) and adx < 0.15:
                continue

            # FILTER: Volume participation
            if pd.notna(vol) and pd.notna(vol_avg) and vol_avg > 0:
                if vol < 0.8 * vol_avg:
                    continue

            # FILTER: Volatility not extreme
            vr = f_row.get("vol_regime", 1.0)
            if pd.notna(vr) and vr > 1.5:
                continue

            # ML probability
            try:
                prob = models[t].predict_proba(f_row.values.reshape(1, -1))[0][1]
            except:
                continue
            if prob < prob_thresh:
                continue

            candidates.append((t, prob, price, atr_val))

        candidates.sort(key=lambda x: -x[1])

        for t, prob, price, atr_val in candidates:
            if len(positions) >= MAX_POS:
                break
            stop_dist = SL_ATR * atr_val
            risk_amt = cash * RISK_PCT
            shares = int(risk_amt / stop_dist)
            if shares < 1:
                continue
            value = shares * price
            if value < MIN_POS_VALUE:
                continue
            ec = commsec(value)
            if value + ec > cash:
                shares = int((cash - 25) * 0.95 / price)
                if shares < 1:
                    continue
                value = shares * price
                if value < MIN_POS_VALUE:
                    continue
                ec = commsec(value)

            # Commission-aware: edge > 2× round-trip
            rt_comm = ec + commsec(value)
            exp_edge = prob * 3.0 * atr_val * shares - (1 - prob) * stop_dist * shares
            if exp_edge < 2.0 * rt_comm:
                continue

            cash -= value + ec
            total_comm += ec
            ticker_tc[t] = ticker_tc.get(t, 0) + 1

            positions.append({
                "ticker": t, "entry_date": day, "entry_price": price,
                "shares": shares, "stop": price - stop_dist,
                "entry_atr": atr_val, "entry_comm": ec,
                "days_held": 0, "at_be": False,
                "trail_on": False, "high_water": price
            })

    # Close remaining
    for pos in positions:
        t = pos["ticker"]
        last = data[t].index[data[t].index <= sim_end]
        if len(last) == 0:
            continue
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
            "gross": gross
        })
        cash += pos["shares"] * price - ec

    # ── Results ──
    final = cash
    roi = (final - capital) / capital * 100
    n = len(trades)

    if len(equity_curve) > 0:
        eq_s = pd.Series([e[1] for e in equity_curve], index=[e[0] for e in equity_curve])
        dd = ((eq_s / eq_s.cummax()) - 1).min() * 100
    else:
        dd = 0

    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    wr = len(wins) / n * 100 if n else 0
    avg_w = np.mean([t["pnl"] for t in wins]) if wins else 0
    avg_l = np.mean([abs(t["pnl"]) for t in losses]) if losses else 0
    pf = sum(t["pnl"] for t in wins) / abs(sum(t["pnl"] for t in losses)) if losses and sum(t["pnl"] for t in losses) != 0 else 99
    rr = avg_w / avg_l if avg_l > 0 else 99
    gross_pnl = sum(t["gross"] for t in trades)
    total_comms = sum(t["comm"] for t in trades)

    print(f"\n  Régimen: {regime_days['full']}d Full, {regime_days['selective']}d Selective, {regime_days['skip']}d Skip")
    if regime_days['cb'] > 0:
        print(f"  🛑 Circuit breaker: {regime_days['cb']}d")
    if regime_days['ls'] > 0:
        print(f"  ⏸️  Loss streak pause: {regime_days['ls']}d")

    print(f"\n  💰 {name}:")
    print(f"     ${capital:,.0f} → ${final:,.2f}")
    print(f"     ROI: {'+' if roi >= 0 else ''}{roi:.2f}% | DD: {abs(dd):.2f}%")
    print(f"     Trades: {n}")
    if n > 0:
        print(f"     WR: {wr:.1f}% | PF: {pf:.2f} | R:R: {rr:.2f}")
        print(f"     Avg W: ${avg_w:.2f} | Avg L: ${avg_l:.2f}")
        print(f"     💵 GROSS: ${gross_pnl:+,.2f}")
        print(f"     💸 CommSec: ${total_comms:,.2f} ({total_comms/capital*100:.1f}% cap)")
        print(f"     📊 NET: ${final - capital:+,.2f}")
        if abs(gross_pnl) > 0:
            print(f"     📉 Commission drag: {total_comms/abs(gross_pnl)*100:.0f}%")

        # Breakeven stop analysis
        be_stops = [t for t in trades if t["reason"] == "BE_STOP"]
        stops = [t for t in trades if t["reason"] == "STOP"]
        if be_stops:
            print(f"     🎯 BE stops: {len(be_stops)}× avg gross ${np.mean([t['gross'] for t in be_stops]):+.2f}")
        if stops:
            print(f"     ❌ Full stops: {len(stops)}× avg gross ${np.mean([t['gross'] for t in stops]):+.2f}")

        # Exit breakdown
        reasons = {}
        for t in trades:
            r = t["reason"]
            if r not in reasons:
                reasons[r] = {"n": 0, "pnl": 0, "gross": 0, "comm": 0, "wins": 0}
            reasons[r]["n"] += 1
            reasons[r]["pnl"] += t["pnl"]
            reasons[r]["gross"] += t["gross"]
            reasons[r]["comm"] += t["comm"]
            if t["pnl"] > 0:
                reasons[r]["wins"] += 1
        print("     Salidas:")
        for r, v in sorted(reasons.items(), key=lambda x: -x[1]["n"]):
            wr_r = v["wins"] / v["n"] * 100 if v["n"] > 0 else 0
            print(f"       {r:14s} {v['n']:3d}× WR:{wr_r:>4.0f}% Net:${v['pnl']:>+8,.2f} Gross:${v['gross']:>+8,.2f} Comm:${v['comm']:>6,.2f}")

    # ── Year-by-year breakdown ──
    if n > 0:
        print(f"\n  📅 AÑO POR AÑO:")
        trade_df = pd.DataFrame(trades)
        trade_df["year"] = pd.to_datetime(trade_df["entry"]).dt.year
        for yr in sorted(trade_df["year"].unique()):
            yr_trades = trade_df[trade_df["year"] == yr]
            yr_n = len(yr_trades)
            yr_net = yr_trades["pnl"].sum()
            yr_gross = yr_trades["gross"].sum()
            yr_comm = yr_trades["comm"].sum()
            yr_wins = (yr_trades["pnl"] > 0).sum()
            yr_wr = yr_wins / yr_n * 100 if yr_n > 0 else 0
            status = "✅" if yr_net >= 0 else "❌"
            print(f"     {status} {yr}: {yr_n:3d} trades | NET:${yr_net:>+8,.2f} | Gross:${yr_gross:>+8,.2f} | Comm:${yr_comm:>6,.2f} | WR:{yr_wr:.0f}%")

    return {
        "name": name, "capital": capital, "final": final,
        "roi": roi, "dd": dd, "trades": n, "wr": wr, "pf": pf, "rr": rr,
        "gross": gross_pnl, "comm": total_comms,
        "trade_list": trades, "equity": equity_curve
    }


# ╔══════════════════════════════════════════════════════════╗
# ║  MAIN                                                    ║
# ╚══════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    results = []
    for name, start, end, train_end in PERIODS:
        r = simulate_period(name, start, end, train_end, CAPITAL)
        results.append(r)

    elapsed = (time.time() - t0) / 60
    now = dt.datetime.now().strftime("%Y%m%d_%H%M")

    print(f"\n\n{'═'*75}")
    print(f"  📊 RESUMEN — DEEPQUANT V9 (NEVER-LOSE YEAR-BY-YEAR)")
    print(f"{'═'*75}")
    hdr = f"  {'Período':20s} {'Cap':>10s} {'ROI':>8s} {'DD':>7s} {'Tr':>4s} {'WR':>6s} {'PF':>5s} {'R:R':>5s} {'Gross':>10s} {'Comm':>8s}"
    print(hdr)
    print(f"  {'─'*75}")
    cum = CAPITAL
    for r in results:
        cum *= (1 + r["roi"] / 100)
        print(f"  {r['name']:20s} ${r['final']:>8,.2f} {r['roi']:>+7.2f}% {abs(r['dd']):>5.2f}% {r['trades']:>4d} {r['wr']:>5.1f}% {r['pf']:>4.2f} {r['rr']:>4.2f} ${r['gross']:>+8,.0f} ${r['comm']:>6,.0f}")
    cum_roi = (cum / CAPITAL - 1) * 100
    print(f"  {'─'*75}")
    print(f"  ACUMULADO: ${cum:,.2f} | ROI: {cum_roi:+.2f}%")

    # Full version comparison
    print(f"\n\n{'═'*85}")
    print(f"  🏆 ALL VERSIONS COMPARISON:")
    print(f"{'═'*85}")
    print(f"  {'Ver':5s} {'P1 ROI':>8s} {'P2 ROI':>8s} {'P3 ROI':>8s} {'Acum':>8s} {'P2 WR':>6s} {'P2 PF':>6s} {'2022':>7s} {'2023':>7s} {'2024':>7s}")
    print(f"  {'─'*5} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*6} {'─'*6} {'─'*7} {'─'*7} {'─'*7}")
    print(f"  V5   -58.71%   +1.94%   -0.33%  -58.07%  62.9%   1.16    N/A     N/A     N/A")
    print(f"  V7   -47.68%  +19.41%   -8.75%  -42.99%  61.9%   2.36    N/A     N/A     N/A")
    p1, p2, p3 = results[0], results[1], results[2]
    # Get year-by-year from P1
    yr_str = {"2022": "N/A", "2023": "N/A", "2024": "N/A"}
    if p1["trade_list"]:
        trade_df = pd.DataFrame(p1["trade_list"])
        trade_df["year"] = pd.to_datetime(trade_df["entry"]).dt.year
        for yr in [2022, 2023, 2024]:
            yr_t = trade_df[trade_df["year"] == yr]
            if len(yr_t) > 0:
                yr_net = yr_t["pnl"].sum()
                yr_str[str(yr)] = f"${yr_net:+.0f}"
            else:
                yr_str[str(yr)] = "$0"
    print(f"  V9   {p1['roi']:>+7.2f}% {p2['roi']:>+7.2f}% {p3['roi']:>+7.2f}% {cum_roi:>+7.2f}% {p2['wr']:>5.1f}% {p2['pf']:>5.2f} {yr_str['2022']:>7s} {yr_str['2023']:>7s} {yr_str['2024']:>7s}")
    print(f"{'═'*85}")

    print(f"\n  📌 V9 KEY DESIGN:")
    print(f"  • 5-layer architecture: Regime → ML → Filters → Management → Safety")
    print(f"  • Regime score (0-100) from VAS + breadth → dynamic prob threshold")
    print(f"  • 25 features including market context (VAS, breadth, relative strength)")
    print(f"  • Target: 5% in 20d with max 4% DD (high hurdle for quality signals)")
    print(f"  • Breakeven stop at +2×ATR → trail 2×ATR (zero-loss after BE)")
    print(f"  • CommSec real fees: $10/$19.95/$29.95/0.12%")
    print(f"  • Circuit breaker -7% → cash 20d | Loss streak 3× → pause 10d")
    print(f"  • Max 2 positions, max 3 trades/ticker, min $2,500/position")

    print(f"\n  ⏱ {elapsed:.1f} min")

    # Save
    fname = f"sim_v9_{now}.xlsx"
    try:
        with pd.ExcelWriter(fname) as writer:
            for r in results:
                sname = r["name"][:12].replace(":", "").replace(" ", "_")
                df_trades = pd.DataFrame(r["trade_list"])
                if len(df_trades) > 0:
                    df_trades.to_excel(writer, sheet_name=sname, index=False)
                df_eq = pd.DataFrame(r["equity"], columns=["date", "equity"])
                df_eq.to_excel(writer, sheet_name=f"{sname}_eq", index=False)
        print(f"  💾 {fname}")
    except Exception as e:
        print(f"  ⚠ {e}")

    print()
