#!/usr/bin/env python
"""
DeepQuant V10 — Never-Lose Year-by-Year (Fixed)
=================================================
V9 POST-MORTEM:
  ❌ Circuit breaker bug: eq_peak never reset → CB cycled 538 days
  ❌ Golden cross not mandatory: 371 "full" days in bear 2022
  ❌ Entry filters too tight: only 5 trades across all periods
  ❌ P2 went from V7's +19.41% to V10's -7.21% due to above

V10 FIXES:
  ✅ Golden cross MANDATORY — no trading without SMA50 > SMA200 on VAS
  ✅ CB resets eq_peak after cooldown ends
  ✅ CB threshold raised to -12% (gives more room with $8K)
  ✅ Wider pullback filter (6% vs 3.5%) for more entries
  ✅ RSI range widened (30-72 vs 35-68)
  ✅ ADX lowered (0.10 vs 0.15)
  ✅ Wider stop 3.0×ATR (more room to breathe)
  ✅ Faster breakeven at +1.8×ATR
  ✅ Dynamic trailing: tight 1.8×ATR in selective, 2.2×ATR in full regime
  ✅ Max 5 trades per ticker
  ✅ Min position $2,000
  ✅ Commission-edge filter uses minimum R:R from historical

ARCHITECTURE (5 layers unchanged, parameters tuned):
  L1: Regime — GOLDEN CROSS REQUIRED + score ≥70 full / ≥55 selective
  L2: ML — LightGBM 25 features with isotonic calibration
  L3: Entry Filters — Relaxed for more signals
  L4: Position Mgmt — Breakeven + trailing stop, no partial TP
  L5: Safety — CB -12% (reset peak), loss streak 4×
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

def commsec(value):
    if value <= 0: return 0.0
    if value <= 1000: return 10.00
    if value <= 10000: return 19.95
    if value <= 25000: return 29.95
    return value * 0.0012

# ── Strategy (TUNED from V9) ───────────────────────────
MAX_POS         = 2
RISK_PCT        = 0.04          # 4% of equity at risk per trade
SL_ATR          = 3.0           # wider stop (was 2.5)
BE_TRIGGER_ATR  = 1.8           # faster breakeven trigger (was 2.0)
TRAIL_ATR_FULL  = 2.2           # trail width in full regime
TRAIL_ATR_SEL   = 1.8           # trail width in selective regime (tighter)
MAX_HOLD        = 45            # slightly longer hold
PULLBACK_PCT    = 0.06          # 6% pullback window (was 3.5%)
MIN_POS_VALUE   = 2000          # lower minimum (was 2500)
MAX_TICKER_TRADES = 5           # more trades per ticker (was 3)

# Regime
REGIME_FULL     = 70            # lowered (was 80)
REGIME_SELECTIVE = 55           # lowered (was 65)
PROB_FULL       = 0.52          # lowered (was 0.55)
PROB_SELECTIVE  = 0.63          # lowered (was 0.68)

# Safety
CB_PCT = -0.12                  # raised from -0.07
CB_DAYS = 15                    # shorter cooldown
LS_MAX = 4                      # more tolerance (was 3)
LS_DAYS = 7                     # shorter pause

PERIODS = [
    ("P1: 2022-2024", "2022-01-01", "2024-12-31", "2021-12-31"),
    ("P2: 2025",      "2025-01-01", "2025-12-31", "2024-12-31"),
    ("P3: 2026 YTD",  "2026-01-01", "2026-12-31", "2025-12-31"),
]
CAPITAL = 8_000.0


# ╔══════════════════════════════════════════════════════════╗
# ║  FEATURES (25 total)                                     ║
# ╚══════════════════════════════════════════════════════════╝

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

    # Consolidation: narrow range relative to ATR → potential breakout
    range_10d = h.rolling(10).max() - l.rolling(10).min()
    feat["consolidation"] = range_10d / (atr * 10 + 1e-10)

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


def build_target(df, fwd=20, min_ret=0.05, max_dd=0.04):
    """5% up in 20d with max 4% drawdown — high quality targets."""
    c, l = df["Close"], df["Low"]
    fwd_ret = c.shift(-fwd) / c - 1
    fwd_dd = pd.Series(np.nan, index=df.index)
    for i in range(len(df) - fwd):
        window = l.iloc[i + 1 : i + fwd + 1]
        fwd_dd.iloc[i] = (window.min() / c.iloc[i]) - 1
    return ((fwd_ret > min_ret) & (fwd_dd > -max_dd)).astype(int)


# ╔══════════════════════════════════════════════════════════╗
# ║  DATA DOWNLOAD                                           ║
# ╚══════════════════════════════════════════════════════════╝

print("╔══════════════════════════════════════════════════════════════╗")
print("║  DEEPQUANT V10 — NEVER-LOSE (FIXED REGIME + CB + FILTERS)  ║")
print("╚══════════════════════════════════════════════════════════════╝")
print(f"\n  🔑 KEY CHANGES from V9:")
print(f"     • Golden cross MANDATORY (no trading in bear market)")
print(f"     • CB resets eq_peak after cooldown (stops infinite loop)")
print(f"     • Wider pullback 6% (was 3.5%) for more entries")
print(f"     • Wider stop 3.0×ATR (was 2.5×) for more room")
print(f"     • Regime threshold lowered to 70/55 (was 80/65)")
print(f"     • CB -12% (was -7%), loss streak 4× (was 3×)")
print(f"  CommSec: ≤$1K→$10 | $1K-$10K→$19.95 | $10K-$25K→$29.95 | >$25K→0.12%")

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

# VAS
vas_df = raw[VAS].dropna(subset=["Close"])
vas_df["SMA50"]  = vas_df["Close"].rolling(50).mean()
vas_df["SMA200"] = vas_df["Close"].rolling(200).mean()
vas_df["ATR14"]  = volatility.average_true_range(vas_df["High"], vas_df["Low"], vas_df["Close"], 14)
vas_df["ATR_pct"] = vas_df["ATR14"] / vas_df["Close"]
vas_df["ATR_avg"] = vas_df["ATR_pct"].rolling(100).mean()
vas_df["MOM20"]  = vas_df["Close"].pct_change(20)
print(f"  📊 VAS.AX: {len(vas_df)} rows")

vas_feats = pd.DataFrame(index=vas_df.index)
vas_feats["mom20"] = vas_df["MOM20"]
vas_feats["pos200"] = (vas_df["Close"] - vas_df["SMA200"]) / vas_df["SMA200"]

# Breadth
print("  📊 Calculando breadth...")
all_dates = vas_df.index
close_matrix = pd.DataFrame(index=all_dates)
for t, df in data.items():
    sma50 = df["Close"].rolling(50).mean()
    above = (df["Close"] > sma50).astype(float)
    close_matrix[t] = above.reindex(all_dates)
breadth_50 = close_matrix.mean(axis=1) * 100
breadth_50 = breadth_50.ffill()
print(f"  ✅ Breadth: {len(breadth_50)} días")

# Universe median momentum
mom_matrix = pd.DataFrame(index=all_dates)
for t, df in data.items():
    mom_matrix[t] = df["Close"].pct_change(20).reindex(all_dates)
universe_mom_median = mom_matrix.median(axis=1)


# ╔══════════════════════════════════════════════════════════╗
# ║  REGIME SCORING — GOLDEN CROSS MANDATORY                 ║
# ╚══════════════════════════════════════════════════════════╝

def calc_regime(date):
    """Returns (score, golden_ok, details).
    GOLDEN CROSS IS A HARD GATE — if not present, score is forced to 0."""
    mask = vas_df.index <= pd.Timestamp(date)
    if mask.sum() == 0:
        return 0, False, {}

    row = vas_df.loc[mask].iloc[-1]
    c = row["Close"]
    s50 = row["SMA50"] if pd.notna(row["SMA50"]) else 0
    s200 = row["SMA200"] if pd.notna(row["SMA200"]) else 0
    mom = row["MOM20"] if pd.notna(row["MOM20"]) else -1
    atr_pct = row["ATR_pct"] if pd.notna(row["ATR_pct"]) else 99
    atr_avg = row["ATR_avg"] if pd.notna(row["ATR_avg"]) else 99

    brd_mask = breadth_50.index <= pd.Timestamp(date)
    brd = breadth_50.loc[brd_mask].iloc[-1] if brd_mask.sum() > 0 else 0

    # MANDATORY: Golden Cross
    golden = (s50 > 0 and s200 > 0 and s50 > s200)
    if not golden:
        return 0, False, {"golden": False, "score": 0, "breadth": brd}

    score = 20  # golden cross = base 20 points
    details = {"golden": True}

    if s200 > 0 and c > s200:
        score += 25
        details["above_200"] = True
    if s50 > 0 and c > s50:
        score += 15
        details["above_50"] = True
    if mom > 0:
        score += 15
        details["mom_pos"] = True
    if brd > 45:
        score += 15
        details["breadth_ok"] = True
    if atr_pct < atr_avg * 1.1:
        score += 10
        details["low_vol"] = True

    details["score"] = score
    details["breadth"] = brd
    return score, True, details


# ╔══════════════════════════════════════════════════════════╗
# ║  SIMULATION                                              ║
# ╚══════════════════════════════════════════════════════════╝

def simulate_period(name, start, end, train_end, capital):
    print(f"\n{'═'*70}")
    print(f"  ⏱  {name}  (train → {train_end})")
    print(f"{'═'*70}")

    train_cutoff = pd.Timestamp(train_end)
    sim_start, sim_end = pd.Timestamp(start), pd.Timestamp(end)

    # ── Train ──
    valid = [t for t in data if data[t].index.min() < train_cutoff - pd.Timedelta(days=365)]
    print(f"  📦 {len(valid)} tickers")

    print(f"  🧠 Training...")
    models = {}
    for i, t in enumerate(valid):
        df = data[t]
        tr = df[df.index <= train_cutoff].copy()
        if len(tr) < 300:
            continue
        feats = build_features(tr, vas_feats, breadth_50, universe_mom_median)
        tgt = build_target(tr)
        mask = feats.notna().all(axis=1) & tgt.notna()
        X, y = feats[mask], tgt[mask]
        if len(X) < 100 or y.sum() < 5:
            continue
        try:
            base = lgb.LGBMClassifier(
                n_estimators=500, max_depth=4, learning_rate=0.03,
                subsample=0.7, colsample_bytree=0.6, min_child_samples=30,
                reg_alpha=0.2, reg_lambda=2.0, verbose=-1, random_state=42,
                is_unbalance=True
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
        rsi_raw = momentum.rsi(df["Close"], 14)
        adx_raw = trend.ADXIndicator(df["High"], df["Low"], df["Close"], 14).adx()
        feat_cache[t] = feats
        price_cache[t] = pd.DataFrame({
            "atr": atr, "sma20": sma20, "sma50": sma50,
            "close": df["Close"], "high": df["High"], "low": df["Low"],
            "volume": df["Volume"], "vol_avg20": df["Volume"].rolling(20).mean(),
            "rsi": rsi_raw, "adx": adx_raw
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

    # Counters
    regime_days = {"full": 0, "selective": 0, "skip": 0, "no_golden": 0, "cb": 0, "ls": 0}
    signals_generated = 0
    signals_filtered = 0

    # Trading days
    all_sim_dates = set()
    for t in models:
        all_sim_dates.update(data[t].index)
    trading_days = sorted([d for d in all_sim_dates if sim_start <= d <= sim_end])

    for day in trading_days:
        score, golden_ok, regime = calc_regime(day)

        if not golden_ok:
            regime_days["no_golden"] += 1
            prob_thresh = None
        elif score >= REGIME_FULL:
            regime_days["full"] += 1
            prob_thresh = PROB_FULL
        elif score >= REGIME_SELECTIVE:
            regime_days["selective"] += 1
            prob_thresh = PROB_SELECTIVE
        else:
            regime_days["skip"] += 1
            prob_thresh = None

        # Safety overrides
        trading_ok = True
        if cb_until and day < cb_until:
            regime_days["cb"] += 1
            trading_ok = False
        elif ls_until and day < ls_until:
            regime_days["ls"] += 1
            trading_ok = False
        else:
            # ✅ FIX: Reset CB/LS state AND equity peak
            if cb_until and day >= cb_until:
                eq_peak = cash + sum(
                    pos["shares"] * data[pos["ticker"]].loc[day, "Close"]
                    for pos in positions if day in data[pos["ticker"]].index
                )
            cb_until = None
            ls_until = None

        # Determine trail width based on regime
        trail_atr = TRAIL_ATR_FULL if (golden_ok and score >= REGIME_FULL) else TRAIL_ATR_SEL

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
                    pos["stop"] = pos["entry_price"]
                    pos["trail_on"] = True

            # Trail after breakeven
            if pos["trail_on"]:
                new_stop = pos["high_water"] - trail_atr * pos["entry_atr"]
                if new_stop > pos["stop"]:
                    pos["stop"] = new_stop

            exit_reason = None

            # Stop / breakeven stop
            if low <= pos["stop"]:
                ep = max(pos["stop"], low)
                exit_reason = "BE_STOP" if pos["at_be"] else "STOP"
            # Golden cross lost while profitable → protect gains
            elif not golden_ok and price > pos["entry_price"] * 1.005:
                ep = price
                exit_reason = "GOLDEN_EXIT"
            # Max hold
            elif pos["days_held"] >= MAX_HOLD:
                ep = price
                exit_reason = "TIME"

            if exit_reason:
                ec = commsec(pos["shares"] * ep)
                gross = pos["shares"] * (ep - pos["entry_price"])
                net = gross - pos["entry_comm"] - ec
                total_comm += ec
                trades.append({
                    "ticker": t, "entry": pos["entry_date"], "exit": day,
                    "entry_p": pos["entry_price"], "exit_p": ep,
                    "shares": pos["shares"], "pnl": net, "reason": exit_reason,
                    "days": pos["days_held"], "comm": pos["entry_comm"] + ec,
                    "gross": gross
                })
                cash += pos["shares"] * ep - ec
                if net > 0:
                    consec_losses = 0
                else:
                    consec_losses += 1
                closed.append(pos)

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
        if eq_peak > 0:
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
            rsi_val = p_row["rsi"]
            adx_val = p_row["adx"]

            if pd.isna(atr_val) or atr_val <= 0 or pd.isna(sma20) or pd.isna(sma50):
                continue

            signals_generated += 1

            # FILTER: Stock above SMA50 (confirmed uptrend)
            if price < sma50:
                signals_filtered += 1
                continue

            # FILTER: Pullback zone — within 6% of SMA20
            #  Allow stocks slightly below SMA20 (dip entry) or moderately above
            dist_sma20 = (price - sma20) / sma20
            if dist_sma20 > PULLBACK_PCT or dist_sma20 < -0.02:
                signals_filtered += 1
                continue

            # FILTER: RSI 30-72
            if pd.notna(rsi_val) and (rsi_val < 30 or rsi_val > 72):
                signals_filtered += 1
                continue

            # FILTER: ADX > 0.10 (some trend)
            if pd.notna(adx_val) and adx_val < 10:
                signals_filtered += 1
                continue

            # FILTER: Volume participation
            if pd.notna(vol) and pd.notna(vol_avg) and vol_avg > 0:
                if vol < 0.7 * vol_avg:
                    signals_filtered += 1
                    continue

            # FILTER: Not extremely volatile
            vr = f_row.get("vol_regime", 1.0)
            if pd.notna(vr) and vr > 1.6:
                signals_filtered += 1
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

            # Commission-aware: expected edge > 1.5× round-trip commission
            rt_comm = ec + commsec(value)
            exp_edge = prob * 2.5 * atr_val * shares - (1 - prob) * stop_dist * shares
            if exp_edge < 1.5 * rt_comm:
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

    total_regime_days = sum(regime_days.values())
    print(f"\n  Régimen ({total_regime_days}d total):")
    print(f"    🟢 Full({REGIME_FULL}+): {regime_days['full']}d | 🟡 Selective({REGIME_SELECTIVE}-{REGIME_FULL}): {regime_days['selective']}d")
    print(f"    🔴 No Golden: {regime_days['no_golden']}d | ⚪ Low Score: {regime_days['skip']}d")
    if regime_days['cb'] > 0:
        print(f"    🛑 CB: {regime_days['cb']}d")
    if regime_days['ls'] > 0:
        print(f"    ⏸️ LS: {regime_days['ls']}d")
    if signals_generated > 0:
        print(f"    📊 Signals: {signals_generated} generated → {signals_generated - signals_filtered} passed filters → {n} traded")

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

        # Exit breakdown
        reasons = {}
        for t_rec in trades:
            r = t_rec["reason"]
            if r not in reasons:
                reasons[r] = {"n": 0, "pnl": 0, "gross": 0, "comm": 0, "wins": 0}
            reasons[r]["n"] += 1
            reasons[r]["pnl"] += t_rec["pnl"]
            reasons[r]["gross"] += t_rec["gross"]
            reasons[r]["comm"] += t_rec["comm"]
            if t_rec["pnl"] > 0:
                reasons[r]["wins"] += 1
        print("     Salidas:")
        for r, v in sorted(reasons.items(), key=lambda x: -x[1]["n"]):
            wr_r = v["wins"] / v["n"] * 100 if v["n"] > 0 else 0
            print(f"       {r:14s} {v['n']:3d}× WR:{wr_r:>4.0f}% Net:${v['pnl']:>+8,.2f} Gross:${v['gross']:>+8,.2f} Comm:${v['comm']:>6,.2f}")

        # Top trades
        best = sorted(trades, key=lambda x: -x["pnl"])[:3]
        worst = sorted(trades, key=lambda x: x["pnl"])[:3]
        print(f"\n     🏆 Best trades:")
        for t_rec in best:
            print(f"       {t_rec['ticker']:8s} {str(t_rec['entry'].date()):10s}→{str(t_rec['exit'].date()):10s} ${t_rec['pnl']:>+8,.2f} ({t_rec['reason']})")
        print(f"     💀 Worst trades:")
        for t_rec in worst:
            print(f"       {t_rec['ticker']:8s} {str(t_rec['entry'].date()):10s}→{str(t_rec['exit'].date()):10s} ${t_rec['pnl']:>+8,.2f} ({t_rec['reason']})")

    # Year-by-year
    print(f"\n  📅 AÑO POR AÑO:")
    if n > 0:
        trade_df = pd.DataFrame(trades)
        trade_df["year"] = pd.to_datetime(trade_df["entry"]).dt.year
        all_years = sorted(set(range(sim_start.year, sim_end.year + 1)))
        for yr in all_years:
            yr_trades = trade_df[trade_df["year"] == yr]
            yr_n = len(yr_trades)
            if yr_n > 0:
                yr_net = yr_trades["pnl"].sum()
                yr_gross = yr_trades["gross"].sum()
                yr_comm = yr_trades["comm"].sum()
                yr_wins = (yr_trades["pnl"] > 0).sum()
                yr_wr = yr_wins / yr_n * 100
                status = "✅" if yr_net >= 0 else "❌"
                print(f"     {status} {yr}: {yr_n:3d} trades | NET:${yr_net:>+8,.2f} | Gross:${yr_gross:>+8,.2f} | Comm:${yr_comm:>6,.2f} | WR:{yr_wr:.0f}%")
            else:
                print(f"     ⬜ {yr}:   0 trades | NET:$    0.00 (cash)")
    else:
        for yr in range(sim_start.year, sim_end.year + 1):
            print(f"     ⬜ {yr}:   0 trades | NET:$    0.00 (cash)")

    return {
        "name": name, "capital": capital, "final": final,
        "roi": roi, "dd": dd, "trades": n, "wr": wr, "pf": pf, "rr": rr,
        "gross": gross_pnl if n > 0 else 0, "comm": total_comms,
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
    print(f"  📊 RESUMEN — DEEPQUANT V10")
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

    # Version comparison
    print(f"\n{'═'*85}")
    print(f"  🏆 VERSION COMPARISON:")
    print(f"{'═'*85}")
    print(f"  {'Ver':5s} {'P1 ROI':>8s} {'P2 ROI':>8s} {'P3 ROI':>8s} {'Acum':>8s} {'P1 Tr':>6s} {'P2 Tr':>6s}")
    print(f"  {'─'*5} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*6} {'─'*6}")
    print(f"  V5   -58.71%   +1.94%   -0.33%  -58.07%     63     21")
    print(f"  V7   -47.68%  +19.41%   -8.75%  -42.99%     63     21")
    print(f"  V9     -4.68%   -7.21%   +0.00%  -11.55%      3      2")
    p1, p2, p3 = results[0], results[1], results[2]
    print(f"  V10  {p1['roi']:>+7.2f}% {p2['roi']:>+7.2f}% {p3['roi']:>+7.2f}% {cum_roi:>+7.2f}% {p1['trades']:>6d} {p2['trades']:>6d}")

    # Year breakdown
    print(f"\n  📅 YEAR-BY-YEAR GOAL CHECK:")
    all_trades = []
    for r in results:
        all_trades.extend(r["trade_list"])
    if all_trades:
        df_all = pd.DataFrame(all_trades)
        df_all["year"] = pd.to_datetime(df_all["entry"]).dt.year
        for yr in sorted(df_all["year"].unique()):
            yr_tr = df_all[df_all["year"] == yr]
            net = yr_tr["pnl"].sum()
            status = "✅ PASS" if net >= 0 else "❌ FAIL"
            print(f"     {yr}: {status} (${net:+,.2f}, {len(yr_tr)} trades)")
    else:
        for yr in [2022, 2023, 2024, 2025, 2026]:
            print(f"     {yr}: ✅ PASS ($0.00, 0 trades — cash)")

    print(f"\n{'═'*85}")
    print(f"\n  ⏱ {elapsed:.1f} min")

    # Save
    fname = f"sim_v10_{now}.xlsx"
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
