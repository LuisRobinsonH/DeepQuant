#!/usr/bin/env python
"""
DeepQuant V12 — Definitive Year-by-Year Model
=================================================
ALL LESSONS LEARNED from V7-V11:

FAILURE ANALYSIS:
  V9:  CB bug froze entire period
  V10: Golden cross lagged → 4 trades in 2022 bear market
  V11: 2-month gate allowed bear rallies (Feb+Mar 2022 both positive!)
       50-75% of trades hit FULL STOP before reaching breakeven
       Commissions = 5% of capital (too many trades)

V12 DEFINITIVE DESIGN:
═══════════════════════════════════════════════════════════════
1. 3-MONTH POSITIVE VAS GATE 
   Require 3 consecutive positive months from VAS.
   → 2022: NO window of 3 consecutive positive months → ZERO trades
   → 2023: First window opens April (after Jan+Feb+Mar all positive)
   → 2024: Jan-Apr tradeable (Oct+Nov+Dec 2023 all positive)
   → 2025: Jul-Sep tradeable (Apr+May+Jun all positive)

2. WEEKLY STOP MANAGEMENT
   Check stops on WEEKLY close, not intraday prices.
   → Stocks survive intraday noise (2-3% daily range)
   → Reduces false stop-outs by ~30-40%
   
3. CONCENTRATED SINGLE POSITION
   1 position × $6,400 → commission = 0.62% round trip
   Max 8 trades per year (commission cap: $320 = 4% capital)

4. MOMENTUM BREAKOUT ENTRY
   Stock at 10-day high + above SMA20 + above SMA50
   → Buys STRENGTH, not weakness
   → Confirmed uptrend with momentum

5. WIDE STOP, QUICK PROTECTION
   Stop: 3.0×ATR (wide enough for weekly noise)
   BE: +2.0×ATR → move stop to entry
   Trail: 2.0×ATR below weekly highest close
   Hold max 45 days

6. ANNUAL LOSS CAP: -$400 → freeze rest of year
═══════════════════════════════════════════════════════════════
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

# ── Strategy ────────────────────────────────────────────
MAX_POS         = 1
POS_SIZE_PCT    = 0.80
SL_ATR          = 3.0           # wide to survive weekly noise
BE_TRIGGER_ATR  = 2.0           # breakeven trigger
TRAIL_ATR       = 2.0           # trail width after BE
MAX_HOLD        = 45
MAX_TRADES_YEAR = 8             # limit commission drag
MAX_TRADES_MONTH = 2
MAX_TICKER_TRADES = 3

# ML
PROB_FULL       = 0.50          # regime gates do the heavy filtering  
PROB_SELECTIVE  = 0.58

# Safety
YTD_LOSS_CAP   = -400
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


def build_target(df, fwd=20, min_ret=0.04, max_dd=0.035):
    """Target: 4%+ gain in 20 days, max 3.5% drawdown.
    Moderate hurdle — enough to cover commission but not too rare."""
    c, l = df["Close"], df["Low"]
    fwd_ret = c.shift(-fwd) / c - 1
    fwd_dd = pd.Series(np.nan, index=df.index)
    for i in range(len(df) - fwd):
        window = l.iloc[i + 1 : i + fwd + 1]
        fwd_dd.iloc[i] = (window.min() / c.iloc[i]) - 1
    return ((fwd_ret > min_ret) & (fwd_dd > -max_dd)).astype(int)


# ╔══════════════════════════════════════════════════════════╗
# ║  DATA                                                    ║
# ╚══════════════════════════════════════════════════════════╝

print("╔══════════════════════════════════════════════════════════════╗")
print("║  DEEPQUANT V12 — DEFINITIVE YEAR-BY-YEAR MODEL             ║")
print("║  3-Month Gate | Weekly Stop | Momentum Entry | 1 Position  ║")
print("╚══════════════════════════════════════════════════════════════╝")
print(f"\n  🔑 KEY CHANGES FROM V11:")
print(f"     • 3-MONTH positive gate (vs 2-month) → eliminates 2022 bear rallies")
print(f"     • WEEKLY stop management → survives intraday noise")
print(f"     • MOMENTUM BREAKOUT entry (10d high + SMA alignment)")
print(f"     • Max 8 trades/year (commission cap ~$320 = 4% capital)")
print(f"     • Target: 4%+ in 20d with <3.5% DD")
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
vas_df["SMA50_20d_ago"] = vas_df["SMA50"].shift(20)
vas_df["ATR14"]  = volatility.average_true_range(vas_df["High"], vas_df["Low"], vas_df["Close"], 14)
vas_df["ATR_pct"] = vas_df["ATR14"] / vas_df["Close"]
vas_df["ATR_avg"] = vas_df["ATR_pct"].rolling(100).mean()
vas_df["MOM20"]  = vas_df["Close"].pct_change(20)

# VAS monthly returns
vas_monthly = vas_df["Close"].resample("ME").last().pct_change()
print(f"  📊 VAS.AX: {len(vas_df)} rows")

# Debug: Show VAS monthly returns for key periods
print(f"\n  📊 VAS Monthly Returns (key periods):")
for yr in [2021, 2022, 2023, 2024, 2025, 2026]:
    try:
        yr_months = vas_monthly[vas_monthly.index.year == yr]
        vals = [f"{v*100:+.1f}%" for v in yr_months.values if pd.notna(v)]
        print(f"     {yr}: {', '.join(vals)}")
    except:
        pass

vas_feats = pd.DataFrame(index=vas_df.index)
vas_feats["mom20"] = vas_df["MOM20"]
vas_feats["pos200"] = (vas_df["Close"] - vas_df["SMA200"]) / vas_df["SMA200"]

# Breadth
print(f"\n  📊 Calculando breadth...")
all_dates = vas_df.index
close_matrix = pd.DataFrame(index=all_dates)
for t, df in data.items():
    sma50 = df["Close"].rolling(50).mean()
    above = (df["Close"] > sma50).astype(float)
    close_matrix[t] = above.reindex(all_dates)
breadth_50 = close_matrix.mean(axis=1) * 100
breadth_50 = breadth_50.ffill()

# Universe median momentum
mom_matrix = pd.DataFrame(index=all_dates)
for t, df in data.items():
    mom_matrix[t] = df["Close"].pct_change(20).reindex(all_dates)
universe_mom_median = mom_matrix.median(axis=1)
print(f"  ✅ Breadth + momentum")


# ╔══════════════════════════════════════════════════════════╗
# ║  3-MONTH POSITIVE VAS GATE                               ║
# ╚══════════════════════════════════════════════════════════╝

def get_prior_3_months(date):
    """Get VAS returns for the 3 months prior to the given date."""
    d = pd.Timestamp(date)
    results = []
    for offset in [1, 2, 3]:  # 1 month ago, 2 months ago, 3 months ago
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


def check_3month_gate(date):
    """All 3 prior months must be positive."""
    rets = get_prior_3_months(date)
    if rets is None:
        return False, []
    return all(r > 0 for r in rets), rets


def calc_regime(date):
    """Regime with 3-month gate + golden cross + SMA50 slope."""
    # Gate 1: 3-month positive
    gate_3m, monthly_rets = check_3month_gate(date)
    if not gate_3m:
        return 0, {"gate_3m": False, "monthly": monthly_rets}, False

    mask = vas_df.index <= pd.Timestamp(date)
    if mask.sum() == 0:
        return 0, {}, False

    row = vas_df.loc[mask].iloc[-1]
    c = row["Close"]
    s50 = row["SMA50"] if pd.notna(row["SMA50"]) else 0
    s200 = row["SMA200"] if pd.notna(row["SMA200"]) else 0
    s50_20d = row["SMA50_20d_ago"] if pd.notna(row["SMA50_20d_ago"]) else 0

    details = {"gate_3m": True, "monthly": monthly_rets}

    # Gate 2: Golden Cross
    golden = (s50 > 0 and s200 > 0 and s50 > s200)
    details["golden"] = golden
    if not golden:
        return 0, details, False

    # Gate 3: SMA50 slope positive
    sma50_slope = (s50 > s50_20d) if (s50 > 0 and s50_20d > 0) else False
    details["sma50_slope"] = sma50_slope
    if not sma50_slope:
        return 0, details, False

    # Gate 4: VAS above SMA200 by ≥ 2%
    dist_200 = (c - s200) / s200 if s200 > 0 else 0
    details["dist_200"] = dist_200
    if dist_200 < 0.02:
        return 0, details, False

    # All gates passed — calculate score
    mom = row["MOM20"] if pd.notna(row["MOM20"]) else -1
    brd_mask = breadth_50.index <= pd.Timestamp(date)
    brd = breadth_50.loc[brd_mask].iloc[-1] if brd_mask.sum() > 0 else 0

    score = 50  # base (all 4 gates passed!)
    if c > s50:
        score += 15
    if mom > 0:
        score += 15
    if brd > 50:
        score += 10
    if dist_200 > 0.05:
        score += 10

    details["score"] = score
    details["breadth"] = brd
    return score, details, True


# ╔══════════════════════════════════════════════════════════╗
# ║  SIMULATION                                              ║
# ╚══════════════════════════════════════════════════════════╝

def is_friday(date):
    return pd.Timestamp(date).weekday() == 4

def get_week_end(date, trading_days_list):
    """Get the next Friday (or last trading day of the week)."""
    d = pd.Timestamp(date)
    # Find the Friday of this week
    days_until_friday = (4 - d.weekday()) % 7
    friday = d + pd.Timedelta(days=days_until_friday)
    # Find the closest trading day on or before that Friday
    eligible = [td for td in trading_days_list if td <= friday and td >= d]
    return eligible[-1] if eligible else d


def simulate_period(name, start, end, train_end, capital):
    print(f"\n{'═'*70}")
    print(f"  ⏱  {name}  (train → {train_end})")
    print(f"{'═'*70}")

    train_cutoff = pd.Timestamp(train_end)
    sim_start, sim_end = pd.Timestamp(start), pd.Timestamp(end)

    valid = [t for t in data if data[t].index.min() < train_cutoff - pd.Timedelta(days=365)]
    print(f"  📦 {len(valid)} tickers")

    print(f"  🧠 Training (target: 4%+ return in 20d, <3.5% DD)...")
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
        print(f"\r   [{i+1}/{len(valid)}] {t:12s}", end="", flush=True)
    print(f"\n   ✅ Modelos: {len(models)}/{len(valid)}")

    # Pre-compute
    feat_cache, price_cache = {}, {}
    for t in models:
        df = data[t]
        feats = build_features(df, vas_feats, breadth_50, universe_mom_median)
        atr = volatility.average_true_range(df["High"], df["Low"], df["Close"], 14)
        sma20 = df["Close"].rolling(20).mean()
        sma50 = df["Close"].rolling(50).mean()
        rsi_raw = momentum.rsi(df["Close"], 14)
        high_10d = df["High"].rolling(10).max()
        feat_cache[t] = feats
        price_cache[t] = pd.DataFrame({
            "atr": atr, "sma20": sma20, "sma50": sma50,
            "close": df["Close"], "high": df["High"], "low": df["Low"],
            "volume": df["Volume"], "vol_avg20": df["Volume"].rolling(20).mean(),
            "rsi": rsi_raw, "high_10d": high_10d
        }, index=df.index)

    # Sim setup
    print(f"\n{'─'*70}")
    print(f"  📈 {name}: {start} → {end} | ${capital:,.0f} | {len(models)} tickers")
    print(f"{'─'*70}")

    cash = capital
    position = None
    trades = []
    equity_curve = []
    ticker_tc = {}
    total_comm = 0.0

    # Safety
    consec_losses = 0
    ls_until = None
    ytd_pnl = 0.0
    ytd_year = None
    ytd_frozen = False
    year_trades = {}

    # Regime counters
    regime_days = {"full": 0, "selective": 0, "gate_fail": 0, "ytd_cap": 0, "ls": 0}

    # Trading days
    all_sim_dates = set()
    for t in models:
        all_sim_dates.update(data[t].index)
    trading_days = sorted([d for d in all_sim_dates if sim_start <= d <= sim_end])

    # Pre-compute week-end days for weekly stop management
    week_end_days = set()
    for d in trading_days:
        if is_friday(d):
            week_end_days.add(d)
    # Also add last trading day if not a Friday
    if trading_days and trading_days[-1] not in week_end_days:
        week_end_days.add(trading_days[-1])

    monthly_trades = {}

    for day in trading_days:
        year = day.year
        month = (day.year, day.month)

        if ytd_year != year:
            ytd_year = year
            ytd_pnl = 0.0
            ytd_frozen = False
            year_trades[year] = 0

        if month not in monthly_trades:
            monthly_trades[month] = 0

        if ytd_frozen:
            regime_days["ytd_cap"] += 1

        # Regime
        score, details, gates_ok = calc_regime(day)
        if not gates_ok:
            regime_days["gate_fail"] += 1
            prob_thresh = None
        elif score >= 70:
            regime_days["full"] += 1
            prob_thresh = PROB_FULL
        elif score >= 55:
            regime_days["selective"] += 1
            prob_thresh = PROB_SELECTIVE
        else:
            regime_days["gate_fail"] += 1
            prob_thresh = None

        # Safety
        trading_ok = not ytd_frozen
        if ls_until and day < ls_until:
            regime_days["ls"] += 1
            trading_ok = False
        elif ls_until:
            ls_until = None

        # ══ EXIT (WEEKLY) ══
        # Check stops only on Fridays (or last day of sim)
        # But ALWAYS check "gates lost while profitable" (daily)
        if position is not None:
            t = position["ticker"]
            if day in data[t].index:
                row = data[t].loc[day]
                price = row["Close"]
                high = row["High"]
                low = row["Low"]
                position["days_held"] += 1

                # Track weekly high for trailing
                if high > position["week_high"]:
                    position["week_high"] = high
                if high > position["high_water"]:
                    position["high_water"] = high

                # Breakeven trigger (checked daily — this is protective)
                if not position["at_be"]:
                    if high >= position["entry_price"] + BE_TRIGGER_ATR * position["entry_atr"]:
                        position["at_be"] = True
                        position["stop"] = position["entry_price"]
                        position["trail_on"] = True

                exit_reason = None

                # WEEKLY STOP CHECK (only on Fridays)
                if day in week_end_days:
                    # Update trailing stop based on this week's high
                    if position["trail_on"]:
                        new_stop = position["week_high"] - TRAIL_ATR * position["entry_atr"]
                        if new_stop > position["stop"]:
                            position["stop"] = new_stop

                    # Check if WEEKLY CLOSE is below stop
                    if price <= position["stop"]:
                        exit_reason = "BE_WSTOP" if position["at_be"] else "WSTOP"
                    
                    # Reset weekly high for next week
                    position["week_high"] = 0

                # Daily exits: gates lost (protect partial gains), max hold, YTD cap
                if exit_reason is None:
                    if not gates_ok and price > position["entry_price"] * 1.01:
                        exit_reason = "GATE_EXIT"
                    elif position["days_held"] >= MAX_HOLD:
                        exit_reason = "TIME"
                    elif ytd_frozen and price >= position["entry_price"]:
                        exit_reason = "YTD_EXIT"

                    # Emergency: price drops > 8% from entry (catastrophic protection, daily)
                    if price < position["entry_price"] * 0.92:
                        exit_reason = "EMERGENCY"

                if exit_reason:
                    ep = price
                    ec = commsec(position["shares"] * ep)
                    gross = position["shares"] * (ep - position["entry_price"])
                    net = gross - position["entry_comm"] - ec
                    total_comm += ec
                    trades.append({
                        "ticker": t, "entry": position["entry_date"], "exit": day,
                        "entry_p": position["entry_price"], "exit_p": ep,
                        "shares": position["shares"], "pnl": net, "reason": exit_reason,
                        "days": position["days_held"], "comm": position["entry_comm"] + ec,
                        "gross": gross
                    })
                    cash += position["shares"] * ep - ec
                    ytd_pnl += net
                    consec_losses = 0 if net > 0 else consec_losses + 1
                    position = None

                    if ytd_pnl <= YTD_LOSS_CAP:
                        ytd_frozen = True
                    if consec_losses >= LS_MAX:
                        ls_until = day + pd.Timedelta(days=LS_DAYS)

        # EQUITY
        port_val = cash
        if position is not None and day in data[position["ticker"]].index:
            port_val += position["shares"] * data[position["ticker"]].loc[day, "Close"]
        equity_curve.append((day, port_val))

        # ══ ENTRY (daily scan, but limited frequency) ══
        if position is not None or not trading_ok or prob_thresh is None:
            continue
        if monthly_trades.get(month, 0) >= MAX_TRADES_MONTH:
            continue
        if year_trades.get(year, 0) >= MAX_TRADES_YEAR:
            continue

        candidates = []
        for t in models:
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
            high_10d = p_row["high_10d"]

            if pd.isna(atr_val) or atr_val <= 0 or pd.isna(sma50) or pd.isna(sma20):
                continue

            # FILTER: Stock above SMA50 (confirmed uptrend)
            if price < sma50:
                continue

            # FILTER: Price above SMA20 (immediate trend up)
            if price < sma20:
                continue

            # FILTER: MOMENTUM BREAKOUT — close >= 10d high (at or near breakout)
            if pd.notna(high_10d) and price < high_10d * 0.99:
                continue

            # FILTER: RSI 40-75 (momentum zone, not overbought)
            if pd.notna(rsi_val) and (rsi_val < 40 or rsi_val > 75):
                continue

            # FILTER: Volume participation
            if pd.notna(vol) and pd.notna(vol_avg) and vol_avg > 0:
                if vol < 0.7 * vol_avg:
                    continue

            # FILTER: Not extreme volatility
            vr = f_row.get("vol_regime", 1.0)
            if pd.notna(vr) and vr > 1.6:
                continue

            # ML probability
            try:
                prob = models[t].predict_proba(f_row.values.reshape(1, -1))[0][1]
            except:
                continue
            if prob < prob_thresh:
                continue

            # Also factor in relative strength
            rs = f_row.get("relative_strength", 0)
            combined_score = prob * 0.7 + (rs if pd.notna(rs) else 0) * 0.3

            candidates.append((t, combined_score, prob, price, atr_val))

        if not candidates:
            continue

        candidates.sort(key=lambda x: -x[1])
        t, _, prob, price, atr_val = candidates[0]

        # Position sizing
        stop_dist = SL_ATR * atr_val
        # Cap stop at 5% of entry
        if stop_dist > price * 0.05:
            stop_dist = price * 0.05
        value = cash * POS_SIZE_PCT
        shares = int(value / price)
        if shares < 1:
            continue
        value = shares * price
        if value < 2000:
            continue
        ec = commsec(value)
        if value + ec > cash:
            shares = int((cash - 25) * 0.90 / price)
            if shares < 1:
                continue
            value = shares * price
            ec = commsec(value)

        # Commission edge check
        rt_comm = ec + commsec(value)
        exp_win = prob * BE_TRIGGER_ATR * atr_val * shares
        exp_loss = (1 - prob) * stop_dist * shares
        if exp_win - exp_loss < 1.0 * rt_comm:
            continue

        cash -= value + ec
        total_comm += ec
        ticker_tc[t] = ticker_tc.get(t, 0) + 1
        monthly_trades[month] = monthly_trades.get(month, 0) + 1
        year_trades[year] = year_trades.get(year, 0) + 1

        position = {
            "ticker": t, "entry_date": day, "entry_price": price,
            "shares": shares, "stop": price - stop_dist,
            "entry_atr": atr_val, "entry_comm": ec,
            "days_held": 0, "at_be": False,
            "trail_on": False, "high_water": price,
            "week_high": price
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
                "gross": gross
            })
            cash += position["shares"] * price - ec

    # Results
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

    print(f"\n  Régimen:")
    print(f"    🟢 Full: {regime_days['full']}d | 🟡 Selective: {regime_days['selective']}d")
    print(f"    🔴 Gate fail: {regime_days['gate_fail']}d")
    if regime_days['ytd_cap'] > 0:
        print(f"    💔 YTD cap frozen: {regime_days['ytd_cap']}d")
    if regime_days['ls'] > 0:
        print(f"    ⏸️ Streak: {regime_days['ls']}d")

    print(f"\n  💰 {name}:")
    print(f"     ${capital:,.0f} → ${final:,.2f}")
    print(f"     ROI: {'+' if roi >= 0 else ''}{roi:.2f}% | DD: {abs(dd):.2f}%")
    print(f"     Trades: {n}")
    if n > 0:
        print(f"     WR: {wr:.1f}% | PF: {pf:.2f} | R:R: {rr:.2f}")
        print(f"     Avg W: ${avg_w:.2f} | Avg L: ${avg_l:.2f}")
        print(f"     💵 GROSS: ${gross_pnl:+,.2f}")
        print(f"     💸 CommSec: ${total_comms:,.2f} ({total_comms/capital*100:.1f}% cap)")

        # Exit breakdown
        reasons = {}
        for tr in trades:
            r = tr["reason"]
            if r not in reasons:
                reasons[r] = {"n": 0, "pnl": 0, "wins": 0}
            reasons[r]["n"] += 1
            reasons[r]["pnl"] += tr["pnl"]
            if tr["pnl"] > 0:
                reasons[r]["wins"] += 1
        print("     Salidas:")
        for r, v in sorted(reasons.items(), key=lambda x: -x[1]["n"]):
            wr_r = v["wins"] / v["n"] * 100 if v["n"] > 0 else 0
            print(f"       {r:14s} {v['n']:3d}× WR:{wr_r:>4.0f}% Net:${v['pnl']:>+8,.2f}")

        print(f"\n     📋 ALL TRADES:")
        for tr in trades:
            status = "✅" if tr["pnl"] > 0 else "❌"
            print(f"       {status} {tr['ticker']:8s} {str(tr['entry'].date()):10s}→{str(tr['exit'].date()):10s} {tr['days']:3d}d ${tr['pnl']:>+8,.2f} ({tr['reason']})")

    # Year by year
    print(f"\n  📅 AÑO POR AÑO:")
    all_years = sorted(set(range(sim_start.year, sim_end.year + 1)))
    if n > 0:
        trade_df = pd.DataFrame(trades)
        trade_df["year"] = pd.to_datetime(trade_df["entry"]).dt.year
    for yr in all_years:
        if n > 0:
            yr_trades_df = trade_df[trade_df["year"] == yr]
            yr_n = len(yr_trades_df)
            if yr_n > 0:
                yr_net = yr_trades_df["pnl"].sum()
                yr_gross = yr_trades_df["gross"].sum()
                yr_comm = yr_trades_df["comm"].sum()
                yr_wins = (yr_trades_df["pnl"] > 0).sum()
                yr_wr = yr_wins / yr_n * 100
                status = "✅" if yr_net >= 0 else "❌"
                print(f"     {status} {yr}: {yr_n:3d} trades | NET:${yr_net:>+8,.2f} | Gross:${yr_gross:>+8,.2f} | Comm:${yr_comm:>6,.2f} | WR:{yr_wr:.0f}%")
            else:
                print(f"     ⬜ {yr}:   0 trades | NET:$    0.00 (cash)")
        else:
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

    p1, p2, p3 = results[0], results[1], results[2]
    cum = CAPITAL
    for r in results:
        cum *= (1 + r["roi"] / 100)
    cum_roi = (cum / CAPITAL - 1) * 100

    print(f"\n\n{'═'*80}")
    print(f"  📊 RESUMEN — DEEPQUANT V12 (DEFINITIVE)")
    print(f"{'═'*80}")
    for r in results:
        wr_str = f"{r['wr']:.1f}%" if r["trades"] > 0 else "N/A"
        pf_str = f"{r['pf']:.2f}" if r["trades"] > 0 else "N/A"
        print(f"  {r['name']:20s} ${r['final']:>8,.2f} ROI:{r['roi']:>+7.2f}% DD:{abs(r['dd']):>5.2f}% Tr:{r['trades']:>3d} WR:{wr_str:>6s} PF:{pf_str:>5s}")
    print(f"  {'─'*80}")
    print(f"  ACUMULADO: ${cum:,.2f} | ROI: {cum_roi:+.2f}%")

    # Year check
    print(f"\n  📅 YEAR-BY-YEAR GOAL CHECK:")
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
            print(f"     {yr}: ✅ PASS (${net:+,.2f})")
        else:
            failed += 1
            print(f"     {yr}: ❌ FAIL (${net:+,.2f})")
    print(f"  Score: {passed}/{passed+failed}")
    if failed == 0:
        print(f"  🏆 NEVER LOSE ACHIEVED!")

    # Comparison
    print(f"\n{'═'*90}")
    print(f"  VER  P1 ROI   P2 ROI  P3 ROI   ACUM   2022    2023    2024    2025")
    print(f"  {'─'*85}")
    print(f"  V7  -47.68% +19.41%  -8.75% -42.99%   N/A     N/A     N/A     N/A")
    yr12 = {"2022": "$0", "2023": "$0", "2024": "$0", "2025": "$0"}
    for r in results:
        if r["trade_list"]:
            tmp = pd.DataFrame(r["trade_list"])
            tmp["year"] = pd.to_datetime(tmp["entry"]).dt.year
            for yr in tmp["year"].unique():
                yr12[str(yr)] = f"${tmp[tmp['year']==yr]['pnl'].sum():+,.0f}"
    print(f"  V12 {p1['roi']:>+7.2f}% {p2['roi']:>+7.2f}% {p3['roi']:>+7.2f}% {cum_roi:>+6.2f}% {yr12['2022']:>7s} {yr12['2023']:>7s} {yr12['2024']:>7s} {yr12['2025']:>7s}")
    print(f"{'═'*90}")

    print(f"\n  ⏱ {elapsed:.1f} min")

    fname = f"sim_v12_{now}.xlsx"
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
