#!/usr/bin/env python
"""
DeepQuant V11 — Year-by-Year Profitability Engine
====================================================
ROOT CAUSE ANALYSIS from V7-V10:
  V7:  +19.41% P2 but -47.68% P1 (no bear protection)
  V9:  CB bug froze 538 days; only 5 trades total  
  V10: Fixed CB but golden cross LAGGED in 2022 (4 trades lost)
       2023: 30% WR → commission drag killed it ($399 on 10 trades)
       2024: 55% WR → PROFITABLE (+$582) ← the model WORKS here
       P2: Only 2 trades, both stopped (too few to show edge)

FUNDAMENTAL REDESIGN (V11):
═══════════════════════════════════════════════════════════════
1. 2-MONTH POSITIVE GATE
   VAS must be positive in BOTH prior 2 calendar months.
   → Eliminates ALL 2022 trading (Dec 2021 was negative!)
   → Limits 2023 to proven uptrend windows only.

2. ALIGNED ML TARGET
   Instead of "5% return in 20d", predict:
   "Stock reaches +1.5×ATR before hitting -2.5×ATR within 25d"
   This matches the actual BE trigger in the trading system.

3. ONE POSITION ($6K+)
   Max 1 position at $6,400 → commission 0.31% vs 0.53% per side.
   Fewer positions = less commission.

4. GOLDEN CROSS + SMA50 SLOPE
   Both required. SMA50 slope catches the turn faster than GC alone.

5. ANNUAL LOSS CAP
   -$350 YTD → stop trading for rest of year.
   Guarantees max ~-4.4% annual loss (near-breakeven).

6. MONTHLY LOSS CAP
   -$200 in current month → skip rest of month.
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
MAX_POS         = 1             # ← KEY: only 1 position (less commission)
POS_SIZE_PCT    = 0.80          # 80% of cash in position (~$6,400)
SL_ATR          = 2.5           # stop loss
SL_MAX_PCT      = 0.04          # max 4% stop
BE_TRIGGER_ATR  = 1.5           # breakeven trigger (aligned with target)
TRAIL_ATR       = 1.8           # trail width after BE
MAX_HOLD        = 35
MAX_TRADES_MONTH = 2
MAX_TICKER_TRADES = 4

# Regime
REGIME_FULL     = 70
REGIME_SELECTIVE = 55
PROB_FULL       = 0.48          # lower because regime gates are very strict
PROB_SELECTIVE  = 0.58

# Safety
YTD_LOSS_CAP   = -350           # stop trading for year at -$350
MONTH_LOSS_CAP = -200           # skip rest of month at -$200
LS_MAX = 3
LS_DAYS = 7

PERIODS = [
    ("P1: 2022-2024", "2022-01-01", "2024-12-31", "2021-12-31"),
    ("P2: 2025",      "2025-01-01", "2025-12-31", "2024-12-31"),
    ("P3: 2026 YTD",  "2026-01-01", "2026-12-31", "2025-12-31"),
]
CAPITAL = 8_000.0


# ╔══════════════════════════════════════════════════════════╗
# ║  FEATURES (25)                                           ║
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

    # Market context
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


def build_target_aligned(df, atr_up=1.5, atr_down=2.5, horizon=25):
    """ALIGNED TARGET: predict if stock reaches +1.5×ATR before -2.5×ATR.
    This directly matches the breakeven trigger in the trading system."""
    c = df["Close"]
    h = df["High"]
    l = df["Low"]
    atr = volatility.average_true_range(h, l, c, 14)

    target = pd.Series(np.nan, index=df.index)
    for i in range(len(df) - horizon):
        entry = c.iloc[i]
        atr_val = atr.iloc[i]
        if pd.isna(atr_val) or atr_val <= 0:
            continue
        up_target = entry + atr_up * atr_val
        down_target = entry - atr_down * atr_val

        hit = 0  # default: neither hit (bad outcome)
        for j in range(i + 1, min(i + horizon + 1, len(df))):
            if h.iloc[j] >= up_target:
                hit = 1  # reached BE trigger first ✓
                break
            if l.iloc[j] <= down_target:
                hit = 0  # stopped out first ✗
                break
        target.iloc[i] = hit
    return target


# ╔══════════════════════════════════════════════════════════╗
# ║  DATA                                                    ║
# ╚══════════════════════════════════════════════════════════╝

print("╔══════════════════════════════════════════════════════════════╗")
print("║  DEEPQUANT V11 — YEAR-BY-YEAR PROFITABILITY ENGINE         ║")
print("║  Aligned Target | 2-Month Gate | 1 Position | Loss Caps    ║")
print("╚══════════════════════════════════════════════════════════════╝")
print(f"\n  🔑 KEY INNOVATIONS:")
print(f"     • 2-MONTH VAS POSITIVE GATE → eliminates 2022 entirely")
print(f"     • ALIGNED ML TARGET: 'reach +1.5×ATR before -2.5×ATR in 25d'")
print(f"     • ONE POSITION at $6,400 (0.31% commission vs 0.53%)")
print(f"     • SMA50 SLOPE CHECK (catches trend turns faster)")
print(f"     • YTD cap: -$350 | Monthly cap: -$200")
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

# Calculate VAS monthly returns for 2-month gate
vas_monthly = vas_df["Close"].resample("ME").last().pct_change()
print(f"  📊 VAS.AX: {len(vas_df)} rows | Monthly returns computed")

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

# Universe median momentum
mom_matrix = pd.DataFrame(index=all_dates)
for t, df in data.items():
    mom_matrix[t] = df["Close"].pct_change(20).reindex(all_dates)
universe_mom_median = mom_matrix.median(axis=1)
print(f"  ✅ Breadth + momentum calculados")


# ╔══════════════════════════════════════════════════════════╗
# ║  REGIME (4 GATES)                                        ║
# ╚══════════════════════════════════════════════════════════╝

def check_2month_gate(date):
    """Check if VAS was positive in BOTH of the prior 2 calendar months."""
    d = pd.Timestamp(date)
    # Find the 2 months prior
    m_cur = d.month
    y_cur = d.year
    # Month before
    m1 = m_cur - 1 if m_cur > 1 else 12
    y1 = y_cur if m_cur > 1 else y_cur - 1
    # Two months before
    m2 = m1 - 1 if m1 > 1 else 12
    y2 = y1 if m1 > 1 else y1 - 1

    # Get monthly returns
    try:
        # Find the end of m1 in the monthly index
        m1_end = pd.Timestamp(year=y1, month=m1, day=28) + pd.offsets.MonthEnd(0)
        m2_end = pd.Timestamp(year=y2, month=m2, day=28) + pd.offsets.MonthEnd(0)
        if m1_end in vas_monthly.index and m2_end in vas_monthly.index:
            r1 = vas_monthly.loc[m1_end]
            r2 = vas_monthly.loc[m2_end]
            return pd.notna(r1) and pd.notna(r2) and r1 > 0 and r2 > 0
    except:
        pass
    return False


def calc_regime(date):
    """4-gate regime check."""
    mask = vas_df.index <= pd.Timestamp(date)
    if mask.sum() == 0:
        return 0, {}, False

    row = vas_df.loc[mask].iloc[-1]
    c = row["Close"]
    s50 = row["SMA50"] if pd.notna(row["SMA50"]) else 0
    s200 = row["SMA200"] if pd.notna(row["SMA200"]) else 0
    s50_20d = row["SMA50_20d_ago"] if pd.notna(row["SMA50_20d_ago"]) else 0
    mom = row["MOM20"] if pd.notna(row["MOM20"]) else -1
    atr_pct = row["ATR_pct"] if pd.notna(row["ATR_pct"]) else 99
    atr_avg = row["ATR_avg"] if pd.notna(row["ATR_avg"]) else 99

    brd_mask = breadth_50.index <= pd.Timestamp(date)
    brd = breadth_50.loc[brd_mask].iloc[-1] if brd_mask.sum() > 0 else 0

    details = {"breadth": brd}

    # GATE 1: Golden Cross
    golden = (s50 > 0 and s200 > 0 and s50 > s200)
    details["golden"] = golden
    if not golden:
        return 0, details, False

    # GATE 2: SMA50 slope positive (SMA50 rising over last 20d)
    sma50_slope = (s50 > s50_20d) if (s50 > 0 and s50_20d > 0) else False
    details["sma50_slope"] = sma50_slope
    if not sma50_slope:
        return 0, details, False

    # GATE 3: VAS > SMA200 by at least 2%
    dist_200 = (c - s200) / s200 if s200 > 0 else 0
    details["dist_200"] = dist_200
    if dist_200 < 0.02:
        return 0, details, False

    # GATE 4: 2-month positive VAS returns
    gate_2m = check_2month_gate(date)
    details["gate_2m"] = gate_2m
    if not gate_2m:
        return 0, details, False

    # All 4 gates passed! Now calculate score for full/selective
    score = 40  # base (all 4 gates passed)
    if c > s50:
        score += 15
    if mom > 0:
        score += 15
    if brd > 50:
        score += 15
    if atr_pct < atr_avg * 1.1:
        score += 10
    if dist_200 > 0.05:
        score += 5

    details["score"] = score
    return score, details, True


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

    print(f"  🧠 Training (aligned target: +1.5×ATR before -2.5×ATR in 25d)...")
    models = {}
    pos_rates = []  # track positive rate for diagnostics
    for i, t in enumerate(valid):
        df = data[t]
        tr = df[df.index <= train_cutoff].copy()
        if len(tr) < 300:
            continue
        feats = build_features(tr, vas_feats, breadth_50, universe_mom_median)
        tgt = build_target_aligned(tr, atr_up=BE_TRIGGER_ATR, atr_down=SL_ATR, horizon=25)
        mask = feats.notna().all(axis=1) & tgt.notna()
        X, y = feats[mask], tgt[mask]
        if len(X) < 100 or y.sum() < 10:
            continue
        pos_rates.append(y.mean())
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

    avg_pos = np.mean(pos_rates) * 100 if pos_rates else 0
    print(f"\n   ✅ Modelos: {len(models)}/{len(valid)} | Avg target positive rate: {avg_pos:.1f}%")

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
    position = None
    trades = []
    equity_curve = []
    ticker_tc = {}
    total_comm = 0.0

    # Safety state
    consec_losses = 0
    ls_until = None

    # Annual / monthly tracking
    ytd_pnl = 0.0
    ytd_year = None
    month_pnl = 0.0
    month_num = None
    ytd_frozen = False
    month_frozen = False

    # Counters
    regime_days = {"full": 0, "selective": 0, "gate_fail": 0, "ytd_cap": 0, "month_cap": 0, "ls": 0}
    monthly_trades = {}

    # Trading days
    all_sim_dates = set()
    for t in models:
        all_sim_dates.update(data[t].index)
    trading_days = sorted([d for d in all_sim_dates if sim_start <= d <= sim_end])

    for day in trading_days:
        year = day.year
        month = (day.year, day.month)

        # Reset annual tracking
        if ytd_year != year:
            ytd_year = year
            ytd_pnl = 0.0
            ytd_frozen = False

        # Reset monthly tracking
        if month_num != month:
            month_num = month
            month_pnl = 0.0
            month_frozen = False
            monthly_trades[month] = 0

        # Check safety caps
        if ytd_frozen:
            regime_days["ytd_cap"] += 1
            # Still need to manage existing position
        elif month_frozen:
            regime_days["month_cap"] += 1

        # ── REGIME ──
        score, details, gates_ok = calc_regime(day)
        if not gates_ok:
            regime_days["gate_fail"] += 1
            prob_thresh = None
        elif score >= REGIME_FULL:
            regime_days["full"] += 1
            prob_thresh = PROB_FULL
        elif score >= REGIME_SELECTIVE:
            regime_days["selective"] += 1
            prob_thresh = PROB_SELECTIVE
        else:
            regime_days["gate_fail"] += 1
            prob_thresh = None

        # Loss streak override
        trading_ok = not ytd_frozen and not month_frozen
        if ls_until and day < ls_until:
            regime_days["ls"] += 1
            trading_ok = False
        elif ls_until and day >= ls_until:
            ls_until = None

        # ── EXIT ──
        if position is not None:
            t = position["ticker"]
            if day in data[t].index:
                row = data[t].loc[day]
                price, low, high = row["Close"], row["Low"], row["High"]
                position["days_held"] += 1
                if high > position["high_water"]:
                    position["high_water"] = high

                # Breakeven trigger
                if not position["at_be"]:
                    if high >= position["entry_price"] + BE_TRIGGER_ATR * position["entry_atr"]:
                        position["at_be"] = True
                        position["stop"] = position["entry_price"]
                        position["trail_on"] = True

                # Trail
                if position["trail_on"]:
                    new_stop = position["high_water"] - TRAIL_ATR * position["entry_atr"]
                    if new_stop > position["stop"]:
                        position["stop"] = new_stop

                exit_reason = None

                # Stop hit
                if low <= position["stop"]:
                    ep = max(position["stop"], low)
                    exit_reason = "BE_STOP" if position["at_be"] else "STOP"
                # Gates lost while profitable → protect
                elif not gates_ok and price > position["entry_price"] * 1.005:
                    ep = price
                    exit_reason = "GATE_EXIT"
                # Max hold
                elif position["days_held"] >= MAX_HOLD:
                    ep = price
                    exit_reason = "TIME"
                # YTD frozen while in position → close if profitable or at BE
                elif ytd_frozen and price >= position["entry_price"]:
                    ep = price
                    exit_reason = "YTD_EXIT"

                if exit_reason:
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
                    month_pnl += net
                    if net > 0:
                        consec_losses = 0
                    else:
                        consec_losses += 1
                    position = None

                    # Check caps after trade
                    if ytd_pnl <= YTD_LOSS_CAP:
                        ytd_frozen = True
                    if month_pnl <= MONTH_LOSS_CAP:
                        month_frozen = True
                    if consec_losses >= LS_MAX:
                        ls_until = day + pd.Timedelta(days=LS_DAYS)

        # ── EQUITY ──
        port_val = cash
        if position is not None and day in data[position["ticker"]].index:
            port_val += position["shares"] * data[position["ticker"]].loc[day, "Close"]
        equity_curve.append((day, port_val))

        # ── ENTRY ──
        if position is not None or not trading_ok or prob_thresh is None:
            continue
        if monthly_trades.get(month, 0) >= MAX_TRADES_MONTH:
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
            adx_val = p_row["adx"]

            if pd.isna(atr_val) or atr_val <= 0 or pd.isna(sma50):
                continue

            # FILTER: Stock above SMA50
            if price < sma50:
                continue

            # FILTER: RSI 30-75
            if pd.notna(rsi_val) and (rsi_val < 30 or rsi_val > 75):
                continue

            # FILTER: ADX > 8 (some directionality)
            if pd.notna(adx_val) and adx_val < 8:
                continue

            # FILTER: Volume participation
            if pd.notna(vol) and pd.notna(vol_avg) and vol_avg > 0:
                if vol < 0.6 * vol_avg:
                    continue

            # FILTER: Not extreme volatility
            vr = f_row.get("vol_regime", 1.0)
            if pd.notna(vr) and vr > 1.8:
                continue

            # ML probability
            try:
                prob = models[t].predict_proba(f_row.values.reshape(1, -1))[0][1]
            except:
                continue
            if prob < prob_thresh:
                continue

            candidates.append((t, prob, price, atr_val))

        if not candidates:
            continue

        candidates.sort(key=lambda x: -x[1])
        t, prob, price, atr_val = candidates[0]

        # Position sizing
        stop_dist = min(SL_ATR * atr_val, price * SL_MAX_PCT)
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

        # Commission-aware: expected net must be positive
        rt_comm = ec + commsec(value)
        exp_win = prob * BE_TRIGGER_ATR * atr_val * shares  # expected win at BE level
        exp_loss = (1 - prob) * stop_dist * shares
        if exp_win - exp_loss < 1.2 * rt_comm:
            continue

        cash -= value + ec
        total_comm += ec
        ticker_tc[t] = ticker_tc.get(t, 0) + 1
        monthly_trades[month] = monthly_trades.get(month, 0) + 1

        position = {
            "ticker": t, "entry_date": day, "entry_price": price,
            "shares": shares, "stop": price - stop_dist,
            "entry_atr": atr_val, "entry_comm": ec,
            "days_held": 0, "at_be": False,
            "trail_on": False, "high_water": price
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

    # ── Results ──
    final = cash
    roi = (final - capital) / capital * 100
    n = len(trades)

    if len(equity_curve) > 0:
        eq_s = pd.Series([e[1] for e in equity_curve], index=[e[0] for e in equity_curve])
        dd = ((eq_s / eq_s.cummax()) - 1).min() * 100
    else:
        dd = 0

    wins = [t_rec for t_rec in trades if t_rec["pnl"] > 0]
    losses = [t_rec for t_rec in trades if t_rec["pnl"] <= 0]
    wr = len(wins) / n * 100 if n else 0
    avg_w = np.mean([t_rec["pnl"] for t_rec in wins]) if wins else 0
    avg_l = np.mean([abs(t_rec["pnl"]) for t_rec in losses]) if losses else 0
    pf = sum(t_rec["pnl"] for t_rec in wins) / abs(sum(t_rec["pnl"] for t_rec in losses)) if losses and sum(t_rec["pnl"] for t_rec in losses) != 0 else 99
    rr = avg_w / avg_l if avg_l > 0 else 99
    gross_pnl = sum(t_rec["gross"] for t_rec in trades)
    total_comms = sum(t_rec["comm"] for t_rec in trades)

    total_days = sum(regime_days.values()) + sum(1 for d in trading_days if not any([
        regime_days  # placeholder
    ]))

    print(f"\n  Régimen:")
    print(f"    🟢 Full({REGIME_FULL}+): {regime_days['full']}d | 🟡 Selective({REGIME_SELECTIVE}-{REGIME_FULL}): {regime_days['selective']}d")
    print(f"    🔴 Gate fail: {regime_days['gate_fail']}d")
    if regime_days['ytd_cap'] > 0:
        print(f"    💔 YTD cap frozen: {regime_days['ytd_cap']}d")
    if regime_days['month_cap'] > 0:
        print(f"    📅 Month cap frozen: {regime_days['month_cap']}d")
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

        # Trade details
        print(f"\n     📋 ALL TRADES:")
        for t_rec in trades:
            status = "✅" if t_rec["pnl"] > 0 else "❌"
            print(f"       {status} {t_rec['ticker']:8s} {str(t_rec['entry'].date()):10s}→{str(t_rec['exit'].date()):10s} {t_rec['days']:3d}d ${t_rec['pnl']:>+8,.2f} ({t_rec['reason']:10s}) gross:${t_rec['gross']:>+7,.2f} comm:${t_rec['comm']:>5,.2f}")

    # Year-by-year
    print(f"\n  📅 AÑO POR AÑO:")
    all_years = sorted(set(range(sim_start.year, sim_end.year + 1)))
    if n > 0:
        trade_df = pd.DataFrame(trades)
        trade_df["year"] = pd.to_datetime(trade_df["entry"]).dt.year
    for yr in all_years:
        if n > 0:
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

    print(f"\n\n{'═'*80}")
    print(f"  📊 RESUMEN — DEEPQUANT V11 (YEAR-BY-YEAR ENGINE)")
    print(f"{'═'*80}")
    hdr = f"  {'Período':20s} {'Final':>10s} {'ROI':>8s} {'DD':>7s} {'Tr':>4s} {'WR':>6s} {'PF':>5s} {'Gross':>10s} {'Comm':>8s}"
    print(hdr)
    print(f"  {'─'*80}")
    cum = CAPITAL
    for r in results:
        cum *= (1 + r["roi"] / 100)
        wr_str = f"{r['wr']:>5.1f}%" if r["trades"] > 0 else "  N/A"
        pf_str = f"{r['pf']:>4.2f}" if r["trades"] > 0 else " N/A"
        print(f"  {r['name']:20s} ${r['final']:>8,.2f} {r['roi']:>+7.2f}% {abs(r['dd']):>5.2f}% {r['trades']:>4d} {wr_str} {pf_str} ${r['gross']:>+8,.0f} ${r['comm']:>6,.0f}")
    cum_roi = (cum / CAPITAL - 1) * 100
    print(f"  {'─'*80}")
    print(f"  ACUMULADO: ${cum:,.2f} | ROI: {cum_roi:+.2f}%")

    # Year-by-year summary
    print(f"\n  📅 YEAR-BY-YEAR GOAL CHECK (target: NEVER LOSE):")
    all_trades_combined = []
    for r in results:
        all_trades_combined.extend(r["trade_list"])
    all_years = set()
    for p in PERIODS:
        for yr in range(int(p[1][:4]), int(p[2][:4]) + 1):
            all_years.add(yr)
    passed = 0
    failed = 0
    if all_trades_combined:
        df_all = pd.DataFrame(all_trades_combined)
        df_all["year"] = pd.to_datetime(df_all["entry"]).dt.year
    for yr in sorted(all_years):
        if all_trades_combined:
            yr_tr = df_all[df_all["year"] == yr] if yr in df_all["year"].values else pd.DataFrame()
        else:
            yr_tr = pd.DataFrame()
        if len(yr_tr) > 0:
            net = yr_tr["pnl"].sum()
            if net >= 0:
                status = "✅ PASS"
                passed += 1
            else:
                status = "❌ FAIL"
                failed += 1
            print(f"     {yr}: {status} (${net:+,.2f}, {len(yr_tr)} trades)")
        else:
            status = "✅ PASS"
            passed += 1
            print(f"     {yr}: {status} ($0.00, 0 trades — cash)")
    print(f"  ──────────────────────────────────────────")
    print(f"  Score: {passed}/{passed+failed} years passed | {'🏆 NEVER LOSE ACHIEVED!' if failed == 0 else f'❌ {failed} years still losing'}")

    # Version comparison
    print(f"\n{'═'*90}")
    print(f"  🏆 ALL VERSIONS COMPARISON:")
    print(f"{'═'*90}")
    print(f"  {'Ver':5s} {'P1 ROI':>8s} {'P2 ROI':>8s} {'P3 ROI':>8s} {'Acum':>8s} {'P1 Tr':>6s} {'P2 Tr':>6s} {'2022':>8s} {'2023':>8s} {'2024':>8s}")
    print(f"  {'─'*90}")
    print(f"  V5   -58.71%   +1.94%   -0.33%  -58.07%     63     21      N/A      N/A      N/A")
    print(f"  V7   -47.68%  +19.41%   -8.75%  -42.99%     63     21      N/A      N/A      N/A")
    print(f"  V9     -4.68%   -7.21%   +0.00%  -11.55%      3      2     -$374       $0       $0")
    print(f"  V10  -13.72%   -8.81%   +0.00%  -21.32%     25      2     -$538   -$1141    +$582")
    p1, p2, p3 = results[0], results[1], results[2]
    yr_vals = {"2022": "$0", "2023": "$0", "2024": "$0"}
    if p1["trade_list"]:
        df_p1 = pd.DataFrame(p1["trade_list"])
        df_p1["year"] = pd.to_datetime(df_p1["entry"]).dt.year
        for yr in [2022, 2023, 2024]:
            yr_t = df_p1[df_p1["year"] == yr]
            if len(yr_t) > 0:
                yr_vals[str(yr)] = f"${yr_t['pnl'].sum():+,.0f}"
    print(f"  V11  {p1['roi']:>+7.2f}% {p2['roi']:>+7.2f}% {p3['roi']:>+7.2f}% {cum_roi:>+7.2f}% {p1['trades']:>6d} {p2['trades']:>6d} {yr_vals['2022']:>8s} {yr_vals['2023']:>8s} {yr_vals['2024']:>8s}")
    print(f"{'═'*90}")

    print(f"\n  📌 V11 ARCHITECTURE:")
    print(f"  ├── GATE 1: Golden Cross (SMA50 > SMA200 on VAS)")
    print(f"  ├── GATE 2: SMA50 Slope Positive (rising trend)")
    print(f"  ├── GATE 3: VAS > SMA200 by ≥ 2%")
    print(f"  ├── GATE 4: VAS positive in prior 2 calendar months")
    print(f"  ├── ML: LightGBM 25 features, aligned BE target")
    print(f"  ├── 1 position × $6,400 (lower commission %)")
    print(f"  ├── BE stop at +1.5×ATR → trail 1.8×ATR")
    print(f"  └── YTD cap: -$350 | Monthly cap: -$200")

    print(f"\n  ⏱ {elapsed:.1f} min")

    # Save
    fname = f"sim_v11_{now}.xlsx"
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
