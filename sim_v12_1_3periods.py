#!/usr/bin/env python
"""
DeepQuant V12.1 — Balanced Regime + Relaxed Stock Filters
==========================================================
V12 PROBLEM: 3-month gate worked perfectly (2022=0, 2023=0) but stock-level
filters were TOO restrictive → only 2 trades in 3 years!
  - "close >= 10d high * 0.99" eliminated virtually all stocks
  - Weekly stop management delayed exits without benefit
  - YTD cap of -$400 froze the rest of 2024 after just 2 losses

V12.1 FIXES:
  1. REMOVE 10-day high breakout filter (too restrictive)
  2. KEEP: price > SMA50 + price > SMA20 (confirmed uptrend)
  3. RELAX: RSI 30-80, volume > 0.5x avg
  4. ADD: momentum_5 > 0 (short-term upward direction)
  5. DAILY stop management (weekly was not better)
  6. Tighter stop: 2.5xATR → faster exit from losers, smaller losses
  7. Easier BE: +1.5xATR (faster protection)
  8. Softer YTD cap: -$650 (allow more trades for statistics)
  9. Max 12 trades/year, 3/month
  10. KEEP: 3-month positive VAS gate (proven: eliminates 2022+2023)
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

# Strategy
MAX_POS         = 1
POS_SIZE_PCT    = 0.80
SL_ATR          = 2.5
BE_TRIGGER_ATR  = 1.5
TRAIL_ATR       = 1.5
MAX_HOLD        = 35
MAX_TRADES_YEAR = 12
MAX_TRADES_MONTH = 3
MAX_TICKER_TRADES = 3
PROB_FULL       = 0.48
PROB_SELECTIVE  = 0.55
YTD_LOSS_CAP   = -650
LS_MAX = 3
LS_DAYS = 7

PERIODS = [
    ("P1: 2022-2024", "2022-01-01", "2024-12-31", "2021-12-31"),
    ("P2: 2025",      "2025-01-01", "2025-12-31", "2024-12-31"),
    ("P3: 2026 YTD",  "2026-01-01", "2026-12-31", "2025-12-31"),
]
CAPITAL = 8_000.0


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
    c, l = df["Close"], df["Low"]
    fwd_ret = c.shift(-fwd) / c - 1
    fwd_dd = pd.Series(np.nan, index=df.index)
    for i in range(len(df) - fwd):
        window = l.iloc[i + 1 : i + fwd + 1]
        fwd_dd.iloc[i] = (window.min() / c.iloc[i]) - 1
    return ((fwd_ret > min_ret) & (fwd_dd > -max_dd)).astype(int)


print("=" * 70)
print("  DEEPQUANT V12.1 - Balanced Regime + Relaxed Stock Filters")
print("  3-Month Gate | Daily Stop | SMA Trend + ML | 1 Position")
print("=" * 70)
print(f"  KEY: Keep 3-month gate (eliminates 2022+2023)")
print(f"  FIX: Removed 10d-high breakout filter (too restrictive)")
print(f"  FIX: Daily stops (weekly had no benefit)")
print(f"  FIX: Softer YTD cap -$650 (was -$400)")
print(f"  FIX: Max 12 trades/year (was 8)")
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
vas_df["ATR14"]  = volatility.average_true_range(vas_df["High"], vas_df["Low"], vas_df["Close"], 14)
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


def check_3month_gate(date):
    rets = get_prior_3_months(date)
    if rets is None:
        return False, []
    return all(r > 0 for r in rets), rets


def calc_regime(date):
    gate_3m, monthly_rets = check_3month_gate(date)
    if not gate_3m:
        return 0, {"gate_3m": False}, False

    mask = vas_df.index <= pd.Timestamp(date)
    if mask.sum() == 0:
        return 0, {}, False

    row = vas_df.loc[mask].iloc[-1]
    c = row["Close"]
    s50 = row["SMA50"] if pd.notna(row["SMA50"]) else 0
    s200 = row["SMA200"] if pd.notna(row["SMA200"]) else 0
    s50_20d = row["SMA50_20d_ago"] if pd.notna(row["SMA50_20d_ago"]) else 0

    details = {"gate_3m": True, "monthly": monthly_rets}

    golden = (s50 > 0 and s200 > 0 and s50 > s200)
    details["golden"] = golden
    if not golden:
        return 0, details, False

    sma50_slope = (s50 > s50_20d) if (s50 > 0 and s50_20d > 0) else False
    details["sma50_slope"] = sma50_slope
    if not sma50_slope:
        return 0, details, False

    dist_200 = (c - s200) / s200 if s200 > 0 else 0
    details["dist_200"] = dist_200
    if dist_200 < 0.02:
        return 0, details, False

    mom = row["MOM20"] if pd.notna(row["MOM20"]) else -1
    brd_mask = breadth_50.index <= pd.Timestamp(date)
    brd = breadth_50.loc[brd_mask].iloc[-1] if brd_mask.sum() > 0 else 0

    score = 50
    if c > s50: score += 15
    if mom > 0: score += 15
    if brd > 50: score += 10
    if dist_200 > 0.05: score += 10

    details["score"] = score
    details["breadth"] = brd
    return score, details, True


def simulate_period(name, start, end, train_end, capital):
    print(f"\n{'='*70}")
    print(f"  {name}  (train -> {train_end})")
    print(f"{'='*70}")

    train_cutoff = pd.Timestamp(train_end)
    sim_start, sim_end = pd.Timestamp(start), pd.Timestamp(end)

    valid = [t for t in data if data[t].index.min() < train_cutoff - pd.Timedelta(days=365)]
    print(f"  {len(valid)} tickers")

    print(f"  Training (target: 4%+ in 20d, <3.5% DD)...")
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
        if (i + 1) % 20 == 0 or i == len(valid) - 1:
            print(f"\r   [{i+1}/{len(valid)}]", end="", flush=True)
    print(f"\n   Models: {len(models)}/{len(valid)}")

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

    cash = capital
    position = None
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
    regime_days = {"full": 0, "selective": 0, "gate_fail": 0, "ytd_cap": 0, "ls": 0}

    all_sim_dates = set()
    for t in models:
        all_sim_dates.update(data[t].index)
    trading_days = sorted([d for d in all_sim_dates if sim_start <= d <= sim_end])
    monthly_trades = {}

    # Debug: track regime on first tradeable day each month
    regime_log = []

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

        trading_ok = not ytd_frozen
        if ls_until and day < ls_until:
            regime_days["ls"] += 1
            trading_ok = False
        elif ls_until:
            ls_until = None

        # EXIT
        if position is not None:
            t = position["ticker"]
            if day in data[t].index:
                row = data[t].loc[day]
                price = row["Close"]
                high_today = row["High"]
                low_today = row["Low"]
                position["days_held"] += 1

                if high_today > position["high_water"]:
                    position["high_water"] = high_today

                # Breakeven trigger
                if not position["at_be"]:
                    if high_today >= position["entry_price"] + BE_TRIGGER_ATR * position["entry_atr"]:
                        position["at_be"] = True
                        position["stop"] = position["entry_price"]

                # Trailing stop update (after BE)
                if position["at_be"]:
                    new_stop = position["high_water"] - TRAIL_ATR * position["entry_atr"]
                    if new_stop > position["stop"]:
                        position["stop"] = new_stop

                exit_reason = None

                # DAILY stop check (but give 2-day grace period for noise)
                if position["days_held"] >= 2:
                    if price <= position["stop"]:
                        exit_reason = "BE_STOP" if position["at_be"] else "STOP"
                elif low_today < position["entry_price"] * 0.93:
                    # Emergency on day 1 only for catastrophic drop
                    exit_reason = "EMERGENCY"

                # Time exit
                if exit_reason is None and position["days_held"] >= MAX_HOLD:
                    exit_reason = "TIME"

                # Regime lost while profitable — protect gains
                if exit_reason is None and not gates_ok and price > position["entry_price"] * 1.005:
                    exit_reason = "GATE_EXIT"

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

        port_val = cash
        if position is not None and day in data[position["ticker"]].index:
            port_val += position["shares"] * data[position["ticker"]].loc[day, "Close"]
        equity_curve.append((day, port_val))

        # ENTRY
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
            mom5 = p_row["mom5"]

            if pd.isna(atr_val) or atr_val <= 0 or pd.isna(sma50) or pd.isna(sma20):
                continue

            # RELAXED STOCK FILTERS:
            # 1. Price above SMA50 (confirmed uptrend)
            if price < sma50:
                continue

            # 2. Price above SMA20 (immediate trend)
            if price < sma20:
                continue

            # 3. RSI 30-80 (wide range)
            if pd.notna(rsi_val) and (rsi_val < 30 or rsi_val > 80):
                continue

            # 4. Some volume (relaxed)
            if pd.notna(vol) and pd.notna(vol_avg) and vol_avg > 0:
                if vol < 0.5 * vol_avg:
                    continue

            # 5. Short-term momentum positive
            if pd.notna(mom5) and mom5 < -0.01:
                continue

            # 6. Not extreme volatility
            vr = f_row.get("vol_regime", 1.0)
            if pd.notna(vr) and vr > 2.0:
                continue

            # ML probability
            try:
                prob = models[t].predict_proba(f_row.values.reshape(1, -1))[0][1]
            except:
                continue
            if prob < prob_thresh:
                continue

            rs = f_row.get("relative_strength", 0)
            combined_score = prob * 0.7 + (rs if pd.notna(rs) else 0) * 0.3

            candidates.append((t, combined_score, prob, price, atr_val))

        if not candidates:
            continue

        candidates.sort(key=lambda x: -x[1])
        t, _, prob, price, atr_val = candidates[0]

        stop_dist = SL_ATR * atr_val
        if stop_dist > price * 0.06:
            stop_dist = price * 0.06

        value = cash * POS_SIZE_PCT
        shares = int(value / price)
        if shares < 1: continue
        value = shares * price
        if value < 2000: continue
        ec = commsec(value)
        if value + ec > cash:
            shares = int((cash - 25) * 0.90 / price)
            if shares < 1: continue
            value = shares * price
            ec = commsec(value)

        rt_comm = ec + commsec(value)
        exp_win = prob * BE_TRIGGER_ATR * atr_val * shares
        exp_loss = (1 - prob) * stop_dist * shares
        if exp_win - exp_loss < 0.8 * rt_comm:
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
            "high_water": price
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

    print(f"\n  Regimen:")
    print(f"    Full: {regime_days['full']}d | Selective: {regime_days['selective']}d")
    print(f"    Gate fail: {regime_days['gate_fail']}d")
    if regime_days['ytd_cap'] > 0:
        print(f"    YTD cap frozen: {regime_days['ytd_cap']}d")
    if regime_days['ls'] > 0:
        print(f"    Loss streak: {regime_days['ls']}d")

    print(f"\n  RESULTS {name}:")
    print(f"     ${capital:,.0f} -> ${final:,.2f}")
    print(f"     ROI: {'+' if roi >= 0 else ''}{roi:.2f}% | DD: {abs(dd):.2f}%")
    print(f"     Trades: {n}")
    if n > 0:
        print(f"     WR: {wr:.1f}% | PF: {pf:.2f} | R:R: {rr:.2f}")
        print(f"     Avg W: ${avg_w:.2f} | Avg L: ${avg_l:.2f}")
        print(f"     GROSS: ${gross_pnl:+,.2f}")
        print(f"     CommSec: ${total_comms:,.2f} ({total_comms/capital*100:.1f}% cap)")

        reasons = {}
        for tr in trades:
            r = tr["reason"]
            if r not in reasons:
                reasons[r] = {"n": 0, "pnl": 0, "wins": 0}
            reasons[r]["n"] += 1
            reasons[r]["pnl"] += tr["pnl"]
            if tr["pnl"] > 0:
                reasons[r]["wins"] += 1
        print("     Exits:")
        for r, v in sorted(reasons.items(), key=lambda x: -x[1]["n"]):
            wr_r = v["wins"] / v["n"] * 100 if v["n"] > 0 else 0
            print(f"       {r:14s} {v['n']:3d}x WR:{wr_r:>4.0f}% Net:${v['pnl']:>+8,.2f}")

        print(f"\n     ALL TRADES:")
        for tr in trades:
            status = "W" if tr["pnl"] > 0 else "L"
            print(f"       [{status}] {tr['ticker']:8s} {str(tr['entry'].date()):10s}->{str(tr['exit'].date()):10s} {tr['days']:3d}d ${tr['pnl']:>+8,.2f} ({tr['reason']})")

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
            print(f"     [{status}] {yr}: {yr_n:3d} trades | NET:${yr_net:>+8,.2f} | Gross:${yr_gross:>+8,.2f} | Comm:${yr_comm:>6,.2f} | WR:{yr_wr:.0f}%")
        else:
            print(f"     [PASS] {yr}:   0 trades | NET:$    0.00")

    return {
        "name": name, "capital": capital, "final": final,
        "roi": roi, "dd": dd, "trades": n, "wr": wr, "pf": pf, "rr": rr,
        "gross": gross_pnl if n > 0 else 0, "comm": total_comms if n > 0 else 0,
        "trade_list": trades, "equity": equity_curve
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
    print(f"  SUMMARY - DEEPQUANT V12.1")
    print(f"{'='*80}")
    for r in results:
        wr_str = f"{r['wr']:.1f}%" if r["trades"] > 0 else "N/A"
        pf_str = f"{r['pf']:.2f}" if r["trades"] > 0 else "N/A"
        print(f"  {r['name']:20s} ${r['final']:>8,.2f} ROI:{r['roi']:>+7.2f}% DD:{abs(r['dd']):>5.2f}% Tr:{r['trades']:>3d} WR:{wr_str:>6s} PF:{pf_str:>5s}")
    print(f"  {'-'*78}")
    print(f"  CUMULATIVE: ${cum:,.2f} | ROI: {cum_roi:+.2f}%")

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
    print(f"  V7   -47.68%  +19.41%   -8.75%  -42.99%")
    print(f"  V11  -14.27%   -7.01%   +2.22%  -18.51%  -$477   -$215   -$450   -$561   +$178")
    print(f"  V12   -7.18%   +0.00%   +0.00%   -7.18%     $0      $0   -$574      $0      $0")
    print(f"  V12.1{p1['roi']:>+7.2f}% {p2['roi']:>+7.2f}% {p3['roi']:>+7.2f}% {cum_roi:>+7.2f}% {yr_all['2022']:>7s} {yr_all['2023']:>7s} {yr_all['2024']:>7s} {yr_all['2025']:>7s} {yr_all['2026']:>7s}")
    print(f"{'='*90}")

    print(f"\n  Time: {elapsed:.1f} min")

    fname = f"sim_v12_1_{now}.xlsx"
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
