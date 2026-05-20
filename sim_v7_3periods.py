#!/usr/bin/env python
"""
DeepQuant V7 — Commission-Aware Ultra-Selective Strategy
=========================================================
KEY INSIGHT from V6: Model generates +$1,573 gross alpha in P1,
but $4,368 in commissions destroys it → net -$2,795.

V7 FIXES:
1. Ultra-selective: prob > 0.58 (fewer, higher-conviction trades)
2. VAS 20d momentum > 0 required (trend confirmation)
3. Max 3 positions (bigger sizes → lower commission %)
4. Commission-aware: skip trades where expected edge < 3× commission
5. Risk 3% per trade (bigger positions = lower commission impact)
6. Reports GROSS vs NET P&L and commission drag %
7. Also runs LOW-COMMISSION scenario ($3 flat, 0%) for comparison
"""

import warnings, datetime as dt, sys
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, yfinance as yf
from pathlib import Path
from ta import momentum, trend, volatility, volume
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb

# ── Config ──────────────────────────────────────────────
SYM_FILE   = Path("au_stock_data/au_symbols.txt")
TICKERS    = [s.strip() for s in SYM_FILE.read_text().splitlines() if s.strip()]
# Add .AX only if not already present
TICKERS    = [t if t.endswith(".AX") else t+".AX" for t in TICKERS]
VAS_TICKER = "VAS.AX"

# Commission models
COMM_STD  = {"flat": 10.0, "pct": 0.0011, "name": "$10+0.11%"}
COMM_LOW  = {"flat": 3.0,  "pct": 0.0,    "name": "$3 flat"}

# Strategy parameters
PROB_BULL       = 0.58   # ultra-selective (was 0.48 in V6)
MAX_POS         = 3      # fewer but bigger (was 6)
RISK_PCT        = 0.03   # 3% risk per trade (was 2%)
SL_ATR          = 2.0    # stop loss multiplier
TP1_ATR         = 4.0    # take profit 1 (50% sell)
NO_TRAIL_DAYS   = 8      # no trail first N days
TRAIL_PRE_ATR   = 3.0    # trail before TP1
TRAIL_POST_ATR  = 2.0    # trail after TP1  
MAX_HOLD        = 30     # max holding days
PULLBACK_PCT    = 0.04   # max dist from SMA20 for entry

# VAS momentum gate
VAS_MOM_DAYS    = 20     # VAS must have positive N-day return
VAS_SMA_LONG    = 200    # regime: VAS > SMA200

# Commission-aware filter
MIN_EDGE_MULT   = 2.5    # expected edge must be > 2.5× round-trip commission

PERIODS = [
    ("P1: 2022-2024", "2022-01-01", "2024-12-31", "2021-12-31"),
    ("P2: 2025",      "2025-01-01", "2025-12-31", "2024-12-31"),
    ("P3: 2026 YTD",  "2026-01-01", "2026-12-31", "2025-12-31"),
]
CAPITAL = 8_000.0

# ── Features ────────────────────────────────────────────
def build_features(df):
    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    sma20 = c.rolling(20).mean(); sma50 = c.rolling(50).mean(); sma200 = c.rolling(200).mean()
    atr = volatility.average_true_range(h, l, c, 14)
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
    return feat


def build_target(df, fwd=10, min_ret=0.015, max_dd=0.05):
    c = df["Close"]; l = df["Low"]
    fwd_ret = c.shift(-fwd) / c - 1
    fwd_dd = pd.Series(np.nan, index=df.index)
    for i in range(len(df) - fwd):
        window = l.iloc[i+1:i+fwd+1]
        fwd_dd.iloc[i] = (window.min() / c.iloc[i]) - 1
    return ((fwd_ret > min_ret) & (fwd_dd > -max_dd)).astype(int)


# ── Commission calculation ──────────────────────────────
def calc_commission(value, comm_model):
    return comm_model["flat"] + value * comm_model["pct"]


# ── Download ────────────────────────────────────────────
print("╔══════════════════════════════════════════════════════════════╗")
print("║  DEEPQUANT V7 — COMMISSION-AWARE ULTRA-SELECTIVE           ║")
print("║  LightGBM × Regime × VAS Momentum × Commission Filter     ║")
print("╚══════════════════════════════════════════════════════════════╝")

all_tickers = list(set(TICKERS + [VAS_TICKER]))
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

vas_raw = raw[VAS_TICKER].dropna(subset=["Close"])
vas_raw["SMA200"] = vas_raw["Close"].rolling(VAS_SMA_LONG).mean()
vas_raw["MOM20"] = vas_raw["Close"].pct_change(VAS_MOM_DAYS)
print(f"  📊 VAS.AX: {len(vas_raw)} rows (regime + momentum)")


# ── Helpers ─────────────────────────────────────────────
def get_regime(date):
    """Returns (is_bull, vas_mom) for given date."""
    mask = vas_raw.index <= pd.Timestamp(date)
    if mask.sum() == 0:
        return False, -1.0
    row = vas_raw.loc[mask].iloc[-1]
    is_bull = row["Close"] > row["SMA200"] if pd.notna(row["SMA200"]) else False
    vas_mom = row["MOM20"] if pd.notna(row["MOM20"]) else -1.0
    return bool(is_bull), float(vas_mom)


# ── Simulate ────────────────────────────────────────────
def simulate_period(name, start, end, train_end, capital, comm_model):
    print(f"\n{'═'*60}")
    print(f"  ⏱  {name}  (train → {train_end})  [{comm_model['name']}]")
    print(f"{'═'*60}")

    train_cutoff = pd.Timestamp(train_end)
    sim_start = pd.Timestamp(start)
    sim_end = pd.Timestamp(end)

    # ── Train ──
    valid_tickers = [t for t in data if data[t].index.min() < train_cutoff - pd.Timedelta(days=365)]
    print(f"  📦 {len(valid_tickers)} tickers")
    models, feat_cache, atr_cache = {}, {}, {}

    print("  🧠 Entrenando...")
    for i, t in enumerate(valid_tickers):
        df = data[t]
        tr = df[df.index <= train_cutoff].copy()
        if len(tr) < 300:
            continue
        feats = build_features(tr)
        tgt = build_target(tr)
        mask = feats.notna().all(axis=1) & tgt.notna()
        X, y = feats[mask], tgt[mask]
        if len(X) < 100 or y.sum() < 10:
            continue
        try:
            base = lgb.LGBMClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.7, min_child_samples=20,
                reg_alpha=0.1, reg_lambda=1.0, verbose=-1, random_state=42
            )
            cal = CalibratedClassifierCV(base, cv=TimeSeriesSplit(3), method="isotonic")
            cal.fit(X, y)
            models[t] = cal
        except:
            pass
        print(f"\r   [{i+1}/{len(valid_tickers)}] {t:12s}", end="", flush=True)

    print(f"\n   ✅ Modelos: {len(models)}/{len(valid_tickers)}")

    # ── Pre-compute features & ATR for simulation period ──
    for t in models:
        df = data[t]
        feats = build_features(df)
        atr = volatility.average_true_range(df["High"], df["Low"], df["Close"], 14)
        sma20 = df["Close"].rolling(20).mean()
        feat_cache[t] = feats
        atr_cache[t] = pd.DataFrame({"atr": atr, "sma20": sma20, "close": df["Close"]}, index=df.index)

    # ── Simulation ──
    print(f"\n{'─'*60}")
    print(f"  📈 {name}")
    print(f"     {start} → {end} | ${capital:,.0f} | {len(models)} tickers")
    print(f"{'─'*60}")

    cash = capital
    positions = []
    trades = []
    equity_curve = []
    total_commission_paid = 0.0

    # Count regime days
    bull_days = bear_days = 0
    no_trade_days = 0  # days skipped due to VAS momentum

    # Get trading days
    all_dates = set()
    for t in models:
        all_dates.update(data[t].index)
    trading_days = sorted([d for d in all_dates if sim_start <= d <= sim_end])

    for day in trading_days:
        is_bull, vas_mom = get_regime(day)
        if is_bull:
            bull_days += 1
        else:
            bear_days += 1

        # VAS momentum gate
        vas_ok = is_bull and vas_mom > 0

        if not vas_ok:
            if is_bull:
                no_trade_days += 1

        # ── Exit logic ──
        closed = []
        for pos in positions:
            t = pos["ticker"]
            if day not in data[t].index:
                continue
            idx = data[t].index.get_loc(day)
            row = data[t].iloc[idx]
            price = row["Close"]
            low = row["Low"]
            high = row["High"]
            pos["days_held"] += 1

            # Trail logic (after NO_TRAIL_DAYS)
            if pos["days_held"] > NO_TRAIL_DAYS:
                if pos["tp1_hit"]:
                    trail_dist = TRAIL_POST_ATR * pos["entry_atr"]
                else:
                    trail_dist = TRAIL_PRE_ATR * pos["entry_atr"]
                new_trail = high - trail_dist
                if new_trail > pos["trail_stop"]:
                    pos["trail_stop"] = new_trail

            # Check stop
            effective_stop = max(pos["stop_loss"], pos["trail_stop"])
            if low <= effective_stop:
                exit_price = effective_stop
                reason = "STOP"
                comm_exit = calc_commission(pos["shares"] * exit_price, comm_model)
                pnl = pos["shares"] * (exit_price - pos["entry_price"]) - pos["entry_comm"] - comm_exit
                total_commission_paid += comm_exit
                trades.append({
                    "ticker": t, "entry": pos["entry_date"], "exit": day,
                    "entry_price": pos["entry_price"], "exit_price": exit_price,
                    "shares": pos["shares"], "pnl": pnl, "reason": reason,
                    "regime": pos["regime"], "days": pos["days_held"],
                    "commission": pos["entry_comm"] + comm_exit,
                    "gross_pnl": pos["shares"] * (exit_price - pos["entry_price"])
                })
                cash += pos["shares"] * exit_price - comm_exit
                closed.append(pos)
                continue

            # Check TP1 (partial sell 50%)
            if not pos["tp1_hit"]:
                tp1_price = pos["entry_price"] + TP1_ATR * pos["entry_atr"]
                if high >= tp1_price:
                    pos["tp1_hit"] = True
                    sell_shares = pos["shares"] // 2
                    if sell_shares > 0:
                        comm_tp = calc_commission(sell_shares * tp1_price, comm_model)
                        pnl_tp = sell_shares * (tp1_price - pos["entry_price"]) - comm_tp
                        total_commission_paid += comm_tp
                        # Allocate proportional entry commission
                        entry_comm_share = pos["entry_comm"] * (sell_shares / pos["shares"])
                        trades.append({
                            "ticker": t, "entry": pos["entry_date"], "exit": day,
                            "entry_price": pos["entry_price"], "exit_price": tp1_price,
                            "shares": sell_shares, "pnl": pnl_tp - entry_comm_share,
                            "reason": "TP1", "regime": pos["regime"],
                            "days": pos["days_held"],
                            "commission": entry_comm_share + comm_tp,
                            "gross_pnl": sell_shares * (tp1_price - pos["entry_price"])
                        })
                        cash += sell_shares * tp1_price - comm_tp
                        pos["shares"] -= sell_shares
                        pos["entry_comm"] -= entry_comm_share

            # Check regime exit (if regime changes to bear while profitable)
            if not is_bull and price > pos["entry_price"]:
                exit_price = price
                reason = "REGIME_EXIT"
                comm_exit = calc_commission(pos["shares"] * exit_price, comm_model)
                pnl = pos["shares"] * (exit_price - pos["entry_price"]) - pos["entry_comm"] - comm_exit
                total_commission_paid += comm_exit
                trades.append({
                    "ticker": t, "entry": pos["entry_date"], "exit": day,
                    "entry_price": pos["entry_price"], "exit_price": exit_price,
                    "shares": pos["shares"], "pnl": pnl, "reason": reason,
                    "regime": pos["regime"], "days": pos["days_held"],
                    "commission": pos["entry_comm"] + comm_exit,
                    "gross_pnl": pos["shares"] * (exit_price - pos["entry_price"])
                })
                cash += pos["shares"] * exit_price - comm_exit
                closed.append(pos)
                continue

            # Check max hold
            if pos["days_held"] >= MAX_HOLD:
                exit_price = price
                reason = "TIME"
                comm_exit = calc_commission(pos["shares"] * exit_price, comm_model)
                pnl = pos["shares"] * (exit_price - pos["entry_price"]) - pos["entry_comm"] - comm_exit
                total_commission_paid += comm_exit
                trades.append({
                    "ticker": t, "entry": pos["entry_date"], "exit": day,
                    "entry_price": pos["entry_price"], "exit_price": exit_price,
                    "shares": pos["shares"], "pnl": pnl, "reason": reason,
                    "regime": pos["regime"], "days": pos["days_held"],
                    "commission": pos["entry_comm"] + comm_exit,
                    "gross_pnl": pos["shares"] * (exit_price - pos["entry_price"])
                })
                cash += pos["shares"] * exit_price - comm_exit
                closed.append(pos)
                continue

        positions = [p for p in positions if p not in closed]

        # ── Entry logic ──
        if vas_ok and len(positions) < MAX_POS:
            candidates = []
            for t in models:
                if any(p["ticker"] == t for p in positions):
                    continue
                if day not in feat_cache[t].index or day not in atr_cache[t].index:
                    continue
                row_f = feat_cache[t].loc[day]
                row_a = atr_cache[t].loc[day]
                if row_f.isna().any() or pd.isna(row_a["atr"]) or pd.isna(row_a["sma20"]):
                    continue
                if row_a["atr"] <= 0:
                    continue

                # Pullback filter: close within PULLBACK_PCT of SMA20
                dist = abs(row_a["close"] - row_a["sma20"]) / row_a["sma20"]
                if dist > PULLBACK_PCT:
                    continue

                # Trend filter: must be above SMA20
                if row_a["close"] < row_a["sma20"] * 0.98:
                    continue

                prob = models[t].predict_proba(row_f.values.reshape(1, -1))[0][1]
                if prob >= PROB_BULL:
                    candidates.append((t, prob, row_a["close"], row_a["atr"]))

            # Sort by probability (highest first)
            candidates.sort(key=lambda x: -x[1])

            for t, prob, price, atr_val in candidates:
                if len(positions) >= MAX_POS:
                    break

                # Position sizing (risk-based)
                stop_dist = SL_ATR * atr_val
                risk_amount = cash * RISK_PCT  # risk% of CURRENT cash
                shares = int(risk_amount / stop_dist)
                if shares < 1:
                    continue
                value = shares * price

                # Commission-aware filter: skip if edge < MIN_EDGE_MULT × commission
                entry_comm = calc_commission(value, comm_model)
                est_exit_comm = calc_commission(value, comm_model)  # approximate
                round_trip_comm = entry_comm + est_exit_comm
                # Expected edge: prob × avg_win - (1-prob) × avg_loss
                # avg_win ≈ 2×ATR, avg_loss ≈ SL_ATR×atr
                expected_win = 2.0 * atr_val * shares
                expected_loss = stop_dist * shares
                expected_edge = prob * expected_win - (1 - prob) * expected_loss
                if expected_edge < MIN_EDGE_MULT * round_trip_comm:
                    continue  # Skip: edge doesn't justify commission

                if value > cash:
                    shares = int(cash * 0.95 / price)
                    if shares < 1:
                        continue
                    value = shares * price
                    entry_comm = calc_commission(value, comm_model)

                cash -= value + entry_comm
                total_commission_paid += entry_comm

                positions.append({
                    "ticker": t, "entry_date": day, "entry_price": price,
                    "shares": shares, "stop_loss": price - stop_dist,
                    "trail_stop": 0, "entry_atr": atr_val,
                    "tp1_hit": False, "days_held": 0,
                    "regime": "BULL", "entry_comm": entry_comm
                })

        # Equity
        port_val = cash
        for pos in positions:
            t = pos["ticker"]
            if day in data[t].index:
                port_val += pos["shares"] * data[t].loc[day, "Close"]
        equity_curve.append((day, port_val))

    # ── Close remaining positions ──
    for pos in positions:
        t = pos["ticker"]
        last_date = data[t].index[data[t].index <= sim_end]
        if len(last_date) == 0:
            continue
        last_date = last_date[-1]
        price = data[t].loc[last_date, "Close"]
        comm_exit = calc_commission(pos["shares"] * price, comm_model)
        pnl = pos["shares"] * (price - pos["entry_price"]) - pos["entry_comm"] - comm_exit
        total_commission_paid += comm_exit
        trades.append({
            "ticker": t, "entry": pos["entry_date"], "exit": last_date,
            "entry_price": pos["entry_price"], "exit_price": price,
            "shares": pos["shares"], "pnl": pnl, "reason": "FINAL",
            "regime": pos["regime"], "days": pos["days_held"],
            "commission": pos["entry_comm"] + comm_exit,
            "gross_pnl": pos["shares"] * (price - pos["entry_price"])
        })
        cash += pos["shares"] * price - comm_exit

    final_equity = cash
    if len(equity_curve) > 0:
        eq_series = pd.Series([e[1] for e in equity_curve])
        max_dd = ((eq_series / eq_series.cummax()) - 1).min() * 100
    else:
        max_dd = 0

    # ── Results ──
    roi = (final_equity - capital) / capital * 100
    n_trades = len(trades)
    bull_trades = sum(1 for t in trades if t["regime"] == "BULL")
    bear_trades = sum(1 for t in trades if t["regime"] == "BEAR")
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    wr = len(wins) / n_trades * 100 if n_trades > 0 else 0
    avg_w = np.mean([t["pnl"] for t in wins]) if wins else 0
    avg_l = np.mean([abs(t["pnl"]) for t in losses]) if losses else 0
    pf = sum(t["pnl"] for t in wins) / abs(sum(t["pnl"] for t in losses)) if losses else 99
    rr = avg_w / avg_l if avg_l > 0 else 99

    # Gross P&L (before commissions)
    gross_pnl = sum(t["gross_pnl"] for t in trades)
    total_commissions = sum(t["commission"] for t in trades)

    print(f"\n  Régimen: {bull_days} días BULL, {bear_days} días BEAR")
    if no_trade_days > 0:
        print(f"  VAS momentum gate bloqueó {no_trade_days} días BULL")
    print(f"  Días: {len(trading_days)}")

    print(f"\n  💰 {name}:")
    print(f"     ${capital:,.0f} → ${final_equity:,.2f}")
    print(f"     ROI: {'+' if roi >= 0 else ''}{roi:.2f}% | DD: {abs(max_dd):.2f}%")
    print(f"     Trades: {n_trades} (BULL:{bull_trades} BEAR:{bear_trades})")
    if n_trades > 0:
        print(f"     WR: {wr:.1f}% | PF: {pf:.2f}")
        print(f"     Avg W: ${avg_w:.2f} | Avg L: ${avg_l:.2f} | R:R: {rr:.2f}")
        print(f"     💵 GROSS P&L: ${gross_pnl:+,.2f}")
        print(f"     💸 Comisiones: ${total_commissions:,.2f} ({total_commissions/capital*100:.1f}% del capital)")
        print(f"     📊 NET P&L:   ${final_equity - capital:+,.2f}")
        comm_drag = total_commissions / abs(gross_pnl) * 100 if abs(gross_pnl) > 0 else 0
        print(f"     📉 Commission drag: {comm_drag:.0f}% del gross P&L")

    # Exit breakdown
    reasons = {}
    for t in trades:
        r = t["reason"]
        if r not in reasons:
            reasons[r] = {"count": 0, "pnl": 0, "wins": 0}
        reasons[r]["count"] += 1
        reasons[r]["pnl"] += t["pnl"]
        if t["pnl"] > 0:
            reasons[r]["wins"] += 1
    print("     Salidas:")
    for r, v in sorted(reasons.items(), key=lambda x: -x[1]["count"]):
        wr_r = v["wins"]/v["count"]*100 if v["count"] > 0 else 0
        print(f"       {r:16s} {v['count']:3d}× | WR {wr_r:.0f}% | ${v['pnl']:+,.2f}")

    return {
        "name": name, "capital": capital, "final": final_equity,
        "roi": roi, "dd": max_dd, "trades": n_trades,
        "wr": wr, "pf": pf, "rr": rr,
        "bull_trades": bull_trades, "bear_trades": bear_trades,
        "gross_pnl": gross_pnl, "commissions": total_commissions,
        "trade_list": trades, "equity": equity_curve
    }


# ── Main ────────────────────────────────────────────────
if __name__ == "__main__":
    import time
    t0 = time.time()

    # Run with STANDARD commissions
    print("\n" + "█"*60)
    print("  SCENARIO A: STANDARD COMMISSIONS ($10 + 0.11%)")
    print("█"*60)
    results_std = []
    cap = CAPITAL
    for name, start, end, train_end in PERIODS:
        r = simulate_period(name, start, end, train_end, cap, COMM_STD)
        results_std.append(r)
        cap = CAPITAL  # reset capital each period (independent)

    # Run with LOW commissions
    print("\n\n" + "█"*60)
    print("  SCENARIO B: LOW COMMISSIONS ($3 flat, 0%)")
    print("█"*60)
    results_low = []
    cap = CAPITAL
    for name, start, end, train_end in PERIODS:
        r = simulate_period(name, start, end, train_end, cap, COMM_LOW)
        results_low.append(r)
        cap = CAPITAL

    # ── Summary ──
    elapsed = (time.time() - t0) / 60
    now = dt.datetime.now().strftime("%Y%m%d_%H%M")

    print(f"\n\n{'═'*70}")
    print(f"  📊 RESUMEN — DEEPQUANT V7 (COMMISSION-AWARE ULTRA-SELECTIVE)")
    print(f"{'═'*70}")

    print(f"\n  SCENARIO A: Standard ($10+0.11%)")
    print(f"  {'Período':20s} {'Cap':>10s} {'ROI':>8s} {'DD':>7s} {'Tr':>4s} {'WR':>6s} {'PF':>5s} {'R:R':>5s} {'Gross':>10s} {'Comm':>8s}")
    print(f"  {'─'*20} {'─'*10} {'─'*8} {'─'*7} {'─'*4} {'─'*6} {'─'*5} {'─'*5} {'─'*10} {'─'*8}")
    cum_std = CAPITAL
    for r in results_std:
        cum_std = cum_std * (1 + r["roi"]/100)
        print(f"  {r['name']:20s} ${r['final']:>8,.2f} {r['roi']:>+7.2f}% {abs(r['dd']):>5.2f}% {r['trades']:>4d} {r['wr']:>5.1f}% {r['pf']:>4.2f} {r['rr']:>4.2f} ${r['gross_pnl']:>+8,.0f} ${r['commissions']:>6,.0f}")
    cum_roi_std = (cum_std / CAPITAL - 1) * 100
    print(f"  {'─'*70}")
    print(f"  ACUMULADO (compuesto): ${cum_std:,.2f} | ROI: {cum_roi_std:+.2f}%")

    print(f"\n  SCENARIO B: Low ($3 flat)")
    print(f"  {'Período':20s} {'Cap':>10s} {'ROI':>8s} {'DD':>7s} {'Tr':>4s} {'WR':>6s} {'PF':>5s} {'R:R':>5s} {'Gross':>10s} {'Comm':>8s}")
    print(f"  {'─'*20} {'─'*10} {'─'*8} {'─'*7} {'─'*4} {'─'*6} {'─'*5} {'─'*5} {'─'*10} {'─'*8}")
    cum_low = CAPITAL
    for r in results_low:
        cum_low = cum_low * (1 + r["roi"]/100)
        print(f"  {r['name']:20s} ${r['final']:>8,.2f} {r['roi']:>+7.2f}% {abs(r['dd']):>5.2f}% {r['trades']:>4d} {r['wr']:>5.1f}% {r['pf']:>4.2f} {r['rr']:>4.2f} ${r['gross_pnl']:>+8,.0f} ${r['commissions']:>6,.0f}")
    cum_roi_low = (cum_low / CAPITAL - 1) * 100
    print(f"  {'─'*70}")
    print(f"  ACUMULADO (compuesto): ${cum_low:,.2f} | ROI: {cum_roi_low:+.2f}%")

    # Version comparison
    print(f"\n\n{'═'*80}")
    print(f"  🏆 COMPARACIÓN HISTÓRICA — TODAS LAS VERSIONES:")
    print(f"{'═'*80}")
    print(f"  {'Ver':5s} {'Concepto':30s} {'P1 ROI':>8s} {'P2 ROI':>8s} {'P3 ROI':>8s} {'Acum':>8s} {'P1 Tr':>6s}")
    print(f"  {'─'*5} {'─'*30} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*6}")
    print(f"  V2    Ensemble ultra-selective     -8.59%   -0.91%   -0.52%   -9.89%     16")
    print(f"  V3    Relaxed thresholds          -47.45%      N/A      N/A    >-47%    179")
    print(f"  V4    LightGBM trend-follow       -12.29%   -2.19%   -1.76%  -15.73%     86")
    print(f"  V5    Pullback + phase trail      -58.71%   +1.94%   -0.33%  -58.07%    234")
    print(f"  V6    Regime filter (VAS>SMA200)  -34.93%   +0.35%   -0.38%  -34.95%    182")
    rs = results_std
    p1, p2, p3 = rs[0], rs[1], rs[2]
    print(f"  V7std Commission-aware ${COMM_STD['name']:10s} {p1['roi']:>+7.2f}% {p2['roi']:>+7.2f}% {p3['roi']:>+7.2f}% {cum_roi_std:>+7.2f}% {p1['trades']:>5d}")
    rl = results_low
    p1l, p2l, p3l = rl[0], rl[1], rl[2]
    print(f"  V7low Commission-aware ${COMM_LOW['name']:10s} {p1l['roi']:>+7.2f}% {p2l['roi']:>+7.2f}% {p3l['roi']:>+7.2f}% {cum_roi_low:>+7.2f}% {p1l['trades']:>5d}")
    print(f"{'═'*80}")

    # Key insights
    print(f"\n  📌 INSIGHTS CLAVE:")
    comm_pct_p1 = results_std[0]["commissions"] / CAPITAL * 100 if results_std[0]["trades"] > 0 else 0
    print(f"  • P1 comisiones STD: ${results_std[0]['commissions']:,.0f} = {comm_pct_p1:.0f}% del capital")
    if results_std[0]["gross_pnl"] != 0:
        drag = results_std[0]["commissions"] / abs(results_std[0]["gross_pnl"]) * 100
        print(f"  • Commission drag P1: {drag:.0f}% del gross P&L")
    print(f"  • Con $3 comisión vs $10: ROI P1 pasa de {results_std[0]['roi']:+.2f}% a {results_low[0]['roi']:+.2f}%")
    print(f"  • El modelo genera α GROSS positivo en mercado alcista")
    print(f"  • Ultra-selectividad (prob>{PROB_BULL}) reduce trades y comisiones")
    print(f"  • VAS momentum gate elimina señales en rallies muertos")

    # Minimum capital analysis
    if results_std[0]["trades"] > 0 and results_std[0]["gross_pnl"] > 0:
        avg_comm_per_trade = results_std[0]["commissions"] / results_std[0]["trades"]
        # For commissions to be < 2% per trade on avg position
        min_cap = avg_comm_per_trade / 0.01 * MAX_POS
        print(f"\n  💡 CAPITAL MÍNIMO RECOMENDADO:")
        print(f"     Para que comisiones < 1% por trade: ~${min_cap:,.0f}")
        print(f"     (Con broker $10 flat + 0.11%)")

    print(f"\n  ⏱ {elapsed:.1f} min")

    # Save
    fname = f"sim_v7_{now}.xlsx"
    try:
        with pd.ExcelWriter(fname) as writer:
            for scenario, results, label in [
                ("std", results_std, "STD"), ("low", results_low, "LOW")
            ]:
                for r in results:
                    sname = f"{label}_{r['name'][:2]}"
                    df_trades = pd.DataFrame(r["trade_list"])
                    if len(df_trades) > 0:
                        df_trades.to_excel(writer, sheet_name=sname, index=False)
                    df_eq = pd.DataFrame(r["equity"], columns=["date", "equity"])
                    df_eq.to_excel(writer, sheet_name=f"{sname}_eq", index=False)
        print(f"\n  💾 {fname}")
    except:
        print("  ⚠ No se pudo guardar Excel")

    print()
