#!/usr/bin/env python
"""
DeepQuant V8 — CommSec-Optimized Swing Trading
================================================
ROOT CAUSE FIXES from V7 analysis:
1. CommSec REAL tiered commissions ($10/$19.95/$29.95/0.12%)
2. Target 4% fwd return over 15 days (covers ~2% commission)
3. Triple VAS regime: VAS>SMA200 + VAS>SMA50 + SMA50>SMA200
4. NO partial TP1 (causes split win/loss = net negative)
5. Breakeven stop after +2×ATR → trail from there
6. Max 2 positions (bigger size → less commission drag)
7. Max 3 trades per ticker per period (SHL had 11!)
8. Equity circuit breaker: -10% from peak → cash 30 days
9. Loss streak: 3 consecutive losses → pause 15 days
10. Stock must be ABOVE SMA50 (stronger trend confirmation)
"""

import warnings, datetime as dt, sys, time
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, yfinance as yf
from pathlib import Path
from ta import momentum, trend, volatility
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
import lightgbm as lgb

# ── Config ──────────────────────────────────────────────
SYM_FILE = Path("au_stock_data/au_symbols.txt")
TICKERS  = [s.strip() for s in SYM_FILE.read_text().splitlines() if s.strip()]
TICKERS  = [t if t.endswith(".AX") else t + ".AX" for t in TICKERS]
VAS_TICKER = "VAS.AX"

# ── CommSec Tiered Commission ──────────────────────────
def commsec_commission(trade_value):
    """CommSec standard online brokerage (per trade)."""
    if trade_value <= 0:
        return 0.0
    if trade_value <= 1000:
        return 10.00
    elif trade_value <= 10000:
        return 19.95
    elif trade_value <= 25000:
        return 29.95
    else:
        return trade_value * 0.0012

# ── Strategy Parameters ────────────────────────────────
PROB_THRESH     = 0.62   # ultra-selective (was 0.58 in V7)
MAX_POS         = 2      # max 2 positions (was 3)
RISK_PCT        = 0.04   # 4% risk per trade
SL_ATR          = 2.5    # wider stop (was 2.0) — less noise
BE_ATR          = 2.0    # move stop to breakeven at +2×ATR
TRAIL_ATR       = 2.2    # trail width after breakeven
MAX_HOLD        = 35     # days — longer to let winners run
PULLBACK_PCT    = 0.035  # max distance from SMA20
MIN_POS_VALUE   = 2000   # minimum position value
MAX_TICKER_TRADES = 3    # max trades per ticker per period

# ── VAS Triple Regime ──────────────────────────────────
VAS_SMA_LONG   = 200
VAS_SMA_MED    = 50
VAS_MOM_DAYS   = 10      # short-term momentum check

# ── Risk Management ────────────────────────────────────
CIRCUIT_BREAKER_PCT = -0.10  # -10% from equity peak
CIRCUIT_BREAKER_DAYS = 30    # pause trading days
LOSS_STREAK_MAX = 3          # consecutive losses
LOSS_STREAK_PAUSE = 15       # pause trading days

# ── Periods ────────────────────────────────────────────
PERIODS = [
    ("P1: 2022-2024", "2022-01-01", "2024-12-31", "2021-12-31"),
    ("P2: 2025",      "2025-01-01", "2025-12-31", "2024-12-31"),
    ("P3: 2026 YTD",  "2026-01-01", "2026-12-31", "2025-12-31"),
]
CAPITAL = 8_000.0

# ── Features (same proven 20) ──────────────────────────
def build_features(df):
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
    return feat


def build_target(df, fwd=15, min_ret=0.04, max_dd=0.06):
    """Target: 4% return in 15 days with max 6% drawdown.
    Higher target = model learns to find moves that COVER commissions."""
    c = df["Close"]
    l = df["Low"]
    fwd_ret = c.shift(-fwd) / c - 1
    fwd_dd = pd.Series(np.nan, index=df.index)
    for i in range(len(df) - fwd):
        window = l.iloc[i + 1 : i + fwd + 1]
        fwd_dd.iloc[i] = (window.min() / c.iloc[i]) - 1
    return ((fwd_ret > min_ret) & (fwd_dd > -max_dd)).astype(int)


# ── Download ────────────────────────────────────────────
print("╔══════════════════════════════════════════════════════════════╗")
print("║  DEEPQUANT V8 — COMMSEC-OPTIMIZED SWING TRADING            ║")
print("║  LightGBM × Triple Regime × Breakeven Stop × CommSec Fees  ║")
print("╚══════════════════════════════════════════════════════════════╝")
print(f"\n  CommSec Fees: ≤$1K→$10 | $1K-$10K→$19.95 | $10K-$25K→$29.95 | $25K+→0.12%")
print(f"  Target: {15}d fwd return > 4%, max DD < 6%")
print(f"  Prob > {PROB_THRESH}, Max {MAX_POS} pos, Risk {RISK_PCT*100:.0f}%/trade")

t0 = time.time()
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
vas_raw["SMA50"]  = vas_raw["Close"].rolling(VAS_SMA_MED).mean()
vas_raw["MOM10"]  = vas_raw["Close"].pct_change(VAS_MOM_DAYS)
print(f"  📊 VAS.AX: {len(vas_raw)} rows")


# ── Regime Check ────────────────────────────────────────
def get_regime(date):
    """Triple regime: VAS>SMA200, VAS>SMA50, SMA50>SMA200, MOM10>0"""
    mask = vas_raw.index <= pd.Timestamp(date)
    if mask.sum() == 0:
        return False, {}
    row = vas_raw.loc[mask].iloc[-1]
    c = row["Close"]
    s200 = row["SMA200"] if pd.notna(row["SMA200"]) else 0
    s50  = row["SMA50"]  if pd.notna(row["SMA50"]) else 0
    mom  = row["MOM10"]  if pd.notna(row["MOM10"]) else -1

    above_200 = c > s200 if s200 > 0 else False
    above_50  = c > s50  if s50 > 0 else False
    golden    = s50 > s200 if s200 > 0 and s50 > 0 else False
    mom_pos   = mom > 0

    # STRONG BULL: all four conditions met
    strong_bull = above_200 and above_50 and golden and mom_pos
    # MODERATE BULL: at least above both SMAs + momentum OR golden cross + momentum
    moderate_bull = (above_200 and above_50 and mom_pos) or (golden and above_50 and mom_pos)

    info = {
        "above_200": above_200, "above_50": above_50,
        "golden": golden, "mom_pos": mom_pos,
        "strong_bull": strong_bull, "moderate_bull": moderate_bull
    }
    return strong_bull, info


# ── Simulate ────────────────────────────────────────────
def simulate_period(name, start, end, train_end, capital):
    print(f"\n{'═'*60}")
    print(f"  ⏱  {name}  (train → {train_end})")
    print(f"{'═'*60}")

    train_cutoff = pd.Timestamp(train_end)
    sim_start = pd.Timestamp(start)
    sim_end = pd.Timestamp(end)

    # ── Train ──
    valid = [t for t in data if data[t].index.min() < train_cutoff - pd.Timedelta(days=365)]
    print(f"  📦 {len(valid)} tickers")
    models = {}

    print("  🧠 Entrenando (target: 15d fwd >4%, max DD <6%)...")
    for i, t in enumerate(valid):
        df = data[t]
        tr = df[df.index <= train_cutoff].copy()
        if len(tr) < 300:
            continue
        feats = build_features(tr)
        tgt = build_target(tr)
        mask = feats.notna().all(axis=1) & tgt.notna()
        X, y = feats[mask], tgt[mask]
        if len(X) < 100 or y.sum() < 8:
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
        print(f"\r   [{i+1}/{len(valid)}] {t:12s}", end="", flush=True)
    print(f"\n   ✅ Modelos: {len(models)}/{len(valid)}")

    # ── Pre-compute ──
    feat_cache, price_cache = {}, {}
    for t in models:
        df = data[t]
        feats = build_features(df)
        atr = volatility.average_true_range(df["High"], df["Low"], df["Close"], 14)
        sma20 = df["Close"].rolling(20).mean()
        sma50 = df["Close"].rolling(50).mean()
        feat_cache[t] = feats
        price_cache[t] = pd.DataFrame({
            "atr": atr, "sma20": sma20, "sma50": sma50,
            "close": df["Close"], "high": df["High"], "low": df["Low"]
        }, index=df.index)

    # ── Sim ──
    print(f"\n{'─'*60}")
    print(f"  📈 {name}")
    print(f"     {start} → {end} | ${capital:,.0f} | {len(models)} tickers")
    print(f"{'─'*60}")

    cash = capital
    positions = []
    trades = []
    equity_curve = []
    ticker_trade_count = {}  # track trades per ticker
    total_comm = 0.0

    # Risk management state
    equity_peak = capital
    circuit_breaker_until = None
    consecutive_losses = 0
    loss_streak_until = None

    # Regime counters
    strong_bull_days = 0
    moderate_bull_days = 0
    bear_days = 0
    cb_days = 0
    ls_days = 0

    # Trading days
    all_dates = set()
    for t in models:
        all_dates.update(data[t].index)
    trading_days = sorted([d for d in all_dates if sim_start <= d <= sim_end])

    for day in trading_days:
        strong_bull, regime_info = get_regime(day)

        if strong_bull:
            strong_bull_days += 1
        elif regime_info.get("moderate_bull"):
            moderate_bull_days += 1
        else:
            bear_days += 1

        # Check circuit breaker
        if circuit_breaker_until and day < circuit_breaker_until:
            cb_days += 1
            # Still process exits
            trading_allowed = False
        elif loss_streak_until and day < loss_streak_until:
            ls_days += 1
            trading_allowed = False
        else:
            trading_allowed = True
            circuit_breaker_until = None
            loss_streak_until = None

        # ── EXIT LOGIC ──
        closed = []
        for pos in positions:
            t = pos["ticker"]
            if day not in data[t].index:
                continue
            row = data[t].loc[day]
            price = row["Close"]
            low   = row["Low"]
            high  = row["High"]
            pos["days_held"] += 1

            # Update highest high
            if high > pos["highest_high"]:
                pos["highest_high"] = high

            # ── Breakeven stop logic ──
            # Once price reaches entry + BE_ATR×ATR, move stop to entry price
            if not pos["at_breakeven"]:
                be_price = pos["entry_price"] + BE_ATR * pos["entry_atr"]
                if high >= be_price:
                    pos["at_breakeven"] = True
                    pos["stop_loss"] = pos["entry_price"]  # move to breakeven
                    pos["trail_active"] = True

            # ── Trailing stop (only AFTER breakeven) ──
            if pos["trail_active"]:
                new_trail = pos["highest_high"] - TRAIL_ATR * pos["entry_atr"]
                if new_trail > pos["stop_loss"]:
                    pos["stop_loss"] = new_trail

            # Check stop
            if low <= pos["stop_loss"]:
                exit_price = max(pos["stop_loss"], low)  # realistic fill
                exit_comm = commsec_commission(pos["shares"] * exit_price)
                gross = pos["shares"] * (exit_price - pos["entry_price"])
                net = gross - pos["entry_comm"] - exit_comm
                total_comm += exit_comm
                reason = "BE_STOP" if pos["at_breakeven"] else "STOP"
                trades.append({
                    "ticker": t, "entry": pos["entry_date"], "exit": day,
                    "entry_price": pos["entry_price"], "exit_price": exit_price,
                    "shares": pos["shares"], "pnl": net, "reason": reason,
                    "regime": "BULL", "days": pos["days_held"],
                    "commission": pos["entry_comm"] + exit_comm,
                    "gross_pnl": gross, "at_breakeven": pos["at_breakeven"]
                })
                cash += pos["shares"] * exit_price - exit_comm
                if net <= 0:
                    consecutive_losses += 1
                else:
                    consecutive_losses = 0
                closed.append(pos)
                continue

            # Regime exit: if NOT even moderate bull and profitable
            if not regime_info.get("moderate_bull") and not strong_bull:
                if price > pos["entry_price"]:
                    exit_comm = commsec_commission(pos["shares"] * price)
                    gross = pos["shares"] * (price - pos["entry_price"])
                    net = gross - pos["entry_comm"] - exit_comm
                    total_comm += exit_comm
                    trades.append({
                        "ticker": t, "entry": pos["entry_date"], "exit": day,
                        "entry_price": pos["entry_price"], "exit_price": price,
                        "shares": pos["shares"], "pnl": net, "reason": "REGIME_EXIT",
                        "regime": "BULL", "days": pos["days_held"],
                        "commission": pos["entry_comm"] + exit_comm,
                        "gross_pnl": gross, "at_breakeven": pos["at_breakeven"]
                    })
                    cash += pos["shares"] * price - exit_comm
                    if net <= 0:
                        consecutive_losses += 1
                    else:
                        consecutive_losses = 0
                    closed.append(pos)
                    continue

            # Max hold
            if pos["days_held"] >= MAX_HOLD:
                exit_comm = commsec_commission(pos["shares"] * price)
                gross = pos["shares"] * (price - pos["entry_price"])
                net = gross - pos["entry_comm"] - exit_comm
                total_comm += exit_comm
                trades.append({
                    "ticker": t, "entry": pos["entry_date"], "exit": day,
                    "entry_price": pos["entry_price"], "exit_price": price,
                    "shares": pos["shares"], "pnl": net, "reason": "TIME",
                    "regime": "BULL", "days": pos["days_held"],
                    "commission": pos["entry_comm"] + exit_comm,
                    "gross_pnl": gross, "at_breakeven": pos["at_breakeven"]
                })
                cash += pos["shares"] * price - exit_comm
                if net <= 0:
                    consecutive_losses += 1
                else:
                    consecutive_losses = 0
                closed.append(pos)
                continue

        positions = [p for p in positions if p not in closed]

        # ── EQUITY TRACKING ──
        port_val = cash
        for pos in positions:
            t = pos["ticker"]
            if day in data[t].index:
                port_val += pos["shares"] * data[t].loc[day, "Close"]
        equity_curve.append((day, port_val))

        # Update equity peak
        if port_val > equity_peak:
            equity_peak = port_val

        # Circuit breaker check
        drawdown_from_peak = (port_val - equity_peak) / equity_peak
        if drawdown_from_peak <= CIRCUIT_BREAKER_PCT and circuit_breaker_until is None:
            circuit_breaker_until = day + pd.Timedelta(days=CIRCUIT_BREAKER_DAYS)
            # Close all positions
            for pos in positions:
                t = pos["ticker"]
                if day in data[t].index:
                    price = data[t].loc[day, "Close"]
                    exit_comm = commsec_commission(pos["shares"] * price)
                    gross = pos["shares"] * (price - pos["entry_price"])
                    net = gross - pos["entry_comm"] - exit_comm
                    total_comm += exit_comm
                    trades.append({
                        "ticker": t, "entry": pos["entry_date"], "exit": day,
                        "entry_price": pos["entry_price"], "exit_price": price,
                        "shares": pos["shares"], "pnl": net, "reason": "CIRCUIT_BRK",
                        "regime": "BULL", "days": pos["days_held"],
                        "commission": pos["entry_comm"] + exit_comm,
                        "gross_pnl": gross, "at_breakeven": pos.get("at_breakeven", False)
                    })
                    cash += pos["shares"] * price - exit_comm
            positions = []
            continue

        # Loss streak check
        if consecutive_losses >= LOSS_STREAK_MAX and loss_streak_until is None:
            loss_streak_until = day + pd.Timedelta(days=LOSS_STREAK_PAUSE)

        # ── ENTRY LOGIC ──
        if not trading_allowed or not strong_bull:
            continue
        if len(positions) >= MAX_POS:
            continue

        candidates = []
        for t in models:
            # Already in position
            if any(p["ticker"] == t for p in positions):
                continue
            # Max trades per ticker
            tc = ticker_trade_count.get(t, 0)
            if tc >= MAX_TICKER_TRADES:
                continue
            # Data check
            if day not in feat_cache[t].index or day not in price_cache[t].index:
                continue
            row_f = feat_cache[t].loc[day]
            row_p = price_cache[t].loc[day]
            if row_f.isna().any() or pd.isna(row_p["atr"]) or row_p["atr"] <= 0:
                continue

            price = row_p["close"]
            sma20 = row_p["sma20"]
            sma50 = row_p["sma50"]

            # FILTER: Stock must be ABOVE SMA50 (in uptrend)
            if pd.isna(sma50) or price < sma50:
                continue

            # FILTER: Pullback near SMA20 (within PULLBACK_PCT)
            if pd.isna(sma20):
                continue
            dist_sma20 = abs(price - sma20) / sma20
            if dist_sma20 > PULLBACK_PCT:
                continue

            # FILTER: Price above SMA20 (not below)
            if price < sma20 * 0.99:
                continue

            # FILTER: Volatility not extreme (vol_regime < 1.4)
            vol_regime = row_f.get("vol_regime", 1.0)
            if pd.notna(vol_regime) and vol_regime > 1.4:
                continue

            # FILTER: RSI between 35-70 (not extreme)
            rsi = row_f.get("rsi", 0.5)
            if pd.notna(rsi) and (rsi < 0.35 or rsi > 0.70):
                continue

            # ML probability
            prob = models[t].predict_proba(row_f.values.reshape(1, -1))[0][1]
            if prob >= PROB_THRESH:
                candidates.append((t, prob, price, row_p["atr"]))

        # Sort by probability
        candidates.sort(key=lambda x: -x[1])

        for t, prob, price, atr_val in candidates:
            if len(positions) >= MAX_POS:
                break

            # Position sizing (risk-based)
            stop_dist = SL_ATR * atr_val
            risk_amount = cash * RISK_PCT
            shares = int(risk_amount / stop_dist)
            if shares < 1:
                continue

            value = shares * price
            if value < MIN_POS_VALUE:
                continue

            # Cap at available cash
            entry_comm = commsec_commission(value)
            if value + entry_comm > cash:
                shares = int((cash - 25) * 0.95 / price)  # leave buffer
                if shares < 1:
                    continue
                value = shares * price
                if value < MIN_POS_VALUE:
                    continue
                entry_comm = commsec_commission(value)

            # Commission-aware: expected edge must be > 2× round-trip commission
            est_exit_comm = commsec_commission(value)
            rt_comm = entry_comm + est_exit_comm
            expected_win_gross = 3.0 * atr_val * shares  # expected winner ~3×ATR
            expected_loss_gross = stop_dist * shares
            expected_edge = prob * expected_win_gross - (1 - prob) * expected_loss_gross
            if expected_edge < 2.0 * rt_comm:
                continue

            cash -= value + entry_comm
            total_comm += entry_comm
            ticker_trade_count[t] = ticker_trade_count.get(t, 0) + 1

            positions.append({
                "ticker": t, "entry_date": day, "entry_price": price,
                "shares": shares, "stop_loss": price - stop_dist,
                "entry_atr": atr_val, "entry_comm": entry_comm,
                "days_held": 0, "at_breakeven": False,
                "trail_active": False, "highest_high": price
            })

    # ── Close remaining ──
    for pos in positions:
        t = pos["ticker"]
        last = data[t].index[data[t].index <= sim_end]
        if len(last) == 0:
            continue
        price = data[t].loc[last[-1], "Close"]
        exit_comm = commsec_commission(pos["shares"] * price)
        gross = pos["shares"] * (price - pos["entry_price"])
        net = gross - pos["entry_comm"] - exit_comm
        total_comm += exit_comm
        trades.append({
            "ticker": t, "entry": pos["entry_date"], "exit": last[-1],
            "entry_price": pos["entry_price"], "exit_price": price,
            "shares": pos["shares"], "pnl": net, "reason": "FINAL",
            "regime": "BULL", "days": pos["days_held"],
            "commission": pos["entry_comm"] + exit_comm,
            "gross_pnl": gross, "at_breakeven": pos.get("at_breakeven", False)
        })
        cash += pos["shares"] * price - exit_comm

    # ── Results ──
    final = cash
    roi = (final - capital) / capital * 100
    n = len(trades)
    if len(equity_curve) > 0:
        eq = pd.Series([e[1] for e in equity_curve])
        dd = ((eq / eq.cummax()) - 1).min() * 100
    else:
        dd = 0

    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    wr = len(wins) / n * 100 if n > 0 else 0
    avg_w = np.mean([t["pnl"] for t in wins]) if wins else 0
    avg_l = np.mean([abs(t["pnl"]) for t in losses]) if losses else 0
    pf = sum(t["pnl"] for t in wins) / abs(sum(t["pnl"] for t in losses)) if losses and sum(t["pnl"] for t in losses) != 0 else 99
    rr = avg_w / avg_l if avg_l > 0 else 99

    gross_pnl = sum(t["gross_pnl"] for t in trades)
    total_comms = sum(t["commission"] for t in trades)

    print(f"\n  Régimen: {strong_bull_days}d Strong Bull, {moderate_bull_days}d Moderate, {bear_days}d Bear")
    if cb_days > 0:
        print(f"  🛑 Circuit breaker activo: {cb_days} días")
    if ls_days > 0:
        print(f"  ⏸️  Loss streak pausa: {ls_days} días")

    print(f"\n  💰 {name}:")
    print(f"     ${capital:,.0f} → ${final:,.2f}")
    print(f"     ROI: {'+' if roi >= 0 else ''}{roi:.2f}% | DD: {abs(dd):.2f}%")
    print(f"     Trades: {n}")
    if n > 0:
        print(f"     WR: {wr:.1f}% | PF: {pf:.2f} | R:R: {rr:.2f}")
        print(f"     Avg W: ${avg_w:.2f} | Avg L: ${avg_l:.2f}")
        print(f"     💵 GROSS P&L: ${gross_pnl:+,.2f}")
        print(f"     💸 CommSec fees: ${total_comms:,.2f} ({total_comms/capital*100:.1f}% cap)")
        print(f"     📊 NET P&L:   ${final - capital:+,.2f}")
        if abs(gross_pnl) > 0:
            drag = total_comms / abs(gross_pnl) * 100
            print(f"     📉 Commission drag: {drag:.0f}% del gross")

        # Breakeven effectiveness
        be_stops = [t for t in trades if t["reason"] == "BE_STOP"]
        reg_stops = [t for t in trades if t["reason"] == "STOP"]
        if be_stops:
            avg_be = np.mean([t["gross_pnl"] for t in be_stops])
            print(f"     🎯 BE stops: {len(be_stops)}× avg gross ${avg_be:+.2f}")
        if reg_stops:
            avg_rs = np.mean([t["gross_pnl"] for t in reg_stops])
            print(f"     ❌ Full stops: {len(reg_stops)}× avg gross ${avg_rs:+.2f}")

        # Exit breakdown
        reasons = {}
        for t in trades:
            r = t["reason"]
            if r not in reasons:
                reasons[r] = {"count": 0, "pnl": 0, "gross": 0, "comm": 0, "wins": 0}
            reasons[r]["count"] += 1
            reasons[r]["pnl"] += t["pnl"]
            reasons[r]["gross"] += t["gross_pnl"]
            reasons[r]["comm"] += t["commission"]
            if t["pnl"] > 0:
                reasons[r]["wins"] += 1
        print("     Salidas:")
        for r, v in sorted(reasons.items(), key=lambda x: -x[1]["count"]):
            wr_r = v["wins"] / v["count"] * 100 if v["count"] > 0 else 0
            print(f"       {r:14s} {v['count']:3d}× | WR {wr_r:.0f}% | Net ${v['pnl']:+,.2f} | Gross ${v['gross']:+,.2f} | Comm ${v['comm']:,.2f}")

        # Top tickers
        by_t = {}
        for tr in trades:
            tk = tr["ticker"]
            if tk not in by_t:
                by_t[tk] = {"n": 0, "net": 0, "gross": 0, "comm": 0}
            by_t[tk]["n"] += 1
            by_t[tk]["net"] += tr["pnl"]
            by_t[tk]["gross"] += tr["gross_pnl"]
            by_t[tk]["comm"] += tr["commission"]
        sorted_t = sorted(by_t.items(), key=lambda x: x[1]["net"])
        if len(sorted_t) > 3:
            print(f"     Peores:")
            for tk, v in sorted_t[:3]:
                print(f"       {tk:10s} {v['n']}× Net:${v['net']:+.2f} Gross:${v['gross']:+.2f} Comm:${v['comm']:.2f}")
            print(f"     Mejores:")
            for tk, v in sorted_t[-3:]:
                print(f"       {tk:10s} {v['n']}× Net:${v['net']:+.2f} Gross:${v['gross']:+.2f} Comm:${v['comm']:.2f}")

    return {
        "name": name, "capital": capital, "final": final,
        "roi": roi, "dd": dd, "trades": n, "wr": wr, "pf": pf, "rr": rr,
        "gross_pnl": gross_pnl, "commissions": total_comms,
        "trade_list": trades, "equity": equity_curve
    }


# ── Main ────────────────────────────────────────────────
if __name__ == "__main__":
    results = []
    for name, start, end, train_end in PERIODS:
        r = simulate_period(name, start, end, train_end, CAPITAL)
        results.append(r)

    elapsed = (time.time() - t0) / 60
    now = dt.datetime.now().strftime("%Y%m%d_%H%M")

    print(f"\n\n{'═'*70}")
    print(f"  📊 RESUMEN — DEEPQUANT V8 (COMMSEC-OPTIMIZED)")
    print(f"{'═'*70}")
    hdr = f"  {'Período':20s} {'Cap':>10s} {'ROI':>8s} {'DD':>7s} {'Tr':>4s} {'WR':>6s} {'PF':>5s} {'R:R':>5s} {'Gross':>10s} {'Comm':>8s}"
    sep = f"  {'─'*20} {'─'*10} {'─'*8} {'─'*7} {'─'*4} {'─'*6} {'─'*5} {'─'*5} {'─'*10} {'─'*8}"
    print(hdr)
    print(sep)
    cum = CAPITAL
    for r in results:
        cum *= (1 + r["roi"] / 100)
        rt = r["roi"]
        print(f"  {r['name']:20s} ${r['final']:>8,.2f} {rt:>+7.2f}% {abs(r['dd']):>5.2f}% {r['trades']:>4d} {r['wr']:>5.1f}% {r['pf']:>4.2f} {r['rr']:>4.2f} ${r['gross_pnl']:>+8,.0f} ${r['commissions']:>6,.0f}")
    print(sep)
    cum_roi = (cum / CAPITAL - 1) * 100
    print(f"  ACUMULADO: ${cum:,.2f} | ROI: {cum_roi:+.2f}%")

    # Version comparison
    print(f"\n\n{'═'*80}")
    print(f"  🏆 COMPARACIÓN — TODAS LAS VERSIONES:")
    print(f"{'═'*80}")
    print(f"  {'Ver':5s} {'P1 ROI':>8s} {'P2 ROI':>8s} {'P3 ROI':>8s} {'Acum':>8s} {'P1 Tr':>6s} {'P2 WR':>6s} {'P2 PF':>6s}")
    print(f"  {'─'*5} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*6} {'─'*6} {'─'*6}")
    print(f"  V2    -8.59%   -0.91%   -0.52%   -9.89%     16  68.8%  N/A")
    print(f"  V4   -12.29%   -2.19%   -1.76%  -15.73%     86  40.0%  N/A")
    print(f"  V5   -58.71%   +1.94%   -0.33%  -58.07%    234  62.9%  1.16")
    print(f"  V6   -34.93%   +0.35%   -0.38%  -34.95%    182  40.6%  1.04")
    print(f"  V7   -47.68%  +19.41%   -8.75%  -42.99%     63  61.9%  2.36")
    p1, p2, p3 = results[0], results[1], results[2]
    print(f"  V8   {p1['roi']:>+7.2f}% {p2['roi']:>+7.2f}% {p3['roi']:>+7.2f}% {cum_roi:>+7.2f}% {p1['trades']:>5d} {p2['wr']:>5.1f}% {p2['pf']:>5.2f}")
    print(f"{'═'*80}")

    print(f"\n  📌 V8 CAMBIOS CLAVE:")
    print(f"  • CommSec real: $10/$19.95/$29.95/0.12%")
    print(f"  • Target 4% en 15 días (cubre comisiones)")
    print(f"  • Triple régimen: VAS>SMA200 + VAS>SMA50 + SMA50>SMA200 + MOM10>0")
    print(f"  • Sin TP1 parcial → breakeven stop a +2×ATR, luego trail 2.2×ATR")
    print(f"  • Max 2 posiciones, max 3 trades/ticker, min $2,000/posición")
    print(f"  • Circuit breaker (-10% equity) + Loss streak (3× → pausa 15d)")
    print(f"  • Stock > SMA50, RSI 35-70, vol_regime < 1.4")

    print(f"\n  ⏱ {elapsed:.1f} min")

    # Save
    fname = f"sim_v8_{now}.xlsx"
    try:
        with pd.ExcelWriter(fname) as writer:
            for r in results:
                sname = r["name"][:12].replace(":", "").replace(" ", "_")
                df_trades = pd.DataFrame(r["trade_list"])
                if len(df_trades) > 0:
                    df_trades.to_excel(writer, sheet_name=sname, index=False)
                df_eq = pd.DataFrame(r["equity"], columns=["date", "equity"])
                df_eq.to_excel(writer, sheet_name=f"{sname}_eq", index=False)
        print(f"\n  💾 {fname}")
    except Exception as e:
        print(f"  ⚠ Excel: {e}")

    print()
