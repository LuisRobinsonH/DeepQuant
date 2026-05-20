# sim_v5_3periods.py — V5: Pullback Entry + Asymmetric R:R
# ═══════════════════════════════════════════════════════════════════════
#  V5 FOCUS: Fix the Reward:Risk ratio
#  ──────────────────────────────────────────────────────────
#  V4 PROBLEM: 35% WR but PF 0.70 → avg loss > avg win → negative
#    - Trail 1.8×ATR ≈ SL 2.0×ATR → winners don't run
#    - 83/86 exits were STOP_LOSS → trail is essentially the SL
#
#  V5 FIXES:
#    1. PULLBACK ENTRY: Buy near SMA20 support in uptrend
#       → Better entry = closer to support = more upside room
#    2. NO TRAIL FIRST 10 DAYS: Let trades develop
#    3. PARTIAL TP at 3×ATR: Lock 50% profit early
#    4. AFTER TP: Tight trail (1.5×ATR) on remaining
#    5. BEFORE TP: Only breakeven lock, wide trail (2.5×ATR)
#    6. HOLD UP TO 30 DAYS: More time for trend to unfold
#    7. MACD RELAXED: Allow near-zero (catching early turns)
#    8. Lower prob threshold: 0.48 (more opportunities)
# ═══════════════════════════════════════════════════════════════════════

import os, sys, warnings, time
import numpy as np
import pandas as pd
import yfinance as yf
import ta as ta_lib
from datetime import datetime

import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV

warnings.filterwarnings('ignore')
os.environ['SIMULATE_AU_INVESTMENT'] = '1'

# ═══════════════════════════════════════════════════════════════════════
#  CONFIGURATION V5
# ═══════════════════════════════════════════════════════════════════════
START_CAPITAL   = 8000.0
MIN_BUY         = 500.0
MAX_POSITIONS   = 6
MAX_POS_PCT     = 0.28
RISK_PER_TRADE  = 0.020       # 2% del equity en riesgo
SL_ATR_MULT     = 2.0         # Stop loss = entry - 2.0×ATR
MAX_HOLD_DAYS   = 30          # MÁS TIEMPO para que el trade se desarrolle
COOLDOWN_DAYS   = 4
MIN_STOCK_PRICE = 1.00
COMMISSION_FLAT = 10.0
COMMISSION_RATE = 0.0011

# ── Trail & TP ──
NO_TRAIL_DAYS   = 8           # NO trailing los primeros 8 días
BREAKEVEN_PROFIT = 1.5        # Breakeven lock cuando profit >= 1.5×ATR
BREAKEVEN_LOCK  = 0.3         # Lock stop en entry + 0.3×ATR
TP1_ATR         = 3.0         # TP1 parcial a 3.0×ATR
TP1_SELL_PCT    = 0.50        # Vender 50% en TP1
TRAIL_WIDE      = 2.5         # Pre-TP1: trailing amplio (2.5×ATR from peak)
TRAIL_TIGHT     = 1.5         # Post-TP1: trailing ajustado (1.5×ATR from peak)

# ── Entry filters ──
PROB_THRESHOLD  = 0.48
MIN_RSI         = 33
MAX_RSI         = 72
MIN_ADX         = 15
MIN_VOL_RATIO   = 0.9
PULLBACK_PCT    = 0.035       # Entrada: precio dentro de 3.5% del SMA20

# ── Sector mapping ──
SECTOR_MAP = {
    'BHP.AX':'Mining','RIO.AX':'Mining','FMG.AX':'Mining','S32.AX':'Mining',
    'MIN.AX':'Mining','IGO.AX':'Mining','PLS.AX':'Mining','LYC.AX':'Mining',
    'NST.AX':'Mining','EVN.AX':'Mining','RRL.AX':'Mining','SBM.AX':'Mining',
    'BSL.AX':'Mining','ILU.AX':'Mining','NHC.AX':'Mining','SGM.AX':'Mining',
    'PDN.AX':'Mining','NIC.AX':'Mining','RSG.AX':'Mining',
    'CBA.AX':'Banks','NAB.AX':'Banks','WBC.AX':'Banks','ANZ.AX':'Banks','MQG.AX':'Banks',
    'STO.AX':'Energy','WDS.AX':'Energy','ORG.AX':'Energy','BPT.AX':'Energy','APA.AX':'Energy',
    'CSL.AX':'Health','COH.AX':'Health','RMD.AX':'Health','SHL.AX':'Health',
    'PME.AX':'Health','RHC.AX':'Health','HLS.AX':'Health','MPL.AX':'Health',
    'WES.AX':'Retail','WOW.AX':'Retail','COL.AX':'Retail','HVN.AX':'Retail',
    'SUL.AX':'Retail','PMV.AX':'Retail','DMP.AX':'Retail',
    'XRO.AX':'Tech','REA.AX':'Tech','WTC.AX':'Tech','NXT.AX':'Tech','SEK.AX':'Tech','CAR.AX':'Tech',
    'GMG.AX':'REIT','SCG.AX':'REIT','DXS.AX':'REIT','MGR.AX':'REIT',
    'SGP.AX':'REIT','CHC.AX':'REIT','BWP.AX':'REIT',
    'QBE.AX':'Insurance','SUN.AX':'Insurance',
    'TLS.AX':'Telecom','TPG.AX':'Telecom','SPK.AX':'Telecom',
    'TCL.AX':'Industrial','QAN.AX':'Industrial','AMC.AX':'Industrial',
    'JHX.AX':'Industrial','QUB.AX':'Industrial','AZJ.AX':'Industrial',
}
MAX_SECTOR_POS  = 2

ASX_TICKERS = [
    'BHP.AX','CBA.AX','CSL.AX','WES.AX','NAB.AX','WBC.AX','ANZ.AX',
    'MQG.AX','FMG.AX','TLS.AX','RIO.AX','GMG.AX','STO.AX','WDS.AX',
    'QBE.AX','ALL.AX','SCG.AX','ORG.AX','NST.AX','SUN.AX','MIN.AX',
    'PLS.AX','IGO.AX','TCL.AX','S32.AX','REA.AX','QAN.AX','RMD.AX',
    'AMC.AX','BSL.AX','CPU.AX','ASX.AX','SHL.AX','JHX.AX','WOW.AX',
    'COH.AX','XRO.AX','TWE.AX','CAR.AX','SEK.AX',
    'APA.AX','ALD.AX','ARB.AX','AZJ.AX','BPT.AX','BRG.AX','BWP.AX',
    'CDA.AX','CHC.AX','CIP.AX','CLW.AX','CMW.AX','CNU.AX','COL.AX',
    'CQR.AX','CWY.AX','DMP.AX','DXS.AX','EDV.AX','ELD.AX','EVN.AX',
    'FLT.AX','GNC.AX','GOZ.AX','GWA.AX','HLS.AX','HVN.AX','IEL.AX',
    'ILU.AX','INA.AX','IRE.AX','LLC.AX','LYC.AX','MFG.AX','MGR.AX',
    'MPL.AX','NHC.AX','NHF.AX','NXT.AX','ORA.AX','ORI.AX','PDN.AX',
    'PME.AX','PMV.AX','PNI.AX','QUB.AX','RHC.AX','RRL.AX','SGM.AX',
    'SGP.AX','SOL.AX','SPK.AX','SUL.AX','SVW.AX','TAH.AX','TPG.AX',
    'VCX.AX','VEA.AX','WEB.AX','WOR.AX','WTC.AX',
]


def calc_commission(value: float) -> float:
    return max(COMMISSION_FLAT, value * COMMISSION_RATE)


# ═══════════════════════════════════════════════════════════════════════
#  FEATURES V5 — Same as V4 (20 raw features, no normalization)
# ═══════════════════════════════════════════════════════════════════════
FEATURE_COLS = [
    'dist_sma20', 'dist_sma50', 'dist_sma200',
    'ma_cross_20_50', 'ma_cross_50_200',
    'atr_pct', 'vol_regime',
    'momentum_5', 'momentum_10', 'momentum_20',
    'rsi', 'macd_diff_norm', 'adx',
    'bb_width', 'bb_position',
    'vol_rel_20',
    'close_to_high20', 'close_to_low20',
    'range_pct', 'gap_pct',
]


def engineer_features(raw_df):
    df = raw_df.copy()
    if len(df) < 220:
        return None

    c, h, l, v, o = df['close'], df['high'], df['low'], df['volume'], df['open']

    sma20  = ta_lib.trend.sma_indicator(c, 20)
    sma50  = ta_lib.trend.sma_indicator(c, 50)
    sma200 = ta_lib.trend.sma_indicator(c, 200)
    df['dist_sma20']  = (c / sma20) - 1
    df['dist_sma50']  = (c / sma50) - 1
    df['dist_sma200'] = (c / sma200) - 1
    df['ma_cross_20_50']  = (sma20 > sma50).astype(float)
    df['ma_cross_50_200'] = (sma50 > sma200).astype(float)

    atr_raw = ta_lib.volatility.average_true_range(h, l, c, 14)
    df['atr'] = atr_raw
    df['atr_pct'] = np.where(c > 0, atr_raw / c, 0.02)
    atr_ma = atr_raw.rolling(50).mean()
    df['vol_regime'] = np.where(atr_ma > 0, atr_raw / atr_ma, 1.0)

    for p in [5, 10, 20]:
        df[f'momentum_{p}'] = c.pct_change(p)

    df['rsi'] = ta_lib.momentum.rsi(c, 14) / 100.0
    macd_diff = ta_lib.trend.macd_diff(c)
    df['macd_diff_norm'] = np.where(c > 0, macd_diff / c, 0)
    df['adx'] = ta_lib.trend.adx(h, l, c, 14) / 100.0

    bb_h = ta_lib.volatility.bollinger_hband(c, 20)
    bb_l = ta_lib.volatility.bollinger_lband(c, 20)
    df['bb_width'] = np.where(c > 0, (bb_h - bb_l) / c, 0)
    bb_range = bb_h - bb_l
    df['bb_position'] = np.where(bb_range > 0, (c - bb_l) / bb_range, 0.5)

    vm20 = v.rolling(20).mean()
    df['vol_rel_20'] = np.where(vm20 > 0, v / vm20, 1.0)

    high20 = c.rolling(20).max()
    low20  = c.rolling(20).min()
    df['close_to_high20'] = np.where(high20 > 0, (c / high20) - 1, 0)
    df['close_to_low20']  = np.where(low20 > 0, (c / low20) - 1, 0)

    df['range_pct'] = np.where(c > 0, (h - l) / c, 0)
    df['gap_pct'] = (o / c.shift(1)) - 1

    # ── Target: 10-day forward return > 1.5% AND max DD < 5% ──
    fut = c.shift(-10) / c - 1
    min_p = pd.Series(np.nan, index=df.index)
    for hh in range(1, 11):
        min_p = pd.concat([min_p, l.shift(-hh)], axis=1).min(axis=1)
    max_dd = (min_p / c) - 1

    df['target'] = ((fut > 0.015) & (max_dd > -0.05)).astype(int)
    if df['target'].mean() < 0.08:
        df['target'] = ((fut > 0.01) & (max_dd > -0.06)).astype(int)
    if df['target'].sum() < 20:
        df['target'] = (fut > 0.005).astype(int)

    df['_sma20'] = sma20
    df['_sma50'] = sma50
    df['_sma200'] = sma200

    return df.replace([np.inf, -np.inf], np.nan).dropna()


# ═══════════════════════════════════════════════════════════════════════
#  MODEL V5 — Same as V4 (LightGBM + Isotonic Calibration)
# ═══════════════════════════════════════════════════════════════════════
def train_models(all_raw, train_end, min_rows=300):
    models = {}
    total = len(all_raw)
    for i, (t, raw) in enumerate(all_raw.items(), 1):
        sys.stdout.write(f"\r   [{i:3d}/{total}] {t:<10}")
        sys.stdout.flush()

        df = engineer_features(raw)
        if df is None:
            continue
        train = df[df.index <= train_end]
        if len(train) < min_rows:
            continue
        avail = [f for f in FEATURE_COLS if f in train.columns]
        if len(avail) < 10:
            continue
        X, y = train[avail], train['target']
        if y.sum() < 15 or y.mean() < 0.03:
            continue
        try:
            base = lgb.LGBMClassifier(
                n_estimators=200, learning_rate=0.05, max_depth=5,
                num_leaves=20, subsample=0.8, colsample_bytree=0.8,
                min_child_samples=20, class_weight='balanced',
                random_state=42, verbose=-1, n_jobs=-1)
            cal = CalibratedClassifierCV(base, method='isotonic', cv=3)
            cal.fit(X, y)
            models[t] = {'model': cal, 'features': avail}
        except Exception:
            try:
                cal = CalibratedClassifierCV(base, method='sigmoid', cv=3)
                cal.fit(X, y)
                models[t] = {'model': cal, 'features': avail}
            except Exception:
                continue
    print(f"\n   ✅ Modelos: {len(models)}/{total}")
    return models


# ═══════════════════════════════════════════════════════════════════════
#  SIMULATION V5: PULLBACK + ASYMMETRIC R:R
# ═══════════════════════════════════════════════════════════════════════
def run_simulation(all_raw, models, sim_start, sim_end,
                   start_capital=START_CAPITAL, label=""):
    print(f"\n{'─'*58}")
    print(f"  📈 {label}")
    print(f"     {sim_start} → {sim_end} | ${start_capital:,.0f} | {len(models)} tickers")
    print(f"{'─'*58}")

    # Pre-compute
    FEAT = {}
    PROBS = {}
    for t, m in models.items():
        raw = all_raw.get(t)
        if raw is None:
            continue
        df = engineer_features(raw)
        if df is None:
            continue
        FEAT[t] = df
        feats = m['features']
        sim = df[(df.index >= pd.Timestamp(sim_start)) &
                 (df.index <= pd.Timestamp(sim_end))]
        if sim.empty:
            PROBS[t] = {}
            continue
        X = sim[feats].replace([np.inf, -np.inf], np.nan).dropna()
        if X.empty:
            PROBS[t] = {}
            continue
        try:
            p = m['model'].predict_proba(X.values)[:, 1]
            PROBS[t] = dict(zip(X.index, p))
        except Exception:
            PROBS[t] = {}

    print(f"  Features: {len(FEAT)} | Probs: {len([t for t,p in PROBS.items() if p])}")

    all_dates = set()
    for t in FEAT:
        if t in models:
            d = FEAT[t]
            all_dates.update(
                d[(d.index >= pd.Timestamp(sim_start)) &
                  (d.index <= pd.Timestamp(sim_end))].index.tolist())
    sim_dates = sorted(all_dates)
    if not sim_dates:
        print("  ⚠️ Sin fechas")
        return None
    print(f"  Días: {len(sim_dates)}")

    cash = start_capital
    portfolio = {}
    equity_log = []
    all_trades = []
    trade_rows = []
    sl_history = {}

    for day in sim_dates:
        # ── Mark-to-market ──
        equity = cash
        for t, pos in portfolio.items():
            raw = all_raw.get(t)
            px = (float(raw.loc[day, 'close'])
                  if raw is not None and day in raw.index else pos['bp'])
            equity += pos['sh'] * px
        equity_log.append({'Fecha': day.strftime('%Y-%m-%d'), 'Equity': round(equity, 2)})

        # ════════════════════ EXITS ════════════════════
        to_sell = []
        for t, pos in list(portfolio.items()):
            raw = all_raw.get(t)
            pos['hd'] = pos.get('hd', 0) + 1

            if raw is None or day not in raw.index:
                if pos['hd'] >= MAX_HOLD_DAYS:
                    to_sell.append((t, 'EXPIRE', pos['bp']))
                continue

            row = raw.loc[day]
            px_c = float(row['close'])
            px_h = float(row['high'])
            px_l = float(row['low'])
            atr_e = pos['atr']

            pos['peak'] = max(pos.get('peak', pos['bp']), px_h)
            profit_atr = (pos['peak'] - pos['bp']) / atr_e if atr_e > 0 else 0

            # ── Breakeven lock ──
            if not pos.get('be', False) and pos['hd'] >= 2:
                if px_h >= pos['bp'] + atr_e * BREAKEVEN_PROFIT:
                    pos['stop'] = max(pos['stop'], pos['bp'] + atr_e * BREAKEVEN_LOCK)
                    pos['be'] = True

            # ── Trailing (PHASE-BASED) ──
            if pos['hd'] >= NO_TRAIL_DAYS:
                if pos.get('tp1_hit', False):
                    # Post-TP1: tight trail
                    trail = pos['peak'] - atr_e * TRAIL_TIGHT
                else:
                    # Pre-TP1: wide trail (only if in profit)
                    if px_c > pos['bp']:
                        trail = pos['peak'] - atr_e * TRAIL_WIDE
                    else:
                        trail = pos['stop']  # don't trail if underwater
                pos['stop'] = max(pos['stop'], trail)

            # ── TP1 partial (50% at 3×ATR) ──
            if not pos.get('tp1_hit', False) and px_h >= pos['bp'] + atr_e * TP1_ATR:
                tp1_px = pos['bp'] + atr_e * TP1_ATR
                sell_sh = max(1, int(pos['sh'] * TP1_SELL_PCT))
                if sell_sh > 0 and sell_sh < pos['sh']:
                    gross = sell_sh * tp1_px
                    comm = calc_commission(gross)
                    net = gross - comm
                    frac = sell_sh / pos['sh']
                    buy_cost = sell_sh * pos['bp'] + pos.get('bc', 0) * frac
                    pnl = net - buy_cost
                    pct = (tp1_px - pos['bp']) / pos['bp']

                    cash += net
                    pos['sh'] -= sell_sh
                    pos['bc'] = pos.get('bc', 0) * (1 - frac)
                    pos['tp1_hit'] = True
                    # Lock stop at entry + 1.5×ATR after partial TP
                    pos['stop'] = max(pos['stop'], pos['bp'] + atr_e * 1.5)

                    all_trades.append({
                        'Tipo': '🟡 TP1', 'Ticker': t,
                        'Fecha': day.strftime('%Y-%m-%d'),
                        'Precio': round(tp1_px, 3),
                        'Shares': sell_sh,
                        'Neto': round(net, 2),
                        'P&L': round(pnl, 2),
                        'P&L%': f"+{pct:.2%}",
                        'Razón': 'TP1_50%',
                        'Días': pos['hd'],
                    })
                    trade_rows.append({
                        'ticker': t, 'resultado': 'WIN',
                        'pnl_aud': round(pnl, 2), 'pnl_pct': round(pct*100, 2),
                        'dias': pos['hd'], 'razon': 'TP1', 'prob': pos.get('prob', 0),
                    })

            # ── Main exits ──
            if px_l <= pos['stop']:
                to_sell.append((t, 'STOP', max(pos['stop'], px_l)))
            elif pos['hd'] >= MAX_HOLD_DAYS:
                to_sell.append((t, 'TIME', px_c))

        for t, reason, sell_px in to_sell:
            if t not in portfolio:
                continue
            pos = portfolio.pop(t)
            sh = pos['sh']
            gross = sh * sell_px
            comm = calc_commission(gross)
            net = gross - comm
            buy_cost = sh * pos['bp'] + pos.get('bc', 0)
            pnl = net - buy_cost
            pct = (sell_px - pos['bp']) / pos['bp']
            cash += net
            if reason == 'STOP':
                sl_history[t] = day

            all_trades.append({
                'Tipo': '🔴 VENTA', 'Ticker': t,
                'Fecha': day.strftime('%Y-%m-%d'),
                'Precio': round(sell_px, 3),
                'Shares': sh,
                'Neto': round(net, 2),
                'P&L': round(pnl, 2),
                'P&L%': f"{pct:+.2%}",
                'Razón': reason,
                'Días': pos['hd'],
            })
            trade_rows.append({
                'ticker': t, 'resultado': 'WIN' if pnl > 0 else 'LOSS',
                'pnl_aud': round(pnl, 2), 'pnl_pct': round(pct*100, 2),
                'dias': pos['hd'], 'razon': reason, 'prob': pos.get('prob', 0),
            })

        # ════════════════════ ENTRIES ════════════════════
        slots = MAX_POSITIONS - len(portfolio)
        if cash < MIN_BUY or slots <= 0:
            continue

        candidates = []

        for t_c in models:
            if t_c in portfolio:
                continue
            last_sl = sl_history.get(t_c)
            if last_sl and (day - last_sl).days < COOLDOWN_DAYS:
                continue
            sector = SECTOR_MAP.get(t_c, 'Other')
            if sum(1 for t in portfolio if SECTOR_MAP.get(t, 'X') == sector) >= MAX_SECTOR_POS:
                continue

            raw = all_raw.get(t_c)
            if raw is None or day not in raw.index:
                continue
            rs = raw[raw.index <= day]
            if len(rs) < 210:
                continue

            px = float(raw.loc[day, 'close'])
            if px < MIN_STOCK_PRICE:
                continue

            prob = PROBS.get(t_c, {}).get(day, 0.0)
            if prob < PROB_THRESHOLD:
                continue

            # ═══════════════════════════════════════════
            #  TREND + PULLBACK FILTER
            # ═══════════════════════════════════════════
            rc = rs['close']
            sma20  = float(rc.rolling(20).mean().iloc[-1])
            sma50  = float(rc.rolling(50).mean().iloc[-1])
            sma200 = float(rc.rolling(200).mean().iloc[-1])
            if pd.isna(sma50) or pd.isna(sma200) or pd.isna(sma20):
                continue

            # TREND: SMA20 > SMA50 (short-term uptrend)
            if sma20 < sma50 * 0.99:
                continue

            # TREND: Price above SMA50 OR SMA50 > SMA200 (structural uptrend)
            if px < sma50 and sma50 < sma200:
                continue  # both bearish = skip

            # PULLBACK: Price near SMA20 (buying on support, not at peak)
            dist_sma20 = abs(px / sma20 - 1)
            if dist_sma20 > PULLBACK_PCT:
                continue  # too far from SMA20 (either above or below)

            # RSI range
            rsi_val = float(ta_lib.momentum.rsi(rc, 14).iloc[-1])
            if pd.isna(rsi_val) or rsi_val < MIN_RSI or rsi_val > MAX_RSI:
                continue

            # MACD: Relaxed — allow near-zero (catching early momentum)
            macd_d = float(ta_lib.trend.macd_diff(rc).iloc[-1])
            if pd.isna(macd_d):
                continue
            # Allow MACD > -0.2% of price (nearly zero or positive)
            if macd_d < -px * 0.002:
                continue

            # ADX trending
            adx_val = float(ta_lib.trend.adx(rs['high'], rs['low'], rc, 14).iloc[-1])
            if pd.isna(adx_val) or adx_val < MIN_ADX:
                continue

            # Volume
            vol_avg = float(rs['volume'].rolling(20).mean().iloc[-1])
            vol_now = float(rs['volume'].iloc[-1])
            vol_ratio = vol_now / vol_avg if vol_avg > 0 else 0
            if vol_ratio < MIN_VOL_RATIO:
                continue

            # Momentum: at least one of 5d/10d positive
            mom5  = float(rc.pct_change(5).iloc[-1])
            mom10 = float(rc.pct_change(10).iloc[-1])
            if mom5 < -0.02 and mom10 < -0.01:
                continue  # both negative = skip

            atr_now = float(ta_lib.volatility.average_true_range(
                rs['high'], rs['low'], rc, 14).iloc[-1])
            if pd.isna(atr_now) or atr_now <= 0:
                continue

            # Score: prob + trend quality + pullback quality
            trend_q = min(1.0, max(0, (sma20/sma50 - 1) * 20))
            pullback_q = max(0, 1.0 - dist_sma20 / PULLBACK_PCT)  # closer to SMA20 = better
            score = prob * (1.0 + trend_q * 0.3 + pullback_q * 0.4)

            candidates.append({
                't': t_c, 'px': px, 'prob': prob, 'score': score,
                'atr': atr_now, 'rsi': rsi_val, 'adx': adx_val, 'mom5': mom5,
            })

        candidates.sort(key=lambda x: x['score'], reverse=True)

        for cand in candidates:
            if cash < MIN_BUY or len(portfolio) >= MAX_POSITIONS:
                break

            t = cand['t']
            px = cand['px']
            atr_now = cand['atr']

            # Risk-based position sizing
            risk_amt = equity * RISK_PER_TRADE
            sl_dist = atr_now * SL_ATR_MULT
            sh_risk = int(risk_amt / sl_dist) if sl_dist > 0 else 0
            sh_cap = int((equity * MAX_POS_PCT) / px) if px > 0 else 0
            sh_cash = int((cash - COMMISSION_FLAT * 1.5) / px) if px > 0 else 0
            shares = max(1, min(sh_risk, sh_cap, sh_cash))

            cost = shares * px
            comm = calc_commission(cost)
            total = cost + comm
            if total > cash or cost < MIN_BUY:
                continue

            cash -= total
            portfolio[t] = {
                'sh': shares, 'bp': px, 'bc': comm,
                'stop': px - atr_now * SL_ATR_MULT,
                'atr': atr_now, 'hd': 0,
                'peak': px, 'be': False, 'tp1_hit': False,
                'prob': cand['prob'],
            }

            all_trades.append({
                'Tipo': '🟢 COMPRA', 'Ticker': t,
                'Fecha': day.strftime('%Y-%m-%d'),
                'Precio': round(px, 3),
                'Shares': shares,
                'Neto': round(total, 2),
                'P&L': '—',
                'P&L%': f"p={cand['prob']:.2f}",
                'Razón': 'PULLBACK_TREND',
                'Días': 0,
            })

    # ── Liquidate ──
    if sim_dates:
        fd = sim_dates[-1]
        for t, pos in list(portfolio.items()):
            raw = all_raw.get(t)
            px = (float(raw.loc[fd, 'close'])
                  if raw is not None and fd in raw.index else pos['bp'])
            gross = pos['sh'] * px
            comm = calc_commission(gross)
            net = gross - comm
            buy_cost = pos['sh'] * pos['bp'] + pos.get('bc', 0)
            pnl = net - buy_cost
            pct = (px - pos['bp']) / pos['bp']
            cash += net
            all_trades.append({
                'Tipo': '🔴 VENTA', 'Ticker': t,
                'Fecha': fd.strftime('%Y-%m-%d'),
                'Precio': round(px, 3),
                'Shares': pos['sh'],
                'Neto': round(net, 2),
                'P&L': round(pnl, 2),
                'P&L%': f"{pct:+.2%}",
                'Razón': 'FINAL',
                'Días': pos.get('hd', 0),
            })
            trade_rows.append({
                'ticker': t, 'resultado': 'WIN' if pnl > 0 else 'LOSS',
                'pnl_aud': round(pnl, 2), 'pnl_pct': round(pct*100, 2),
                'dias': pos.get('hd', 0), 'razon': 'FINAL', 'prob': pos.get('prob', 0),
            })

    # ── Results ──
    trades_df = pd.DataFrame(all_trades)
    eq_df = pd.DataFrame(equity_log)
    tlog = pd.DataFrame(trade_rows) if trade_rows else pd.DataFrame()

    sales = trades_df[trades_df['Tipo'].str.contains('VENTA|TP1')] if not trades_df.empty else pd.DataFrame()
    pnl_vals = (sales['P&L'].apply(pd.to_numeric, errors='coerce').dropna()
                if not sales.empty else pd.Series([0.0]))

    roi = (cash - start_capital) / start_capital
    wins = int((pnl_vals > 0).sum())
    losses = int((pnl_vals <= 0).sum())
    n_tr = wins + losses
    eq_s = eq_df['Equity'] if not eq_df.empty else pd.Series([start_capital, cash])
    max_dd = float(((eq_s.cummax() - eq_s) / eq_s.cummax()).max())
    w_sum = float(pnl_vals[pnl_vals > 0].sum()) if wins else 0
    l_sum = abs(float(pnl_vals[pnl_vals <= 0].sum())) if losses else 1
    pf = w_sum / l_sum if l_sum > 0 else float('inf')

    # Avg win / avg loss
    avg_w = w_sum / wins if wins else 0
    avg_l = l_sum / losses if losses else 0

    reason_stats = {}
    if not tlog.empty:
        for r in tlog['razon'].unique():
            rd = tlog[tlog['razon'] == r]
            reason_stats[r] = {
                'n': len(rd),
                'w': int((rd['resultado'] == 'WIN').sum()),
                'pnl': float(rd['pnl_aud'].sum()),
            }

    return {
        'label': label, 'cap_i': start_capital, 'cap_f': round(cash, 2),
        'roi': roi, 'max_dd': max_dd,
        'n_tr': n_tr, 'wins': wins, 'losses': losses,
        'wr': wins / n_tr if n_tr > 0 else 0,
        'pf': pf, 'avg_w': avg_w, 'avg_l': avg_l,
        'pnl_avg': float(pnl_vals.mean()) if n_tr > 0 else 0,
        'best': float(pnl_vals.max()) if n_tr > 0 else 0,
        'worst': float(pnl_vals.min()) if n_tr > 0 else 0,
        'days': len(sim_dates),
        'reasons': reason_stats,
        'trades_df': trades_df, 'eq_df': eq_df, 'tlog': tlog,
    }


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════
def main():
    t0 = time.time()

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  DEEPQUANT V5 — PULLBACK ENTRY + ASYMMETRIC R:R           ║")
    print("║  LightGBM × SMA20-Pullback × Phase-Based Trail × TP1      ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    periods = [
        {'name': 'P1: 2022-2024', 'te': '2021-12-31',
         'ss': '2022-01-01', 'se': '2024-12-31'},
        {'name': 'P2: 2025',      'te': '2024-12-31',
         'ss': '2025-01-01', 'se': '2025-12-31'},
        {'name': 'P3: 2026 YTD',  'te': '2025-12-31',
         'ss': '2026-01-01', 'se': '2026-12-31'},
    ]

    # Download once
    DATA = {}
    print(f"\n  📡 Descargando {len(ASX_TICKERS)} tickers...")
    for i, t in enumerate(ASX_TICKERS, 1):
        raw = yf.download(t, start='2017-01-01', end='2026-12-31',
                          progress=False, auto_adjust=True)
        if raw is not None and not raw.empty:
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            raw.columns = [c.lower() for c in raw.columns]
            raw.index = pd.to_datetime(raw.index)
            if all(c in raw.columns for c in ['close','high','low','open','volume']):
                raw = raw[['close','high','low','open','volume']].copy()
                if len(raw) > 200:
                    DATA[t] = raw
        sys.stdout.write(f"\r   {i}/{len(ASX_TICKERS)} OK:{len(DATA)}")
        sys.stdout.flush()
    print(f"\n  ✅ {len(DATA)} tickers\n")

    results = []

    for p in periods:
        print(f"\n{'═'*58}")
        print(f"  ⏱  {p['name']}  (train → {p['te']})")
        print(f"{'═'*58}")

        ALL = {}
        for t, raw in DATA.items():
            d = raw[raw.index <= pd.Timestamp(p['se'])]
            if len(d) > 200:
                ALL[t] = d

        print(f"  📦 {len(ALL)} tickers")
        print(f"  🧠 Entrenando...")
        models = train_models(ALL, p['te'])

        if not models:
            results.append(None)
            continue

        result = run_simulation(ALL, models, p['ss'], p['se'], label=p['name'])
        results.append(result)

        if result:
            print(f"\n  💰 {p['name']}:")
            print(f"     ${result['cap_i']:,.0f} → ${result['cap_f']:,.2f}")
            print(f"     ROI: {result['roi']:+.2%} | DD: {result['max_dd']:.2%}")
            print(f"     Trades: {result['n_tr']} | WR: {result['wr']:.1%} | PF: {result['pf']:.2f}")
            print(f"     Avg W: ${result['avg_w']:.2f} | Avg L: ${result['avg_l']:.2f} | R:R: {result['avg_w']/result['avg_l']:.2f}" if result['avg_l'] > 0 else "")
            if result['reasons']:
                print(f"     Exit breakdown:")
                for r, st in result['reasons'].items():
                    wr = st['w']/st['n']*100 if st['n'] > 0 else 0
                    print(f"       {r:<12} {st['n']:>3}× | WR {wr:.0f}% | ${st['pnl']:+.2f}")

    # ── Summary ──
    elapsed = time.time() - t0
    print(f"\n\n{'═'*62}")
    print(f"  📊 RESUMEN — DEEPQUANT V5 (PULLBACK + ASYMMETRIC R:R)")
    print(f"{'═'*62}")
    print(f"  {'Período':<16} {'Cap':>9} {'ROI':>8} {'DD':>6} {'Tr':>4} {'WR':>6} {'PF':>5} {'R:R':>5}")
    print(f"  {'─'*16} {'─'*9} {'─'*8} {'─'*6} {'─'*4} {'─'*6} {'─'*5} {'─'*5}")

    for r in results:
        if not r:
            print("  (sin datos)")
            continue
        rr = r['avg_w']/r['avg_l'] if r['avg_l'] > 0 else 0
        print(f"  {r['label']:<16} ${r['cap_f']:>7,.2f} {r['roi']:>+7.2%} "
              f"{r['max_dd']:>5.2%} {r['n_tr']:>3} {r['wr']:>5.1%} "
              f"{r['pf']:>4.2f} {rr:>4.2f}")

    cum = START_CAPITAL
    for r in results:
        if r:
            cum *= (1 + r['roi'])
    cum_roi = (cum - START_CAPITAL) / START_CAPITAL

    print(f"  {'─'*62}")
    print(f"  {'ACUMULADO':<16} ${cum:>7,.2f} {cum_roi:>+7.2%}")
    print(f"  {'INICIAL':<16} ${START_CAPITAL:>7,.2f}")
    print(f"{'═'*62}")
    print(f"  ⏱ {elapsed/60:.1f} min")

    # Save Excel
    ts = datetime.now().strftime('%Y%m%d_%H%M')
    out = f"sim_v5_{ts}.xlsx"
    with pd.ExcelWriter(out, engine='openpyxl') as w:
        rows = []
        for r in results:
            if not r:
                continue
            rr = r['avg_w']/r['avg_l'] if r['avg_l'] > 0 else 0
            rows.append({
                'Período': r['label'],
                'Capital_Ini': r['cap_i'], 'Capital_Fin': r['cap_f'],
                'ROI': f"{r['roi']:+.2%}", 'Max_DD': f"{r['max_dd']:.2%}",
                'Trades': r['n_tr'], 'WR': f"{r['wr']:.1%}",
                'PF': round(r['pf'], 2), 'R:R': round(rr, 2),
                'Avg_Win': round(r['avg_w'], 2), 'Avg_Loss': round(r['avg_l'], 2),
            })
        pd.DataFrame(rows).to_excel(w, sheet_name='Resumen', index=False)
        for i, r in enumerate(results):
            if not r or r['trades_df'].empty:
                continue
            r['trades_df'].to_excel(w, sheet_name=f"Trades_P{i+1}", index=False)
            if not r['eq_df'].empty:
                r['eq_df'].to_excel(w, sheet_name=f"Equity_P{i+1}", index=False)

    print(f"\n  💾 {out}")

    print(f"\n{'═'*62}")
    print("  V5 CAMBIOS CLAVE vs V4:")
    print(f"{'─'*62}")
    print("  ✅ Pullback entry (precio cerca SMA20 en uptrend)")
    print("  ✅ No trailing primeros 8 días (dar tiempo al trade)")
    print("  ✅ TP1 parcial a 3×ATR (50% sell → lock profits)")
    print(f"  ✅ Trail amplio pre-TP1 (2.5×ATR) / tight post-TP1 (1.5×ATR)")
    print(f"  ✅ Hold hasta 30 días (más tiempo para trends)")
    print(f"  ✅ MACD relajado (near-zero OK → early entries)")
    print(f"  ✅ Prob threshold 0.48 (más oportunidades)")
    print(f"{'═'*62}\n")

    # ── Comparison table vs previous versions ──
    print(f"{'═'*70}")
    print(f"  📊 COMPARACIÓN TODAS LAS VERSIONES:")
    print(f"{'═'*70}")
    print(f"  {'Versión':<10} {'P1 ROI':>8} {'P2 ROI':>8} {'P3 ROI':>8} {'Acum':>8} {'P1 WR':>6} {'P1 Tr':>6}")
    print(f"  {'─'*10} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*6} {'─'*6}")
    print(f"  {'V2':<10} {'-8.59%':>8} {'-0.91%':>8} {'-0.52%':>8} {'-9.89%':>8} {'18.8%':>6} {'16':>6}")
    print(f"  {'V3':<10} {'-47.45%':>8} {'N/A':>8} {'N/A':>8} {'>-47%':>8} {'16.8%':>6} {'179':>6}")
    print(f"  {'V4':<10} {'-12.29%':>8} {'-2.19%':>8} {'-1.76%':>8} {'-15.73%':>8} {'34.9%':>6} {'86':>6}")
    if results[0]:
        r1, r2, r3 = results[0], results[1], results[2]
        p1r = f"{r1['roi']:+.2%}" if r1 else 'N/A'
        p2r = f"{r2['roi']:+.2%}" if r2 else 'N/A'
        p3r = f"{r3['roi']:+.2%}" if r3 else 'N/A'
        p1wr = f"{r1['wr']:.1%}" if r1 else 'N/A'
        p1tr = str(r1['n_tr']) if r1 else 'N/A'
        print(f"  {'V5':<10} {p1r:>8} {p2r:>8} {p3r:>8} {cum_roi:>+7.2%} {p1wr:>6} {p1tr:>6}")
    print(f"{'═'*70}\n")


if __name__ == "__main__":
    main()
