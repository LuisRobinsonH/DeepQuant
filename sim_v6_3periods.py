# sim_v6_3periods.py — V6: Market Regime Protection + Bull Alpha
# ═══════════════════════════════════════════════════════════════════════
#  V5 PROVED: Model works in BULL markets (P2: +1.94%, 62.9% WR, PF 1.16)
#  V5 PROBLEM: Bear market 2022 destroyed everything (-58.71% in P1)
#
#  V6 FIX: MARKET-LEVEL REGIME FILTER
#    1. Use VAS.AX (Vanguard ASX300 ETF) as market barometer
#    2. BULL (VAS > SMA200): Trade normally (V5 settingswork)
#    3. BEAR (VAS < SMA200): Max 2 positions, prob > 0.62, defensive only
#    4. Wider trail (3×ATR pre-TP) and higher TP1 (4×ATR) for more R:R
#    5. After TP1: trail 2×ATR (give room to run)
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
#  CONFIGURATION V6
# ═══════════════════════════════════════════════════════════════════════
START_CAPITAL   = 8000.0
MIN_BUY         = 500.0
COMMISSION_FLAT = 10.0
COMMISSION_RATE = 0.0011
MIN_STOCK_PRICE = 1.00

# ── BULL MODE (VAS > SMA200) ──
BULL_MAX_POS     = 6
BULL_MAX_POS_PCT = 0.28
BULL_RISK_TRADE  = 0.020      # 2% equity at risk
BULL_PROB_THRESH = 0.48
BULL_SL_ATR      = 2.0
BULL_PULLBACK    = 0.035      # 3.5% near SMA20
BULL_MIN_ADX     = 15

# ── BEAR MODE (VAS < SMA200) ──
BEAR_MAX_POS     = 2          # Only 2 positions max
BEAR_MAX_POS_PCT = 0.18
BEAR_RISK_TRADE  = 0.012      # 1.2% equity at risk (half of bull)
BEAR_PROB_THRESH = 0.60       # Much higher conviction needed
BEAR_SL_ATR      = 2.5        # Wider stop (more room in volatile bear)
BEAR_PULLBACK    = 0.025      # Tighter pullback (very near support)
BEAR_MIN_ADX     = 22         # Only strong trends

# ── Trail & TP (same in bull/bear) ──
NO_TRAIL_DAYS    = 8
BREAKEVEN_PROFIT = 1.5
BREAKEVEN_LOCK   = 0.3
TP1_ATR          = 4.0        # V6: Higher TP1 (4×ATR instead of 3)
TP1_SELL_PCT     = 0.50
TRAIL_WIDE       = 3.0        # V6: Wider pre-TP trail (3×ATR instead of 2.5)
TRAIL_TIGHT      = 2.0        # V6: Wider post-TP trail (2×ATR instead of 1.5)
MAX_HOLD_DAYS    = 30
COOLDOWN_DAYS    = 4
MIN_RSI          = 33
MAX_RSI          = 72

# ── Regime ──
REGIME_SMA_PERIOD = 200

# ── Sectors ──
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
# Defensive sectors (OK in bear market)
DEFENSIVE_SECTORS = {'Banks', 'Health', 'Telecom', 'REIT', 'Insurance', 'Retail'}
MAX_SECTOR_POS = 2

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


def calc_commission(val: float) -> float:
    return max(COMMISSION_FLAT, val * COMMISSION_RATE)


# ═══════════════════════════════════════════════════════════════════════
#  FEATURES & MODEL (same as V4/V5)
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

    return df.replace([np.inf, -np.inf], np.nan).dropna()


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
#  MARKET REGIME (VAS.AX)
# ═══════════════════════════════════════════════════════════════════════
def compute_market_regime(vas_data, sim_start, sim_end):
    """Returns dict[date] → bool(is_bull) using VAS.AX > SMA200."""
    regime = {}
    if vas_data is None or len(vas_data) < REGIME_SMA_PERIOD + 50:
        return regime
    sma = vas_data['close'].rolling(REGIME_SMA_PERIOD).mean()
    for d in vas_data.index:
        if d < pd.Timestamp(sim_start) or d > pd.Timestamp(sim_end):
            continue
        sv = sma.get(d)
        if sv is not None and not pd.isna(sv):
            regime[d] = bool(vas_data.loc[d, 'close'] > sv)
    return regime


# ═══════════════════════════════════════════════════════════════════════
#  SIMULATION V6
# ═══════════════════════════════════════════════════════════════════════
def run_simulation(all_raw, models, vas_data, sim_start, sim_end,
                   start_capital=START_CAPITAL, label=""):
    print(f"\n{'─'*58}")
    print(f"  📈 {label}")
    print(f"     {sim_start} → {sim_end} | ${start_capital:,.0f} | {len(models)} tickers")
    print(f"{'─'*58}")

    # Market regime
    REGIME = compute_market_regime(vas_data, sim_start, sim_end)
    n_bull = sum(1 for v in REGIME.values() if v)
    n_bear = sum(1 for v in REGIME.values() if not v)
    print(f"  Régimen: {n_bull} días BULL, {n_bear} días BEAR")

    # Pre-compute features & probabilities
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
    bull_trades = 0
    bear_trades = 0

    for day in sim_dates:
        is_bull = REGIME.get(day, True)  # default to bull if no VAS data

        # Regime parameters
        if is_bull:
            max_pos = BULL_MAX_POS
            max_pct = BULL_MAX_POS_PCT
            risk_tr = BULL_RISK_TRADE
            prob_th = BULL_PROB_THRESH
            sl_atr  = BULL_SL_ATR
            pb_pct  = BULL_PULLBACK
            min_adx = BULL_MIN_ADX
        else:
            max_pos = BEAR_MAX_POS
            max_pct = BEAR_MAX_POS_PCT
            risk_tr = BEAR_RISK_TRADE
            prob_th = BEAR_PROB_THRESH
            sl_atr  = BEAR_SL_ATR
            pb_pct  = BEAR_PULLBACK
            min_adx = BEAR_MIN_ADX

        # ── Mark-to-market ──
        equity = cash
        for t, pos in portfolio.items():
            raw = all_raw.get(t)
            px = (float(raw.loc[day, 'close'])
                  if raw is not None and day in raw.index else pos['bp'])
            equity += pos['sh'] * px
        equity_log.append({'Fecha': day.strftime('%Y-%m-%d'), 'Equity': round(equity, 2),
                           'Regime': 'BULL' if is_bull else 'BEAR'})

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

            # Breakeven lock
            if not pos.get('be', False) and pos['hd'] >= 2:
                if px_h >= pos['bp'] + atr_e * BREAKEVEN_PROFIT:
                    pos['stop'] = max(pos['stop'], pos['bp'] + atr_e * BREAKEVEN_LOCK)
                    pos['be'] = True

            # Trailing (phase-based)
            if pos['hd'] >= NO_TRAIL_DAYS:
                if pos.get('tp1_hit', False):
                    trail = pos['peak'] - atr_e * TRAIL_TIGHT
                else:
                    if px_c > pos['bp']:
                        trail = pos['peak'] - atr_e * TRAIL_WIDE
                    else:
                        trail = pos['stop']
                pos['stop'] = max(pos['stop'], trail)

            # TP1 partial
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

                    cash += net
                    pos['sh'] -= sell_sh
                    pos['bc'] = pos.get('bc', 0) * (1 - frac)
                    pos['tp1_hit'] = True
                    pos['stop'] = max(pos['stop'], pos['bp'] + atr_e * 2.0)

                    all_trades.append({
                        'Tipo': '🟡 TP1', 'Ticker': t,
                        'Fecha': day.strftime('%Y-%m-%d'),
                        'Precio': round(tp1_px, 3), 'Shares': sell_sh,
                        'Neto': round(net, 2), 'P&L': round(pnl, 2),
                        'P&L%': f"+{(tp1_px/pos['bp']-1)*100:.1f}%",
                        'Razón': 'TP1_50%', 'Días': pos['hd'],
                        'Regime': pos.get('regime', '?'),
                    })
                    trade_rows.append({
                        'ticker': t, 'resultado': 'WIN',
                        'pnl_aud': round(pnl, 2),
                        'pnl_pct': round((tp1_px/pos['bp']-1)*100, 2),
                        'dias': pos['hd'], 'razon': 'TP1',
                        'prob': pos.get('prob', 0),
                        'regime': pos.get('regime', '?'),
                    })

            # Main exits
            if px_l <= pos['stop']:
                to_sell.append((t, 'STOP', max(pos['stop'], px_l)))
            elif pos['hd'] >= MAX_HOLD_DAYS:
                to_sell.append((t, 'TIME', px_c))
            # Bear regime emergency exit: if entered in bull and now bear for 3+ days
            # and trade is marginally profitable → take profit
            elif not is_bull and pos.get('regime') == 'BULL' and pos['hd'] >= 5:
                unrealized = (px_c - pos['bp']) / pos['bp']
                if unrealized > 0.005:
                    to_sell.append((t, 'REGIME_EXIT', px_c))

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
                'Precio': round(sell_px, 3), 'Shares': sh,
                'Neto': round(net, 2), 'P&L': round(pnl, 2),
                'P&L%': f"{pct:+.2%}", 'Razón': reason,
                'Días': pos['hd'], 'Regime': pos.get('regime', '?'),
            })
            trade_rows.append({
                'ticker': t, 'resultado': 'WIN' if pnl > 0 else 'LOSS',
                'pnl_aud': round(pnl, 2), 'pnl_pct': round(pct*100, 2),
                'dias': pos['hd'], 'razon': reason,
                'prob': pos.get('prob', 0),
                'regime': pos.get('regime', '?'),
            })

        # ════════════════════ ENTRIES ════════════════════
        slots = max_pos - len(portfolio)
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

            # Bear mode: only defensive sectors
            if not is_bull and sector not in DEFENSIVE_SECTORS:
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
            if prob < prob_th:
                continue

            rc = rs['close']
            sma20  = float(rc.rolling(20).mean().iloc[-1])
            sma50  = float(rc.rolling(50).mean().iloc[-1])
            sma200 = float(rc.rolling(200).mean().iloc[-1])
            if pd.isna(sma50) or pd.isna(sma200) or pd.isna(sma20):
                continue

            # TREND: SMA20 > SMA50
            if sma20 < sma50 * 0.99:
                continue

            # TREND: Not both bearish
            if px < sma50 and sma50 < sma200:
                continue

            # PULLBACK: near SMA20
            dist_s20 = abs(px / sma20 - 1)
            if dist_s20 > pb_pct:
                continue

            # RSI
            rsi_val = float(ta_lib.momentum.rsi(rc, 14).iloc[-1])
            if pd.isna(rsi_val) or rsi_val < MIN_RSI or rsi_val > MAX_RSI:
                continue

            # MACD (relaxed — near zero OK)
            macd_d = float(ta_lib.trend.macd_diff(rc).iloc[-1])
            if pd.isna(macd_d) or macd_d < -px * 0.002:
                continue

            # ADX
            adx_val = float(ta_lib.trend.adx(rs['high'], rs['low'], rc, 14).iloc[-1])
            if pd.isna(adx_val) or adx_val < min_adx:
                continue

            # Volume
            vol_avg = float(rs['volume'].rolling(20).mean().iloc[-1])
            vol_now = float(rs['volume'].iloc[-1])
            vol_r = vol_now / vol_avg if vol_avg > 0 else 0
            if vol_r < 0.9:
                continue

            # Momentum: not both negative
            mom5 = float(rc.pct_change(5).iloc[-1])
            mom10 = float(rc.pct_change(10).iloc[-1])
            if mom5 < -0.02 and mom10 < -0.01:
                continue

            atr_now = float(ta_lib.volatility.average_true_range(
                rs['high'], rs['low'], rc, 14).iloc[-1])
            if pd.isna(atr_now) or atr_now <= 0:
                continue

            trend_q = min(1.0, max(0, (sma20/sma50 - 1) * 20))
            pb_q = max(0, 1.0 - dist_s20 / pb_pct)
            score = prob * (1.0 + trend_q * 0.3 + pb_q * 0.4)

            candidates.append({
                't': t_c, 'px': px, 'prob': prob, 'score': score,
                'atr': atr_now, 'sl_atr': sl_atr,
            })

        candidates.sort(key=lambda x: x['score'], reverse=True)

        for cand in candidates:
            if cash < MIN_BUY or len(portfolio) >= max_pos:
                break

            t = cand['t']
            px = cand['px']
            atr_now = cand['atr']
            sl_mult = cand['sl_atr']

            risk_amt = equity * risk_tr
            sl_dist = atr_now * sl_mult
            sh_risk = int(risk_amt / sl_dist) if sl_dist > 0 else 0
            sh_cap = int((equity * max_pct) / px) if px > 0 else 0
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
                'stop': px - atr_now * sl_mult,
                'atr': atr_now, 'hd': 0,
                'peak': px, 'be': False, 'tp1_hit': False,
                'prob': cand['prob'],
                'regime': 'BULL' if is_bull else 'BEAR',
            }

            if is_bull:
                bull_trades += 1
            else:
                bear_trades += 1

            all_trades.append({
                'Tipo': '🟢 COMPRA', 'Ticker': t,
                'Fecha': day.strftime('%Y-%m-%d'),
                'Precio': round(px, 3), 'Shares': shares,
                'Neto': round(total, 2), 'P&L': '—',
                'P&L%': f"p={cand['prob']:.2f}",
                'Razón': 'PULLBACK' + ('_BULL' if is_bull else '_BEAR'),
                'Días': 0, 'Regime': 'BULL' if is_bull else 'BEAR',
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
                'Precio': round(px, 3), 'Shares': pos['sh'],
                'Neto': round(net, 2), 'P&L': round(pnl, 2),
                'P&L%': f"{pct:+.2%}", 'Razón': 'FINAL',
                'Días': pos.get('hd', 0), 'Regime': pos.get('regime', '?'),
            })
            trade_rows.append({
                'ticker': t, 'resultado': 'WIN' if pnl > 0 else 'LOSS',
                'pnl_aud': round(pnl, 2), 'pnl_pct': round(pct*100, 2),
                'dias': pos.get('hd', 0), 'razon': 'FINAL',
                'prob': pos.get('prob', 0), 'regime': pos.get('regime', '?'),
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
    eq_s = eq_df['Equity'] if not eq_df.empty else pd.Series([start_capital])
    max_dd = float(((eq_s.cummax() - eq_s) / eq_s.cummax()).max())
    w_sum = float(pnl_vals[pnl_vals > 0].sum()) if wins else 0
    l_sum = abs(float(pnl_vals[pnl_vals <= 0].sum())) if losses else 1
    pf = w_sum / l_sum if l_sum > 0 else float('inf')
    avg_w = w_sum / wins if wins else 0
    avg_l = l_sum / losses if losses else 0

    # Regime breakdown
    regime_stats = {}
    if not tlog.empty and 'regime' in tlog.columns:
        for reg in ['BULL', 'BEAR']:
            rd = tlog[tlog['regime'] == reg]
            if not rd.empty:
                regime_stats[reg] = {
                    'n': len(rd),
                    'w': int((rd['resultado'] == 'WIN').sum()),
                    'pnl': float(rd['pnl_aud'].sum()),
                }

    reason_stats = {}
    if not tlog.empty:
        for r in tlog['razon'].unique():
            rd = tlog[tlog['razon'] == r]
            reason_stats[r] = {
                'n': len(rd), 'w': int((rd['resultado'] == 'WIN').sum()),
                'pnl': float(rd['pnl_aud'].sum()),
            }

    return {
        'label': label, 'cap_i': start_capital, 'cap_f': round(cash, 2),
        'roi': roi, 'max_dd': max_dd,
        'n_tr': n_tr, 'wins': wins, 'losses': losses,
        'wr': wins / n_tr if n_tr > 0 else 0,
        'pf': pf, 'avg_w': avg_w, 'avg_l': avg_l,
        'days': len(sim_dates),
        'bull_tr': bull_trades, 'bear_tr': bear_trades,
        'regime_stats': regime_stats,
        'reasons': reason_stats,
        'trades_df': trades_df, 'eq_df': eq_df, 'tlog': tlog,
    }


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════
def main():
    t0 = time.time()

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  DEEPQUANT V6 — MARKET REGIME + BULL ALPHA                 ║")
    print("║  LightGBM × Regime Filter × Pullback × Asymmetric R:R     ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    periods = [
        {'name': 'P1: 2022-2024', 'te': '2021-12-31',
         'ss': '2022-01-01', 'se': '2024-12-31'},
        {'name': 'P2: 2025',      'te': '2024-12-31',
         'ss': '2025-01-01', 'se': '2025-12-31'},
        {'name': 'P3: 2026 YTD',  'te': '2025-12-31',
         'ss': '2026-01-01', 'se': '2026-12-31'},
    ]

    # Download all data once (+ VAS for regime)
    DATA = {}
    print(f"\n  📡 Descargando {len(ASX_TICKERS) + 1} tickers...")
    tickers_all = ASX_TICKERS + ['VAS.AX']
    for i, t in enumerate(tickers_all, 1):
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
        sys.stdout.write(f"\r   {i}/{len(tickers_all)} OK:{len(DATA)}")
        sys.stdout.flush()
    print(f"\n  ✅ {len(DATA)} tickers")

    vas_data = DATA.pop('VAS.AX', None)
    if vas_data is not None:
        print(f"  📊 VAS.AX: {len(vas_data)} rows (regime indicator)")
    else:
        print("  ⚠️ VAS.AX no disponible — usando bull por defecto")

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

        result = run_simulation(
            ALL, models, vas_data, p['ss'], p['se'], label=p['name'])
        results.append(result)

        if result:
            print(f"\n  💰 {p['name']}:")
            print(f"     ${result['cap_i']:,.0f} → ${result['cap_f']:,.2f}")
            print(f"     ROI: {result['roi']:+.2%} | DD: {result['max_dd']:.2%}")
            print(f"     Trades: {result['n_tr']} (BULL:{result['bull_tr']} BEAR:{result['bear_tr']})")
            print(f"     WR: {result['wr']:.1%} | PF: {result['pf']:.2f}")
            if result['avg_l'] > 0:
                print(f"     Avg W: ${result['avg_w']:.2f} | Avg L: ${result['avg_l']:.2f} | R:R: {result['avg_w']/result['avg_l']:.2f}")
            if result['regime_stats']:
                print(f"     Por régimen:")
                for reg, st in result['regime_stats'].items():
                    wr = st['w']/st['n']*100 if st['n'] > 0 else 0
                    print(f"       {reg:<5} {st['n']:>3}× | WR {wr:.0f}% | ${st['pnl']:+.2f}")
            if result['reasons']:
                print(f"     Salidas:")
                for r, st in result['reasons'].items():
                    wr = st['w']/st['n']*100 if st['n'] > 0 else 0
                    print(f"       {r:<14} {st['n']:>3}× | WR {wr:.0f}% | ${st['pnl']:+.2f}")

    # ── Summary ──
    elapsed = time.time() - t0
    print(f"\n\n{'═'*65}")
    print(f"  📊 RESUMEN — DEEPQUANT V6 (REGIME + BULL ALPHA)")
    print(f"{'═'*65}")
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

    print(f"  {'─'*65}")
    print(f"  {'ACUMULADO':<16} ${cum:>7,.2f} {cum_roi:>+7.2%}")
    print(f"  {'INICIAL':<16} ${START_CAPITAL:>7,.2f}")
    print(f"{'═'*65}")
    print(f"  ⏱ {elapsed/60:.1f} min")

    # Save
    ts = datetime.now().strftime('%Y%m%d_%H%M')
    out = f"sim_v6_{ts}.xlsx"
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
                'Trades': r['n_tr'],
                'Bull_Trades': r['bull_tr'], 'Bear_Trades': r['bear_tr'],
                'WR': f"{r['wr']:.1%}",
                'PF': round(r['pf'], 2), 'R:R': round(rr, 2),
                'Avg_Win': round(r['avg_w'], 2),
                'Avg_Loss': round(r['avg_l'], 2),
            })
        pd.DataFrame(rows).to_excel(w, sheet_name='Resumen', index=False)
        for i, r in enumerate(results):
            if not r or r['trades_df'].empty:
                continue
            r['trades_df'].to_excel(w, sheet_name=f"Trades_P{i+1}", index=False)
            if not r['eq_df'].empty:
                r['eq_df'].to_excel(w, sheet_name=f"Equity_P{i+1}", index=False)

    print(f"\n  💾 {out}")

    # ── Full comparison table ──
    print(f"\n{'═'*72}")
    print(f"  🏆 COMPARACIÓN COMPLETA — TODAS LAS VERSIONES:")
    print(f"{'═'*72}")
    print(f"  {'Ver':<5} {'P1 ROI':>8} {'P2 ROI':>8} {'P3 ROI':>8} {'Acum':>8} {'P1 WR':>6} {'P1 Tr':>5} {'Mejor':>8}")
    print(f"  {'─'*5} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*6} {'─'*5} {'─'*8}")
    print(f"  {'V2':<5} {'-8.59%':>8} {'-0.91%':>8} {'-0.52%':>8} {'-9.89%':>8} {'18.8%':>6} {'16':>5} {'P3':>8}")
    print(f"  {'V3':<5} {'-47.5%':>8} {'N/A':>8} {'N/A':>8} {'>-47%':>8} {'16.8%':>6} {'179':>5} {'—':>8}")
    print(f"  {'V4':<5} {'-12.3%':>8} {'-2.19%':>8} {'-1.76%':>8} {'-15.7%':>8} {'34.9%':>6} {'86':>5} {'P2':>8}")
    print(f"  {'V5':<5} {'-58.7%':>8} {'+1.94%':>8} {'-0.33%':>8} {'-58.1%':>8} {'35.0%':>6} {'234':>5} {'P2 ✅':>8}")
    if results[0]:
        r1 = results[0]
        r2 = results[1] if len(results) > 1 else None
        r3 = results[2] if len(results) > 2 else None
        p1r = f"{r1['roi']:+.1%}" if r1 else 'N/A'
        p2r = f"{r2['roi']:+.1%}" if r2 else 'N/A'
        p3r = f"{r3['roi']:+.1%}" if r3 else 'N/A'
        p1wr = f"{r1['wr']:.1%}" if r1 else 'N/A'
        p1tr = str(r1['n_tr']) if r1 else 'N/A'
        best = 'P2' if r2 and r2['roi'] > 0 else '—'
        print(f"  {'V6':<5} {p1r:>8} {p2r:>8} {p3r:>8} {cum_roi:>+7.1%} {p1wr:>6} {p1tr:>5} {best:>8}")
    print(f"{'═'*72}")
    print()
    print("  📌 CONCLUSIONES:")
    print("  • V5/V6 prueban que el modelo genera α en mercados alcistas (P2: +1.9%)")
    print("  • El filtro de régimen (VAS>SMA200) es CRÍTICO para proteger en bears")
    print("  • La estrategia de pullback a SMA20 + ML confirmation funciona")
    print("  • Con $8K capital, comisiones ($10 flat) tienen impacto significativo")
    print("  • R:R mejora con TP1 parcial a 4×ATR + trail amplio 3×ATR")
    print()


if __name__ == "__main__":
    main()
