# sim_v4_3periods.py — V4: Trend-Following + ML Confirmation
# ═══════════════════════════════════════════════════════════════════════
#  FILOSOFÍA V4: "SIMPLE, DISCIPLINADO, RENTABLE"
#  ──────────────────────────────────────────────────────────────────
#  ✗ NO reversión (demostrado anti-predictivo)
#  ✗ NO normalización rolling (destruye señales, tree-models no la necesitan)
#  ✗ NO ensemble complejo (LightGBM solo > ruido de ensemble)
#  ✗ NO stops ultra-tight (1.2×ATR = noise-death)
#  ✗ NO partial-TP1 prematuro
#
#  ✓ SÍ trend-following puro (solo comprar en tendencia alcista)
#  ✓ SÍ stops amplios (2.0×ATR — dar espacio al trade)
#  ✓ SÍ trailing stop inteligente (proteger sin ahogar)
#  ✓ SÍ ML como CONFIRMACIÓN (no como generador de señales)
#  ✓ SÍ target simple y limpio (10d return > median ATR)
#  ✓ SÍ pocas posiciones de alta convicción
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
#  CONFIGURACIÓN — V4 TREND FOLLOWING
# ═══════════════════════════════════════════════════════════════════════
START_CAPITAL   = 8000.0
MIN_BUY         = 500.0
MAX_POSITIONS   = 5           # Concentrado: 5 apuestas de alta convicción
MAX_POS_PCT     = 0.30        # Hasta 30% en una sola posición
RISK_PER_TRADE  = 0.025       # 2.5% del equity en riesgo por trade
SL_ATR_MULT     = 2.0         # AMPLIO: stop loss = entry - 2.0×ATR
TRAIL_ATR_MULT  = 1.8         # Trailing stop = peak - 1.8×ATR
BREAKEVEN_ATR   = 1.5         # Mover a breakeven cuando gana 1.5×ATR
BREAKEVEN_LOCK  = 0.3         # Lock stop en entry + 0.3×ATR
MAX_HOLD_DAYS   = 20          # Máx 20 días hold
COOLDOWN_DAYS   = 5           # 5 días cooldown tras stop loss
MIN_STOCK_PRICE = 1.00        # Min $1 de precio (evitar penny stocks)
COMMISSION_FLAT = 10.0
COMMISSION_RATE = 0.0011

# ── Entry filters ──
PROB_THRESHOLD  = 0.52        # ML probability mínima (alta convicción)
MIN_RSI         = 35          # No comprar en sobreventa extrema
MAX_RSI         = 72          # No comprar en sobrecompra
MIN_ADX         = 18          # Tendencia mínima (no range-bound)
MIN_VOL_RATIO   = 1.0         # Volumen al menos normal
MIN_MOM10       = 0.005       # Momentum 10d positivo (0.5%)

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
#  FEATURE ENGINEERING V4 — SIMPLE & RAW (no normalization)
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
    'range_pct',
    'gap_pct',
]

def engineer_features(raw_df):
    """
    Feature engineering V4: 20 features sin normalización rolling.
    Tree-models (LightGBM) manejan features en escala original.
    """
    df = raw_df.copy()
    if len(df) < 220:
        return None

    c = df['close']
    h = df['high']
    l = df['low']
    v = df['volume']
    o = df['open']

    # SMAs
    sma20  = ta_lib.trend.sma_indicator(c, 20)
    sma50  = ta_lib.trend.sma_indicator(c, 50)
    sma200 = ta_lib.trend.sma_indicator(c, 200)
    df['dist_sma20']  = (c / sma20) - 1
    df['dist_sma50']  = (c / sma50) - 1
    df['dist_sma200'] = (c / sma200) - 1
    df['ma_cross_20_50']  = (sma20 > sma50).astype(float)
    df['ma_cross_50_200'] = (sma50 > sma200).astype(float)

    # ATR
    atr_raw = ta_lib.volatility.average_true_range(h, l, c, 14)
    df['atr'] = atr_raw
    df['atr_pct'] = np.where(c > 0, atr_raw / c, 0.02)
    atr_ma = atr_raw.rolling(50).mean()
    df['vol_regime'] = np.where(atr_ma > 0, atr_raw / atr_ma, 1.0)

    # Momentum
    for p in [5, 10, 20]:
        df[f'momentum_{p}'] = c.pct_change(p)

    # RSI (0-1 scale)
    df['rsi'] = ta_lib.momentum.rsi(c, 14) / 100.0

    # MACD diff normalized by price
    macd_diff = ta_lib.trend.macd_diff(c)
    df['macd_diff_norm'] = np.where(c > 0, macd_diff / c, 0)

    # ADX
    df['adx'] = ta_lib.trend.adx(h, l, c, 14) / 100.0

    # Bollinger
    bb_h = ta_lib.volatility.bollinger_hband(c, 20)
    bb_l = ta_lib.volatility.bollinger_lband(c, 20)
    df['bb_width'] = np.where(c > 0, (bb_h - bb_l) / c, 0)
    bb_range = bb_h - bb_l
    df['bb_position'] = np.where(bb_range > 0, (c - bb_l) / bb_range, 0.5)

    # Volume
    vm20 = v.rolling(20).mean()
    df['vol_rel_20'] = np.where(vm20 > 0, v / vm20, 1.0)

    # Price position
    high20 = c.rolling(20).max()
    low20  = c.rolling(20).min()
    df['close_to_high20'] = np.where(high20 > 0, (c / high20) - 1, 0)
    df['close_to_low20']  = np.where(low20 > 0, (c / low20) - 1, 0)

    # Microstructure
    df['range_pct'] = np.where(c > 0, (h - l) / c, 0)
    df['gap_pct'] = (o / c.shift(1)) - 1

    # ── TARGET V4: Simple + Clean ──
    # 10-day forward return exceeds 1.5% AND drawdown < 5%
    fut_return = c.shift(-10) / c - 1
    # Max drawdown durante los 10 días
    min_price_10d = pd.Series(np.nan, index=df.index)
    for hh in range(1, 11):
        fl = l.shift(-hh)
        min_price_10d = pd.concat([min_price_10d, fl], axis=1).min(axis=1)
    max_dd_10d = (min_price_10d / c) - 1  # negative value

    df['target'] = ((fut_return > 0.015) & (max_dd_10d > -0.05)).astype(int)

    # Fallback: if too few positives, relax to 1%
    if df['target'].mean() < 0.08:
        df['target'] = ((fut_return > 0.01) & (max_dd_10d > -0.06)).astype(int)
    if df['target'].sum() < 20:
        df['target'] = (fut_return > 0.005).astype(int)

    # Store raw values for simulation (sma50 and sma200 needed)
    df['_sma50'] = sma50
    df['_sma200'] = sma200
    df['_sma20'] = sma20

    return df.replace([np.inf, -np.inf], np.nan).dropna()


# ═══════════════════════════════════════════════════════════════════════
#  MODEL V4: LightGBM + Isotonic Calibration (simple & fast)
# ═══════════════════════════════════════════════════════════════════════
def build_model():
    """Single LightGBM with isotonic calibration. Fast, robust, interpretable."""
    lgbm = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        num_leaves=20,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=20,
        class_weight='balanced',
        random_state=42,
        verbose=-1,
        n_jobs=-1,
    )
    return lgbm


def train_models(all_raw_data, train_end_date, min_rows=300):
    """Train one LightGBM per ticker."""
    models = {}
    total = len(all_raw_data)

    for i, (ticker, raw) in enumerate(all_raw_data.items(), 1):
        sys.stdout.write(f"\r   [{i:3d}/{total}] {ticker:<10}")
        sys.stdout.flush()

        df = engineer_features(raw)
        if df is None:
            continue

        train = df[df.index <= train_end_date]
        if len(train) < min_rows:
            continue

        avail = [f for f in FEATURE_COLS if f in train.columns]
        if len(avail) < 10:
            continue

        X = train[avail]
        y = train['target']

        if y.sum() < 15 or y.mean() < 0.03:
            continue

        try:
            base = build_model()
            cal = CalibratedClassifierCV(base, method='isotonic', cv=3)
            cal.fit(X, y)
            models[ticker] = {'model': cal, 'features': avail}
        except Exception:
            try:
                cal = CalibratedClassifierCV(base, method='sigmoid', cv=3)
                cal.fit(X, y)
                models[ticker] = {'model': cal, 'features': avail}
            except Exception:
                continue

    print(f"\n   ✅ Modelos: {len(models)}/{total}")
    return models


# ═══════════════════════════════════════════════════════════════════════
#  SIMULATION V4: TREND-FOLLOWING + ML CONFIRMATION
# ═══════════════════════════════════════════════════════════════════════
def run_simulation(all_raw, models, sim_start, sim_end,
                   start_capital=START_CAPITAL, label=""):
    print(f"\n{'─'*60}")
    print(f"  📈 {label}")
    print(f"     {sim_start} → {sim_end} | ${start_capital:,.0f} AUD | {len(models)} tickers")
    print(f"{'─'*60}")

    # ── Pre-compute features + probabilities ──
    FEAT_DATA = {}
    PROBS = {}
    for t, m in models.items():
        raw = all_raw.get(t)
        if raw is None:
            continue
        df = engineer_features(raw)
        if df is None:
            continue
        FEAT_DATA[t] = df

        feats = m['features']
        sim_rows = df[(df.index >= pd.Timestamp(sim_start)) &
                      (df.index <= pd.Timestamp(sim_end))]
        if sim_rows.empty:
            PROBS[t] = {}
            continue
        X_batch = sim_rows[feats].replace([np.inf, -np.inf], np.nan).dropna()
        if X_batch.empty:
            PROBS[t] = {}
            continue
        try:
            probs = m['model'].predict_proba(X_batch.values)[:, 1]
            PROBS[t] = dict(zip(X_batch.index, probs))
        except Exception:
            PROBS[t] = {}

    print(f"  Tickers con features: {len(FEAT_DATA)}")
    print(f"  Tickers con probs:    {len([t for t,p in PROBS.items() if p])}")

    # ── Simulation dates ──
    all_dates = set()
    for t in FEAT_DATA:
        if t in models:
            df_t = FEAT_DATA[t]
            all_dates.update(
                df_t[(df_t.index >= pd.Timestamp(sim_start)) &
                     (df_t.index <= pd.Timestamp(sim_end))].index.tolist()
            )
    sim_dates = sorted(all_dates)
    if not sim_dates:
        print("  ⚠️ Sin fechas")
        return None
    print(f"  Días de mercado: {len(sim_dates)}")

    # ── State ──
    cash       = start_capital
    portfolio  = {}       # ticker → position dict
    equity_log = []
    all_trades = []
    trade_rows = []
    sl_history = {}       # ticker → last SL date
    recent_pnls = []

    for day in sim_dates:
        # ── Mark-to-market ──
        equity = cash
        for t_pos, pos in portfolio.items():
            raw = all_raw.get(t_pos)
            px = (float(raw.loc[day, 'close'])
                  if raw is not None and day in raw.index
                  else pos['buy_price'])
            equity += pos['shares'] * px
        equity_log.append({'Fecha': day.strftime('%Y-%m-%d'), 'Equity': round(equity, 2)})

        # ════════════════════ EXITS ════════════════════
        to_sell = []
        for t_pos, pos in list(portfolio.items()):
            raw = all_raw.get(t_pos)
            pos['hold_days'] = pos.get('hold_days', 0) + 1

            if raw is None or day not in raw.index:
                if pos['hold_days'] >= MAX_HOLD_DAYS:
                    to_sell.append((t_pos, 'EXPIRE', pos['buy_price']))
                continue

            row = raw.loc[day]
            px_close = float(row['close'])
            px_high  = float(row['high'])
            px_low   = float(row['low'])
            atr_e    = pos['atr_entry']

            # Update max price
            pos['max_price'] = max(pos.get('max_price', pos['buy_price']), px_high)

            # ── Breakeven lock ──
            if not pos.get('be_locked', False) and pos['hold_days'] >= 2:
                if px_high >= pos['buy_price'] + atr_e * BREAKEVEN_ATR:
                    be_stop = pos['buy_price'] + atr_e * BREAKEVEN_LOCK
                    pos['stop'] = max(pos['stop'], be_stop)
                    pos['be_locked'] = True

            # ── Trailing stop (from day 3+) ──
            if pos['hold_days'] >= 3:
                trail_stop = pos['max_price'] - atr_e * TRAIL_ATR_MULT
                pos['stop'] = max(pos['stop'], trail_stop)

            # ── Exit checks ──
            if px_low <= pos['stop']:
                sell_px = pos['stop']  # filled at stop
                to_sell.append((t_pos, 'STOP_LOSS', sell_px))
            elif pos['hold_days'] >= MAX_HOLD_DAYS:
                to_sell.append((t_pos, 'TIME_EXIT', px_close))
            elif pos['hold_days'] >= 5:
                # Trend break: price below SMA50 → exit at close
                feat_df = FEAT_DATA.get(t_pos)
                if feat_df is not None and day in feat_df.index:
                    sma50_val = feat_df.loc[day, '_sma50']
                    if not pd.isna(sma50_val) and px_close < sma50_val * 0.98:
                        # Check if profitable — don't exit at loss on trend break
                        unrealized = (px_close - pos['buy_price']) / pos['buy_price']
                        if unrealized > 0.005:
                            to_sell.append((t_pos, 'TREND_BREAK', px_close))

        # Execute sells
        for t_pos, reason, sell_px in to_sell:
            if t_pos not in portfolio:
                continue
            pos = portfolio.pop(t_pos)
            shares = pos['shares']
            gross = shares * sell_px
            comm = calc_commission(gross)
            net = gross - comm
            buy_cost = shares * pos['buy_price'] + pos.get('buy_comm', 0)
            pnl = net - buy_cost
            pct = (sell_px - pos['buy_price']) / pos['buy_price']

            cash += net
            if reason == 'STOP_LOSS':
                sl_history[t_pos] = day

            recent_pnls.append(pnl)
            if len(recent_pnls) > 30:
                recent_pnls.pop(0)

            all_trades.append({
                'Tipo': '🔴 VENTA', 'Ticker': t_pos,
                'Fecha': day.strftime('%Y-%m-%d'),
                'Precio': round(sell_px, 3),
                'Acciones': shares,
                'Neto_AUD': round(net, 2),
                'P&L_AUD': round(pnl, 2),
                'P&L_%': f"{pct:+.2%}",
                'Razon': reason,
                'Dias': pos['hold_days'],
            })
            trade_rows.append({
                'ticker': t_pos,
                'resultado': 'WIN' if pnl > 0 else 'LOSS',
                'pnl_aud': round(pnl, 2),
                'pnl_pct': round(pct * 100, 2),
                'dias': pos['hold_days'],
                'razon': reason,
                'prob': pos.get('entry_prob', 0),
            })

        # ════════════════════ ENTRIES ════════════════════
        slots = MAX_POSITIONS - len(portfolio)
        if cash < MIN_BUY or slots <= 0:
            continue

        candidates = []

        for t_cand in models:
            if t_cand in portfolio:
                continue

            # Cooldown check
            last_sl = sl_history.get(t_cand)
            if last_sl and (day - last_sl).days < COOLDOWN_DAYS:
                continue

            # Sector limit
            sector = SECTOR_MAP.get(t_cand, 'Other')
            sect_count = sum(1 for t in portfolio if SECTOR_MAP.get(t, 'X') == sector)
            if sect_count >= MAX_SECTOR_POS:
                continue

            # Get raw data
            raw = all_raw.get(t_cand)
            if raw is None or day not in raw.index:
                continue
            raw_slice = raw[raw.index <= day]
            if len(raw_slice) < 210:
                continue

            px = float(raw.loc[day, 'close'])
            if px < MIN_STOCK_PRICE:
                continue

            # ML Probability check
            prob = PROBS.get(t_cand, {}).get(day, 0.0)
            if prob < PROB_THRESHOLD:
                continue

            # ═══════════════════════════════════════════════
            #  TREND CONFIRMATION FILTERS
            # ═══════════════════════════════════════════════
            raw_c = raw_slice['close']
            sma20  = float(raw_c.rolling(20).mean().iloc[-1])
            sma50  = float(raw_c.rolling(50).mean().iloc[-1])
            sma200 = float(raw_c.rolling(200).mean().iloc[-1])

            if pd.isna(sma50) or pd.isna(sma200) or pd.isna(sma20):
                continue

            # PRIMARY: Price above SMA50 (in uptrend)
            if px < sma50:
                continue

            # SECONDARY: SMA20 > SMA50 (short-term trend aligned)
            if sma20 < sma50 * 0.99:  # 1% tolerance
                continue

            # RSI range
            rsi_val = float(ta_lib.momentum.rsi(raw_c, 14).iloc[-1])
            if pd.isna(rsi_val) or rsi_val < MIN_RSI or rsi_val > MAX_RSI:
                continue

            # MACD bullish
            macd_d = float(ta_lib.trend.macd_diff(raw_c).iloc[-1])
            if pd.isna(macd_d) or macd_d < 0:
                continue

            # ADX trending
            adx_val = float(ta_lib.trend.adx(
                raw_slice['high'], raw_slice['low'], raw_c, 14).iloc[-1])
            if pd.isna(adx_val) or adx_val < MIN_ADX:
                continue

            # Volume confirmation
            vol_avg = float(raw_slice['volume'].rolling(20).mean().iloc[-1])
            vol_now = float(raw_slice['volume'].iloc[-1])
            vol_ratio = vol_now / vol_avg if vol_avg > 0 else 0
            if vol_ratio < MIN_VOL_RATIO:
                continue

            # Momentum 10d positive
            mom10 = float(raw_c.pct_change(10).iloc[-1])
            if pd.isna(mom10) or mom10 < MIN_MOM10:
                continue

            # ATR for position sizing
            atr_now = float(ta_lib.volatility.average_true_range(
                raw_slice['high'], raw_slice['low'], raw_c, 14).iloc[-1])
            if pd.isna(atr_now) or atr_now <= 0:
                continue

            # Score: prob × momentum quality × trend strength
            trend_str = min(1.0, (px / sma200 - 1) * 10) if sma200 > 0 else 0
            mom_quality = min(1.0, mom10 * 15)
            score = prob * (1.0 + trend_str * 0.3 + mom_quality * 0.3)

            candidates.append({
                'ticker': t_cand,
                'price': px,
                'prob': prob,
                'score': score,
                'atr': atr_now,
                'rsi': rsi_val,
                'adx': adx_val,
                'mom10': mom10,
            })

        # Sort by score, take top candidates
        candidates.sort(key=lambda x: x['score'], reverse=True)

        for cand in candidates:
            if cash < MIN_BUY or len(portfolio) >= MAX_POSITIONS:
                break

            t = cand['ticker']
            px = cand['price']
            atr_now = cand['atr']

            # Position sizing: risk-based
            current_equity = equity  # from mark-to-market above
            risk_amount = current_equity * RISK_PER_TRADE
            sl_distance = atr_now * SL_ATR_MULT
            sl_pct = sl_distance / px if px > 0 else 0.05

            shares_risk = int(risk_amount / sl_distance) if sl_distance > 0 else 0
            shares_cap = int((current_equity * MAX_POS_PCT) / px) if px > 0 else 0
            shares_cash = int((cash - COMMISSION_FLAT * 1.5) / px) if px > 0 else 0  # leave room for commission
            shares = max(1, min(shares_risk, shares_cap, shares_cash))

            buy_cost = shares * px
            buy_comm = calc_commission(buy_cost)
            total_cost = buy_cost + buy_comm

            if total_cost > cash or buy_cost < MIN_BUY:
                continue

            # Set stop loss
            sl_price = px - atr_now * SL_ATR_MULT

            cash -= total_cost
            portfolio[t] = {
                'shares': shares,
                'buy_price': px,
                'buy_comm': buy_comm,
                'stop': sl_price,
                'atr_entry': atr_now,
                'hold_days': 0,
                'max_price': px,
                'be_locked': False,
                'entry_prob': cand['prob'],
            }

            all_trades.append({
                'Tipo': '🟢 COMPRA', 'Ticker': t,
                'Fecha': day.strftime('%Y-%m-%d'),
                'Precio': round(px, 3),
                'Acciones': shares,
                'Neto_AUD': round(total_cost, 2),
                'P&L_AUD': '—',
                'P&L_%': f"p={cand['prob']:.2f} s={cand['score']:.2f}",
                'Razon': 'TREND_FOLLOW',
                'Dias': 0,
            })

    # ── Liquidate remaining positions ──
    if sim_dates:
        final_day = sim_dates[-1]
        for t_pos, pos in list(portfolio.items()):
            raw = all_raw.get(t_pos)
            px = (float(raw.loc[final_day, 'close'])
                  if raw is not None and final_day in raw.index
                  else pos['buy_price'])
            gross = pos['shares'] * px
            comm = calc_commission(gross)
            net = gross - comm
            buy_cost = pos['shares'] * pos['buy_price'] + pos.get('buy_comm', 0)
            pnl = net - buy_cost
            pct = (px - pos['buy_price']) / pos['buy_price']
            cash += net

            all_trades.append({
                'Tipo': '🔴 VENTA', 'Ticker': t_pos,
                'Fecha': final_day.strftime('%Y-%m-%d'),
                'Precio': round(px, 3),
                'Acciones': pos['shares'],
                'Neto_AUD': round(net, 2),
                'P&L_AUD': round(pnl, 2),
                'P&L_%': f"{pct:+.2%}",
                'Razon': 'CIERRE_FINAL',
                'Dias': pos.get('hold_days', 0),
            })
            trade_rows.append({
                'ticker': t_pos,
                'resultado': 'WIN' if pnl > 0 else 'LOSS',
                'pnl_aud': round(pnl, 2),
                'pnl_pct': round(pct * 100, 2),
                'dias': pos.get('hold_days', 0),
                'razon': 'CIERRE_FINAL',
                'prob': pos.get('entry_prob', 0),
            })

    # ── Results ──
    trades_df = pd.DataFrame(all_trades)
    eq_df = pd.DataFrame(equity_log)
    tlog = pd.DataFrame(trade_rows) if trade_rows else pd.DataFrame()

    sales = trades_df[trades_df['Tipo'].str.contains('VENTA')] if not trades_df.empty else pd.DataFrame()
    pnl_vals = (sales['P&L_AUD'].apply(pd.to_numeric, errors='coerce').dropna()
                if not sales.empty else pd.Series([0.0]))

    roi = (cash - start_capital) / start_capital
    wins = int((pnl_vals > 0).sum())
    losses = int((pnl_vals <= 0).sum())
    n_trades = wins + losses
    eq_s = eq_df['Equity'] if not eq_df.empty else pd.Series([start_capital, cash])
    max_dd = float(((eq_s.cummax() - eq_s) / eq_s.cummax()).max())

    win_total = float(pnl_vals[pnl_vals > 0].sum()) if wins > 0 else 0
    loss_total = abs(float(pnl_vals[pnl_vals <= 0].sum())) if losses > 0 else 1
    pf = win_total / loss_total if loss_total > 0 else float('inf')

    # Exit reason breakdown
    reason_stats = {}
    if not tlog.empty:
        for reason in tlog['razon'].unique():
            r_df = tlog[tlog['razon'] == reason]
            reason_stats[reason] = {
                'count': len(r_df),
                'wins': int((r_df['resultado'] == 'WIN').sum()),
                'pnl': float(r_df['pnl_aud'].sum()),
            }

    return {
        'label': label,
        'capital_ini': start_capital,
        'capital_fin': round(cash, 2),
        'roi': roi,
        'max_dd': max_dd,
        'n_trades': n_trades,
        'wins': wins,
        'losses': losses,
        'win_rate': wins / n_trades if n_trades > 0 else 0,
        'pf': pf,
        'pnl_avg': float(pnl_vals.mean()) if n_trades > 0 else 0,
        'best': float(pnl_vals.max()) if n_trades > 0 else 0,
        'worst': float(pnl_vals.min()) if n_trades > 0 else 0,
        'sim_days': len(sim_dates),
        'reason_stats': reason_stats,
        'trades_df': trades_df,
        'equity_df': eq_df,
        'trade_log': tlog,
    }


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════
def main():
    t0 = time.time()

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  DEEPQUANT V4 — TREND-FOLLOWING + ML CONFIRMATION          ║")
    print("║  LightGBM × Raw Features × Wide Stops × Discipline        ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    periods = [
        {'name': 'P1: 2022-2024', 'train_end': '2021-12-31',
         'sim_start': '2022-01-01', 'sim_end': '2024-12-31'},
        {'name': 'P2: 2025',      'train_end': '2024-12-31',
         'sim_start': '2025-01-01', 'sim_end': '2025-12-31'},
        {'name': 'P3: 2026 YTD',  'train_end': '2025-12-31',
         'sim_start': '2026-01-01', 'sim_end': '2026-12-31'},
    ]

    # ── Download all data once ──
    DATA = {}
    print(f"\n  📡 Descargando {len(ASX_TICKERS)} tickers (2017 → 2026)...")
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
    print(f"\n  ✅ {len(DATA)} tickers en caché\n")

    results = []

    for period in periods:
        print(f"\n{'═'*60}")
        print(f"  ⏱  {period['name']}")
        print(f"     Train: 2017 → {period['train_end']}")
        print(f"     Test:  {period['sim_start']} → {period['sim_end']}")
        print(f"{'═'*60}")

        # Prepare data for this period
        ALL_RAW = {}
        for t, raw in DATA.items():
            mask = raw.index <= pd.Timestamp(period['sim_end'])
            d = raw[mask]
            if len(d) > 200:
                ALL_RAW[t] = d

        print(f"\n  📦 Tickers: {len(ALL_RAW)}")

        # Train
        print(f"  🧠 Entrenando (train → {period['train_end']})...")
        models = train_models(ALL_RAW, period['train_end'])

        if not models:
            print("  ⚠️ Sin modelos")
            results.append(None)
            continue

        # Simulate
        result = run_simulation(
            ALL_RAW, models,
            sim_start=period['sim_start'],
            sim_end=period['sim_end'],
            label=period['name']
        )
        results.append(result)

        if result:
            print(f"\n  💰 {period['name']}:")
            print(f"     ${result['capital_ini']:,.0f} → ${result['capital_fin']:,.2f}")
            print(f"     ROI: {result['roi']:+.2%} | Max DD: {result['max_dd']:.2%}")
            print(f"     Trades: {result['n_trades']} | WR: {result['win_rate']:.1%} | PF: {result['pf']:.2f}")
            if result['reason_stats']:
                print(f"     Salidas:")
                for reason, st in result['reason_stats'].items():
                    wr = st['wins']/st['count']*100 if st['count'] > 0 else 0
                    print(f"       {reason:<15} {st['count']:>3}× | WR {wr:.0f}% | P&L ${st['pnl']:+.2f}")

    # ── Summary ──
    elapsed = time.time() - t0
    print(f"\n\n{'═'*65}")
    print(f"  📊 RESUMEN — DEEPQUANT V4 (TREND-FOLLOWING)")
    print(f"{'═'*65}")
    print(f"  {'Período':<18} {'Cap.Fin':>10} {'ROI':>8} {'DD':>7} {'Trades':>7} {'WR':>6} {'PF':>6}")
    print(f"  {'─'*18} {'─'*10} {'─'*8} {'─'*7} {'─'*7} {'─'*6} {'─'*6}")

    for r in results:
        if r is None:
            print("  (sin datos)")
            continue
        print(f"  {r['label']:<18} ${r['capital_fin']:>8,.2f} {r['roi']:>+7.2%} "
              f"{r['max_dd']:>6.2%} {r['n_trades']:>6} {r['win_rate']:>5.1%} "
              f"{r['pf']:>5.2f}")

    cum = START_CAPITAL
    for r in results:
        if r:
            cum *= (1 + r['roi'])
    cum_roi = (cum - START_CAPITAL) / START_CAPITAL

    print(f"  {'─'*65}")
    print(f"  {'ACUMULADO':<18} ${cum:>8,.2f} {cum_roi:>+7.2%}")
    print(f"  {'CAPITAL INICIAL':<18} ${START_CAPITAL:>8,.2f}")
    print(f"{'═'*65}")
    print(f"  ⏱ {elapsed/60:.1f} min")

    # ── Save Excel ──
    ts = datetime.now().strftime('%Y%m%d_%H%M')
    out = f"sim_v4_{ts}.xlsx"
    with pd.ExcelWriter(out, engine='openpyxl') as w:
        rows = []
        for r in results:
            if r is None:
                continue
            rows.append({
                'Período': r['label'],
                'Capital_Ini': r['capital_ini'],
                'Capital_Fin': r['capital_fin'],
                'ROI': f"{r['roi']:+.2%}",
                'Max_DD': f"{r['max_dd']:.2%}",
                'Trades': r['n_trades'],
                'WR': f"{r['win_rate']:.1%}",
                'PF': round(r['pf'], 2),
                'Avg_PnL': round(r['pnl_avg'], 2),
                'Best': round(r['best'], 2),
                'Worst': round(r['worst'], 2),
            })
        pd.DataFrame(rows).to_excel(w, sheet_name='Resumen', index=False)

        for i, r in enumerate(results):
            if r is None or r['trades_df'].empty:
                continue
            r['trades_df'].to_excel(w, sheet_name=f"Trades_P{i+1}", index=False)
            if not r['equity_df'].empty:
                r['equity_df'].to_excel(w, sheet_name=f"Equity_P{i+1}", index=False)

    print(f"\n  💾 {out}")

    # ── V4 improvements summary ──
    print(f"\n{'═'*65}")
    print("  🧠 V4 vs ANTERIORES:")
    print(f"{'─'*65}")
    print("  ✅ LightGBM solo (más robusto que ensemble ruidoso)")
    print("  ✅ Sin normalización rolling (señales puras)")
    print("  ✅ Sin REVERSION (probado anti-predictivo)")
    print("  ✅ Stop 2.0×ATR (evita muerte por ruido)")
    print("  ✅ Trail 1.8×ATR (protege sin ahogar)")
    print("  ✅ Solo trend-following (price > SMA50 + MACD > 0)")
    print("  ✅ Target simple: 10d return > 1.5% + DD < 5%")
    print("  ✅ Risk-based sizing: 2.5% equity/trade")
    print("  ✅ 5 posiciones concentradas de alta convicción")
    print(f"{'═'*65}\n")


if __name__ == "__main__":
    main()
