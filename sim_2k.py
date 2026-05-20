# sim_2k.py — Simulación $2,000 AUD
# ─────────────────────────────────────────────────────────────────
# LÓGICA CORRECTA:
#   ENTRENAMIENTO : 2018-01-01 → 2024-12-31  (datos históricos reales)
#   VALIDACIÓN    : 2025-01-01 → 2025-12-31  (año completo out-of-sample)
#   DATOS         : yfinance scraping EN VIVO (NO archivos CSV locales)
# ─────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import yfinance as yf
import ta as ta_lib
import warnings, os, sys
from datetime import datetime

warnings.filterwarnings('ignore')
os.environ['SIMULATE_AU_INVESTMENT'] = '1'

# ─── PARÁMETROS ───────────────────────────────────────────────────
START_CAPITAL   = 8000.0
TRAIN_START     = "2018-01-01"
TRAIN_END       = "2025-12-31"
SIM_START       = pd.Timestamp("2025-01-01")
SIM_END         = pd.Timestamp("2025-12-31")
PROB_THRESHOLD  = 0.36          # base: relajado para capturar más setups
PROB_MOMENTUM   = 0.42          # trade log: losers avg prob=0.40, winners avg=0.48 → subir umbral
PROB_REVERSION  = 0.33          # estrategia mean-reversion (oversold calidad)
TP1_PCT         = 0.09          # TP parcial 9% — asegurar ganancia en 50% posición
TP2_PCT         = 0.22          # TP final 22% (puede ser mayor por ATR)
SL_PCT          = 0.05          # stop loss 5%
MAX_HOLD_DAYS   = 20            # extendido 15→20: trades EXPIRE promediaban +5.73% — 5 días más alcanzan TP1
MAX_POSITIONS   = 5             # 5 posiciones simultáneas — universo 100 tickers
MIN_BUY         = 500.0         # mínimo por operación — comisión $10 < 2%
RISK_PER_TRADE  = 0.025         # base Kelly: 2.5% equity por trade
MAX_POS_PCT     = 0.28          # máx 28% equity por posición
COOLDOWN_DAYS   = 5             # días sin re-entrada en ticker stop-loss
REGIME_SMA      = 200           # VAS.AX > SMA(200) = mercado alcista
BREAKEVEN_PCT   = 0.035         # cuando sube +3.5% → activar BE lock (grace period ≥2 días)
BREAKEVEN_LOCK  = 0.015         # stop a +1.5%: nivel óptimo para este universo (no interfiere con EXPIRE trades)
VOL_MIN_MOM     = 1.20          # volumen mínimo MOMENTUM (winners=1.83 losers=1.09)
ENTRY_DISCOUNT  = 0.010         # orden límite 1% bajo el close de la señal
#  → Si el low del día siguiente <= close*0.99 → llena al precio límite (mejor entrada)
#  → Si no baja → llena al open del día siguiente (entrada realista)
#  El ATR promedio ASX200 intradía es ~1.5%, así que el 1% se llena ~65% de las veces.

# Comisión CommSec CommShare — $10 AUD flat
COMMISSION_FLAT = 10.0
COMMISSION_RATE = 0.0011

def calc_commission(value_aud: float) -> float:
    """CommSec: max($10 flat, 0.11% del valor)."""
    return max(COMMISSION_FLAT, value_aud * COMMISSION_RATE)

def kelly_fraction(recent_pnls: list) -> float:
    """Half-Kelly dinámico: fracción de equity basada en historial real de trades.
    Con <8 trades usa 20% conservador. Clipeado a rango 10-35%."""
    if len(recent_pnls) < 8:
        return 0.20
    wins   = [p for p in recent_pnls if p > 0]
    losses = [abs(p) for p in recent_pnls if p <= 0]
    if not wins or not losses:
        return 0.20
    p  = len(wins) / len(recent_pnls)
    b  = np.mean(wins) / np.mean(losses)
    f  = (p * b - (1 - p)) / b   # Kelly completo
    return max(0.10, min(0.35, f * 0.5))  # half-Kelly

# ASX200 completo — 101 tickers (universo 2.5× mayor → más oportunidades)
ASX_TICKERS = [
    # --- core blue chips ---
    'BHP.AX','CBA.AX','CSL.AX','WES.AX','NAB.AX','WBC.AX','ANZ.AX',
    'MQG.AX','FMG.AX','TLS.AX','RIO.AX','GMG.AX','STO.AX','WDS.AX',
    'QBE.AX','ALL.AX','SCG.AX','ORG.AX','NST.AX','SUN.AX','MIN.AX',
    'PLS.AX','IGO.AX','TCL.AX','S32.AX','REA.AX','QAN.AX','RMD.AX',
    'AMC.AX','BSL.AX','CPU.AX','ASX.AX','SHL.AX','JHX.AX','WOW.AX',
    'COH.AX','XRO.AX','TWE.AX','CAR.AX','SEK.AX',
    # --- extended ASX200 ---
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

FEATURE_COLS = [
    'dist_sma20','dist_sma50','dist_sma200','ma_cross_20_50','ma_cross_50_200',
    'atr_pct','vol_regime','momentum_3','momentum_5','momentum_10','momentum_20',
    'roc_5','roc_10','rsi','macd_diff','adx',
    'bb_width','bb_upper_dist','bb_lower_dist',
    'stoch_k','stoch_d','williams_r','cci',
    'vol_rel_20','vol_rel_50',
    'close_to_max5','close_to_min5','close_to_max20','close_to_min20',
]

# ─── SCRAPING DIRECTO vía yfinance (ignora CSVs locales) ──────────
def fetch_live(ticker, start, end=None):
    try:
        kw = dict(start=start, progress=False, auto_adjust=True)
        if end:
            kw['end'] = end
        df = yf.download(ticker, **kw)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        df.index   = pd.to_datetime(df.index)
        df.index.name = 'date'
        needed = ['close','high','low','open','volume']
        if not all(c in df.columns for c in needed):
            return None
        return df[needed].copy()
    except Exception:
        return None

# ─── FEATURE ENGINEERING ──────────────────────────────────────────
def engineer(raw_df, horizon=5):
    import ta as ta_lib
    df = raw_df.copy()
    if len(df) < 120:
        return None
    c = df['close']

    df['sma_20']  = ta_lib.trend.sma_indicator(c, 20)
    df['sma_50']  = ta_lib.trend.sma_indicator(c, 50)
    df['sma_200'] = ta_lib.trend.sma_indicator(c, 200)
    df['dist_sma20']  = (c / df['sma_20'])  - 1
    df['dist_sma50']  = (c / df['sma_50'])  - 1
    df['dist_sma200'] = (c / df['sma_200']) - 1
    df['ma_cross_20_50']  = (df['sma_20']  > df['sma_50']).astype(int)
    df['ma_cross_50_200'] = (df['sma_50']  > df['sma_200']).astype(int)

    atr = ta_lib.volatility.average_true_range(df['high'], df['low'], c, 14)
    df['atr']     = atr
    df['atr_pct'] = np.where(c > 0, atr / c, 0)
    atr_ma = atr.rolling(50).mean()
    df['vol_regime'] = np.where(atr_ma > 0, atr / atr_ma, 1.0)

    for p in [3, 5, 10, 20]:
        df[f'momentum_{p}'] = c.pct_change(p)
    df['roc_5']     = ta_lib.momentum.roc(c, 5)
    df['roc_10']    = ta_lib.momentum.roc(c, 10)
    df['rsi']       = ta_lib.momentum.rsi(c, 14) / 100.0
    df['macd_diff'] = ta_lib.trend.macd_diff(c)
    df['adx']       = ta_lib.trend.adx(df['high'], df['low'], c, 14)

    bb_h = ta_lib.volatility.bollinger_hband(c, 20)
    bb_l = ta_lib.volatility.bollinger_lband(c, 20)
    df['bb_width']      = (bb_h - bb_l) / c
    df['bb_upper_dist'] = (bb_h - c) / c
    df['bb_lower_dist'] = (c - bb_l) / c

    df['stoch_k']    = ta_lib.momentum.stoch(df['high'], df['low'], c, 14, 3) / 100.0
    df['stoch_d']    = ta_lib.momentum.stoch_signal(df['high'], df['low'], c, 14, 3) / 100.0
    df['williams_r'] = ta_lib.momentum.williams_r(df['high'], df['low'], c, 14)
    df['cci']        = ta_lib.trend.cci(df['high'], df['low'], c, 20)

    vol  = df['volume']
    vm20 = vol.rolling(20).mean()
    vm50 = vol.rolling(50).mean()
    df['vol_rel_20'] = np.where(vm20 > 0, vol / vm20, 1.0)
    df['vol_rel_50'] = np.where(vm50 > 0, vol / vm50, 1.0)

    for w, sfx in [(5, '5'), (20, '20')]:
        df[f'max_{sfx}'] = c.rolling(w).max()
        df[f'min_{sfx}'] = c.rolling(w).min()
    df['close_to_max5']  = (c / df['max_5'])  - 1
    df['close_to_min5']  = (c / df['min_5'])  - 1
    df['close_to_max20'] = (c / df['max_20']) - 1
    df['close_to_min20'] = (c / df['min_20']) - 1

    # TARGET ROBUSTO: win = retorno futuro > 1x ATR
    fut     = df['close'].shift(-horizon) / df['close'] - 1
    raw_atr = np.where(df['close'] > 0, atr / df['close'], 0.02)
    raw_atr = pd.Series(raw_atr, index=df.index).fillna(0.02)
    df['target'] = (fut > raw_atr * 1.0).astype(int)
    if df['target'].mean() < 0.05:
        df['target'] = (fut > raw_atr * 0.5).astype(int)
    if df['target'].sum() == 0:
        df['target'] = (fut > 0.01).astype(int)

    # NORMALIZACIÓN ROLLING (sin data leakage)
    skip = {'target','close','open','high','low','volume',
            'sma_20','sma_50','sma_200','atr','max_5','min_5','max_20','min_20'}
    for col in df.columns:
        if col in skip:
            continue
        if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
            rm = df[col].rolling(252, min_periods=50).mean()
            rs = df[col].rolling(252, min_periods=50).std()
            df[col] = np.where(rs > 0, (df[col] - rm) / rs, 0.0)

    return df.replace([np.inf, -np.inf], np.nan).dropna()


def get_feats(df):
    return [f for f in FEATURE_COLS if f in df.columns]


def build_model():
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    lr  = Pipeline([('sc', StandardScaler()),
                    ('lr', LogisticRegression(C=0.3, solver='liblinear',
                                             class_weight='balanced', random_state=42))])
    rf  = RandomForestClassifier(n_estimators=150, max_depth=6, min_samples_leaf=8,
                                  class_weight='balanced', random_state=42, n_jobs=-1)
    gb  = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05,
                                      max_depth=3, subsample=0.8, random_state=42)
    ens = VotingClassifier([('rf', rf), ('lr', lr), ('gb', gb)],
                           voting='soft', weights=[2, 1, 2])
    try:
        return CalibratedClassifierCV(ens, method='isotonic', cv=3)
    except Exception:
        return CalibratedClassifierCV(ens, method='sigmoid', cv=3)


# ═══════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("💰  SIMULACIÓN TITAN AI — $8,000 AUD")
print(f"    Entrenamiento  : {TRAIN_START}  →  {TRAIN_END}")
print(f"    Validación     : {SIM_START.date()}  →  {SIM_END.date()}  (AÑO COMPLETO out-of-sample)")
print(f"    Fuente de datos: yfinance LIVE (no CSVs locales)")
print(f"{'='*65}\n")

# ─── PASO 1: Descargar datos del período de simulación ────────────────
ALL_FULL = {}
# Solo necesitamos datos desde antes del SIM_START para calcular indicadores
DATA_FETCH_START = (SIM_START - pd.Timedelta(days=500)).strftime('%Y-%m-%d')
print(f"📡 Descargando {len(ASX_TICKERS)} tickers ({DATA_FETCH_START} → {SIM_END.date()})...")
for i, t in enumerate(ASX_TICKERS):
    df_raw = fetch_live(t, start=DATA_FETCH_START, end=SIM_END.strftime('%Y-%m-%d'))
    if df_raw is not None and len(df_raw) > 50:
        ALL_FULL[t] = df_raw
    sys.stdout.write(f"\r   {i+1}/{len(ASX_TICKERS)}  descargados:{len(ALL_FULL)}")
    sys.stdout.flush()

print(f"\n✅ Tickers con datos disponibles: {len(ALL_FULL)}")

# ─── RÉGIMEN MACRO: VAS.AX (ASX300 ETF) > SMA200 → solo comprar en bull ──
print("\n🌐 Calculando régimen de mercado (VAS.AX > SMA200)...")
MARKET_REGIME = {}  # date → True=alcista, False=defensivo
_vas = fetch_live('VAS.AX', start=DATA_FETCH_START, end=SIM_END.strftime('%Y-%m-%d'))
if _vas is not None and len(_vas) > REGIME_SMA:
    _vas_sma = _vas['close'].rolling(REGIME_SMA).mean()
    for _d in _vas.index:
        _sv = _vas_sma.get(_d)
        if _sv is not None and not pd.isna(_sv):
            MARKET_REGIME[_d] = bool(_vas.loc[_d, 'close'] > _sv)
    _bull = sum(MARKET_REGIME.values())
    print(f"   ✅ ASX alcista {_bull}/{len(MARKET_REGIME)} días del período")
else:
    print("   ⚠️  VAS.AX no disponible — asumiendo mercado alcista")

# ─── PASO 2: Feature engineering ─────────────────────────────────
PROCESSED = {}
print("\n⚙️  Feature engineering...")
for t, raw in ALL_FULL.items():
    df_eng = engineer(raw)
    if df_eng is not None and len(df_eng) > 50:
        PROCESSED[t] = df_eng
    sys.stdout.write(f"\r   Procesados: {len(PROCESSED)}/{len(ALL_FULL)}")
    sys.stdout.flush()
print(f"\n✅ Tickers con features válidas: {len(PROCESSED)}")

# ─── PASO 3: Cargar modelos desde caché (ya entrenados) ────────────
MODELS = {}
FEATS  = {}
MODEL_CACHE_FILE   = 'models_cache.joblib'
FEATURE_CACHE_FILE = 'features_cache.joblib'
if not os.path.exists(MODEL_CACHE_FILE) or not os.path.exists(FEATURE_CACHE_FILE):
    print(f"\n❌ ERROR: No se encontró {MODEL_CACHE_FILE}.")
    print("   Ejecuta primero: python train_model_2021_2025.py")
    sys.exit(1)

import joblib
import csv as _csv

# ─── LOG DE ANÁLISIS DE TRADES ──────────────────────────────────
# Registra CADA trade cerrado con contexto completo de entrada
# → para aprender qué indicadores predicen pérdidas
TRADE_LOG_FILE = f"trade_log_{SIM_START.date()}_{SIM_END.date()}.csv"
MODELS = joblib.load(MODEL_CACHE_FILE)
FEATS  = joblib.load(FEATURE_CACHE_FILE)
print(f"\n🧠 Modelos cargados desde caché: {len(MODELS)} tickers")
print(f"   {MODEL_CACHE_FILE}  ({os.path.getsize(MODEL_CACHE_FILE)//1024} KB)")
# Solo usar tickers que tienen tanto modelo como datos descargados
MODELS = {t: m for t, m in MODELS.items() if t in PROCESSED}
FEATS  = {t: f for t, f in FEATS.items()  if t in PROCESSED}
print(f"   Tickers listos para simular: {len(MODELS)}\n")

# ─── PASO 3b: Pre-computar TODAS las probabilidades en batch ─────
# 40 llamadas batch en lugar de 252×40 = 10,080 llamadas individuales
print("⚡ Pre-computando probabilidades en batch (más rápido)...")
PROBS = {}   # PROBS[ticker][date] = prob
ok = 0
for t, model in MODELS.items():
    feats_t = FEATS.get(t, [])
    df_sim  = PROCESSED[t]
    sim_rows = df_sim[(df_sim.index >= SIM_START) & (df_sim.index <= SIM_END)]
    if sim_rows.empty or not all(f in sim_rows.columns for f in feats_t):
        PROBS[t] = {}
        continue
    X_batch = sim_rows[feats_t].replace([np.inf, -np.inf], np.nan).dropna()
    if X_batch.empty:
        PROBS[t] = {}
        continue
    try:
        probs_arr = model.predict_proba(X_batch.values)[:, 1]
        PROBS[t] = dict(zip(X_batch.index, probs_arr))
    except Exception:
        PROBS[t] = {}
    ok += 1
    sys.stdout.write(f"\r   Procesados: {ok}/{len(MODELS)}")
    sys.stdout.flush()
print(f"\n✅ Probabilidades pre-computadas para {ok} tickers\n")

# ─── PASO 4: Simular en el período configurado (días reales de mercado ASX) ───
all_sim = set()
for t, df in PROCESSED.items():
    if t in MODELS:
        all_sim.update(df[(df.index >= SIM_START) & (df.index <= SIM_END)].index.tolist())
sim_dates = sorted(all_sim)

print(f"{'─'*65}")
print(f"📅 Período validado: {len(sim_dates)} días de mercado ASX")
print(f"   Desde {sim_dates[0].date()} hasta {sim_dates[-1].date()}")
print(f"{'─'*65}")

cash        = START_CAPITAL
portfolio   = {}
equity_log  = []
all_trades  = []
_trade_rows = []   # log detallado de todos los trades cerrados
SL_HISTORY  = {}   # ticker → último día de stop-loss (cooldown re-entrada)
RECENT_PNLS = []   # últimos 30 P&L cerrados para half-Kelly dinámico
pending_entries = {}  # ticker → datos de la señal del día anterior (limit order next-day)

for day in sim_dates:
    # ── Mark-to-market ──
    equity = cash
    for t_pos, pos in portfolio.items():
        raw_day = ALL_FULL.get(t_pos)
        price   = raw_day.loc[day, 'close'] if raw_day is not None and day in raw_day.index else pos['buy_price']
        equity += pos['shares'] * price
    equity_log.append({'Fecha': day.strftime('%Y-%m-%d'), 'Equity_AUD': round(equity, 2)})

    # ── FILL PENDING LIMIT ORDERS (señales del día anterior) ────────────
    # El día anterior detectamos señal al cierre. Hoy intentamos llenar:
    #   • Si low de hoy <= señal×(1-ENTRY_DISCOUNT) → llena al precio límite (mejor precio)
    #   • Si no baja → llena al open de hoy (precio de mercado realista)
    for t_pend in list(pending_entries.keys()):
        # Cancelar si ya tenemos posición, o portafolio lleno
        if t_pend in portfolio or len(portfolio) >= MAX_POSITIONS:
            pending_entries.pop(t_pend, None)
            continue
        # Cancelar cooldown activo
        last_sl = SL_HISTORY.get(t_pend)
        if last_sl is not None and (day - last_sl).days < COOLDOWN_DAYS:
            pending_entries.pop(t_pend, None)
            continue

        raw_fill = ALL_FULL.get(t_pend)
        if raw_fill is None or day not in raw_fill.index:
            pending_entries.pop(t_pend, None)
            continue

        pend       = pending_entries.pop(t_pend)
        today_low  = float(raw_fill.loc[day, 'low'])
        today_open = float(raw_fill.loc[day, 'open'])
        limit_px   = pend['limit']

        # Determinar precio real de entrada
        if today_low <= limit_px:
            entry_price = limit_px       # ✅ limit order LLENADO (mejor precio)
            fill_type   = 'LIMIT'
        else:
            entry_price = today_open     # 📋 market at open
            fill_type   = 'OPEN'

        # Calcular tamaño de posición al precio real de entrada
        current_equity = cash + sum(
            p['shares'] * (ALL_FULL[t].loc[day, 'close']
                           if ALL_FULL.get(t) is not None and day in ALL_FULL[t].index
                           else p['buy_price'])
            for t, p in portfolio.items()
        )
        kf           = kelly_fraction(RECENT_PNLS)
        risk_amount  = current_equity * kf
        shares_kelly = int(risk_amount / (entry_price * SL_PCT))
        shares_cap   = int((current_equity * MAX_POS_PCT) / entry_price)
        shares       = max(1, min(shares_kelly, shares_cap))

        buy_comm = calc_commission(shares * entry_price)
        cost     = shares * entry_price + buy_comm
        if cost > cash or cost < MIN_BUY:
            continue

        estrategia  = pend['estrategia']
        atr_now     = pend['atr_now']
        tp1_use     = pend['tp1_use']
        atr_tp2_use = pend['atr_tp2_use']
        signal_close = pend['signal_close']

        cash -= cost
        portfolio[t_pend] = {
            'shares':      shares,
            'buy_price':   entry_price,
            'buy_comm':    buy_comm,
            'stop':        entry_price * (1 - SL_PCT),
            'tp1':         entry_price * (1 + tp1_use),
            'tp1_hit':     False,
            'tp2':         entry_price * (1 + atr_tp2_use),
            'trail_high':  entry_price,
            'hold_days':   0,
            'estrategia':  estrategia,
            'max_price':   entry_price,
            'min_price':   entry_price,
            'entry_context': {
                'fecha_entrada':  day.strftime('%Y-%m-%d'),
                'fecha_señal':    pend['fecha_señal'],
                'fill_type':      fill_type,
                'signal_close':   round(signal_close, 3),
                'entry_discount': round((signal_close - entry_price) / signal_close * 100, 3),
                'prob':           round(pend['prob'], 4),
                'rsi':            round(float(pend['rsi_raw']), 1),
                'macd':           round(float(pend['macd_d_raw']), 5),
                'adx':            round(float(pend['adx_raw']), 1),
                'vol_ratio':      round(float(pend['vol_ratio']), 3),
                'dist_sma50':     round((signal_close / float(pend['sma50_raw']) - 1) * 100, 2),
                'dist_sma200':    round((signal_close / float(pend['sma200_raw']) - 1) * 100, 2),
                'momentum5':      round(float(pend['mom5']) * 100, 2),
            },
        }
        all_trades.append({
            'Tipo':        '🟢 COMPRA',
            'Ticker':      t_pend,
            'Estrategia':  estrategia,
            'Fecha':       day.strftime('%Y-%m-%d'),
            'Precio':      round(entry_price, 3),
            'Acciones':    shares,
            'Monto_AUD':   round(cost, 2),
            'P&L_AUD':     '—',
            'P&L_%':       f"IA:{pend['prob']:.1%} [{fill_type}] desc:{(signal_close-entry_price)/signal_close*100:.2f}%",
            'Razon':       'SEÑAL_IA',
            'Dias':        0,
        })

    # ── EXITS ──────────────────────────────────────────────────
    to_sell = []
    for t_pos, pos in list(portfolio.items()):
        raw_day = ALL_FULL.get(t_pos)
        pos['hold_days'] = pos.get('hold_days', 0) + 1

        if raw_day is None or day not in raw_day.index:
            if pos['hold_days'] >= MAX_HOLD_DAYS:
                to_sell.append((t_pos, 'EXPIRE_NO_DATA', pos['buy_price']))
            continue

        row_raw = raw_day.loc[day]
        price   = row_raw['close']
        high    = row_raw['high']
        low     = row_raw['low']

        # ── Rastrear máx/mín durante la vida del trade ───────────────
        pos['max_price'] = max(pos.get('max_price', pos['buy_price']), high)
        pos['min_price'] = min(pos.get('min_price', pos['buy_price']), low)

        # ── Breakeven lock: si sube +3.5% (y pasaron ≥2 días) mover stop a +1.5% ──
        # Grace period de 2 días: evita que spikes del día 1 activen el lock prematuramente
        # Trade log: BRG/REA/TCL/WTC peakó +3.3-4.8% en día 4-10 → rescatar esas pérdidas
        if not pos.get('be_lock', False) and pos['hold_days'] >= 2:
            if high >= pos['buy_price'] * (1 + BREAKEVEN_PCT):
                be_price = pos['buy_price'] * (1 + BREAKEVEN_LOCK)
                if be_price > pos['stop']:
                    pos['stop']    = be_price
                    pos['be_lock'] = True

        # ── Trailing stop update ─────────────────────────────────────
        if high > pos.get('trail_high', pos['buy_price']):
            pos['trail_high'] = high
            # Tras TP1 parcial: trail 3% del máximo para proteger ganancias aseguradas
            # Antes de TP1: trail 5% (SL_PCT) — da espacio al movimiento inicial
            trail_pct = 0.03 if pos.get('tp1_hit', False) else SL_PCT
            pos['stop'] = max(pos.get('stop', 0), pos['trail_high'] * (1 - trail_pct))

        # ── TP1 parcial: cerrar 50% al 8% de ganancia ────────────────────
        if not pos.get('tp1_hit', False) and high >= pos.get('tp1', float('inf')):
            shares_sell = pos['shares'] // 2
            tp1_value   = shares_sell * pos['tp1']
            # Solo ejecutar TP parcial si el lote justifica la comisión (< 3% del valor)
            if shares_sell >= 1 and tp1_value > 0 and calc_commission(tp1_value) / tp1_value < 0.03:
                sell_gross   = shares_sell * pos['tp1']
                sell_comm    = calc_commission(sell_gross)
                net_part     = sell_gross - sell_comm
                pct_frac     = shares_sell / pos['shares']
                buy_cost_p   = shares_sell * pos['buy_price'] + pos.get('buy_comm', 0) * pct_frac
                pnl_part     = net_part - buy_cost_p
                cash        += net_part
                RECENT_PNLS.append(pnl_part)
                if len(RECENT_PNLS) > 30: RECENT_PNLS.pop(0)
                pos['shares']   -= shares_sell
                pos['buy_comm']  = pos.get('buy_comm', 0) * (1 - pct_frac)
                pos['tp1_hit']   = True
                pos['stop']      = max(pos['stop'], pos['buy_price'] * 1.02)  # mover stop a BE+2%
                all_trades.append({
                    'Tipo': '🟡 TP PARCIAL', 'Ticker': t_pos,
                    'Fecha': day.strftime('%Y-%m-%d'), 'Precio': round(pos['tp1'], 3),
                    'Acciones': shares_sell, 'Monto_AUD': round(net_part, 2),
                    'P&L_AUD': round(pnl_part, 2), 'P&L_%': f"+{TP1_PCT:.0%}",
                    'Razon': 'TP1_PARCIAL', 'Dias': pos['hold_days'],
                })

        # ── Salidas principales (elif → solo una acción por día) ──────────
        if   low  <= pos['stop']:
            to_sell.append((t_pos, 'STOP_LOSS',   pos['stop']))
        elif high >= pos.get('tp2', float('inf')):
            to_sell.append((t_pos, 'TAKE_PROFIT', pos['tp2']))
        elif pos['hold_days'] >= MAX_HOLD_DAYS:
            to_sell.append((t_pos, 'EXPIRE',      price))
        elif pos['hold_days'] >= 3:
            # MACD exit: salir si momentum se invierte
            # ─ Si hay BE lock activo: umbral más bajo (0.5%) — proteger ganancia asegurada
            # ─ Sin BE lock: sólo salir si hay ganancia > 1% (evitar salida prematura)
            raw_pos = ALL_FULL.get(t_pos)
            if raw_pos is not None:
                slice_p = raw_pos.loc[raw_pos.index <= day, 'close']
                if len(slice_p) >= 30:
                    macd_now   = ta_lib.trend.macd_diff(slice_p).iloc[-1]
                    unrealized = (price - pos['buy_price']) / pos['buy_price']
                    min_profit = 0.005 if pos.get('be_lock', False) else 0.01
                    if macd_now < 0 and unrealized > min_profit:
                        to_sell.append((t_pos, 'MACD_EXIT', price))

    for t_pos, reason, sell_price in to_sell:
        if t_pos not in portfolio:
            continue  # ya liquidado (p.ej. TP1 parcial en mismo día)
        pos         = portfolio.pop(t_pos)
        n_shares    = pos['shares']
        sell_gross  = n_shares * sell_price
        sell_comm   = calc_commission(sell_gross)
        net_proceed = sell_gross - sell_comm
        buy_cost    = n_shares * pos['buy_price'] + pos.get('buy_comm', 0)
        pnl         = net_proceed - buy_cost
        pct         = (sell_price - pos['buy_price']) / pos['buy_price']
        cash       += net_proceed
        if reason == 'STOP_LOSS':
            SL_HISTORY[t_pos] = day  # registrar para cooldown
        RECENT_PNLS.append(pnl)
        if len(RECENT_PNLS) > 30: RECENT_PNLS.pop(0)
        all_trades.append({
            'Tipo':      '🔴 VENTA',
            'Ticker':    t_pos,
            'Fecha':     day.strftime('%Y-%m-%d'),
            'Precio':    round(sell_price, 3),
            'Acciones':  n_shares,
            'Monto_AUD': round(net_proceed, 2),
            'P&L_AUD':   round(pnl, 2),
            'P&L_%':     f"{pct:+.2%}",
            'Razon':     reason,
            'Dias':      pos['hold_days'],
        })
        # ── Log de análisis: capturar contexto completo del trade ────────
        ec = pos.get('entry_context', {})
        max_p = pos.get('max_price', sell_price)
        min_p = pos.get('min_price', sell_price)
        bp    = pos['buy_price']
        max_ganancia_pct = round((max_p - bp) / bp * 100, 2)
        max_perdida_pct  = round((min_p - bp) / bp * 100, 2)
        # ¿Llegó a tocar TP1 antes de caer?
        tp1_pct = (pos.get('tp1', bp * 1.09) - bp) / bp
        toco_tp1 = pos.get('tp1_hit', False)
        # ¿Cuánto faltó para el TP? (si perdió)
        dist_tp2_al_cierre = round((pos.get('tp2', bp) - sell_price) / bp * 100, 2)
        _trade_rows.append({
            'fecha_entrada':      ec.get('fecha_entrada', '?'),
            'fecha_señal':        ec.get('fecha_señal', ec.get('fecha_entrada', '?')),
            'fill_type':          ec.get('fill_type', 'CLOSE'),   # LIMIT / OPEN / CLOSE (legacy)
            'entry_discount_pct': ec.get('entry_discount', 0.0),  # % ahorrado vs señal close
            'fecha_salida':       day.strftime('%Y-%m-%d'),
            'ticker':             t_pos,
            'estrategia':         pos.get('estrategia', '?'),
            'resultado':          'WIN' if pnl > 0 else 'LOSS',
            'pnl_aud':            round(pnl, 2),
            'pnl_pct':            round(pct * 100, 2),
            'dias_trade':         pos['hold_days'],
            'razon_salida':       reason,
            # Indicadores en entrada
            'prob_ia':            ec.get('prob', 0),
            'rsi_entrada':        ec.get('rsi', 0),
            'macd_entrada':       ec.get('macd', 0),
            'adx_entrada':        ec.get('adx', 0),
            'vol_ratio_entrada':  ec.get('vol_ratio', 0),
            'dist_sma50_pct':     ec.get('dist_sma50', 0),
            'dist_sma200_pct':    ec.get('dist_sma200', 0),
            'momentum5_pct':      ec.get('momentum5', 0),
            # Comportamiento del precio durante el trade
            'precio_señal':       ec.get('signal_close', round(bp, 3)),
            'precio_entrada':     round(bp, 3),
            'precio_salida':      round(sell_price, 3),
            'precio_stop':        round(pos.get('stop', bp * 0.95), 3),
            'precio_tp1':         round(pos.get('tp1', 0), 3),
            'precio_tp2':         round(pos.get('tp2', 0), 3),
            'max_ganancia_pct':   max_ganancia_pct,
            'max_perdida_pct':    max_perdida_pct,
            'toco_tp1':           toco_tp1,
            'dist_tp2_al_cierre': dist_tp2_al_cierre,
        })

    # ── ENTRIES ────────────────────────────────────────────────
    if cash >= MIN_BUY and len(portfolio) + len(pending_entries) < MAX_POSITIONS and MARKET_REGIME.get(day, True):
        candidates = []
        regime_bull = MARKET_REGIME.get(day, True)

        for t_cand in MODELS:
            if t_cand in portfolio:
                continue
            raw_cand = ALL_FULL.get(t_cand)
            if raw_cand is None or day not in raw_cand.index:
                continue

            prob = PROBS.get(t_cand, {}).get(day, 0.0)

            # ─ Prob demasiado baja → saltar ────────────────────────────
            if prob < PROB_REVERSION:
                continue

            real_price = raw_cand.loc[day, 'close']
            if real_price <= 0:
                continue

            raw_slice  = raw_cand.loc[raw_cand.index <= day]
            if len(raw_slice) < 210:
                continue
            raw_close  = raw_slice['close']

            sma20_raw  = raw_close.rolling(20).mean().iloc[-1]
            sma50_raw  = raw_close.rolling(50).mean().iloc[-1]
            sma200_raw = raw_close.rolling(200).mean().iloc[-1]
            rsi_raw    = ta_lib.momentum.rsi(raw_close, 14).iloc[-1]
            macd_d_raw = ta_lib.trend.macd_diff(raw_close).iloc[-1]
            adx_raw    = ta_lib.trend.adx(raw_slice['high'], raw_slice['low'], raw_close, 14).iloc[-1]
            vol_avg20  = raw_slice['volume'].rolling(20).mean().iloc[-1]
            vol_ratio  = float(raw_slice['volume'].iloc[-1]) / vol_avg20 if vol_avg20 > 0 else 0.0
            mom5       = float(raw_close.pct_change(5).iloc[-1])
            atr_now    = ta_lib.volatility.average_true_range(
                raw_slice['high'], raw_slice['low'], raw_close, 14).iloc[-1]

            if pd.isna(sma50_raw) or pd.isna(sma200_raw) or pd.isna(rsi_raw):
                continue

            # ═══════ ESTRATEGIA 1: MOMENTUM BREAKOUT ═════════════════
            # Precio en tendencia alcista, MACD positivo, volumen confirma
            rejeccion_mom = None
            if prob >= PROB_MOMENTUM:
                if   real_price <= sma50_raw:          rejeccion_mom = 'bajo_sma50'
                elif sma50_raw < sma200_raw * 0.95:    rejeccion_mom = 'sma50_bajo_sma200'
                elif rsi_raw >= 78:                    rejeccion_mom = 'rsi_alto'
                elif macd_d_raw <= 0:                  rejeccion_mom = 'macd_negativo'
                elif adx_raw < 15:                     rejeccion_mom = 'adx_bajo'
                elif vol_ratio < VOL_MIN_MOM:          rejeccion_mom = 'vol_bajo'  # >=1.20 (análisis: winners=1.83)
                elif mom5 < 0.015:                     rejeccion_mom = 'momentum_debil'  # trade log: losers=0.91% winners=2.35%
                else:
                    score = prob * (1.0 + max(0.0, min(1.0, mom5 * 15)))
                    candidates.append((t_cand, real_price, prob, score, 'MOMENTUM', atr_now,
                                       rsi_raw, macd_d_raw, adx_raw, vol_ratio, mom5, sma50_raw, sma200_raw))
                    continue
            else:
                rejeccion_mom = 'prob_baja_mom'

            # ═══════ ESTRATEGIA 2: MEAN-REVERSION (oversold calidad) ════════
            # RSI < 38, precio cerca de soporte SMA200, MACD girando al alza
            rejeccion_rev = None
            if prob >= PROB_REVERSION:
                near_support = real_price >= sma200_raw * 0.92 and real_price <= sma200_raw * 1.08
                macd_turning = macd_d_raw > -0.05 * real_price * 0.001  # cerca de cruce
                if   rsi_raw >= 38:           rejeccion_rev = 'rsi_no_oversold'
                elif not near_support:        rejeccion_rev = 'lejos_soporte'
                elif not macd_turning:        rejeccion_rev = 'macd_muy_negativo'
                elif vol_ratio < 0.8:         rejeccion_rev = 'vol_muy_bajo'
                elif adx_raw > 35:            rejeccion_rev = 'tendencia_bajista_fuerte'
                else:
                    # Score menor que momentum (más riesgo contratendencia)
                    score = prob * 0.85 * (1.0 + max(0.0, min(0.5, (38 - rsi_raw) / 38)))
                    candidates.append((t_cand, real_price, prob, score, 'REVERSION', atr_now,
                                       rsi_raw, macd_d_raw, adx_raw, vol_ratio, mom5, sma50_raw, sma200_raw))
                    continue
            else:
                rejeccion_rev = 'prob_baja_rev'

        candidates.sort(key=lambda x: x[3], reverse=True)  # mejor score primero

        for t_cand, price, prob, score, estrategia, atr_now, rsi_raw, macd_d_raw, adx_raw, vol_ratio, mom5, sma50_raw, sma200_raw in candidates:
            if cash < MIN_BUY or len(portfolio) + len(pending_entries) >= MAX_POSITIONS:
                break

            # ── Cooldown: no re-entrar tras un stop-loss reciente ─────────
            last_sl = SL_HISTORY.get(t_cand)
            if last_sl is not None and (day - last_sl).days < 7:  # ~5 días de mercado
                continue

            # ── Saltar si ya está en cola o en portfolio ─────────────────
            if t_cand in pending_entries:
                continue

            # ── TP ajustado según estrategia ─────────────────────────────
            if estrategia == 'REVERSION':
                tp1_use     = TP1_PCT * 0.8
                atr_tp2_use = max(TP2_PCT * 0.75, float(atr_now / price) * 2.5)
            else:
                tp1_use     = TP1_PCT
                atr_tp2_use = max(TP2_PCT, float(atr_now / price) * 3.5)

            # ── Señal detectada al cierre → queued como limit order D+1 ──
            # La compra real ocurrirá mañana al inicio del día (fill pending section)
            pending_entries[t_cand] = {
                'limit':       price * (1 - ENTRY_DISCOUNT),   # 1% bajo el close
                'signal_close': price,
                'fecha_señal': day.strftime('%Y-%m-%d'),
                'estrategia':  estrategia,
                'atr_now':     atr_now,
                'tp1_use':     tp1_use,
                'atr_tp2_use': atr_tp2_use,
                'prob':        prob,
                'rsi_raw':     rsi_raw,
                'macd_d_raw':  macd_d_raw,
                'adx_raw':     adx_raw,
                'vol_ratio':   vol_ratio,
                'mom5':        mom5,
                'sma50_raw':   sma50_raw,
                'sma200_raw':  sma200_raw,
            }

# ─── LIQUIDAR POSICIONES ABIERTAS ────────────────────────────────
if sim_dates:
    final_day = sim_dates[-1]
    for t_pos, pos in list(portfolio.items()):
        raw_d = ALL_FULL.get(t_pos)
        price       = raw_d.loc[final_day, 'close'] if raw_d is not None and final_day in raw_d.index else pos['buy_price']
        sell_gross  = pos['shares'] * price
        sell_comm   = calc_commission(sell_gross)
        net_proceed = sell_gross - sell_comm
        buy_cost    = pos['shares'] * pos['buy_price'] + pos.get('buy_comm', 0)
        pnl         = net_proceed - buy_cost
        pct         = (price - pos['buy_price']) / pos['buy_price']
        cash       += net_proceed
        all_trades.append({
            'Tipo':      '🔴 VENTA',
            'Ticker':    t_pos,
            'Fecha':     final_day.strftime('%Y-%m-%d'),
            'Precio':    round(price, 3),
            'Acciones':  pos['shares'],
            'Monto_AUD': round(net_proceed, 2),
            'P&L_AUD':   round(pnl, 2),
            'P&L_%':     f"{pct:+.2%}",
            'Razon':     'CIERRE_FINAL',
            'Dias':      pos.get('hold_days', 0),
        })
        # Log de análisis para posiciones cerradas al final del período
        ec = pos.get('entry_context', {})
        bp = pos['buy_price']
        max_p = pos.get('max_price', price)
        min_p = pos.get('min_price', price)
        _trade_rows.append({
            'fecha_entrada':      ec.get('fecha_entrada', '?'),
            'fecha_señal':        ec.get('fecha_señal', ec.get('fecha_entrada', '?')),
            'fill_type':          ec.get('fill_type', 'CLOSE'),
            'entry_discount_pct': ec.get('entry_discount', 0.0),
            'fecha_salida':       final_day.strftime('%Y-%m-%d'),
            'ticker':             t_pos,
            'estrategia':         pos.get('estrategia', '?'),
            'resultado':          'WIN' if pnl > 0 else 'LOSS',
            'pnl_aud':            round(pnl, 2),
            'pnl_pct':            round(pct * 100, 2),
            'dias_trade':         pos.get('hold_days', 0),
            'razon_salida':       'CIERRE_FINAL',
            'prob_ia':            ec.get('prob', 0),
            'rsi_entrada':        ec.get('rsi', 0),
            'macd_entrada':       ec.get('macd', 0),
            'adx_entrada':        ec.get('adx', 0),
            'vol_ratio_entrada':  ec.get('vol_ratio', 0),
            'dist_sma50_pct':     ec.get('dist_sma50', 0),
            'dist_sma200_pct':    ec.get('dist_sma200', 0),
            'momentum5_pct':      ec.get('momentum5', 0),
            'precio_señal':       ec.get('signal_close', round(bp, 3)),
            'precio_entrada':     round(bp, 3),
            'precio_salida':      round(price, 3),
            'precio_stop':        round(pos.get('stop', bp * 0.95), 3),
            'precio_tp1':         round(pos.get('tp1', 0), 3),
            'precio_tp2':         round(pos.get('tp2', 0), 3),
            'max_ganancia_pct':   round((max_p - bp) / bp * 100, 2),
            'max_perdida_pct':    round((min_p - bp) / bp * 100, 2),
            'toco_tp1':           pos.get('tp1_hit', False),
            'dist_tp2_al_cierre': round((pos.get('tp2', bp) - price) / bp * 100, 2),
        })

# ═══════════════════════════════════════════════════════════════════
#  RESULTADOS
# ═══════════════════════════════════════════════════════════════════
trades_df = pd.DataFrame(all_trades)
eq_df     = pd.DataFrame(equity_log)

ventas  = trades_df[trades_df['Tipo'].str.contains('VENTA|TP PARCIAL')]  if not trades_df.empty else pd.DataFrame()
compras = trades_df[trades_df['Tipo'].str.contains('COMPRA')]              if not trades_df.empty else pd.DataFrame()

pnl_vals = ventas['P&L_AUD'].apply(pd.to_numeric, errors='coerce').dropna() if not ventas.empty else pd.Series([0.0])

roi    = (cash - START_CAPITAL) / START_CAPITAL
wins   = (pnl_vals > 0).sum()
losses = (pnl_vals <= 0).sum()
n_ops  = wins + losses

eq_s   = eq_df['Equity_AUD'] if not eq_df.empty else pd.Series([START_CAPITAL, cash])
max_dd = ((eq_s.cummax() - eq_s) / eq_s.cummax()).max()

print(f"\n{'═'*65}")
print(f"  💰  RESULTADO FINAL — $8,000 AUD  ({SIM_START.date()} → {SIM_END.date()})")
print(f"{'═'*65}")
print(f"  Capital inicial:           $8,000.00 AUD")
print(f"  Capital final:             ${cash:,.2f} AUD")
gol = 'GANANCIA' if roi > 0 else 'PÉRDIDA'
print(f"  Resultado neto:            {roi:+.2%}  ({gol}: ${abs(cash-START_CAPITAL):.2f} AUD)")
print(f"  Max Drawdown:              {max_dd:.2%}")
print(f"  Días simulados (2026):     {len(sim_dates)}")
print(f"  Tickers en universo:       {len(MODELS)}")

print(f"\n{'─'*65}")
print(f"  📊  ESTADÍSTICAS DE TRADING")
print(f"{'─'*65}")
print(f"  Total compras:             {len(compras)}")
print(f"  Total ventas cerradas:     {n_ops}")
if n_ops > 0:
    print(f"  ✅ Ganadoras:              {wins}  ({wins/n_ops:.1%})")
    print(f"  ❌ Perdedoras:             {losses}  ({losses/n_ops:.1%})")
    print(f"  PnL promedio / trade:     ${pnl_vals.mean():+.2f} AUD")
    print(f"  Mejor trade:              ${pnl_vals.max():+.2f} AUD")
    print(f"  Peor trade:               ${pnl_vals.min():+.2f} AUD")
else:
    print("  ℹ️  Sin operaciones cerradas — Capital preservado al 100 %")
    print("     (El modelo no superó el umbral de confianza en 2026)")

if not trades_df.empty:
    print(f"\n{'─'*65}")
    print(f"  🗒   OPERACIONES DETALLADAS ({len(trades_df)})")
    print(f"{'─'*65}")
    pd.set_option('display.max_rows', 300)
    pd.set_option('display.width', 130)
    print(trades_df.to_string(index=False))

# ── Estadísticas de ejecución de órdenes (limit vs open) ──────────
if _trade_rows:
    tlog = pd.DataFrame(_trade_rows)
    if 'fill_type' in tlog.columns:
        n_limit = (tlog['fill_type'] == 'LIMIT').sum()
        n_open  = (tlog['fill_type'] == 'OPEN').sum()
        n_total = len(tlog)
        avg_disc_limit = tlog.loc[tlog['fill_type'] == 'LIMIT', 'entry_discount_pct'].mean()
        avg_disc_open  = tlog.loc[tlog['fill_type'] == 'OPEN',  'entry_discount_pct'].mean()
        avg_disc_all   = tlog['entry_discount_pct'].mean()
        print(f"\n{'─'*65}")
        print(f"  ⚡  EJECUCIÓN DE ÓRDENES (limit order {ENTRY_DISCOUNT*100:.0f}% bajo señal)")
        print(f"{'─'*65}")
        print(f"  LIMIT (mejor precio):  {n_limit}/{n_total} trades  ({n_limit/max(1,n_total):.0%})")
        print(f"  OPEN  (a la apertura): {n_open}/{n_total} trades  ({n_open/max(1,n_total):.0%})")
        if n_limit > 0:
            print(f"  Descuento promedio (LIMIT): {avg_disc_limit:.3f}%")
        print(f"  Descuento promedio total:   {avg_disc_all:.3f}%")

        # P&L comparativo
        pnl_limit = tlog.loc[tlog['fill_type'] == 'LIMIT', 'pnl_aud']
        pnl_open  = tlog.loc[tlog['fill_type'] == 'OPEN',  'pnl_aud']
        if len(pnl_limit) > 0:
            print(f"  P&L promedio LIMIT: ${pnl_limit.mean():+.2f} AUD")
        if len(pnl_open) > 0:
            print(f"  P&L promedio OPEN:  ${pnl_open.mean():+.2f} AUD")

if not eq_df.empty:
    print(f"\n{'─'*65}")
    print(f"  📈  CURVA DE EQUITY 2026 (cada 5 días)")
    print(f"{'─'*65}")
    step = max(1, len(eq_df) // 15)
    for _, row in eq_df.iloc[::step].iterrows():
        delta  = row['Equity_AUD'] - START_CAPITAL
        bar    = ('▲' if delta >= 0 else '▼') * min(20, int(abs(delta) / 10))
        sign   = '+' if delta >= 0 else ''
        print(f"  {row['Fecha']}  ${row['Equity_AUD']:>8.2f}  ({sign}{delta:.0f})  {bar}")
    last = eq_df.iloc[-1]
    print(f"  {last['Fecha']}  ${last['Equity_AUD']:>8.2f}  ← HOY")

# Guardar Excel
ts  = datetime.now().strftime('%Y%m%d_%H%M')
out = f"sim_2k_2026_{ts}.xlsx"
with pd.ExcelWriter(out, engine='openpyxl') as w:
    pd.DataFrame([{
        'Capital_Inicial_AUD': START_CAPITAL,
        'Capital_Final_AUD':   round(cash, 2),
        'ROI':                 f"{roi:+.2%}",
        'Max_Drawdown':        f"{max_dd:.2%}",
        'Dias_Simulados':      len(sim_dates),
        'Tickers_Universo':    len(MODELS),
        'Total_Compras':       len(compras),
        'Total_Ventas':        n_ops,
        'Win_Rate':            f"{wins/n_ops:.1%}" if n_ops > 0 else 'N/A',
        'PnL_Promedio_AUD':    round(pnl_vals.mean(), 2) if n_ops > 0 else 0,
    }]).to_excel(w, sheet_name='Resumen', index=False)
    if not trades_df.empty:
        trades_df.to_excel(w, sheet_name='Operaciones', index=False)
    if not eq_df.empty:
        eq_df.to_excel(w, sheet_name='Equity_Curve', index=False)

print(f"\n💾 Guardado en: {out}")

# ─── GUARDAR Y ANALIZAR TRADE LOG ────────────────────────────────────
if _trade_rows:
    tlog = pd.DataFrame(_trade_rows)
    tlog.to_csv(TRADE_LOG_FILE, index=False)

    wins_log  = tlog[tlog['resultado'] == 'WIN']
    loss_log  = tlog[tlog['resultado'] == 'LOSS']

    print(f"\n{'─'*65}")
    print(f"  📋  ANÁLISIS DE TRADES — ¿POR QUÉ PERDEMOS?")
    print(f"{'─'*65}")
    print(f"  Trades registrados: {len(tlog)}  (🟢 {len(wins_log)} wins / 🔴 {len(loss_log)} losses)")

    if not loss_log.empty:
        print(f"\n  ── CAUSAS DE PÉRDIDA:")
        for razón, cnt in loss_log['razon_salida'].value_counts().items():
            print(f"     {razón:<20} {cnt:>3} veces")

        print(f"\n  ── INDICADORES EN ENTRADA (Pérdidas vs Ganancias):")
        cols_cmp = ['prob_ia', 'rsi_entrada', 'adx_entrada', 'vol_ratio_entrada',
                    'dist_sma50_pct', 'momentum5_pct']
        for col in cols_cmp:
            if col in tlog.columns:
                w_mean = wins_log[col].mean() if not wins_log.empty else float('nan')
                l_mean = loss_log[col].mean()
                diff_sign = '⚠️ ' if abs(w_mean - l_mean) > 0.5 * abs(w_mean + l_mean + 0.001) / 2 else '  '
                print(f"     {diff_sign}{col:<25}  WIN avg={w_mean:>7.2f}  LOSS avg={l_mean:>7.2f}")

        print(f"\n  ── COMPORTAMIENTO DEL PRECIO (solo pérdidas):")
        print(f"     Máx ganancia no realizada antes de caer: {loss_log['max_ganancia_pct'].mean():>+.2f}% prom")
        print(f"     Profundidad máxima de pérdida:           {loss_log['max_perdida_pct'].mean():>+.2f}% prom")
        print(f"     ¿Tocó TP1 antes de perder?:              {loss_log['toco_tp1'].sum()} de {len(loss_log)} trades")
        print(f"     Días promedio antes de salir con pérdida: {loss_log['dias_trade'].mean():.1f} días")

        # Trades que llegaron a ser positivos pero terminaron en pérdida
        casi_ganadores = loss_log[loss_log['max_ganancia_pct'] > 2.0]
        if not casi_ganadores.empty:
            print(f"\n  ── OPORTUNIDADES PERDIDAS (llegaron a +2% pero cerraron en LOSS):")
            for _, row in casi_ganadores.iterrows():
                print(f"     {row['ticker']:<8} {row['estrategia']:<10}  entró {row['fecha_entrada']}  "
                      f"max +{row['max_ganancia_pct']:.1f}%  cerró {row['pnl_pct']:+.1f}%  "
                      f"({row['razon_salida']}, día {int(row['dias_trade'])})")

    if not wins_log.empty:
        print(f"\n  ── PERFIL DE OPERACIONES GANADORAS:")
        print(f"     Máx ganancia no realizada:   {wins_log['max_ganancia_pct'].mean():>+.2f}% prom")
        print(f"     Profundidad máxima sufrida:  {wins_log['max_perdida_pct'].mean():>+.2f}% prom")
        print(f"     Días promedio hasta la venta: {wins_log['dias_trade'].mean():.1f} días")

    print(f"\n  💾 Trade log guardado en: {TRADE_LOG_FILE}")
print(f"{'═'*65}\n")
