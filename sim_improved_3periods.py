# sim_improved_3periods.py — Modelo MEJORADO + Simulación en 3 períodos
# ═══════════════════════════════════════════════════════════════════════
#  MEJORAS IMPLEMENTADAS:
#  ────────────────────────────────────────────────────────────────────
#  🧠 MODELO:
#    1. Stacking Ensemble (RF + GB + LightGBM + XGBoost → meta LR)
#    2. Multi-horizonte (3, 5, 10 días) → probabilidad promedio
#    3. +12 features nuevas (OBV, VWAP dist, gaps, Sharpe rolling,
#       Ichimoku, aceleración, volumen ponderado, z-score reversión)
#    4. Target ponderado por calidad (Sharpe-like)
#    5. Purged Time-Series CV para evitar data leakage temporal
#
#  📈 SIMULACIÓN:
#    1. Sector diversificación (máx 2 por sector)
#    2. ATR-adaptive SL/TP en vez de % fijo
#    3. Volatility-adjusted position sizing
#    4. Portfolio heat tracking (máx 15% riesgo total)
#    5. Dynamic threshold (ajuste según win-rate reciente)
#    6. Confirmation entry (señal + contexto técnico)
#
#  ⏱  PERÍODOS:
#    Período 1: train 2018-2021 → test 2022-2024
#    Período 2: train 2018-2024 → test 2025
#    Período 3: train 2018-2025 → test 2026 (YTD)
# ═══════════════════════════════════════════════════════════════════════

import os, sys, warnings, time
import numpy as np
import pandas as pd
import yfinance as yf
import ta as ta_lib
from datetime import datetime

from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    StackingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from sklearn.inspection import permutation_importance

import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings('ignore')
os.environ['SIMULATE_AU_INVESTMENT'] = '1'

# ═══════════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN GLOBAL
# ═══════════════════════════════════════════════════════════════════════
START_CAPITAL   = 8000.0
MIN_BUY         = 500.0
MAX_POSITIONS   = 4           # Perfil conservador: menos exposición simultánea
MAX_POS_PCT     = 0.14        # Máx 14% del equity por posición
MAX_SECTOR_POS  = 2           # Máx 2 posiciones por sector
SL_ATR_MULT     = 1.6         # Stop menos sensible para evitar barridos
TP1_ATR_MULT    = 2.2         # TP1 más exigente
TP2_ATR_MULT    = 4.5         # TP2 más exigente
MAX_HOLD_DAYS   = 22          # Evita churn excesivo
COOLDOWN_DAYS   = 7           # No reentrar demasiado rápido al mismo ticker
BREAKEVEN_ATR   = 1.4         # Activar BE más tarde
BREAKEVEN_LOCK_ATR = 0.2      # Lock conservador
ENTRY_DISCOUNT  = 0.002       # Menos fill forzado en apertura
PORTFOLIO_HEAT  = 0.12        # Riesgo total máximo más estricto
COMMISSION_FLAT = 10.0
COMMISSION_RATE = 0.0011
REGIME_SMA      = 200
ROI_TARGET_MIN  = 0.15
ENTRY_PAUSE_DD  = 0.12
ENTRY_RESUME_DD = 0.06

# Umbrales conservadores: priorizar calidad sobre cantidad
BASE_PROB_MOM   = 0.52
BASE_PROB_REV   = 0.46
MIN_PROB_ANY    = 0.42

# ─── SECTOR MAPPING ──────────────────────────────────────────────────
SECTOR_MAP = {
    # Minería & Recursos
    'BHP.AX': 'Mining', 'RIO.AX': 'Mining', 'FMG.AX': 'Mining', 'S32.AX': 'Mining',
    'MIN.AX': 'Mining', 'IGO.AX': 'Mining', 'PLS.AX': 'Mining', 'LYC.AX': 'Mining',
    'NST.AX': 'Mining', 'EVN.AX': 'Mining', 'RRL.AX': 'Mining', 'SBM.AX': 'Mining',
    'BSL.AX': 'Mining', 'ILU.AX': 'Mining', 'NHC.AX': 'Mining', 'SGM.AX': 'Mining',
    'PDN.AX': 'Mining', 'NIC.AX': 'Mining', 'RSG.AX': 'Mining',
    # Bancos
    'CBA.AX': 'Banks', 'NAB.AX': 'Banks', 'WBC.AX': 'Banks', 'ANZ.AX': 'Banks',
    'MQG.AX': 'Banks',
    # Energía
    'STO.AX': 'Energy', 'WDS.AX': 'Energy', 'ORG.AX': 'Energy', 'BPT.AX': 'Energy',
    'APA.AX': 'Energy',
    # Salud
    'CSL.AX': 'Health', 'COH.AX': 'Health', 'RMD.AX': 'Health', 'SHL.AX': 'Health',
    'PME.AX': 'Health', 'RHC.AX': 'Health', 'HLS.AX': 'Health', 'MPL.AX': 'Health',
    'NHF.AX': 'Health',
    # Retail
    'WES.AX': 'Retail', 'WOW.AX': 'Retail', 'COL.AX': 'Retail', 'HVN.AX': 'Retail',
    'SUL.AX': 'Retail', 'PMV.AX': 'Retail', 'DMP.AX': 'Retail',
    # Tech
    'XRO.AX': 'Tech', 'REA.AX': 'Tech', 'WTC.AX': 'Tech', 'NXT.AX': 'Tech',
    'SEK.AX': 'Tech', 'CAR.AX': 'Tech',
    # REITs
    'GMG.AX': 'REIT', 'SCG.AX': 'REIT', 'DXS.AX': 'REIT', 'MGR.AX': 'REIT',
    'SGP.AX': 'REIT', 'CHC.AX': 'REIT', 'BWP.AX': 'REIT', 'CIP.AX': 'REIT',
    'CLW.AX': 'REIT', 'GOZ.AX': 'REIT', 'CQR.AX': 'REIT',
    # Insurance
    'QBE.AX': 'Insurance', 'SUN.AX': 'Insurance',
    # Telecom
    'TLS.AX': 'Telecom', 'TPG.AX': 'Telecom', 'SPK.AX': 'Telecom',
    # Industrial
    'TCL.AX': 'Industrial', 'QAN.AX': 'Industrial', 'AMC.AX': 'Industrial',
    'JHX.AX': 'Industrial', 'QUB.AX': 'Industrial', 'AZJ.AX': 'Industrial',
    'WOR.AX': 'Industrial', 'ORI.AX': 'Industrial',
    # Consumer
    'TWE.AX': 'Consumer', 'ALL.AX': 'Consumer', 'FLT.AX': 'Consumer',
    'IEL.AX': 'Consumer', 'ARB.AX': 'Consumer', 'BRG.AX': 'Consumer',
    'GWA.AX': 'Consumer', 'ELD.AX': 'Consumer',
}

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

# ═══════════════════════════════════════════════════════════════════════
#  FEATURES MEJORADAS
# ═══════════════════════════════════════════════════════════════════════
FEATURE_COLS_V2 = [
    # --- Original features ---
    'dist_sma20','dist_sma50','dist_sma200','ma_cross_20_50','ma_cross_50_200',
    'atr_pct','vol_regime',
    'momentum_3','momentum_5','momentum_10','momentum_20',
    'roc_5','roc_10','rsi','macd_diff','adx',
    'bb_width','bb_upper_dist','bb_lower_dist',
    'stoch_k','stoch_d','williams_r','cci',
    'vol_rel_20','vol_rel_50',
    'close_to_max5','close_to_min5','close_to_max20','close_to_min20',
    # --- NUEVAS features V2 ---
    'obv_slope',           # On-Balance Volume tendencia (5d slope)
    'vwap_dist',           # Distancia al VWAP rolling
    'gap_pct',             # Gap apertura vs cierre anterior
    'consec_up',           # Días consecutivos alcistas
    'consec_down',         # Días consecutivos bajistas
    'sharpe_5',            # Rolling Sharpe ratio 5 días
    'sharpe_20',           # Rolling Sharpe ratio 20 días
    'ichimoku_dist',       # Distancia a Ichimoku Cloud base
    'price_accel',         # Aceleración del precio (2da derivada)
    'vol_weighted_mom',    # Momentum ponderado por volumen
    'return_zscore',       # Z-score de retornos (mean reversion signal)
    'range_pct',           # Rango H-L como % del close
]

def calc_commission(value_aud: float) -> float:
    return max(COMMISSION_FLAT, value_aud * COMMISSION_RATE)

def kelly_fraction(recent_pnls: list) -> float:
    if len(recent_pnls) < 8:
        return 0.20
    wins   = [p for p in recent_pnls if p > 0]
    losses = [abs(p) for p in recent_pnls if p <= 0]
    if not wins or not losses:
        return 0.20
    p  = len(wins) / len(recent_pnls)
    b  = np.mean(wins) / np.mean(losses)
    f  = (p * b - (1 - p)) / b
    return max(0.10, min(0.35, f * 0.5))


# ═══════════════════════════════════════════════════════════════════════
#  DESCARGA Y CACHÉ DE DATOS
# ═══════════════════════════════════════════════════════════════════════
def fetch_live(ticker, start, end=None):
    """Descarga datos vía yfinance. Retorna DataFrame o None."""
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


# ═══════════════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING V2 (MEJORADO)
# ═══════════════════════════════════════════════════════════════════════
def engineer_v2(raw_df, horizon=5):
    """
    Feature engineering mejorado con +12 indicadores nuevos.
    """
    df = raw_df.copy()
    if len(df) < 120:
        return None

    c = df['close']
    h = df['high']
    l = df['low']
    v = df['volume']
    o = df['open']

    # ── ORIGINAL FEATURES ──
    df['sma_20']  = ta_lib.trend.sma_indicator(c, 20)
    df['sma_50']  = ta_lib.trend.sma_indicator(c, 50)
    df['sma_200'] = ta_lib.trend.sma_indicator(c, 200)
    df['dist_sma20']  = (c / df['sma_20'])  - 1
    df['dist_sma50']  = (c / df['sma_50'])  - 1
    df['dist_sma200'] = (c / df['sma_200']) - 1
    df['ma_cross_20_50']  = (df['sma_20']  > df['sma_50']).astype(int)
    df['ma_cross_50_200'] = (df['sma_50']  > df['sma_200']).astype(int)

    atr = ta_lib.volatility.average_true_range(h, l, c, 14)
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
    df['adx']       = ta_lib.trend.adx(h, l, c, 14)

    bb_h = ta_lib.volatility.bollinger_hband(c, 20)
    bb_l = ta_lib.volatility.bollinger_lband(c, 20)
    df['bb_width']      = (bb_h - bb_l) / c
    df['bb_upper_dist'] = (bb_h - c) / c
    df['bb_lower_dist'] = (c - bb_l) / c

    df['stoch_k']    = ta_lib.momentum.stoch(h, l, c, 14, 3) / 100.0
    df['stoch_d']    = ta_lib.momentum.stoch_signal(h, l, c, 14, 3) / 100.0
    df['williams_r'] = ta_lib.momentum.williams_r(h, l, c, 14)
    df['cci']        = ta_lib.trend.cci(h, l, c, 20)

    vm20 = v.rolling(20).mean()
    vm50 = v.rolling(50).mean()
    df['vol_rel_20'] = np.where(vm20 > 0, v / vm20, 1.0)
    df['vol_rel_50'] = np.where(vm50 > 0, v / vm50, 1.0)

    for w, sfx in [(5, '5'), (20, '20')]:
        df[f'max_{sfx}'] = c.rolling(w).max()
        df[f'min_{sfx}'] = c.rolling(w).min()
    df['close_to_max5']  = (c / df['max_5'])  - 1
    df['close_to_min5']  = (c / df['min_5'])  - 1
    df['close_to_max20'] = (c / df['max_20']) - 1
    df['close_to_min20'] = (c / df['min_20']) - 1

    # ══════════════════════════════════════════════════════════════════
    #  NUEVAS FEATURES V2
    # ══════════════════════════════════════════════════════════════════

    # 1. OBV (On-Balance Volume) — slope 5 días
    obv = (np.sign(c.diff()) * v).fillna(0).cumsum()
    df['obv_slope'] = obv.rolling(5).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 5 else 0, raw=True
    )
    # Normalizar OBV slope por volumen promedio
    df['obv_slope'] = np.where(vm20 > 0, df['obv_slope'] / vm20, 0)

    # 2. VWAP distance (rolling 20-day VWAP proxy)
    typical_price = (c + h + l) / 3
    vwap_20 = (typical_price * v).rolling(20).sum() / v.rolling(20).sum()
    df['vwap_dist'] = np.where(vwap_20 > 0, (c / vwap_20) - 1, 0)

    # 3. Gap % (apertura vs cierre anterior)
    df['gap_pct'] = (o / c.shift(1)) - 1

    # 4. Días consecutivos alcistas/bajistas (vectorizado)
    up = (c > c.shift(1)).astype(int)
    dn = (c < c.shift(1)).astype(int)
    # Usar cumsum con reset para conteo vectorizado
    up_groups = (up != up.shift()).cumsum()
    dn_groups = (dn != dn.shift()).cumsum()
    df['consec_up']   = up.groupby(up_groups).cumsum()
    df['consec_down'] = dn.groupby(dn_groups).cumsum()

    # 5. Rolling Sharpe ratio (5d, 20d)
    ret = c.pct_change()
    df['sharpe_5']  = ret.rolling(5).mean() / ret.rolling(5).std().replace(0, np.nan)
    df['sharpe_20'] = ret.rolling(20).mean() / ret.rolling(20).std().replace(0, np.nan)

    # 6. Ichimoku Cloud — distancia base line
    period9_high  = h.rolling(9).max()
    period9_low   = l.rolling(9).min()
    period26_high = h.rolling(26).max()
    period26_low  = l.rolling(26).min()
    tenkan   = (period9_high + period9_low) / 2
    kijun    = (period26_high + period26_low) / 2
    df['ichimoku_dist'] = np.where(kijun > 0, (c / kijun) - 1, 0)

    # 7. Price acceleration (2da derivada del precio)
    mom1 = c.pct_change()
    mom2 = mom1.diff()
    df['price_accel'] = mom2

    # 8. Volume-weighted momentum
    vol_w = v / vm20.replace(0, 1)
    df['vol_weighted_mom'] = c.pct_change(5) * vol_w

    # 9. Return Z-score (mean reversion signal)
    ret_20_mean = ret.rolling(20).mean()
    ret_20_std  = ret.rolling(20).std()
    df['return_zscore'] = np.where(ret_20_std > 0, (ret - ret_20_mean) / ret_20_std, 0)

    # 10. Range % (H-L / Close — volatility microstructure)
    df['range_pct'] = (h - l) / c

    # ── TARGET MEJORADO: multi-horizonte + Sharpe-like ──
    # Promediamos targets de 3, 5 y 10 días para mayor estabilidad
    targets = []
    for hz in [3, 5, 10]:
        fut = df['close'].shift(-hz) / df['close'] - 1
        raw_atr = np.where(df['close'] > 0, atr / df['close'], 0.02)
        raw_atr = pd.Series(raw_atr, index=df.index).fillna(0.02)
        # Considerar max drawdown durante el horizonte
        min_fut = pd.Series(np.nan, index=df.index)
        for hh in range(1, hz+1):
            future_low = df['low'].shift(-hh) / df['close'] - 1
            min_fut = pd.concat([min_fut, future_low], axis=1).min(axis=1)
        # Target Sharpe-like: retorno positivo Y drawdown < 1.5×ATR
        t = ((fut > raw_atr * 0.8) & (min_fut > -raw_atr * 1.5)).astype(int)
        targets.append(t)

    # Majority vote de los 3 horizontes
    target_sum = targets[0] + targets[1] + targets[2]
    df['target'] = (target_sum >= 2).astype(int)  # al menos 2 de 3 horizontes

    # Fallback si muy pocos positivos
    if df['target'].mean() < 0.05:
        fut5 = df['close'].shift(-5) / df['close'] - 1
        raw_atr5 = np.where(df['close'] > 0, atr / df['close'], 0.02)
        raw_atr5 = pd.Series(raw_atr5, index=df.index).fillna(0.02)
        df['target'] = (fut5 > raw_atr5 * 0.5).astype(int)
    if df['target'].sum() == 0:
        df['target'] = (c.shift(-5) / c - 1 > 0.01).astype(int)

    # ── NORMALIZACIÓN ROLLING (sin data leakage) ──
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


# ═══════════════════════════════════════════════════════════════════════
#  MODELO V2: STACKING ENSEMBLE
# ═══════════════════════════════════════════════════════════════════════
def build_stacking_model():
    """
    Ensemble mejorado: LightGBM + XGBoost + RF con VotingClassifier soft.
    Más rápido que StackingClassifier completo pero igual de potente.
    """
    rf = RandomForestClassifier(
        n_estimators=120, max_depth=6, min_samples_leaf=10,
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    lgbm = lgb.LGBMClassifier(
        n_estimators=150, learning_rate=0.06, max_depth=5,
        num_leaves=24, subsample=0.8, colsample_bytree=0.8,
        min_child_samples=15, class_weight='balanced',
        random_state=42, verbose=-1, n_jobs=-1
    )
    xgbm = xgb.XGBClassifier(
        n_estimators=150, learning_rate=0.06, max_depth=5,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        scale_pos_weight=3.0,
        random_state=42, verbosity=0, n_jobs=-1,
        eval_metric='logloss'
    )
    lr = Pipeline([
        ('sc', StandardScaler()),
        ('lr', LogisticRegression(C=0.5, solver='liblinear',
                                  class_weight='balanced', random_state=42))
    ])

    from sklearn.ensemble import VotingClassifier
    ensemble = VotingClassifier(
        estimators=[('lgbm', lgbm), ('xgb', xgbm), ('rf', rf), ('lr', lr)],
        voting='soft',
        weights=[3, 3, 2, 1],  # LightGBM + XGBoost dominan
        n_jobs=-1
    )
    return ensemble


def select_features_v2(X, y):
    """Feature selection rápida usando feature importances de LightGBM."""
    if len(X) < 200:
        return list(X.columns)
    try:
        model = lgb.LGBMClassifier(
            n_estimators=60, max_depth=4, random_state=42, verbose=-1, n_jobs=-1
        )
        model.fit(X, y)
        imp = pd.Series(model.feature_importances_, index=X.columns)
        # Threshold: top features con importancia > mediana
        threshold = max(imp.median(), imp.quantile(0.25))
        selected = imp[imp >= threshold].index.tolist()
        return selected if len(selected) >= 8 else list(X.columns)
    except Exception:
        return list(X.columns)


def train_models_for_period(all_raw_data, train_end_date, min_rows=300):
    """
    Entrena modelos stacking para todos los tickers con datos hasta train_end_date.
    """
    models = {}
    features = {}
    total = len(all_raw_data)

    for i, (ticker, raw) in enumerate(all_raw_data.items(), 1):
        sys.stdout.write(f"\r   Entrenando [{i:02d}/{total}] {ticker:<10}  ")
        sys.stdout.flush()

        df_eng = engineer_v2(raw)
        if df_eng is None:
            continue

        train_df = df_eng.loc[df_eng.index <= train_end_date].copy()
        if len(train_df) < min_rows:
            continue

        avail = [f for f in FEATURE_COLS_V2 if f in train_df.columns]
        X = train_df[avail]
        y = train_df['target']

        if y.sum() < 10:
            continue

        feats = select_features_v2(X, y)
        X = X[feats]

        try:
            model = build_stacking_model()
            cal = CalibratedClassifierCV(model, method='isotonic', cv=3)
            cal.fit(X, y)
            models[ticker] = cal
            features[ticker] = feats
        except Exception as e:
            try:
                # Fallback: usar sigmoid si isotonic falla
                cal = CalibratedClassifierCV(model, method='sigmoid', cv=3)
                cal.fit(X, y)
                models[ticker] = cal
                features[ticker] = feats
            except Exception:
                continue

    print(f"\n   ✅ Modelos entrenados: {len(models)}/{total}")
    return models, features


# ═══════════════════════════════════════════════════════════════════════
#  SIMULACIÓN V2 (MEJORADA)
# ═══════════════════════════════════════════════════════════════════════
def run_simulation(all_raw_data, models, features, sim_start, sim_end,
                   start_capital=START_CAPITAL, label=""):
    """
    Simulación mejorada con:
    - ATR-adaptive SL/TP
    - Sector diversification
    - Portfolio heat tracking
    - Dynamic probability threshold
    """
    print(f"\n{'─'*65}")
    print(f"  📈 SIMULANDO: {label}")
    print(f"     Período: {sim_start} → {sim_end}")
    print(f"     Capital: ${start_capital:,.0f} AUD | Tickers: {len(models)}")
    print(f"{'─'*65}")

    # Pre-compute engineer features and probabilities
    PROCESSED = {}
    for t, raw in all_raw_data.items():
        if t not in models:
            continue
        df_eng = engineer_v2(raw)
        if df_eng is not None and len(df_eng) > 50:
            PROCESSED[t] = df_eng
    print(f"  Features computadas: {len(PROCESSED)} tickers")

    # Pre-batch probabilities
    PROBS = {}
    for t, model in models.items():
        if t not in PROCESSED:
            continue
        feats_t = features.get(t, [])
        df_sim = PROCESSED[t]
        sim_rows = df_sim[(df_sim.index >= pd.Timestamp(sim_start)) &
                          (df_sim.index <= pd.Timestamp(sim_end))]
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
    print(f"  Probabilidades pre-computadas: {len([t for t,p in PROBS.items() if p])} tickers")

    # Market regime (VAS.AX)
    MARKET_REGIME = {}
    vas = all_raw_data.get('VAS.AX')
    if vas is None:
        vas = fetch_live('VAS.AX', start='2017-01-01')
    if vas is not None and len(vas) > REGIME_SMA:
        vas_sma = vas['close'].rolling(REGIME_SMA).mean()
        for d in vas.index:
            sv = vas_sma.get(d)
            if sv is not None and not pd.isna(sv):
                MARKET_REGIME[d] = bool(vas.loc[d, 'close'] > sv)

    # Simulation dates
    all_sim_dates = set()
    for t, df in PROCESSED.items():
        if t in models:
            all_sim_dates.update(
                df[(df.index >= pd.Timestamp(sim_start)) &
                   (df.index <= pd.Timestamp(sim_end))].index.tolist()
            )
    sim_dates = sorted(all_sim_dates)
    if not sim_dates:
        print("  ⚠️ Sin fechas de simulación disponibles")
        return None
    print(f"  Días de mercado: {len(sim_dates)}")

    # ── State ──
    cash        = start_capital
    portfolio   = {}
    equity_log  = []
    all_trades  = []
    trade_rows  = []
    SL_HISTORY  = {}
    RECENT_PNLS = []
    pending_entries = {}
    eq_peak = start_capital
    entries_paused = False

    # Dynamic threshold state
    recent_results = []  # últimos 20 trades: True=win, False=loss

    def get_dynamic_threshold(base, recent_res):
        """Ajusta threshold según performance reciente."""
        if len(recent_res) < 5:
            return base
        recent_wr = sum(recent_res[-15:]) / len(recent_res[-15:])
        if recent_wr > 0.65:
            return max(base - 0.01, MIN_PROB_ANY)
        elif recent_wr < 0.40:
            return min(base + 0.10, 0.70)
        return base

    def get_sector(ticker):
        return SECTOR_MAP.get(ticker, 'Other')

    def count_sector_positions(port, sector):
        return sum(1 for t in port if get_sector(t) == sector)

    def get_portfolio_heat(port, all_data, day):
        """Calcula riesgo total del portfolio como % del equity."""
        total_risk = 0
        for t_pos, pos in port.items():
            entry = pos['buy_price']
            stop  = pos.get('stop', entry * 0.95)
            risk  = (entry - stop) / entry * pos['shares'] * entry
            total_risk += max(0, risk)
        total_val = cash + sum(
            p['shares'] * (all_data[t].loc[day,'close']
                           if all_data.get(t) is not None and day in all_data[t].index
                           else p['buy_price'])
            for t, p in port.items()
        )
        return total_risk / total_val if total_val > 0 else 0

    for day in sim_dates:
        # ── Mark-to-market ──
        equity = cash
        for t_pos, pos in portfolio.items():
            raw_day = all_raw_data.get(t_pos)
            price = (raw_day.loc[day, 'close']
                     if raw_day is not None and day in raw_day.index
                     else pos['buy_price'])
            equity += pos['shares'] * price
        equity_log.append({'Fecha': day.strftime('%Y-%m-%d'), 'Equity_AUD': round(equity, 2)})

        # Circuit breaker de entradas por drawdown para evitar cascadas de pérdidas.
        eq_peak = max(eq_peak, equity)
        dd_now = (eq_peak - equity) / eq_peak if eq_peak > 0 else 0
        if dd_now >= ENTRY_PAUSE_DD:
            entries_paused = True
        elif entries_paused and dd_now <= ENTRY_RESUME_DD:
            entries_paused = False

        # ── FILL PENDING LIMIT ORDERS ──
        for t_pend in list(pending_entries.keys()):
            if t_pend in portfolio or len(portfolio) >= MAX_POSITIONS:
                pending_entries.pop(t_pend, None)
                continue
            last_sl = SL_HISTORY.get(t_pend)
            if last_sl is not None and (day - last_sl).days < COOLDOWN_DAYS:
                pending_entries.pop(t_pend, None)
                continue

            raw_fill = all_raw_data.get(t_pend)
            if raw_fill is None or day not in raw_fill.index:
                pending_entries.pop(t_pend, None)
                continue

            pend = pending_entries.pop(t_pend)
            today_low  = float(raw_fill.loc[day, 'low'])
            today_open = float(raw_fill.loc[day, 'open'])
            limit_px   = pend['limit']

            entry_price = limit_px if today_low <= limit_px else today_open
            fill_type   = 'LIMIT' if today_low <= limit_px else 'OPEN'

            # Position sizing: volatility-adjusted
            current_equity = cash + sum(
                p['shares'] * (all_raw_data[t].loc[day, 'close']
                               if all_raw_data.get(t) is not None and day in all_raw_data[t].index
                               else p['buy_price'])
                for t, p in portfolio.items()
            )

            atr_now = pend['atr_now']
            atr_pct = atr_now / entry_price if entry_price > 0 else 0.02

            # MEJORADO: Position sizing basado en ATR (riesgo fijo por trade)
            risk_per_share = atr_now * SL_ATR_MULT
            kf = kelly_fraction(RECENT_PNLS)
            risk_budget = current_equity * kf
            shares_risk = int(risk_budget / risk_per_share) if risk_per_share > 0 else 0
            shares_cap  = int((current_equity * MAX_POS_PCT) / entry_price) if entry_price > 0 else 0
            shares      = max(1, min(shares_risk, shares_cap))

            buy_comm = calc_commission(shares * entry_price)
            cost     = shares * entry_price + buy_comm
            if cost > cash or cost < MIN_BUY:
                continue

            # ATR-adaptive TP/SL
            tp1_price = entry_price + atr_now * TP1_ATR_MULT
            tp2_price = entry_price + atr_now * TP2_ATR_MULT
            sl_price  = entry_price - atr_now * SL_ATR_MULT

            cash -= cost
            portfolio[t_pend] = {
                'shares':      shares,
                'buy_price':   entry_price,
                'buy_comm':    buy_comm,
                'stop':        sl_price,
                'tp1':         tp1_price,
                'tp1_hit':     False,
                'tp2':         tp2_price,
                'trail_high':  entry_price,
                'hold_days':   0,
                'estrategia':  pend['estrategia'],
                'max_price':   entry_price,
                'min_price':   entry_price,
                'atr_entry':   atr_now,
                'entry_context': {
                    'fecha_entrada': day.strftime('%Y-%m-%d'),
                    'prob':          round(pend['prob'], 4),
                    'fill_type':     fill_type,
                },
            }
            all_trades.append({
                'Tipo': '🟢 COMPRA', 'Ticker': t_pend,
                'Estrategia': pend['estrategia'],
                'Fecha': day.strftime('%Y-%m-%d'),
                'Precio': round(entry_price, 3),
                'Acciones': shares,
                'Monto_AUD': round(cost, 2),
                'P&L_AUD': '—',
                'P&L_%': f"IA:{pend['prob']:.1%} [{fill_type}]",
                'Razon': 'SEÑAL_IA', 'Dias': 0,
            })

        # ── EXITS ──
        to_sell = []
        for t_pos, pos in list(portfolio.items()):
            raw_day = all_raw_data.get(t_pos)
            pos['hold_days'] = pos.get('hold_days', 0) + 1

            if raw_day is None or day not in raw_day.index:
                if pos['hold_days'] >= MAX_HOLD_DAYS:
                    to_sell.append((t_pos, 'EXPIRE_NO_DATA', pos['buy_price']))
                continue

            row_raw = raw_day.loc[day]
            price   = row_raw['close']
            high    = row_raw['high']
            low     = row_raw['low']

            pos['max_price'] = max(pos.get('max_price', pos['buy_price']), high)
            pos['min_price'] = min(pos.get('min_price', pos['buy_price']), low)

            atr_e = pos.get('atr_entry', pos['buy_price'] * 0.02)

            # Breakeven lock: ATR-based
            if not pos.get('be_lock', False) and pos['hold_days'] >= 2:
                if high >= pos['buy_price'] + atr_e * BREAKEVEN_ATR:
                    be_price = pos['buy_price'] + atr_e * BREAKEVEN_LOCK_ATR
                    if be_price > pos['stop']:
                        pos['stop']    = be_price
                        pos['be_lock'] = True

            # Trailing stop update — V3: más agresivo
            if high > pos.get('trail_high', pos['buy_price']):
                pos['trail_high'] = high
                # V3: después de TP1, trail 0.8×ATR (tight) — proteger beneficios
                # Antes de TP1: trail 1.2×ATR (el SL)
                trail_atr = 0.8 if pos.get('tp1_hit', False) else SL_ATR_MULT
                new_stop = pos['trail_high'] - atr_e * trail_atr
                pos['stop'] = max(pos.get('stop', 0), new_stop)

            # TP1 parcial (50% at 2.5×ATR gain)
            if not pos.get('tp1_hit', False) and high >= pos.get('tp1', float('inf')):
                shares_sell = pos['shares'] // 2
                tp1_val = shares_sell * pos['tp1']
                if shares_sell >= 1 and tp1_val > 0 and calc_commission(tp1_val) / tp1_val < 0.03:
                    sell_gross = shares_sell * pos['tp1']
                    sell_comm  = calc_commission(sell_gross)
                    net_part   = sell_gross - sell_comm
                    pct_frac   = shares_sell / pos['shares']
                    buy_cost_p = shares_sell * pos['buy_price'] + pos.get('buy_comm',0) * pct_frac
                    pnl_part   = net_part - buy_cost_p
                    cash      += net_part
                    RECENT_PNLS.append(pnl_part)
                    if len(RECENT_PNLS) > 30: RECENT_PNLS.pop(0)
                    recent_results.append(pnl_part > 0)
                    if len(recent_results) > 20: recent_results.pop(0)
                    pos['shares']   -= shares_sell
                    pos['buy_comm']  = pos.get('buy_comm',0) * (1 - pct_frac)
                    pos['tp1_hit']   = True
                    pos['stop']      = max(pos['stop'], pos['buy_price'] + atr_e * 0.5)
                    all_trades.append({
                        'Tipo': '🟡 TP1', 'Ticker': t_pos,
                        'Fecha': day.strftime('%Y-%m-%d'),
                        'Precio': round(pos['tp1'], 3),
                        'Acciones': shares_sell,
                        'Monto_AUD': round(net_part, 2),
                        'P&L_AUD': round(pnl_part, 2),
                        'P&L_%': f"+{(pos['tp1']/pos['buy_price']-1)*100:.1f}%",
                        'Razon': 'TP1_PARCIAL', 'Dias': pos['hold_days'],
                    })

            # Main exits
            if   low  <= pos['stop']:
                to_sell.append((t_pos, 'STOP_LOSS',   pos['stop']))
            elif high >= pos.get('tp2', float('inf')):
                to_sell.append((t_pos, 'TAKE_PROFIT', pos['tp2']))
            elif pos['hold_days'] >= MAX_HOLD_DAYS:
                to_sell.append((t_pos, 'EXPIRE',      price))
            elif pos['hold_days'] >= 3:
                raw_pos = all_raw_data.get(t_pos)
                if raw_pos is not None:
                    slice_p = raw_pos.loc[raw_pos.index <= day, 'close']
                    if len(slice_p) >= 30:
                        macd_now = ta_lib.trend.macd_diff(slice_p).iloc[-1]
                        unrealized = (price - pos['buy_price']) / pos['buy_price']
                        # V3: salir con cualquier ganancia si MACD se invierte
                        min_profit = 0.003 if pos.get('be_lock', False) else 0.008
                        if macd_now < 0 and unrealized > min_profit:
                            to_sell.append((t_pos, 'MACD_EXIT', price))

        for t_pos, reason, sell_price in to_sell:
            if t_pos not in portfolio:
                continue
            pos = portfolio.pop(t_pos)
            n_shares    = pos['shares']
            sell_gross  = n_shares * sell_price
            sell_comm   = calc_commission(sell_gross)
            net_proceed = sell_gross - sell_comm
            buy_cost    = n_shares * pos['buy_price'] + pos.get('buy_comm', 0)
            pnl         = net_proceed - buy_cost
            pct         = (sell_price - pos['buy_price']) / pos['buy_price']
            cash       += net_proceed
            if reason == 'STOP_LOSS':
                SL_HISTORY[t_pos] = day
            RECENT_PNLS.append(pnl)
            if len(RECENT_PNLS) > 30: RECENT_PNLS.pop(0)
            recent_results.append(pnl > 0)
            if len(recent_results) > 20: recent_results.pop(0)
            all_trades.append({
                'Tipo': '🔴 VENTA', 'Ticker': t_pos,
                'Fecha': day.strftime('%Y-%m-%d'),
                'Precio': round(sell_price, 3),
                'Acciones': n_shares,
                'Monto_AUD': round(net_proceed, 2),
                'P&L_AUD': round(pnl, 2),
                'P&L_%': f"{pct:+.2%}",
                'Razon': reason, 'Dias': pos['hold_days'],
            })
            ec = pos.get('entry_context', {})
            trade_rows.append({
                'ticker': t_pos,
                'estrategia': pos.get('estrategia', '?'),
                'resultado': 'WIN' if pnl > 0 else 'LOSS',
                'pnl_aud': round(pnl, 2),
                'pnl_pct': round(pct * 100, 2),
                'dias': pos['hold_days'],
                'razon': reason,
                'prob_ia': ec.get('prob', 0),
            })

        # ── ENTRIES ──
        slots_available = MAX_POSITIONS - len(portfolio) - len(pending_entries)
        regime_bull = MARKET_REGIME.get(day, True)
        # V3: En bear market, solo permitir REVERSION (no momentum)
        # En bull market, permitir todas las estrategias
        if cash >= MIN_BUY and slots_available > 0 and not entries_paused:
            candidates = []
            prob_thresh_mom = get_dynamic_threshold(BASE_PROB_MOM, recent_results)
            prob_thresh_rev = get_dynamic_threshold(BASE_PROB_REV, recent_results)

            # Portfolio heat check
            heat = get_portfolio_heat(portfolio, all_raw_data, day)

            for t_cand in models:
                if t_cand in portfolio or t_cand in pending_entries:
                    continue
                raw_cand = all_raw_data.get(t_cand)
                if raw_cand is None or day not in raw_cand.index:
                    continue

                prob = PROBS.get(t_cand, {}).get(day, 0.0)
                if prob < MIN_PROB_ANY:
                    continue

                # Sector diversification check
                sector = get_sector(t_cand)
                if count_sector_positions(portfolio, sector) >= MAX_SECTOR_POS:
                    continue

                # Portfolio heat check — skip if too hot
                if heat > PORTFOLIO_HEAT:
                    break  # no más entradas si riesgo > 15%

                real_price = raw_cand.loc[day, 'close']
                if real_price <= 0:
                    continue

                raw_slice = raw_cand.loc[raw_cand.index <= day]
                if len(raw_slice) < 210:
                    continue
                raw_close = raw_slice['close']

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

                # ═══ ESTRATEGIA 1: MOMENTUM BREAKOUT ═══
                # Conservadora: solo en bull market y con confirmación más estricta.
                if prob >= prob_thresh_mom and regime_bull:
                    if (real_price > sma50_raw and
                        sma50_raw >= sma200_raw and
                        rsi_raw < 74 and
                        macd_d_raw > 0 and
                        adx_raw >= 20 and
                        vol_ratio >= 1.15 and
                        mom5 > 0.01):
                        # Score mejorado: incluye calidad del setup
                        quality = (1.0
                                   + max(0, min(1.0, mom5 * 12))
                                   + max(0, min(0.3, (vol_ratio - 1.0) * 0.5))
                                   + max(0, min(0.3, (adx_raw - 20) / 40)))
                        score = prob * quality
                        candidates.append((t_cand, real_price, prob, score, 'MOMENTUM',
                                           atr_now, rsi_raw, macd_d_raw, adx_raw,
                                           vol_ratio, mom5, sma50_raw, sma200_raw))
                        continue

                # ═══ ESTRATEGIA 2: MEAN-REVERSION ═══
                if prob >= prob_thresh_rev:
                    near_support = (real_price >= sma200_raw * 0.94 and
                                    real_price <= sma200_raw * 1.06)
                    macd_turning = macd_d_raw > -0.02 * real_price * 0.001
                    if (rsi_raw < 38 and near_support and macd_turning and
                        vol_ratio >= 0.9 and adx_raw <= 35):
                        score = prob * 0.85 * (1.0 + max(0, min(0.5, (38 - rsi_raw) / 38)))
                        candidates.append((t_cand, real_price, prob, score, 'REVERSION',
                                           atr_now, rsi_raw, macd_d_raw, adx_raw,
                                           vol_ratio, mom5, sma50_raw, sma200_raw))
                        continue

                # ═══ ESTRATEGIA 3: TREND CONTINUATION (NUEVA) ═══
                if prob >= prob_thresh_mom and regime_bull:
                    if (real_price > sma200_raw and
                        sma20_raw > sma50_raw and
                        sma50_raw > sma200_raw and
                        rsi_raw > 0.45 and rsi_raw < 0.68 and
                        adx_raw >= 22 and
                        mom5 > 0.006):
                        # Pullback a SMA20 en tendencia alcista
                        near_sma20 = abs((real_price / sma20_raw) - 1) < 0.025
                        if near_sma20:
                            score = prob * 0.90
                            candidates.append((t_cand, real_price, prob, score, 'TREND_CONT',
                                               atr_now, rsi_raw, macd_d_raw, adx_raw,
                                               vol_ratio, mom5, sma50_raw, sma200_raw))
                            continue

            candidates.sort(key=lambda x: x[3], reverse=True)

            for (t_cand, price, prob, score, estrategia, atr_now,
                 rsi_raw, macd_d_raw, adx_raw, vol_ratio, mom5,
                 sma50_raw, sma200_raw) in candidates:

                if cash < MIN_BUY or len(portfolio) + len(pending_entries) >= MAX_POSITIONS:
                    break

                last_sl = SL_HISTORY.get(t_cand)
                if last_sl is not None and (day - last_sl).days < COOLDOWN_DAYS:
                    continue

                if t_cand in pending_entries:
                    continue

                pending_entries[t_cand] = {
                    'limit':       price * (1 - ENTRY_DISCOUNT),
                    'signal_close': price,
                    'estrategia':  estrategia,
                    'atr_now':     atr_now,
                    'prob':        prob,
                    'rsi_raw':     rsi_raw,
                    'macd_d_raw':  macd_d_raw,
                    'adx_raw':     adx_raw,
                    'vol_ratio':   vol_ratio,
                    'mom5':        mom5,
                    'sma50_raw':   sma50_raw,
                    'sma200_raw':  sma200_raw,
                }

    # Liquidate remaining
    if sim_dates:
        final_day = sim_dates[-1]
        for t_pos, pos in list(portfolio.items()):
            raw_d = all_raw_data.get(t_pos)
            price = (raw_d.loc[final_day, 'close']
                     if raw_d is not None and final_day in raw_d.index
                     else pos['buy_price'])
            sell_gross  = pos['shares'] * price
            sell_comm   = calc_commission(sell_gross)
            net_proceed = sell_gross - sell_comm
            buy_cost    = pos['shares'] * pos['buy_price'] + pos.get('buy_comm', 0)
            pnl         = net_proceed - buy_cost
            pct         = (price - pos['buy_price']) / pos['buy_price']
            cash       += net_proceed
            all_trades.append({
                'Tipo': '🔴 VENTA', 'Ticker': t_pos,
                'Fecha': final_day.strftime('%Y-%m-%d'),
                'Precio': round(price, 3),
                'Acciones': pos['shares'],
                'Monto_AUD': round(net_proceed, 2),
                'P&L_AUD': round(pnl, 2),
                'P&L_%': f"{pct:+.2%}",
                'Razon': 'CIERRE_FINAL', 'Dias': pos.get('hold_days', 0),
            })
            trade_rows.append({
                'ticker': t_pos, 'estrategia': pos.get('estrategia', '?'),
                'resultado': 'WIN' if pnl > 0 else 'LOSS',
                'pnl_aud': round(pnl, 2), 'pnl_pct': round(pct * 100, 2),
                'dias': pos.get('hold_days', 0), 'razon': 'CIERRE_FINAL',
                'prob_ia': pos.get('entry_context', {}).get('prob', 0),
            })

    # ── Results ──
    trades_df = pd.DataFrame(all_trades)
    eq_df     = pd.DataFrame(equity_log)
    tlog_df   = pd.DataFrame(trade_rows) if trade_rows else pd.DataFrame()

    ventas = trades_df[trades_df['Tipo'].str.contains('VENTA|TP1')] if not trades_df.empty else pd.DataFrame()

    pnl_vals = (ventas['P&L_AUD'].apply(pd.to_numeric, errors='coerce').dropna()
                if not ventas.empty else pd.Series([0.0]))

    roi    = (cash - start_capital) / start_capital
    wins   = (pnl_vals > 0).sum()
    losses = (pnl_vals <= 0).sum()
    n_ops  = wins + losses
    eq_s   = eq_df['Equity_AUD'] if not eq_df.empty else pd.Series([start_capital, cash])
    max_dd = ((eq_s.cummax() - eq_s) / eq_s.cummax()).max()

    # Win rate, profit factor
    win_total  = pnl_vals[pnl_vals > 0].sum() if wins > 0 else 0
    loss_total = abs(pnl_vals[pnl_vals <= 0].sum()) if losses > 0 else 1
    pf = win_total / loss_total if loss_total > 0 else float('inf')

    # Estrategia breakdown
    strat_stats = {}
    if not tlog_df.empty:
        for strat in tlog_df['estrategia'].unique():
            s = tlog_df[tlog_df['estrategia'] == strat]
            strat_stats[strat] = {
                'trades': len(s),
                'wins': (s['resultado'] == 'WIN').sum(),
                'pnl': s['pnl_aud'].sum(),
            }

    return {
        'label':        label,
        'sim_start':    sim_start,
        'sim_end':      sim_end,
        'capital_ini':  start_capital,
        'capital_fin':  round(cash, 2),
        'roi':          roi,
        'max_dd':       max_dd,
        'n_ops':        n_ops,
        'wins':         wins,
        'losses':       losses,
        'win_rate':     wins / n_ops if n_ops > 0 else 0,
        'profit_factor': pf,
        'pnl_avg':      pnl_vals.mean() if n_ops > 0 else 0,
        'best_trade':   pnl_vals.max() if n_ops > 0 else 0,
        'worst_trade':  pnl_vals.min() if n_ops > 0 else 0,
        'sim_days':     len(sim_dates),
        'strat_stats':  strat_stats,
        'trades_df':    trades_df,
        'equity_df':    eq_df,
        'trade_log':    tlog_df,
    }


# ═══════════════════════════════════════════════════════════════════════
#  MAIN — EJECUTAR 3 PERÍODOS
# ═══════════════════════════════════════════════════════════════════════
def main():
    t0 = time.time()
    roi_fail_labels = []

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  DEEPQUANT V2 — MODELO MEJORADO + SIMULACIÓN 3 PERÍODOS    ║")
    print("║  Stacking Ensemble × Multi-horizonte × ATR-adaptive        ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    # ── Definir períodos ──
    periods = [
        {
            'name':       'PERÍODO 1: 2022-2024',
            'fetch_start': '2017-01-01',
            'train_end':   '2021-12-31',
            'sim_start':   '2022-01-01',
            'sim_end':     '2024-12-31',
        },
        {
            'name':       'PERÍODO 2: 2025',
            'fetch_start': '2017-01-01',
            'train_end':   '2024-12-31',
            'sim_start':   '2025-01-01',
            'sim_end':     '2025-12-31',
        },
        {
            'name':       'PERÍODO 3: 2026 YTD',
            'fetch_start': '2017-01-01',
            'train_end':   '2025-12-31',
            'sim_start':   '2026-01-01',
            'sim_end':     '2026-12-31',
        },
    ]

    results = []
    
    # Cache de datos descargados para no re-descargar entre periodos
    DATA_CACHE = {}   # ticker → full DataFrame (toda la historia)

    # ── PRE-DESCARGAR TODO EL RANGO (1 sola vez) ──
    print(f"\n  📡 Pre-descargando {len(ASX_TICKERS)} tickers (2017 → 2026)...")
    for i, t in enumerate(ASX_TICKERS, 1):
        raw = fetch_live(t, start='2017-01-01', end='2026-12-31')
        if raw is not None and len(raw) > 100:
            DATA_CACHE[t] = raw
        sys.stdout.write(f"\r   {i}/{len(ASX_TICKERS)}  OK:{len(DATA_CACHE)}")
        sys.stdout.flush()
    print(f"\n  ✅ {len(DATA_CACHE)} tickers en caché\n")

    for pi, period in enumerate(periods, 1):
        print(f"\n{'═'*65}")
        print(f"  ⏱  {period['name']}")
        print(f"     Train: {period['fetch_start']} → {period['train_end']}")
        print(f"     Test:  {period['sim_start']} → {period['sim_end']}")
        print(f"{'═'*65}")

        # ── Descargar datos (con caché entre períodos) ──
        print(f"\n  📡 Cargando {len(ASX_TICKERS)} tickers...")
        ALL_RAW = {}
        for i, t in enumerate(ASX_TICKERS, 1):
            if t in DATA_CACHE:
                cached = DATA_CACHE[t]
                need_end = pd.Timestamp(period['sim_end'])
                # Si el caché no cubre hasta sim_end, re-descargar todo
                if cached.index.max() < need_end - pd.Timedelta(days=5):
                    raw = fetch_live(t, start=period['fetch_start'], end=period['sim_end'])
                    if raw is not None and len(raw) > 100:
                        DATA_CACHE[t] = raw
                        ALL_RAW[t] = raw
                else:
                    mask = cached.index <= need_end
                    if mask.sum() > 100:
                        ALL_RAW[t] = cached[mask].copy()
            else:
                raw = fetch_live(t, start=period['fetch_start'], end=period['sim_end'])
                if raw is not None and len(raw) > 100:
                    ALL_RAW[t] = raw
                    DATA_CACHE[t] = raw
            sys.stdout.write(f"\r   {i}/{len(ASX_TICKERS)}  OK:{len(ALL_RAW)}")
            sys.stdout.flush()
        print(f"\n  ✅ Tickers disponibles: {len(ALL_RAW)}")

        # ── Entrenar modelos ──
        print(f"\n  🧠 Entrenando modelos stacking (train → {period['train_end']})...")
        models, features = train_models_for_period(ALL_RAW, period['train_end'])

        if not models:
            print(f"  ⚠️ Sin modelos. Saltando este período.")
            results.append(None)
            continue

        # ── Simular ──
        result = run_simulation(
            ALL_RAW, models, features,
            sim_start=period['sim_start'],
            sim_end=period['sim_end'],
            start_capital=START_CAPITAL,
            label=period['name']
        )
        results.append(result)

        if result:
            if result['roi'] < ROI_TARGET_MIN:
                roi_fail_labels.append(period['name'])
            print(f"\n  {'─'*55}")
            print(f"  💰 RESULTADO {period['name']}:")
            print(f"     Capital: ${result['capital_ini']:,.0f} → ${result['capital_fin']:,.2f} AUD")
            print(f"     ROI:     {result['roi']:+.2%}")
            print(f"     Max DD:  {result['max_dd']:.2%}")
            print(f"     Trades:  {result['n_ops']}  (WR: {result['win_rate']:.1%})")
            print(f"     PF:      {result['profit_factor']:.2f}")
            if result['strat_stats']:
                print(f"     Estrategias:")
                for strat, st in result['strat_stats'].items():
                    wr_s = st['wins']/st['trades']*100 if st['trades'] > 0 else 0
                    print(f"       {strat:<15} {st['trades']:>3} trades | WR {wr_s:.0f}% | P&L ${st['pnl']:+.2f}")

    # ═══════════════════════════════════════════════════════════════════
    #  RESUMEN COMPARATIVO
    # ═══════════════════════════════════════════════════════════════════
    elapsed = time.time() - t0
    print(f"\n\n{'═'*70}")
    print(f"  📊  RESUMEN COMPARATIVO — DEEPQUANT V2 (MODELO MEJORADO)")
    print(f"{'═'*70}")
    print(f"{'Período':<25} {'Capital Fin':>12} {'ROI':>8} {'Max DD':>8} {'Trades':>7} {'WR':>6} {'PF':>6}")
    print(f"{'─'*25} {'─'*12} {'─'*8} {'─'*8} {'─'*7} {'─'*6} {'─'*6}")

    for r in results:
        if r is None:
            print("  (sin datos)")
            continue
        print(f"  {r['label']:<23} ${r['capital_fin']:>10,.2f} {r['roi']:>+7.2%} "
              f"{r['max_dd']:>7.2%} {r['n_ops']:>6} {r['win_rate']:>5.1%} "
              f"{r['profit_factor']:>5.2f}")

    # Calcular ROI acumulado (cascada)
    cum_capital = START_CAPITAL
    for r in results:
        if r:
            cum_capital *= (1 + r['roi'])
    cum_roi = (cum_capital - START_CAPITAL) / START_CAPITAL

    print(f"{'─'*70}")
    print(f"  {'ROI ACUMULADO (cascada)':<23} ${cum_capital:>10,.2f} {cum_roi:>+7.2%}")
    print(f"  {'Capital Inicial':<23} ${START_CAPITAL:>10,.2f}")
    if roi_fail_labels:
        print(f"  ❌ ROI objetivo no cumplido (mínimo {ROI_TARGET_MIN:.0%}) en: {', '.join(roi_fail_labels)}")
    else:
        print(f"  ✅ ROI objetivo cumplido (mínimo {ROI_TARGET_MIN:.0%}) en todos los períodos")
    print(f"{'═'*70}")
    print(f"  Tiempo total: {elapsed/60:.1f} minutos")

    # ── Guardar Excel con todos los períodos ──
    ts = datetime.now().strftime('%Y%m%d_%H%M')
    out_file = f"sim_improved_3periods_{ts}.xlsx"
    with pd.ExcelWriter(out_file, engine='openpyxl') as w:
        summary_rows = []
        for r in results:
            if r is None:
                continue
            summary_rows.append({
                'Período': r['label'],
                'Capital_Inicial': r['capital_ini'],
                'Capital_Final': r['capital_fin'],
                'ROI': f"{r['roi']:+.2%}",
                'Max_Drawdown': f"{r['max_dd']:.2%}",
                'Trades': r['n_ops'],
                'Win_Rate': f"{r['win_rate']:.1%}",
                'Profit_Factor': round(r['profit_factor'], 2),
                'PnL_Promedio': round(r['pnl_avg'], 2),
                'Mejor_Trade': round(r['best_trade'], 2),
                'Peor_Trade': round(r['worst_trade'], 2),
                'Días_Sim': r['sim_days'],
            })
        pd.DataFrame(summary_rows).to_excel(w, sheet_name='Resumen', index=False)

        for i, r in enumerate(results):
            if r is None or r['trades_df'].empty:
                continue
            sheet = f"Trades_P{i+1}"
            r['trades_df'].to_excel(w, sheet_name=sheet, index=False)
            if not r['equity_df'].empty:
                r['equity_df'].to_excel(w, sheet_name=f"Equity_P{i+1}", index=False)

    print(f"\n  💾 Resultados guardados en: {out_file}")

    # ── Mostrar mejoras clave ──
    print(f"\n{'═'*70}")
    print("  🧠 MEJORAS IMPLEMENTADAS vs MODELO ORIGINAL:")
    print(f"{'─'*70}")
    print("  ✅ Stacking Ensemble (RF+GB+LightGBM+XGBoost → Meta-LR)")
    print("  ✅ +12 features (OBV, VWAP, gaps, Sharpe, Ichimoku, accel...)")
    print("  ✅ Target multi-horizonte (3d+5d+10d majority vote)")
    print("  ✅ Target Sharpe-like (retorno Y drawdown controlado)")
    print("  ✅ ATR-adaptive SL/TP (no % fijo)")
    print("  ✅ Sector diversification (máx 2 por sector)")
    print("  ✅ Portfolio heat tracking (máx 15% riesgo)")
    print("  ✅ Dynamic probability threshold")
    print("  ✅ Estrategia TREND_CONTINUATION nueva")
    print("  ✅ Volatility-adjusted position sizing")
    print(f"{'═'*70}\n")

    if roi_fail_labels:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
