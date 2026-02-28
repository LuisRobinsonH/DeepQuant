# train_model_2021_2025.py
# ══════════════════════════════════════════════════════════════════
#  ENTRENAMIENTO DEL MODELO — FASE ÚNICA
#  Datos: 2021-01-01 → 2025-12-31  (histórico completo pre-2026)
#  Output: models_cache.joblib + features_cache.joblib
#
#  Ejecutar UNA sola vez. predict_2026.py carga los modelos guardados.
# ══════════════════════════════════════════════════════════════════
import os, warnings, sys
import numpy as np
import pandas as pd
import yfinance as yf
import joblib
import ta

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings('ignore')

# ─── CONFIGURACIÓN ────────────────────────────────────────────────
TRAIN_START = "2021-01-01"
TRAIN_END   = "2025-12-31"
MIN_ROWS    = 300           # mínimo de filas para entrenar

MODEL_CACHE_FILE   = "models_cache.joblib"
FEATURE_CACHE_FILE = "features_cache.joblib"

# ASX200 completo — 101 tickers (mismo universo que sim_2k.py)
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
    'atr_pct','vol_regime',
    'momentum_3','momentum_5','momentum_10','momentum_20',
    'roc_5','roc_10',
    'rsi','macd_diff','adx',
    'bb_width','bb_upper_dist','bb_lower_dist',
    'stoch_k','stoch_d','williams_r','cci',
    'vol_rel_20','vol_rel_50',
    'close_to_max5','close_to_min5','close_to_max20','close_to_min20',
]

# ─── DOWNLOAD ─────────────────────────────────────────────────────
def download(ticker, start, end):
    try:
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
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
    except Exception as e:
        print(f"  ⚠️  Error descargando {ticker}: {e}")
        return None

# ─── FEATURE ENGINEERING (mismo que TitanBrain) ───────────────────
def engineer(raw_df, horizon=5):
    df = raw_df.copy()
    if len(df) < 120:
        return None
    c = df['close']

    df['sma_20']  = ta.trend.sma_indicator(c, 20)
    df['sma_50']  = ta.trend.sma_indicator(c, 50)
    df['sma_200'] = ta.trend.sma_indicator(c, 200)
    df['dist_sma20']  = (c / df['sma_20'])  - 1
    df['dist_sma50']  = (c / df['sma_50'])  - 1
    df['dist_sma200'] = (c / df['sma_200']) - 1
    df['ma_cross_20_50']  = (df['sma_20']  > df['sma_50']).astype(int)
    df['ma_cross_50_200'] = (df['sma_50']  > df['sma_200']).astype(int)

    atr = ta.volatility.average_true_range(df['high'], df['low'], c, 14)
    df['atr']     = atr
    df['atr_pct'] = np.where(c > 0, atr / c, 0)
    atr_ma = atr.rolling(50).mean()
    df['vol_regime'] = np.where(atr_ma > 0, atr / atr_ma, 1.0)

    for p in [3, 5, 10, 20]:
        df[f'momentum_{p}'] = c.pct_change(p)
    df['roc_5']     = ta.momentum.roc(c, 5)
    df['roc_10']    = ta.momentum.roc(c, 10)
    df['rsi']       = ta.momentum.rsi(c, 14) / 100.0
    df['macd_diff'] = ta.trend.macd_diff(c)
    df['adx']       = ta.trend.adx(df['high'], df['low'], c, 14)

    bb_h = ta.volatility.bollinger_hband(c, 20)
    bb_l = ta.volatility.bollinger_lband(c, 20)
    df['bb_width']      = (bb_h - bb_l) / c
    df['bb_upper_dist'] = (bb_h - c) / c
    df['bb_lower_dist'] = (c - bb_l) / c

    df['stoch_k']    = ta.momentum.stoch(df['high'], df['low'], c, 14, 3) / 100.0
    df['stoch_d']    = ta.momentum.stoch_signal(df['high'], df['low'], c, 14, 3) / 100.0
    df['williams_r'] = ta.momentum.williams_r(df['high'], df['low'], c, 14)
    df['cci']        = ta.trend.cci(df['high'], df['low'], c, 20)

    vol  = df['volume']
    vm20 = vol.rolling(20).mean()
    vm50 = vol.rolling(50).mean()
    df['vol_rel_20'] = np.where(vm20 > 0, vol / vm20, 1.0)
    df['vol_rel_50'] = np.where(vm50 > 0, vol / vm50, 1.0)

    for w, s in [(5,'5'), (20,'20')]:
        df[f'max_{s}'] = c.rolling(w).max()
        df[f'min_{s}'] = c.rolling(w).min()
    df['close_to_max5']  = (c / df['max_5'])  - 1
    df['close_to_min5']  = (c / df['min_5'])  - 1
    df['close_to_max20'] = (c / df['max_20']) - 1
    df['close_to_min20'] = (c / df['min_20']) - 1

    # TARGET ROBUSTO: win = retorno futuro > 1x ATR (ganancia real, no micro-ruido)
    fut     = df['close'].shift(-horizon) / df['close'] - 1
    raw_atr = np.where(df['close'] > 0, atr / df['close'], 0.02)
    raw_atr = pd.Series(raw_atr, index=df.index).fillna(0.02)
    df['target'] = (fut > raw_atr * 1.0).astype(int)
    if df['target'].mean() < 0.05:
        df['target'] = (fut > raw_atr * 0.5).astype(int)
    if df['target'].sum() == 0:
        df['target'] = (fut > 0.01).astype(int)

    # NORMALIZACIÓN ROLLING (evita data leakage)
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

# ─── FEATURE SELECTION (estabilidad temporal) ─────────────────────
def select_features(X, y):
    if len(X) < 200:
        return list(X.columns)
    rf = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42, n_jobs=-1)
    splits = TimeSeriesSplit(n_splits=3)
    scores = {c: [] for c in X.columns}
    for tr_idx, _ in splits.split(X):
        if len(tr_idx) < 100:
            continue
        Xf, yf = X.iloc[tr_idx], y.iloc[tr_idx]
        if yf.sum() < 5:
            continue
        try:
            rf.fit(Xf, yf)
            pi = permutation_importance(rf, Xf, yf, n_repeats=5, random_state=42)
            for i, name in enumerate(X.columns):
                scores[name].append(pi.importances_mean[i])
        except Exception:
            pass
    stable = [f for f, s in scores.items()
              if len(s) > 0 and np.median(s) > 0.003 and np.std(s) < 0.30]
    return stable if len(stable) >= 5 else list(X.columns)

# ─── ENTRENAR UN MODELO ───────────────────────────────────────────
def train_ticker(ticker, df_eng):
    """
    Entrena el modelo ensemble+calibración con datos 2021-2025.
    Retorna (model, feature_list) o None si no hay suficientes datos.
    """
    # Solo datos de entrenamiento (excluir los últimos 5 días de 2025 por leakage)
    train_df = df_eng.loc[df_eng.index <= TRAIN_END].copy()
    if len(train_df) < MIN_ROWS:
        return None, None

    avail_feats = [f for f in FEATURE_COLS if f in train_df.columns]
    X = train_df[avail_feats]
    y = train_df['target']

    if y.sum() < 10:
        return None, None

    # Selección de features estables
    feats = select_features(X, y)
    X = X[feats]

    # Ensemble con class_weight='balanced' para manejar clases desbalanceadas
    lr = Pipeline([
        ('sc', StandardScaler()),
        ('lr', LogisticRegression(C=0.3, solver='liblinear',
                                  class_weight='balanced', random_state=42))
    ])
    rf = RandomForestClassifier(n_estimators=150, max_depth=6,
                                 min_samples_leaf=8,
                                 class_weight='balanced',
                                 random_state=42, n_jobs=-1)
    gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05,
                                     max_depth=3, subsample=0.8, random_state=42)
    ens = VotingClassifier([('rf', rf), ('lr', lr), ('gb', gb)],
                           voting='soft', weights=[2, 1, 2])

    try:
        cal = CalibratedClassifierCV(ens, method='isotonic', cv=3)
        cal.fit(X, y)
    except Exception:
        cal = CalibratedClassifierCV(ens, method='sigmoid', cv=3)
        cal.fit(X, y)

    return cal, feats

# ─── MAIN ─────────────────────────────────────────────────────────
def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   TITAN AI — ENTRENAMIENTO 2021-2025                    ║")
    print("║   Todos los modelos se entrenan UNA VEZ y se guardan.   ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"\n  Tickers  : {len(ASX_TICKERS)}")
    print(f"  Período  : {TRAIN_START}  →  {TRAIN_END}")
    print(f"  Guardando: {MODEL_CACHE_FILE}  +  {FEATURE_CACHE_FILE}\n")

    trained_models  = {}
    feature_cache   = {}
    failed          = []

    total = len(ASX_TICKERS)
    for i, ticker in enumerate(ASX_TICKERS, 1):
        print(f"[{i:02d}/{total}] {ticker:<10}", end=" ... ")
        sys.stdout.flush()

        raw = download(ticker, TRAIN_START, TRAIN_END)
        if raw is None or len(raw) < 120:
            print("❌  Sin datos")
            failed.append(ticker)
            continue

        df_eng = engineer(raw)
        if df_eng is None or len(df_eng) < MIN_ROWS:
            print(f"❌  Features insuficientes ({len(df_eng) if df_eng is not None else 0} filas)")
            failed.append(ticker)
            continue

        model, feats = train_ticker(ticker, df_eng)
        if model is None:
            print("❌  Entrenamiento fallido (pocos positivos)")
            failed.append(ticker)
            continue

        trained_models[ticker] = model
        feature_cache[ticker]  = feats
        pos_rate = df_eng.loc[df_eng.index <= TRAIN_END, 'target'].mean()
        print(f"✅  {len(df_eng)} filas | {len(feats)} features | target={pos_rate:.1%}")

    # Guardar caché
    joblib.dump(trained_models,  MODEL_CACHE_FILE)
    joblib.dump(feature_cache,   FEATURE_CACHE_FILE)

    print("\n" + "═"*60)
    print(f"✅  Modelos guardados : {len(trained_models)}")
    if failed:
        print(f"⚠️   Fallidos         : {len(failed)}  →  {failed}")
    print(f"\n  {MODEL_CACHE_FILE}   ({os.path.getsize(MODEL_CACHE_FILE)/1024:.0f} KB)")
    print(f"  {FEATURE_CACHE_FILE}  ({os.path.getsize(FEATURE_CACHE_FILE)/1024:.0f} KB)")
    print("\n  Listo. Ahora ejecuta:  python predict_2026.py\n")

if __name__ == "__main__":
    main()
