# predict_2026.py
# ══════════════════════════════════════════════════════════════════
#  PREDICCIONES 2026 — SOLO INFERENCIA, SIN SIMULACIÓN
#
#  Requiere: models_cache.joblib + features_cache.joblib
#            (generados por train_model_2021_2025.py)
#
#  Lógica:
#   1. Carga modelos ya entrenados en 2021-2025
#   2. Descarga datos 2025-06-01 → hoy (lookback para indicadores)
#   3. Para cada día de trading en 2026, predice probabilidad
#   4. Muestra tabla de señales (sin comprar, sin vender)
#   5. Valida: ¿fue correcto el BUY signal? (precio subió en 5 días)
# ══════════════════════════════════════════════════════════════════
import os, sys, warnings
import numpy as np
import pandas as pd
import yfinance as yf
import joblib
import ta
from datetime import datetime

warnings.filterwarnings('ignore')

# ─── CONFIGURACIÓN ────────────────────────────────────────────────
MODEL_CACHE_FILE   = "models_cache.joblib"
FEATURE_CACHE_FILE = "features_cache.joblib"

LOOKBACK_START  = "2025-04-01"   # necesitamos ~200 días para SMA200 + indicadores
SIM_START       = pd.Timestamp("2026-01-01")
SIM_END         = pd.Timestamp(datetime.now().strftime('%Y-%m-%d'))
PROB_THRESHOLD  = 0.50           # umbral para emitir señal BUY

ASX_TICKERS = [
    'BHP.AX','CBA.AX','CSL.AX','WES.AX','NAB.AX','WBC.AX','ANZ.AX',
    'MQG.AX','FMG.AX','TLS.AX','RIO.AX','GMG.AX','STO.AX','WDS.AX',
    'QBE.AX','ALL.AX','SCG.AX','ORG.AX','NST.AX','SUN.AX','MIN.AX',
    'PLS.AX','IGO.AX','TCL.AX','S32.AX','REA.AX','QAN.AX','RMD.AX',
    'AMC.AX','BSL.AX','CPU.AX','ASX.AX','SHL.AX','JHX.AX','WOW.AX',
    'COH.AX','XRO.AX','TWE.AX','CAR.AX','SEK.AX',
]

# ─── DESCARGA ─────────────────────────────────────────────────────
def download(ticker, start):
    try:
        df = yf.download(ticker, start=start, progress=False, auto_adjust=True)
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

# ─── FEATURE ENGINEERING (idéntico al de entrenamiento) ───────────
def engineer(raw_df):
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

    # ── IMPORTANTE: usar la MISMA normalización que en entrenamiento.
    # El modelo fue entrenado con datos z-scored de 2021-2025.
    # En inferencia, normalizamos con la media/std del lookback reciente
    # (solo columnas numéricas excepto 'close' raw que no existe ya).
    for col in df.select_dtypes(include=[np.number]).columns:
        std = df[col].std()
        if std > 0:
            df[col] = (df[col] - df[col].mean()) / std

    return df.replace([np.inf, -np.inf], np.nan).dropna()

# ─── PREDICCIÓN PARA UNA FILA ─────────────────────────────────────
def predict_row(model, feats, row):
    try:
        avail = [f for f in feats if f in row.index]
        if len(avail) < 3:
            return None
        X = row[avail].values.reshape(1, -1)
        prob = model.predict_proba(X)[0][1]
        return prob
    except Exception:
        return None

# ─── MAIN ─────────────────────────────────────────────────────────
def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   TITAN AI — PREDICCIONES 2026  (solo inferencia)       ║")
    print("║   Sin simulación. Sin compras virtuales.                 ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # ── Verificar caché ────────────────────────────────────────────
    if not os.path.exists(MODEL_CACHE_FILE) or not os.path.exists(FEATURE_CACHE_FILE):
        print("\n❌  No se encontraron modelos entrenados.")
        print("    Primero ejecuta:  python train_model_2021_2025.py\n")
        sys.exit(1)

    print(f"\n🧠 Cargando modelos desde {MODEL_CACHE_FILE} ...")
    trained_models = joblib.load(MODEL_CACHE_FILE)
    feature_cache  = joblib.load(FEATURE_CACHE_FILE)
    print(f"   Modelos cargados: {len(trained_models)} tickers")

    days_range = pd.date_range(SIM_START, SIM_END, freq='B')  # solo días hábiles
    print(f"\n📅 Período de predicción: {SIM_START.date()}  →  {SIM_END.date()}")
    print(f"   Días hábiles estimados: {len(days_range)}")
    print(f"   Umbral BUY: prob >= {PROB_THRESHOLD:.0%}\n")

    tickers_to_predict = [t for t in ASX_TICKERS if t in trained_models]
    print(f"📡 Descargando datos (lookback desde {LOOKBACK_START}) para {len(tickers_to_predict)} tickers...\n")

    all_signals   = []   # señales BUY emitidas
    all_daily     = []   # todas las predicciones (para análisis)

    for i, ticker in enumerate(tickers_to_predict, 1):
        print(f"  [{i:02d}/{len(tickers_to_predict)}] {ticker:<10}", end=" ... ")
        sys.stdout.flush()

        raw = download(ticker, LOOKBACK_START)
        if raw is None or len(raw) < 80:
            print("❌ sin datos")
            continue

        df_eng = engineer(raw)
        if df_eng is None or len(df_eng) < 20:
            print("❌ features insuficientes")
            continue

        model = trained_models[ticker]
        feats = feature_cache[ticker]

        # Solo fechas en el período de validación 2026
        df_2026 = df_eng[df_eng.index >= SIM_START]
        if df_2026.empty:
            print("⚠️  sin datos 2026")
            continue

        # También necesitamos el precio real para validar si la predicción fue correcta
        # "correcto" = precio sube en los próximos 5 días hábiles
        raw_2026 = raw[raw.index >= SIM_START]

        signals_ticker = 0
        for date, row in df_2026.iterrows():
            prob = predict_row(model, feats, row)
            if prob is None:
                continue

            price = raw.loc[date, 'close'] if date in raw.index else np.nan

            # Validación: ¿subió en 5 días?
            future_idx = raw.index.searchsorted(date)
            outcome = None
            outcome_ret = np.nan
            if future_idx + 5 < len(raw):
                future_price = raw.iloc[future_idx + 5]['close']
                outcome_ret  = (future_price / price) - 1 if not np.isnan(price) else np.nan
                if not np.isnan(outcome_ret):
                    outcome = "✅ SUBIÓ" if outcome_ret > 0 else "❌ BAJÓ"

            signal = "BUY" if prob >= PROB_THRESHOLD else "hold"

            all_daily.append({
                'Fecha':   date.strftime('%Y-%m-%d'),
                'Ticker':  ticker,
                'Precio':  round(price, 3) if not np.isnan(price) else None,
                'Prob':    round(prob, 4),
                'Señal':   signal,
                'Ret5d':   round(outcome_ret * 100, 2) if not np.isnan(outcome_ret) else None,
                'Correcto': outcome,
            })

            if signal == "BUY":
                signals_ticker += 1
                all_signals.append({
                    'Fecha':    date.strftime('%Y-%m-%d'),
                    'Ticker':   ticker,
                    'Precio':   round(price, 3) if not np.isnan(price) else None,
                    'Prob':     round(prob, 4),
                    'Ret5d_%':  round(outcome_ret * 100, 2) if not np.isnan(outcome_ret) else "pendiente",
                    'Correcto': outcome if outcome else "pendiente",
                })

        print(f"✅ {len(df_2026)} días | {signals_ticker} señales BUY")

    # ─── RESULTADOS ───────────────────────────────────────────────
    print("\n" + "═"*70)
    print("  📊  SEÑALES BUY EMITIDAS EN 2026")
    print("═"*70)

    if not all_signals:
        print("  (ninguna señal superó el umbral)")
    else:
        sig_df = pd.DataFrame(all_signals).sort_values(['Fecha','Prob'], ascending=[True, False])
        # Mostrar en consola
        print(f"\n  Total señales: {len(sig_df)}")
        print(f"  Tickers únicos: {sig_df['Ticker'].nunique()}\n")
        print(f"  {'Fecha':<12} {'Ticker':<10} {'Precio':>8} {'Prob':>6}  {'Ret5d%':>8}  Correcto")
        print("  " + "-"*65)
        for _, r in sig_df.iterrows():
            ret_str = f"{r['Ret5d_%']:>7.2f}%" if isinstance(r['Ret5d_%'], float) else f"{'pendiente':>8}"
            cor_str = r['Correcto'] if r['Correcto'] else "pendiente"
            print(f"  {r['Fecha']:<12} {r['Ticker']:<10} {r['Precio']:>8.3f}  {r['Prob']:>5.1%}  {ret_str}  {cor_str}")

    # ─── ESTADÍSTICAS DE VALIDACIÓN ───────────────────────────────
    if all_signals:
        sig_df = pd.DataFrame(all_signals)
        with_outcome = sig_df[sig_df['Correcto'].notna() & (sig_df['Correcto'] != "pendiente")]
        if not with_outcome.empty:
            correct = (with_outcome['Correcto'] == "✅ SUBIÓ").sum()
            total_v = len(with_outcome)
            win_rate = correct / total_v
            avg_ret  = pd.to_numeric(with_outcome['Ret5d_%'], errors='coerce').mean()
            print(f"\n  📈 VALIDACIÓN (señales con resultado conocido: {total_v})")
            print(f"     Win rate     : {win_rate:.1%}  ({correct}/{total_v} correctas)")
            print(f"     Retorno medio: {avg_ret:+.2f}% en 5 días")

    # ─── GUARDAR EN EXCEL ──────────────────────────────────────────
    if all_daily:
        out_file = f"predictions_2026_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"
        all_df  = pd.DataFrame(all_daily)
        sig_df2 = pd.DataFrame(all_signals) if all_signals else pd.DataFrame()

        with pd.ExcelWriter(out_file, engine='openpyxl') as writer:
            if not sig_df2.empty:
                sig_df2.sort_values(['Fecha','Prob'], ascending=[True,False]).to_excel(
                    writer, sheet_name='Señales_BUY', index=False)
            all_df.sort_values(['Fecha','Ticker']).to_excel(
                writer, sheet_name='Todas_Predicciones', index=False)

            # Resumen por ticker
            if not sig_df2.empty:
                resumen = sig_df2.groupby('Ticker').agg(
                    Señales=('Prob','count'),
                    Prob_Media=('Prob','mean'),
                ).sort_values('Señales', ascending=False).reset_index()
                resumen.to_excel(writer, sheet_name='Resumen_Tickers', index=False)

        print(f"\n  💾 Guardado en: {out_file}")

    print()

if __name__ == "__main__":
    main()
