# validate_model.py - TITAN AI: WALK-FORWARD VALIDATION & MODEL EVALUATION
"""
Evalúa el modelo predictivo TitanBrain con validación walk-forward real (sin data leakage):

MÉTRICAS DE CLASIFICACIÓN (por fold y global):
  - AUC ROC, Precision, Recall, F1
  - Tasa de señales generadas

MÉTRICAS DE TRADING (simulación out-of-sample):
  - Win Rate
  - Sharpe Ratio (anualizado)
  - Max Drawdown
  - Promedio PnL por operación
  - Curva de equity

REPORTES:
  - Por ticker: mejores y peores activos
  - Por fold temporal: tendencia del modelo en el tiempo
  - Guardado en validation_results_YYYY-MM-DD.xlsx

Uso:
    python validate_model.py
    python validate_model.py --tickers BHP.AX CBA.AX CSL.AX --start 2021-01-01
"""

import pandas as pd
import numpy as np
import os
import sys
import warnings
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────
START_DATE       = "2020-01-01"
N_SPLITS         = 5         # Folds walk-forward
MIN_TRAIN_ROWS   = 300       # Mínimo filas de entrenamiento por fold
PROB_THRESHOLD   = 0.48      # Umbral para generar señal (igual que orchestrator)
TP_PCT           = 0.10      # Take profit por operación
SL_PCT           = 0.05      # Stop loss por operación
HOLD_DAYS        = 10        # Días máximos de holding si no se toca TP ni SL
START_CAPITAL    = 1000.0    # Capital virtual para curva de equity


def get_feature_cols(df):
    """Retorna la lista de features disponibles para el modelo."""
    CANDIDATE_FEATURES = [
        'dist_sma20', 'dist_sma50', 'dist_sma200',
        'ma_cross_20_50', 'ma_cross_50_200',
        'atr_pct', 'vol_regime',
        'momentum_3', 'momentum_5', 'momentum_10', 'momentum_20',
        'roc_5', 'roc_10',
        'rsi', 'macd_diff', 'adx',
        'bb_width', 'bb_upper_dist', 'bb_lower_dist',
        'stoch_k', 'stoch_d', 'williams_r', 'cci',
        'vol_rel_20', 'vol_rel_50',
        'close_to_max5', 'close_to_min5',
        'close_to_max20', 'close_to_min20',
    ]
    return [f for f in CANDIDATE_FEATURES if f in df.columns]


def build_model():
    """Construye el ensemble calibrado idéntico al usado en producción."""
    lr_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(C=0.5, solver='liblinear', random_state=42))
    ])
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=5, min_samples_leaf=10,
        random_state=42, n_jobs=-1
    )
    gb = GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42
    )
    ensemble = VotingClassifier(
        estimators=[('rf', rf), ('lr', lr_pipe), ('gb', gb)],
        voting='soft'
    )
    try:
        return CalibratedClassifierCV(ensemble, method='isotonic', cv=3)
    except Exception:
        return CalibratedClassifierCV(ensemble, method='sigmoid', cv=3)


def simulate_trade(test_df, entry_idx, entry_price):
    """
    Simula una operación desde entry_idx hasta TP, SL o HOLD_DAYS.
    Retorna (exit_price, outcome, pnl_pct).
    """
    for j in range(1, min(HOLD_DAYS + 1, len(test_df) - entry_idx)):
        future_price = test_df['close'].iloc[entry_idx + j]
        ret = (future_price - entry_price) / entry_price
        if ret >= TP_PCT:
            return entry_price * (1 + TP_PCT), 'TAKE_PROFIT', TP_PCT
        elif ret <= -SL_PCT:
            return entry_price * (1 - SL_PCT), 'STOP_LOSS', -SL_PCT
    # Expiró el holding máximo
    exit_price = test_df['close'].iloc[min(entry_idx + HOLD_DAYS, len(test_df) - 1)]
    pnl = (exit_price - entry_price) / entry_price
    return exit_price, 'EXPIRE', pnl


def validate_ticker(ticker, df, brain_instance):
    """
    Ejecuta validación walk-forward completa para un ticker.
    Retorna (predictions_list, trades_list).
    """
    feat_cols = get_feature_cols(df)
    if not feat_cols or len(df) < MIN_TRAIN_ROWS + 50:
        return [], []

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    predictions = []
    trades = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(df)):
        if len(train_idx) < MIN_TRAIN_ROWS:
            continue

        train_df = df.iloc[train_idx]
        test_df  = df.iloc[test_idx]

        X_train = train_df[feat_cols]
        y_train = train_df['target']

        if y_train.sum() < 5:
            continue

        model = build_model()
        try:
            model.fit(X_train, y_train)
        except Exception:
            continue

        X_test = test_df[feat_cols]
        try:
            probs = model.predict_proba(X_test)[:, 1]
        except Exception:
            continue

        preds = (probs >= PROB_THRESHOLD).astype(int)

        # ── Registrar predicciones ──
        for i, (date, prob) in enumerate(zip(test_df.index, probs)):
            predictions.append({
                'ticker': ticker,
                'fold':   fold,
                'date':   date,
                'prob':   prob,
                'pred':   int(preds[i]),
                'actual': int(test_df['target'].iloc[i]),
            })

        # ── Simular trades ──
        positions_open = {}  # {entry_date: True} para no abrir 2 posiciones simultáneas
        i = 0
        while i < len(test_df):
            date  = test_df.index[i]
            prob  = probs[i]

            if prob >= PROB_THRESHOLD and len(positions_open) == 0:
                entry_price = test_df['close'].iloc[i]
                if entry_price > 0:
                    exit_price, outcome, pnl_pct = simulate_trade(test_df, i, entry_price)
                    trades.append({
                        'ticker':      ticker,
                        'fold':        fold,
                        'date':        date,
                        'entry_price': entry_price,
                        'exit_price':  exit_price,
                        'pnl_pct':     pnl_pct,
                        'prob':        prob,
                        'outcome':     outcome,
                        'win':         1 if pnl_pct > 0 else 0,
                    })
                    # Saltar los días de holding para no solapar operaciones
                    i += HOLD_DAYS
                    continue
            i += 1

    return predictions, trades


def equity_metrics(pnl_series):
    """
    Calcula Sharpe anualizado y Max Drawdown de una serie de PnLs porcentuales.
    """
    equity = [START_CAPITAL]
    for pnl in pnl_series:
        equity.append(equity[-1] * (1 + pnl))
    eq = pd.Series(equity)
    daily_ret = eq.pct_change().dropna()
    sharpe    = (daily_ret.mean() / daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else 0.0
    max_dd    = ((eq.cummax() - eq) / eq.cummax()).max()
    return sharpe, max_dd, eq.iloc[-1]


def run_validation(tickers=None, start_date=START_DATE):
    print("\n" + "=" * 70)
    print("🔬  TITAN AI — WALK-FORWARD VALIDATION")
    print(f"    Parámetros: prob≥{PROB_THRESHOLD} | TP={TP_PCT:.0%} | SL={SL_PCT:.0%} | hold={HOLD_DAYS}d")
    print("=" * 70)

    from core.brain import TitanBrain
    brain = TitanBrain()

    # ── Cargar tickers ──
    if tickers is None:
        csv_files = [f for f in os.listdir('au_stock_data')
                     if f.endswith('.csv') and not f.startswith('au_')]
        tickers = [f.replace('.csv', '') + '.AX' for f in csv_files]

    print(f"\n📂 Cargando datos de {len(tickers)} tickers desde {start_date}...")
    raw_data = brain.get_data(tickers, start_date=start_date)

    all_predictions = []
    all_trades      = []
    ticker_summary  = []

    # ── Validar ticker por ticker ──
    processed = 0
    for ticker in tickers:
        if ticker not in raw_data:
            continue

        df = brain.engineer_features(raw_data[ticker].copy())
        if df is None or len(df) < MIN_TRAIN_ROWS + 50:
            continue

        preds, trades = validate_ticker(ticker, df, brain)
        all_predictions.extend(preds)
        all_trades.extend(trades)
        processed += 1

        # Métricas por ticker
        if preds:
            pdf = pd.DataFrame(preds)
            try:
                auc  = roc_auc_score(pdf['actual'], pdf['prob']) if pdf['actual'].sum() > 0 else 0.5
            except Exception:
                auc = 0.5
            prec = precision_score(pdf['actual'], pdf['pred'], zero_division=0)
            rec  = recall_score(pdf['actual'], pdf['pred'], zero_division=0)
            f1   = f1_score(pdf['actual'], pdf['pred'], zero_division=0)
            n_sig = int(pdf['pred'].sum())

            trade_metrics = {}
            if trades:
                tdf = pd.DataFrame(trades)
                sharpe, max_dd, final_eq = equity_metrics(tdf['pnl_pct'].tolist())
                trade_metrics = {
                    'win_rate': tdf['win'].mean(),
                    'avg_pnl':  tdf['pnl_pct'].mean(),
                    'n_trades': len(tdf),
                    'sharpe':   sharpe,
                    'max_dd':   max_dd,
                    'final_eq': final_eq,
                }

            ticker_summary.append({
                'ticker':    ticker,
                'auc':       auc,
                'precision': prec,
                'recall':    rec,
                'f1':        f1,
                'n_signals': n_sig,
                **trade_metrics,
            })

        sys.stdout.write(f"\r   Procesados: {processed}/{len(tickers)}")
        sys.stdout.flush()

    print(f"\n\n✅ Tickers procesados: {processed}")

    if not all_predictions:
        print("❌ No se generaron predicciones. Verifica los datos en au_stock_data/")
        return

    pred_df   = pd.DataFrame(all_predictions)
    trades_df = pd.DataFrame(all_trades) if all_trades else pd.DataFrame()
    results_df = pd.DataFrame(ticker_summary) if ticker_summary else pd.DataFrame()

    # ──────────────────────────────────────────
    # MÉTRICAS GLOBALES DE CLASIFICACIÓN
    # ──────────────────────────────────────────
    global_prec = precision_score(pred_df['actual'], pred_df['pred'], zero_division=0)
    global_rec  = recall_score(pred_df['actual'], pred_df['pred'], zero_division=0)
    global_f1   = f1_score(pred_df['actual'], pred_df['pred'], zero_division=0)
    try:
        global_auc = roc_auc_score(pred_df['actual'], pred_df['prob'])
    except Exception:
        global_auc = 0.5

    total_signals = int(pred_df['pred'].sum())
    total_rows    = len(pred_df)
    signal_rate   = total_signals / total_rows if total_rows > 0 else 0

    print(f"\n{'─'*60}")
    print("📊  MÉTRICAS GLOBALES  (Out-of-Sample Walk-Forward)")
    print(f"{'─'*60}")
    print(f"   AUC ROC:              {global_auc:.4f}   {_auc_badge(global_auc)}")
    print(f"   Precisión:            {global_prec:.4f}")
    print(f"   Recall:               {global_rec:.4f}")
    print(f"   F1 Score:             {global_f1:.4f}")
    print(f"   Total predicciones:   {total_rows:,}")
    print(f"   Total señales (BUY):  {total_signals:,}  ({signal_rate:.1%} del tiempo)")

    # ──────────────────────────────────────────
    # MÉTRICAS DE TRADING
    # ──────────────────────────────────────────
    if not trades_df.empty:
        win_rate  = trades_df['win'].mean()
        avg_pnl   = trades_df['pnl_pct'].mean()
        n_trades  = len(trades_df)
        sharpe, max_dd, final_eq = equity_metrics(trades_df['pnl_pct'].tolist())

        print(f"\n{'─'*60}")
        print("💰  MÉTRICAS DE TRADING  (Simulación ${:.0f} inicial)".format(START_CAPITAL))
        print(f"{'─'*60}")
        print(f"   Total operaciones:         {n_trades:,}")
        print(f"   Win Rate:                  {win_rate:.2%}  {_badge(win_rate, 0.50, 0.55)}")
        print(f"   PnL promedio / operación:  {avg_pnl:+.2%}")
        print(f"   PnL total acumulado:       {trades_df['pnl_pct'].sum():+.2%}")
        print(f"   Sharpe Ratio (anual.):     {sharpe:.2f}    {_badge(sharpe, 1.0, 1.5, is_ratio=True)}")
        print(f"   Max Drawdown:              {max_dd:.2%}")
        print(f"   Capital final (${START_CAPITAL:.0f}):    ${final_eq:.2f}")

        # Desglose por resultado
        print(f"\n{'─'*60}")
        print("📋  DESGLOSE POR RESULTADO")
        print(f"{'─'*60}")
        outcome_df = (trades_df.groupby('outcome')
                               .agg(count=('pnl_pct', 'count'), avg_pnl=('pnl_pct', 'mean'))
                               .reset_index())
        for _, row in outcome_df.iterrows():
            bar = "█" * int(row['count'] / n_trades * 30)
            print(f"   {row['outcome']:12s}: {int(row['count']):5d} trades "
                  f"| avg PnL: {row['avg_pnl']:+.2%}  {bar}")

        # PnL por fold (tendencia temporal del modelo)
        print(f"\n{'─'*60}")
        print("📅  PnL POR FOLD (¿mejora el modelo en el tiempo?)")
        print(f"{'─'*60}")
        fold_stats = (trades_df.groupby('fold')
                               .agg(n=('pnl_pct', 'count'), win=('win', 'mean'),
                                    avg_pnl=('pnl_pct', 'mean'))
                               .reset_index())
        for _, row in fold_stats.iterrows():
            sign = "+" if row['avg_pnl'] >= 0 else ""
            print(f"   Fold {int(row['fold'])}: {int(row['n']):4d} trades "
                  f"| Win: {row['win']:.1%} | Avg PnL: {sign}{row['avg_pnl']:.2%}")

    # ──────────────────────────────────────────
    # TOP / WORST TICKERS
    # ──────────────────────────────────────────
    if not results_df.empty:
        print(f"\n{'─'*60}")
        print("🏆  TOP 10 TICKERS POR WIN RATE")
        print(f"{'─'*60}")
        print(f"   {'Ticker':12s} {'AUC':>6} {'WinRate':>8} {'AvgPnL':>8} {'Trades':>7} {'Sharpe':>7}")
        print(f"   {'─'*12} {'─'*6} {'─'*8} {'─'*8} {'─'*7} {'─'*7}")

        top = (results_df[results_df.get('n_trades', results_df.get('n_signals', pd.Series([0]*len(results_df)))) > 3]
               .sort_values('win_rate', ascending=False)
               .head(10) if 'win_rate' in results_df.columns else
               results_df.sort_values('auc', ascending=False).head(10))

        for _, r in top.iterrows():
            wr  = f"{r.get('win_rate', 0):.1%}" if 'win_rate' in r else 'N/A'
            ap  = f"{r.get('avg_pnl', 0):+.2%}" if 'avg_pnl' in r else 'N/A'
            nt  = str(int(r.get('n_trades', r.get('n_signals', 0))))
            sh  = f"{r.get('sharpe', 0):.2f}" if 'sharpe' in r else 'N/A'
            print(f"   {r['ticker']:12s} {r['auc']:>6.3f} {wr:>8} {ap:>8} {nt:>7} {sh:>7}")

        if 'avg_pnl' in results_df.columns and 'n_trades' in results_df.columns:
            print(f"\n{'─'*60}")
            print("⚠️   BOTTOM 5 TICKERS POR WIN RATE (candidatos a eliminar del universo)")
            print(f"{'─'*60}")
            bottom = (results_df[results_df['n_trades'] > 3]
                      .sort_values('win_rate', ascending=True)
                      .head(5))
            for _, r in bottom.iterrows():
                print(f"   {r['ticker']:12s} WinRate:{r.get('win_rate',0):.1%} "
                      f"AvgPnL:{r.get('avg_pnl',0):+.2%} Trades:{int(r.get('n_trades',0))}")

    # ──────────────────────────────────────────
    # RECOMENDACIONES
    # ──────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("💡  RECOMENDACIONES PARA MEJORAR EL MODELO")
    print(f"{'─'*60}")
    if not trades_df.empty:
        if win_rate < 0.45:
            print("   ⬆️  Win Rate bajo: considera aumentar PROB_THRESHOLD a 0.52-0.55")
        elif win_rate > 0.60:
            print("   ✅  Win Rate sólido. Puedes probar bajar PROB_THRESHOLD para más señales.")
        if sharpe < 0.5:
            print("   ⚠️  Sharpe bajo: revisa el ratio TP/SL (actual TP={:.0%} SL={:.0%})".format(TP_PCT, SL_PCT))
        if max_dd > 0.25:
            print("   🛑  Max Drawdown alto: añade filtro de régimen de mercado (solo operar en BULL)")
        if signal_rate < 0.02:
            print("   📉  Muy pocas señales ({:.1%}): baja PROB_THRESHOLD o revisa el target en engineer_features".format(signal_rate))

    # ──────────────────────────────────────────
    # GUARDAR RESULTADOS EN EXCEL
    # ──────────────────────────────────────────
    output_file = f"validation_results_{datetime.now().strftime('%Y-%m-%d')}.xlsx"
    summary_data = {
        'Metrica': ['AUC ROC', 'Precisión', 'Recall', 'F1', 'Total Señales', 'Tasa Señal',
                    'Win Rate', 'Avg PnL/Operación', 'Sharpe', 'Max Drawdown',
                    'Total Operaciones', 'Capital Final'],
        'Valor': [
            f"{global_auc:.4f}", f"{global_prec:.4f}", f"{global_rec:.4f}", f"{global_f1:.4f}",
            str(total_signals), f"{signal_rate:.2%}",
            f"{win_rate:.2%}"  if not trades_df.empty else 'N/A',
            f"{avg_pnl:+.2%}" if not trades_df.empty else 'N/A',
            f"{sharpe:.2f}"   if not trades_df.empty else 'N/A',
            f"{max_dd:.2%}"   if not trades_df.empty else 'N/A',
            str(n_trades)      if not trades_df.empty else '0',
            f"${final_eq:.2f}" if not trades_df.empty else 'N/A',
        ]
    }

    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        pd.DataFrame(summary_data).to_excel(writer, sheet_name='Resumen', index=False)
        pred_df.to_excel(writer, sheet_name='Predicciones', index=False)
        if not trades_df.empty:
            trades_df.to_excel(writer, sheet_name='Trades_Simulados', index=False)
        if not results_df.empty:
            results_df.sort_values('auc', ascending=False).to_excel(
                writer, sheet_name='Por_Ticker', index=False)

    print(f"\n💾 Resultados guardados en: {output_file}")
    print("─" * 70 + "\n")


# ─────────────────────────────────────────────
# HELPERS VISUALES
# ─────────────────────────────────────────────
def _auc_badge(auc):
    if auc >= 0.65: return "🟢 Bueno"
    if auc >= 0.55: return "🟡 Aceptable"
    return "🔴 Débil (cercano a azar)"

def _badge(val, warn, good, is_ratio=False):
    if is_ratio:
        if val >= good:  return "🟢"
        if val >= warn:  return "🟡"
        return "🔴"
    if val >= good:  return "🟢"
    if val >= warn:  return "🟡"
    return "🔴"


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Walk-Forward Validation — TitanBrain")
    parser.add_argument('--tickers', nargs='+', default=None,
                        help='Lista de tickers (ej: BHP.AX CBA.AX). Default: todos los CSV en au_stock_data/')
    parser.add_argument('--start', default=START_DATE,
                        help=f'Fecha de inicio (default: {START_DATE})')
    parser.add_argument('--prob', type=float, default=PROB_THRESHOLD,
                        help=f'Umbral de probabilidad (default: {PROB_THRESHOLD})')
    parser.add_argument('--tp', type=float, default=TP_PCT,
                        help=f'Take profit %% (default: {TP_PCT})')
    parser.add_argument('--sl', type=float, default=SL_PCT,
                        help=f'Stop loss %% (default: {SL_PCT})')
    args = parser.parse_args()

    PROB_THRESHOLD = args.prob
    TP_PCT         = args.tp
    SL_PCT         = args.sl

    run_validation(tickers=args.tickers, start_date=args.start)
