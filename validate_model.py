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
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, brier_score_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────
START_DATE       = "2020-01-01"
N_SPLITS         = 5         # Folds walk-forward
MIN_TRAIN_ROWS   = 300       # Mínimo filas de entrenamiento por fold
PROB_THRESHOLD   = 0.52      # Umbral por defecto alineado con V37
SL_ATR           = 2.5       # Stop = entrada - 2.5*ATR
BE_TRIGGER_ATR   = 1.5       # Trigger de breakeven
TP1_PCT          = 0.09      # Referencia V37
TP2_PCT          = 0.22      # Salida objetivo final V37
MAX_HOLD_DAYS    = 35
MACD_MIN_PROFIT  = 0.01
MACD_MIN_PROFIT_BE = 0.005
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


def build_model(seed=42):
    """Replica el ensemble calibrado del pipeline de entrenamiento principal."""
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=2,
        min_samples_split=5,
        class_weight='balanced_subsample',
        random_state=seed,
        n_jobs=-1,
    )
    rf_calibrated = CalibratedClassifierCV(rf, method='isotonic', cv=5)

    lr = LogisticRegression(
        C=0.5,
        class_weight='balanced',
        random_state=seed,
        max_iter=1000,
    )
    lr_calibrated = CalibratedClassifierCV(lr, method='sigmoid', cv=5)

    gb = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        random_state=seed,
    )

    return VotingClassifier(
        estimators=[('rf', rf_calibrated), ('lr', lr_calibrated), ('gb', gb)],
        voting='soft',
        weights=[0.40, 0.35, 0.25],
    )


def simulate_trade(test_df, entry_idx, entry_price):
    """
    Simula una operación con lógica de salida alineada a V37.
    Retorna (exit_price, outcome, pnl_pct).
    """
    atr = float(test_df['atr'].iloc[entry_idx]) if 'atr' in test_df.columns else entry_price * 0.02
    if not np.isfinite(atr) or atr <= 0:
        atr = entry_price * 0.02

    stop_price = entry_price - SL_ATR * atr
    be_price = entry_price + BE_TRIGGER_ATR * atr
    breakeven_active = False

    horizon = min(MAX_HOLD_DAYS + 1, len(test_df) - entry_idx)
    for j in range(1, horizon):
        idx = entry_idx + j
        future_price = float(test_df['close'].iloc[idx])
        pnl = (future_price - entry_price) / entry_price

        if (not breakeven_active) and future_price >= be_price:
            stop_price = max(stop_price, entry_price)
            breakeven_active = True

        if future_price <= stop_price:
            stop_pnl = (stop_price - entry_price) / entry_price
            return stop_price, 'STOP_LOSS', stop_pnl

        if pnl >= TP2_PCT:
            return entry_price * (1 + TP2_PCT), 'TAKE_PROFIT', TP2_PCT

        if 'macd_diff' in test_df.columns and idx > 0:
            prev_macd = float(test_df['macd_diff'].iloc[idx - 1])
            curr_macd = float(test_df['macd_diff'].iloc[idx])
            macd_invertido = prev_macd > 0 and curr_macd < 0
            min_profit = MACD_MIN_PROFIT_BE if breakeven_active else MACD_MIN_PROFIT
            if macd_invertido and pnl >= min_profit:
                return future_price, 'MACD_INVERTIDO', pnl

    exit_idx = min(entry_idx + MAX_HOLD_DAYS, len(test_df) - 1)
    exit_price = float(test_df['close'].iloc[exit_idx])
    pnl = (exit_price - entry_price) / entry_price
    return exit_price, 'MAX_DIAS', pnl


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
                    i += MAX_HOLD_DAYS
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


def expected_calibration_error(y_true, y_prob, n_bins=10):
    """ECE simple para medir descalibración de probabilidades."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins, right=True) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    ece = 0.0
    n = len(y_true)
    if n == 0:
        return 0.0
    for b in range(n_bins):
        mask = bin_ids == b
        if not np.any(mask):
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def population_stability_index(reference, current, bins=10):
    """PSI para cuantificar drift de distribución de probabilidad."""
    reference = np.asarray(reference)
    current = np.asarray(current)
    if len(reference) == 0 or len(current) == 0:
        return 0.0

    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.quantile(reference, quantiles)
    edges = np.unique(edges)
    if len(edges) < 3:
        edges = np.linspace(0.0, 1.0, bins + 1)

    ref_hist, _ = np.histogram(reference, bins=edges)
    cur_hist, _ = np.histogram(current, bins=edges)

    eps = 1e-6
    ref_pct = ref_hist / max(ref_hist.sum(), 1) + eps
    cur_pct = cur_hist / max(cur_hist.sum(), 1) + eps
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def run_validation(tickers=None, start_date=START_DATE):
    print("\n" + "=" * 70)
    print("🔬  TITAN AI — WALK-FORWARD VALIDATION")
    print(
        f"    Parámetros: prob≥{PROB_THRESHOLD:.2f} | "
        f"SL_ATR={SL_ATR:.1f} | BE_ATR={BE_TRIGGER_ATR:.1f} | "
        f"TP2={TP2_PCT:.0%} | hold={MAX_HOLD_DAYS}d"
    )
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
    global_brier = brier_score_loss(pred_df['actual'], pred_df['prob'])
    global_ece = expected_calibration_error(pred_df['actual'], pred_df['prob'])

    pred_sorted = pred_df.sort_values('date')
    mid = len(pred_sorted) // 2
    probs_ref = pred_sorted['prob'].iloc[:mid].values
    probs_cur = pred_sorted['prob'].iloc[mid:].values
    global_psi = population_stability_index(probs_ref, probs_cur) if mid > 100 else 0.0

    total_signals = int(pred_df['pred'].sum())
    total_rows    = len(pred_df)
    signal_rate   = total_signals / total_rows if total_rows > 0 else 0

    print(f"\n{'─'*60}")
    print("📊  MÉTRICAS GLOBALES  (Out-of-Sample Walk-Forward)")
    print(f"{'─'*60}")
    print(f"   AUC ROC:              {global_auc:.4f}   {_auc_badge(global_auc)}")
    print(f"   Brier Score:          {global_brier:.4f}   {'🟢' if global_brier <= 0.20 else '🟡' if global_brier <= 0.25 else '🔴'}")
    print(f"   ECE (10 bins):        {global_ece:.4f}   {'🟢' if global_ece <= 0.05 else '🟡' if global_ece <= 0.10 else '🔴'}")
    print(f"   PSI (1ra vs 2da mitad): {global_psi:.4f}   {'🟢' if global_psi < 0.10 else '🟡' if global_psi < 0.25 else '🔴'}")
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
            print("   ⚠️  Sharpe bajo: revisa filtros de entrada o sube PROB_THRESHOLD (0.53-0.56)")
        if max_dd > 0.25:
            print("   🛑  Max Drawdown alto: añade filtro de régimen de mercado (solo operar en BULL)")
        if signal_rate < 0.02:
            print("   📉  Muy pocas señales ({:.1%}): baja PROB_THRESHOLD o revisa el target en engineer_features".format(signal_rate))
    if global_ece > 0.10:
        print("   🎯 Calibración floja: reentrena calibrador (isotonic/sigmoid) con ventana más reciente")
    if global_psi >= 0.25:
        print("   🌊 Drift alto: reentrena con más peso a datos recientes o segmenta por régimen")

    # ──────────────────────────────────────────
    # GUARDAR RESULTADOS EN EXCEL
    # ──────────────────────────────────────────
    output_file = f"validation_results_{datetime.now().strftime('%Y-%m-%d')}.xlsx"
    summary_data = {
        'Metrica': ['AUC ROC', 'Brier Score', 'ECE', 'PSI', 'Precisión', 'Recall', 'F1', 'Total Señales', 'Tasa Señal',
                    'Win Rate', 'Avg PnL/Operación', 'Sharpe', 'Max Drawdown',
                    'Total Operaciones', 'Capital Final'],
        'Valor': [
            f"{global_auc:.4f}", f"{global_brier:.4f}", f"{global_ece:.4f}", f"{global_psi:.4f}",
            f"{global_prec:.4f}", f"{global_rec:.4f}", f"{global_f1:.4f}",
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
    parser.add_argument('--sl-atr', type=float, default=SL_ATR,
                        help=f'Stop en ATR (default: {SL_ATR})')
    parser.add_argument('--be-atr', type=float, default=BE_TRIGGER_ATR,
                        help=f'Break-even trigger en ATR (default: {BE_TRIGGER_ATR})')
    parser.add_argument('--tp2', type=float, default=TP2_PCT,
                        help=f'Take profit final %% (default: {TP2_PCT})')
    parser.add_argument('--max-hold', type=int, default=MAX_HOLD_DAYS,
                        help=f'Días máximos de holding (default: {MAX_HOLD_DAYS})')
    args = parser.parse_args()

    PROB_THRESHOLD = args.prob
    SL_ATR         = args.sl_atr
    BE_TRIGGER_ATR = args.be_atr
    TP2_PCT        = args.tp2
    MAX_HOLD_DAYS  = args.max_hold

    run_validation(tickers=args.tickers, start_date=args.start)
