# core/recommendation.py - Lógica centralizada de recomendaciones y alertas
import pandas as pd
from core.brain import TitanBrain
from core.risk_engine import RiskEngine
import datetime
import os
import warnings
warnings.filterwarnings('ignore')

def get_recommendations(tickers=None, start_date=None, portfolio_path='current_portfolio.json', min_ev_ratio=0.0, min_prob=0.05):
    """
    Devuelve recomendaciones de compra y venta, y el estado del portafolio, usando el flujo centralizado.
    """
    if tickers is None:
        # Usar solo los tickers que tienen CSV en au_stock_data
        tickers = []
        data_dir = 'au_stock_data'
        for fname in os.listdir(data_dir):
            if fname.endswith('.csv'):
                symbol = fname.replace('.csv','') + '.AX'
                tickers.append(symbol)
    if start_date is None:
        start_date = (datetime.datetime.now() - datetime.timedelta(days=365)).strftime('%Y-%m-%d')
    brain = TitanBrain()
    risk_engine = RiskEngine()
    # Cargar portafolio
    portfolio = {}
    if os.path.exists(portfolio_path):
        try:
            with open(portfolio_path, 'r') as f:
                portfolio = pd.read_json(f, typ='dict') if portfolio_path.endswith('.json') else {}
                if isinstance(portfolio, dict):
                    for t, pos in portfolio.items():
                        pos['shares'] = int(pos['shares'])
                        pos['buy_price'] = float(pos['buy_price'])
                        pos['highest_price'] = float(pos['highest_price'])
                        pos['stop_price'] = float(pos['stop_price'])
                        pos['take_profit_price'] = float(pos['take_profit_price'])
                        pos['atr_at_buy'] = float(pos['atr_at_buy'])
        except Exception:
            portfolio = {}
    # 1. Usar CSV solo para entrenar el modelo
    raw_data = brain.get_data(tickers, start_date=start_date)
    processed_data = {}
    valid_tickers = []
    if hasattr(raw_data, 'columns'):
        tickers_iter = [t for t in tickers if t in raw_data.columns]
        get_df = lambda t: raw_data[t].copy()
    else:
        tickers_iter = [t for t in tickers if t in raw_data]
        get_df = lambda t: raw_data[t].copy()
    for t in tickers_iter:
        df = brain.engineer_features(get_df(t))
        if df is not None and len(df) > 100:
            processed_data[t] = df
            valid_tickers.append(t)
    if not valid_tickers:
        return [], [], [], None

    # 2. Obtener datos actuales en tiempo real para predicción de compra
    try:
        from fetch_realtime_au import fetch_realtime_au
        realtime_data = fetch_realtime_au(valid_tickers)
    except Exception as e:
        print(f"[ERROR] No se pudo obtener datos en tiempo real: {e}")
        realtime_data = {}
    # Usar datos en tiempo real para predicción de compra
    day_data = {}
    for t in valid_tickers:
        if t in realtime_data:
            # Crear un DataFrame de una fila con las features del modelo y los datos actuales
            df_hist = processed_data[t]
            # Tomar la última fila histórica y actualizar con los valores actuales
            last_hist = df_hist.iloc[-1].copy()
            for k in ['close','high','low','open','volume']:
                if k in last_hist and k in realtime_data[t]:
                    last_hist[k] = realtime_data[t][k]
            # Recalcular features técnicas si es necesario (opcional: aquí solo actualizamos precios)
            day_data[t] = last_hist
        else:
            # Si no hay dato en tiempo real, usar el último histórico
            day_data[t] = processed_data[t].iloc[-1]
    # Para ventas y posiciones, seguir usando el último dato histórico
    latest_date = max(df.index[-1] for df in processed_data.values())
    buy_recommendations = []
    sell_recommendations = []
    positions_status = []
    positions_to_remove = []
    for t, pos in portfolio.items():
        if t not in day_data:
            continue
        row = day_data[t]
        current_price = row['Close'] if 'Close' in row else row['close']
        high_today = row.get('High', row.get('high', current_price))
        low_today  = row.get('Low',  row.get('low',  current_price))
        # Actualizar trailing stop antes de evaluar la posición
        pos = risk_engine.update_trailing_stop(pos, current_price)
        portfolio[t] = pos  # persistir el stop actualizado
        current_value = pos['shares'] * current_price
        buy_value = pos['shares'] * pos['buy_price']
        unrealized_pnl = current_value - buy_value
        growth_pct = (current_price / pos['buy_price']) - 1
        sell_signal = ""
        if low_today <= pos['stop_price']:
            sell_signal = "STOP_LOSS"
        elif high_today >= pos['take_profit_price']:
            sell_signal = "TAKE_PROFIT"
        if sell_signal:
            positions_to_remove.append(t)
        positions_status.append({
            'ticker': t,
            'shares': pos['shares'],
            'buy_price': pos['buy_price'],
            'current_price': current_price,
            'current_value': current_value,
            'unrealized_pnl': unrealized_pnl,
            'growth_pct': growth_pct,
            'sell_signal': sell_signal,
            'stop_price': pos['stop_price'],
            'take_profit_price': pos['take_profit_price']
        })
    for t in positions_to_remove:
        if t in portfolio:
            del portfolio[t]
    for pos in positions_status:
        if pos['sell_signal']:
            sell_price = pos['stop_price'] if pos['sell_signal'] == "STOP_LOSS" else pos['take_profit_price']
            sell_recommendations.append({
                'ticker': pos['ticker'],
                'action': 'SELL',
                'reason': pos['sell_signal'],
                'price': sell_price,
                'shares': pos['shares'],
                'pnl': pos['unrealized_pnl'],
                'pnl_pct': pos['growth_pct']
            })
    available_tickers = [t for t in valid_tickers if t not in portfolio and t in day_data]
    discard_log = []
    for t in available_tickers:
        row = day_data[t]
        motivo = None
        # Usar la fila actualizada (con datos en tiempo real) para la predicción
        # Se pasa la fecha de hoy para forzar predicción con datos actuales
        today = datetime.datetime.now().strftime('%Y-%m-%d')
        prob_win, features, atr_pct = brain.train_and_predict_calibrated(t, processed_data[t], today)
        print(f"[DEBUG] {t}: prob_win={prob_win:.4f}, features={features}, ATR_Pct={atr_pct:.4f}")
        if prob_win < min_prob:
            motivo = f'Descartado: Probabilidad {prob_win:.4f} < min_prob {min_prob}'
        else:
            regime = risk_engine.get_market_regime(today)
            sl_pct, tp_pct = risk_engine.get_dynamic_stops(regime, atr_pct)
            buy_recommendations.append({
                'ticker': t,
                'prob': prob_win,
                'price': row['close'] if 'close' in row else row['Close'],
                'sl_pct': sl_pct,
                'tp_pct': tp_pct,
                'regime': regime
            })
        if motivo:
            discard_log.append({'ticker': t, 'motivo': motivo, 'prob_win': prob_win, 'features': features, 'ATR_Pct': atr_pct})
    buy_recommendations.sort(key=lambda x: x.get('prob', 0), reverse=True)
    sell_recommendations.sort(key=lambda x: x['pnl_pct'], reverse=True)
    return buy_recommendations, sell_recommendations, positions_status, latest_date, discard_log
    buy_recommendations.sort(key=lambda x: x['ev'], reverse=True)
    sell_recommendations.sort(key=lambda x: x['pnl_pct'], reverse=True)
    return buy_recommendations, sell_recommendations, positions_status, latest_date
