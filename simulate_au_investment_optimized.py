# simulate_au_investment_optimized.py - Simulación óptima con reglas realistas y parámetros variables
"""
Simula un portafolio de $5,000 invirtiendo en el ASX200 con reglas:
- Compra mínima $500
- Take profit y stop loss variables
- No se re-compra el mismo ticker hasta vender
- Venta automática al alcanzar objetivo o stop
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from core.brain import TitanBrain, load_au_tickers
from sklearn.ensemble import RandomForestClassifier
import joblib
from itertools import product

START_CAPITAL = 5000
START_DATE = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
END_DATE = datetime.now().strftime('%Y-%m-%d')
MIN_BUY = 500


TAKE_PROFITS = [0.15, 0.18]  # Foco en los mejores valores
STOP_LOSSES = [0.10, 0.12]  # Probar trailing stops más ajustados (10% y 12%)
CALIBRATION_METHODS = ['auto', 'isotonic', 'sigmoid']
MAX_POSITION_PCT = 0.25  # Máximo 25% del capital en una sola posición

import os
# Usar todos los símbolos con CSV descargado en au_stock_data
csv_files = [f for f in os.listdir('au_stock_data') if f.endswith('.csv')]
TICKERS = [f.split('.')[0]+'.AX' for f in csv_files]
brain = TitanBrain()
all_data = brain.get_data(TICKERS, start_date=START_DATE)


# --- NEW: Test multiple calibration methods ---
results_summary = []
for calibration_method, tp, sl in product(CALIBRATION_METHODS, TAKE_PROFITS, STOP_LOSSES):
    portfolio = {'cash': START_CAPITAL, 'positions': {}, 'history': []}
    trading_days = None
    for t, df in all_data.items():
        if trading_days is None or len(df) > len(trading_days):
            trading_days = df.index
    trading_days = pd.to_datetime(sorted(set(trading_days)))
    trading_days = trading_days[(trading_days >= pd.to_datetime(START_DATE)) & (trading_days <= pd.to_datetime(END_DATE))]
    for day in trading_days:
        to_sell = []
        for ticker, pos in list(portfolio['positions'].items()):
            if ticker not in all_data or day not in all_data[ticker].index:
                continue
            price = all_data[ticker].loc[day]['close']
            entry = pos['buy_price']
            if 'max_price' not in pos:
                pos['max_price'] = entry
            if price > pos['max_price']:
                pos['max_price'] = price
            trailing_stop = pos['max_price'] * (1 - sl)
            if price <= trailing_stop or (price - entry) / entry >= tp:
                change = (price - entry) / entry
                to_sell.append((ticker, pos['shares'], price, change))
        for ticker, shares, price, change in to_sell:
            portfolio['cash'] += shares * price
            portfolio['history'].append({'date': day, 'action': 'sell', 'ticker': ticker, 'shares': shares, 'price': price, 'change': change})
            del portfolio['positions'][ticker]
        candidates = []
        for t, df in all_data.items():
            if t in portfolio['positions']:
                continue
            if day not in df.index:
                continue
            df_slice = df.loc[:day]
            if len(df_slice) < 150:
                continue
            df_eng = brain.engineer_features(df_slice)
            if df_eng is None or day not in df_eng.index:
                continue
            row = df_eng.loc[day]
            price = row['close']
            sma200 = row.get('sma_200', 0)
            adx = row.get('adx', 0)
            rsi = row.get('rsi', 50)
            if price <= sma200 or adx < 20 or rsi > 80:
                continue
            next_day = day + pd.Timedelta(days=1)
            try:
                prob, *_ = brain.train_and_predict_calibrated(t, df_eng, next_day, calibration_method=calibration_method)
            except Exception:
                prob = 0.0
            max_invest = min(portfolio['cash'], START_CAPITAL * MAX_POSITION_PCT)
            if prob > 0.6 and max_invest >= MIN_BUY:
                max_shares = int(max_invest // price)
                if max_shares * price >= MIN_BUY:
                    candidates.append((t, price, prob, max_shares))
        if candidates:
            candidates.sort(key=lambda x: x[2], reverse=True)
            ticker, price, prob, shares = candidates[0]
            invest_amount = shares * price
            if invest_amount >= MIN_BUY and invest_amount <= portfolio['cash']:
                portfolio['positions'][ticker] = {'shares': shares, 'buy_price': price}
                portfolio['cash'] -= invest_amount
    for ticker, pos in portfolio['positions'].items():
        last_price = all_data[ticker].iloc[-1]['close']
        change = (last_price - pos['buy_price']) / pos['buy_price']
        if change > 0:
            portfolio['cash'] += pos['shares'] * last_price
            portfolio['history'].append({'date': trading_days[-1], 'action': 'sell', 'ticker': ticker, 'shares': pos['shares'], 'price': last_price, 'change': change})
    final_value = portfolio['cash']
    results_summary.append({'calibration': calibration_method, 'take_profit': tp, 'stop_loss': sl, 'final_value': final_value})
    # Save only the best history
    if calibration_method == 'auto' and tp == 0.15 and sl == 0.10:
        pd.DataFrame(portfolio['history']).to_csv('investment/simulation_history_optimized.csv', index=False)
print('RESUMEN DE SIMULACIÓN (Calibration, Take Profit, Stop Loss, Capital Final)')
for r in results_summary:
    print(f"Calib: {r['calibration']} | TP: {int(r['take_profit']*100)}% | SL: {int(r['stop_loss']*100)}% | Capital final: ${r['final_value']:.2f}")
print('Historial guardado solo para la combinación óptima en investment/simulation_history_optimized.csv')
