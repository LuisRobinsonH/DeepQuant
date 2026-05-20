
# SIMULATE_AU_INVESTMENT.PY - Simulación anual profesional de inversión AU Stock (ASX-200)
# Simula portafolio anual, genera reporte .txt profesional y logs claros. Outputs en investment/.

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from core.brain import TitanBrain, load_au_tickers

INVESTMENT_DIR = 'investment'
os.makedirs(INVESTMENT_DIR, exist_ok=True)
START_CAPITAL = 5000
START_DATE = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
END_DATE = datetime.now().strftime('%Y-%m-%d')
BRANDING = '\n---\nDeepQuant AU Stock Simulation\nhttps://github.com/yourbrand\n'

def main():
    print(f"\n[SIMULACIÓN] Iniciando simulación anual AU Stock - {datetime.now()}")
    # 1. Cargar tickers y datos históricos
    TICKERS = load_au_tickers()
    brain = TitanBrain()
    all_data = brain.get_data(TICKERS, start_date=START_DATE)

    # 2. Inicializar portafolio
    portfolio = {'cash': START_CAPITAL, 'positions': {}, 'history': []}

    # 3. Simulación diaria
    trading_days = None
    for t, df in all_data.items():
        if trading_days is None or len(df) > len(trading_days):
            trading_days = df.index
    trading_days = pd.to_datetime(sorted(set(trading_days)))
    trading_days = trading_days[(trading_days >= pd.to_datetime(START_DATE)) & (trading_days <= pd.to_datetime(END_DATE))]

    for day in trading_days:
        # Generar recomendaciones para el día
        candidates = []
        for t, df in all_data.items():
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
            if price <= sma200 or adx < 25 or rsi > 75:
                continue
            # Predicción real del modelo
            try:
                prob, *_ = brain.train_and_predict_calibrated(t, df_eng, day)
            except Exception as e:
                prob = 0
            if prob > 0.55:
                candidates.append((t, price, prob))
        # Seleccionar top 1 recomendación
        if candidates:
            candidates.sort(key=lambda x: x[2], reverse=True)
            ticker, price, prob = candidates[0]
            # Si no tenemos posición, comprar con todo el cash
            if ticker not in portfolio['positions'] and portfolio['cash'] > 0:
                shares = int(portfolio['cash'] // price)
                if shares > 0:
                    portfolio['positions'][ticker] = {'shares': shares, 'buy_price': price}
                    portfolio['cash'] -= shares * price
                    portfolio['history'].append({'date': day, 'action': 'buy', 'ticker': ticker, 'shares': shares, 'price': price})
        # Simular venta al final del año
        if day == trading_days[-1]:
            for ticker, pos in portfolio['positions'].items():
                last_price = all_data[ticker].loc[day]['close']
                portfolio['cash'] += pos['shares'] * last_price
                portfolio['history'].append({'date': day, 'action': 'sell', 'ticker': ticker, 'shares': pos['shares'], 'price': last_price})
            portfolio['positions'] = {}

    # 4. Resultados
    final_value = portfolio['cash']
    print(f"\nSimulación anual terminada. Capital final: ${final_value:.2f}")

    # Guardar historial
    hist_df = pd.DataFrame(portfolio['history'])
    hist_path = os.path.join(INVESTMENT_DIR, 'simulation_history.csv')
    hist_df.to_csv(hist_path, index=False)
    print(f"Historial de operaciones guardado en {hist_path}")

    # Generar reporte profesional
    report_path = os.path.join(INVESTMENT_DIR, 'simulation_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"Simulación anual AU Stock\n")
        f.write(f"Periodo: {START_DATE} a {END_DATE}\n")
        f.write(f"Capital inicial: ${START_CAPITAL}\n")
        f.write(f"Capital final: ${final_value:.2f}\n")
        f.write(f"Total operaciones: {len(hist_df)}\n")
        f.write(f"\n---\nResumen de operaciones:\n")
        for _, row in hist_df.iterrows():
            f.write(f"{row['date']} | {row['action'].upper()} | {row['ticker']} | Shares: {row['shares']} | Price: ${row['price']:.2f}\n")
        f.write(BRANDING)
    print(f"Reporte profesional guardado en {report_path}")

if __name__ == "__main__":
    main()

START_CAPITAL = 5000
START_DATE = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
END_DATE = datetime.now().strftime('%Y-%m-%d')

# 1. Cargar tickers y datos históricos
TICKERS = load_au_tickers()
brain = TitanBrain()
all_data = brain.get_data(TICKERS, start_date=START_DATE)

# 2. Inicializar portafolio
portfolio = {'cash': START_CAPITAL, 'positions': {}, 'history': []}

# 3. Simulación diaria
trading_days = None
for t, df in all_data.items():
    if trading_days is None or len(df) > len(trading_days):
        trading_days = df.index

trading_days = pd.to_datetime(sorted(set(trading_days)))
# Limitar a solo el último año
trading_days = trading_days[(trading_days >= pd.to_datetime(START_DATE)) & (trading_days <= pd.to_datetime(END_DATE))]

for day in trading_days:
    # Generar recomendaciones para el día
    candidates = []
    for t, df in all_data.items():
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
        if price <= sma200 or adx < 25 or rsi > 75:
            continue
        # Predicción real del modelo
        try:
            prob, *_ = brain.train_and_predict_calibrated(t, df_eng, day)
        except Exception as e:
            prob = 0
        if prob > 0.55:
            candidates.append((t, price, prob))
    # Seleccionar top 1 recomendación
    if candidates:
        candidates.sort(key=lambda x: x[2], reverse=True)
        ticker, price, prob = candidates[0]
        # Si no tenemos posición, comprar con todo el cash
        if ticker not in portfolio['positions'] and portfolio['cash'] > 0:
            shares = int(portfolio['cash'] // price)
            if shares > 0:
                portfolio['positions'][ticker] = {'shares': shares, 'buy_price': price}
                portfolio['cash'] -= shares * price
                portfolio['history'].append({'date': day, 'action': 'buy', 'ticker': ticker, 'shares': shares, 'price': price})
    # Simular venta al final del año
    if day == trading_days[-1]:
        for ticker, pos in portfolio['positions'].items():
            last_price = all_data[ticker].loc[day]['close']
            portfolio['cash'] += pos['shares'] * last_price
            portfolio['history'].append({'date': day, 'action': 'sell', 'ticker': ticker, 'shares': pos['shares'], 'price': last_price})
        portfolio['positions'] = {}

# 4. Resultados
final_value = portfolio['cash']
print(f"\nSimulación anual terminada. Capital final: ${final_value:.2f}")

# Guardar historial
hist_df = pd.DataFrame(portfolio['history'])
hist_df.to_csv('investment/simulation_history.csv', index=False)
print("Historial de operaciones guardado en investment/simulation_history.csv")
