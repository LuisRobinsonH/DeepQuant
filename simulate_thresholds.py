# SIMULATE_THRESHOLDS.PY - Simulate different probability thresholds and calculate accuracy/gains
import pandas as pd
from core.brain import TitanBrain, load_au_tickers
import yfinance as yf
import numpy as np
from datetime import datetime

THRESHOLDS = [0.4, 0.5, 0.6, 0.7]
START_DATE = "2025-01-01"
END_DATE = "2026-01-01"

results = []

tickers = load_au_tickers()
brain = TitanBrain()

def simulate_for_threshold(threshold):
    total_signals = 0
    total_gains = 0
    total_trades = 0
    for t in tickers:
        try:
            df = yf.download(t, start=START_DATE, end=END_DATE, interval='1d', progress=False)
            if df.empty or 'Close' not in df.columns:
                continue
            df = df.rename(columns={
                'Close': 'close', 'High': 'high', 'Low': 'low', 'Open': 'open', 'Volume': 'volume'
            })
            df_eng = brain.engineer_features(df)
            if df_eng is None:
                continue
            for idx, row in df_eng.iterrows():
                prob, feats, atr = brain.train_and_predict_calibrated(t, df_eng, idx)
                if prob > threshold:
                    # Simulate buy at close, sell at next day's close
                    if idx in df.index and (df.index.get_loc(idx) + 1) < len(df.index):
                        next_idx = df.index[df.index.get_loc(idx) + 1]
                        buy_price = row['close']
                        sell_price = df.loc[next_idx]['close']
                        gain = (sell_price - buy_price) / buy_price
                        total_gains += gain
                        total_trades += 1
                    total_signals += 1
        except Exception as e:
            continue
    avg_gain = total_gains / total_trades if total_trades > 0 else 0
    return {
        'threshold': threshold,
        'signals': total_signals,
        'trades': total_trades,
        'avg_gain': avg_gain
    }

for th in THRESHOLDS:
    res = simulate_for_threshold(th)
    results.append(res)
    print(f"Threshold: {th} | Signals: {res['signals']} | Trades: {res['trades']} | Avg Gain: {res['avg_gain']:.4f}")

# Save results
pd.DataFrame(results).to_csv('simulation_results.csv', index=False)
print("Simulation complete. Results saved to simulation_results.csv")
