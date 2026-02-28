# daily_recommend_au.py - Recomendaciones diarias para AU Stock

# Recomendaciones AU Stock usando TitanBrain y flujo centralizado

from core.brain import TitanBrain
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import os

import os
def load_au_tickers():
    # Usar todos los símbolos con CSV descargado
    return [f.split('.')[0]+'.AX' for f in os.listdir('au_stock_data') if f.endswith('.csv')]

def get_daily_recommendations():
    print(f"\n🔮 Recomendaciones AU Stock - {datetime.now()}")
    brain = TitanBrain()
    TICKERS = load_au_tickers()
    full_data = brain.get_data(TICKERS)
    candidates = []
    for t in tqdm(TICKERS):
        try:
            if t in full_data:
                df = full_data[t].copy()
            else:
                continue
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            df.index = df.index.normalize()
            df_eng = brain.engineer_features(df)
            if df_eng is None:
                continue
            last_date = df_eng.index[-1]
            row = df_eng.iloc[-1]
            price = row['Close']
            sma200 = row.get('SMA_200', 0)
            adx = row.get('ADX', 0)
            rsi = row.get('RSI', 50)
            tendencia_fuerte = (adx >= 30) and (sma200 > 0) and (price > sma200)
            if price <= sma200 or adx < 25 or rsi > 75:
                continue
            next_day = last_date + pd.Timedelta(days=1)
            prob, *_ = brain.train_and_predict_calibrated(t, df_eng, next_day)
            if prob > 0.6 and tendencia_fuerte:
                candidates.append({
                    'Ticker': t,
                    'Precio': price,
                    'Prob_IA': prob * 100,
                    'RSI': rsi,
                    'ADX': adx,
                    'Fecha_Datos': str(last_date.date())
                })
        except Exception:
            continue
    if not candidates:
        print("⚠️  No hay señales de compra claras.")
    else:
        candidates.sort(key=lambda x: x['Prob_IA'], reverse=True)
        print(f"{'TICKER':<10} {'PRECIO':<10} {'CONF.(%)':<10} {'RSI':<8} {'ADX':<8}")
        print("-" * 50)
        for c in candidates[:3]:
            print(f"{c['Ticker']:<10} ${c['Precio']:<9.2f} {c['Prob_IA']:<9.1f}% {c['RSI']:<8.1f} {c['ADX']:<8.1f}")
        print(f"\n💡 RECOMENDACIÓN TOP: {candidates[0]['Ticker']} (Confianza: {candidates[0]['Prob_IA']:.1f}%)")

if __name__ == "__main__":
    get_daily_recommendations()
