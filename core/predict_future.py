# predict_future.py
import pandas as pd
from core.brain import TitanBrain
import warnings
from tqdm import tqdm
import datetime

warnings.filterwarnings('ignore')

def get_market_opportunities():
    print("\n🔮 TITAN AI: ESCANEANDO OPORTUNIDADES PARA MAÑANA...")
    print("=====================================================")
    
    brain = TitanBrain()
    
    # Usar tickers centralizados
    from core.brain import load_au_tickers
    TICKERS = load_au_tickers()
    print(f"📡 Descargando datos en tiempo real ({len(TICKERS)} activos)...")
    full_data = brain.get_data(TICKERS)
    
    candidates = []
    
    print("🧠 La IA está analizando patrones recientes...")
    for t in tqdm(TICKERS):
        try:
            if t in full_data.columns:
                df = full_data[t].copy()
            else: continue
            
            if df.index.tz is not None: df.index = df.index.tz_localize(None)
            df.index = df.index.normalize()
            
            df_eng = brain.engineer_features(df)
            if df_eng is None: continue
            
            # Último día disponible
            last_date = df_eng.index[-1]
            row = df_eng.iloc[-1]
            
            price = row['Close']
            sma200 = row.get('SMA_200', 0)
            adx = row.get('ADX', 0)
            rsi = row.get('RSI', 50)
            
            # Filtros
            if price <= sma200: continue
            if adx < 25: continue
            if rsi > 75: continue
            
            # Predicción para el siguiente día hábil
            next_day = last_date + pd.Timedelta(days=1)
            prob, score = brain.train_and_predict(t, df_eng, next_day)
            
            if prob > 0.55: # Mostrar todo lo que tenga > 55%
                candidates.append({
                    'Ticker': t,
                    'Precio': price,
                    'Prob_IA': prob * 100,
                    'Score': int(score),
                    'RSI': rsi,
                    'ADX': adx,
                    'Fecha_Datos': str(last_date.date())
                })
                
        except Exception:
            continue

    print("\n📋 RESULTADOS DEL ESCÁNER (Predicción Próxima Sesión):")
    
    if not candidates:
        print("⚠️  No hay señales de compra claras. Mercado peligroso o lateral.")
    else:
        candidates.sort(key=lambda x: x['Prob_IA'], reverse=True)
        
        # Formato bonito
        print(f"{'TICKER':<10} {'PRECIO':<10} {'CONF.(%)':<10} {'RSI':<8} {'ADX':<8}")
        print("-" * 50)
        for c in candidates:
            print(f"{c['Ticker']:<10} ${c['Precio']:<9.2f} {c['Prob_IA']:<9.1f}% {c['RSI']:<8.1f} {c['ADX']:<8.1f}")
        
        print("\n💡 RECOMENDACIÓN TOP:")
        top = candidates[0]
        print(f"👉 {top['Ticker']} (Confianza: {top['Prob_IA']:.1f}%)")

if __name__ == "__main__":
    get_market_opportunities()