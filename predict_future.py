# predict_future.py - FIXED VERSION
import pandas as pd
from core.brain import TitanBrain
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')


def get_market_opportunities():
    """
    Scan ASX market for trading opportunities for next session.
    
    Uses TitanBrain's calibrated predictions to identify high-probability setups.
    """
    print("\n" + "="*60)
    print("🔮 TITAN AI: ESCANEANDO OPORTUNIDADES PARA MAÑANA")
    print("="*60)
    
    brain = TitanBrain()
    
    # ASX 50 Universe
    TICKERS = [
        'BHP.AX', 'CBA.AX', 'CSL.AX', 'WES.AX', 'NAB.AX', 'WBC.AX', 'ANZ.AX', 
        'MQG.AX', 'FMG.AX', 'WOW.AX', 'TLS.AX', 'RIO.AX', 'GMG.AX', 'STO.AX', 
        'WDS.AX', 'XRO.AX', 'QBE.AX', 'ALL.AX', 'SCG.AX', 'COH.AX', 'S32.AX', 
        'TCL.AX', 'BSL.AX', 'ORG.AX', 'NST.AX', 'SUN.AX', 'CPU.AX', 'RMD.AX', 
        'AMC.AX', 'MIN.AX', 'PLS.AX', 'IGO.AX', 'TWE.AX', 'REA.AX', 'CAR.AX', 
        'SEK.AX', 'ASX.AX', 'SHL.AX', 'JHX.AX', 'QAN.AX'
    ]

    print(f"\n📡 Descargando datos recientes ({len(TICKERS)} activos)...")
    full_data = brain.get_data(TICKERS, start_date="2020-01-01")
    
    candidates = []
    
    print("🧠 Analizando patrones con IA calibrada...")
    for t in tqdm(TICKERS):
        try:
            # Extract ticker data
            if t in full_data.columns:
                df = full_data[t].copy()
            else:
                continue
            
            # Normalize timezone
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            df.index = df.index.normalize()
            
            # Engineer features
            df_eng = brain.engineer_features(df)
            if df_eng is None or len(df_eng) < 300:
                continue
            
            # Get last available date
            last_date = df_eng.index[-1]
            row = df_eng.iloc[-1]
            
            price = row['Close']
            sma200 = row.get('SMA_200', 0)
            adx = row.get('ADX', 0)
            rsi = row.get('rsi', 50)
            
            # --- FILTERS ---
            # 1. Trend: Price above 200 SMA (uptrend)
            if price <= sma200:
                continue
            
            # 2. Trend Strength: ADX > 25 (strong trend)
            if adx < 25:
                continue
            
            # 3. Not Overbought: rsi < 75
            if rsi > 75:
                continue
            
            # --- AI PREDICTION ---
            # FIXED: Use correct function name
            next_day = last_date + pd.Timedelta(days=1)
            prob, features, atr_pct = brain.train_and_predict_calibrated(
                t, df_eng, next_day
            )
            
            # Calculate confidence score (0-100)
            score = int(prob * 100)
            
            # Only show predictions with > 55% confidence
            if prob > 0.55:
                candidates.append({
                    'Ticker': t,
                    'Precio': price,
                    'Prob_IA': prob * 100,
                    'Score': score,
                    'rsi': rsi,
                    'ADX': adx,
                    'ATR': atr_pct * 100,  # As percentage
                    'Fecha_Datos': str(last_date.date()),
                    'Features': len(features)
                })
                
        except Exception as e:
            # Silently skip errors (most are data issues)
            continue

    # --- DISPLAY RESULTS ---
    print("\n" + "="*60)
    print("📋 RESULTADOS DEL ESCÁNER")
    print("="*60)
    
    if not candidates:
        print("\n⚠️  No hay señales de compra claras.")
        print("    El mercado puede estar lateral o sin momentum.")
        print("\n💡 Sugerencia: Espera mejor setup o revisa filtros.")
    else:
        # Sort by confidence
        candidates.sort(key=lambda x: x['Prob_IA'], reverse=True)
        
        print(f"\n✅ Encontradas {len(candidates)} oportunidades\n")
        
        # Header
        print(f"{'TICKER':<10} {'PRECIO':<10} {'CONF.(%)':<10} {'rsi':<8} {'ADX':<8} {'ATR%':<8}")
        print("-" * 62)
        
        # Display all candidates
        for c in candidates:
            print(
                f"{c['Ticker']:<10} "
                f"${c['Precio']:<9.2f} "
                f"{c['Prob_IA']:<9.1f}% "
                f"{c['RSI']:<8.1f} "
                f"{c['ADX']:<8.1f} "
                f"{c['ATR']:<7.2f}%"
            )
        
        # Highlight top pick
        print("\n" + "="*60)
        print("💡 RECOMENDACIÓN TOP")
        print("="*60)
        
        top = candidates[0]
        print(f"\n🎯 {top['Ticker']}")
        print(f"   • Precio:         ${top['Precio']:.2f}")
        print(f"   • Confianza IA:   {top['Prob_IA']:.1f}%")
        print(f"   • RSI:            {top['RSI']:.1f}")
        print(f"   • ADX (Trend):    {top['ADX']:.1f}")
        print(f"   • Volatilidad:    {top['ATR']:.2f}%")
        print(f"   • Features:       {top['Features']}")
        print(f"   • Datos hasta:    {top['Fecha_Datos']}")
        
        # Risk suggestion
        suggested_stop = top['Precio'] * (1 - (top['ATR'] / 100) * 2)
        suggested_target = top['Precio'] * (1 + (top['ATR'] / 100) * 4)
        
        print(f"\n📊 Niveles Sugeridos:")
        print(f"   • Stop Loss:      ${suggested_stop:.2f} ({-2 * top['ATR']:.1f}%)")
        print(f"   • Take Profit:    ${suggested_target:.2f} ({+4 * top['ATR']:.1f}%)")
        
        print("\n⚠️  DISCLAIMER: Esta es una predicción algorítmica.")
        print("    Siempre usa gestión de riesgo y no inviertas más de lo que puedas perder.")


if __name__ == "__main__":
    get_market_opportunities()