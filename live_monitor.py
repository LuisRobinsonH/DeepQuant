# live_monitor.py

import time
import datetime
from core.recommendation import get_recommendations
from core.telegram_alert import send_telegram_alert

def run_hourly_check():
    print(f"\n⌚ INICIANDO ESCANEO: {datetime.datetime.now().strftime('%H:%M:%S')}")
    
    brain = TitanBrain()
    risk = RiskEngine()
    
    # Lista reducida para velocidad (Top Liquid Stocks)
    tickers = ['BHP.AX', 'CBA.AX', 'CSL.AX', 'WES.AX', 'NAB.AX', 'RIO.AX', 'FMG.AX', 'XRO.AX', 'WDS.AX', 'QBE.AX', 'NST.AX', 'PLS.AX']
    
    # 1. Descarga Rápida (Solo último año para features, no 5 años)
    print("📡 Verificando precios de mercado...")
    # Usamos start_date reciente para que sea rápido
    one_year_ago = (datetime.datetime.now() - datetime.timedelta(days=365)).strftime('%Y-%m-%d')
    data = brain.get_data(tickers, start_date=one_year_ago)
    
    # 2. Análisis

    def run_hourly_check():
        print(f"\n⌚ INICIANDO ESCANEO: {datetime.datetime.now().strftime('%H:%M:%S')}")
        # Usar solo tickers principales para velocidad
        tickers = ['BHP.AX', 'CBA.AX', 'CSL.AX', 'WES.AX', 'NAB.AX', 'RIO.AX', 'FMG.AX', 'XRO.AX', 'WDS.AX', 'QBE.AX', 'NST.AX', 'PLS.AX']
        one_year_ago = (datetime.datetime.now() - datetime.timedelta(days=365)).strftime('%Y-%m-%d')
        buy_recommendations, _, _, _ = get_recommendations(tickers=tickers, start_date=one_year_ago, min_prob=0.7)
        if buy_recommendations:
            for rec in buy_recommendations:
                print(f"🔥 SEÑAL DETECTADA: {rec['ticker']} ({rec['prob']:.1%})")
                msg = (f"🚨 TITAN ALERT: BUY {rec['ticker']}\n"
                       f"Precio: ${rec['price']:.2f}\n"
                       f"Probabilidad: {rec['prob']:.1%}\n"
                       f"EV: {rec['ev']:.2f}\n"
                       f"Régimen: {rec.get('regime','')}\n")
                send_telegram_alert(msg)
        else:
            print("No se detectaron señales de compra con alta confianza.")
        print("✅ Escaneo finalizado. Durmiendo 60 minutos...")