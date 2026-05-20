# backtest_au_alerts.py - Backtest de alertas AU Stock en las últimas 2 semanas
from core.brain import TitanBrain
import pandas as pd
from tqdm import tqdm
from datetime import datetime, timedelta
import os

TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN', None)
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', None)

WINDOW_DAYS = 365  # 1 año para test
TRAIN_DAYS = 3 * 365  # 3 años para entrenamiento

def backtest_alerts():
    print(f"\n🔬 Backtest de alertas AU Stock (Telegram) - {datetime.now()}")
    brain = TitanBrain()
    TICKERS = [
        'BHP.AX', 'CBA.AX', 'CSL.AX', 'WES.AX', 'NAB.AX', 'WBC.AX', 'ANZ.AX', 'MQG.AX', 'FMG.AX', 'WOW.AX',
        'TLS.AX', 'RIO.AX', 'GMG.AX', 'STO.AX', 'WDS.AX', 'XRO.AX', 'QBE.AX', 'ALL.AX', 'SCG.AX', 'COH.AX',
        'S32.AX', 'TCL.AX', 'BSL.AX', 'ORG.AX', 'NST.AX', 'SUN.AX', 'CPU.AX', 'RMD.AX', 'AMC.AX', 'MIN.AX',
        'PLS.AX', 'IGO.AX', 'TWE.AX', 'REA.AX', 'CAR.AX', 'SEK.AX', 'ASX.AX', 'SHL.AX', 'JHX.AX', 'QAN.AX'
    ]
    # Descargar solo los datos necesarios (últimos 2 semanas + buffer para entrenamiento)
    # Descargar 4 años de datos (3 años train + 1 año test)
    start_date = (datetime.now() - timedelta(days=WINDOW_DAYS + TRAIN_DAYS)).strftime('%Y-%m-%d')
    full_data = brain.get_data(TICKERS, start_date)
    from core.recommendation import get_recommendations
    print(f"\n🔬 Backtest de inversión AU Stock (modo investment) - {datetime.now()}")
    buy_recommendations, sell_recommendations, positions_status, latest_date, discard_log = get_recommendations(tickers=TICKERS, start_date=(datetime.now() - timedelta(days=WINDOW_DAYS)).strftime('%Y-%m-%d'))
    resumen = [
        f"Periodo simulado: {WINDOW_DAYS} días",
        f"Tickers analizados: {len(TICKERS)}",
        f"Fecha final: {latest_date}",
        f"Total recomendaciones de compra: {len(buy_recommendations)}",
        f"Total recomendaciones de venta: {len(sell_recommendations)}"
    ]
    with open("report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(resumen)+"\n\n")
        f.write("--- Recomendaciones de compra ---\n")
        for rec in buy_recommendations:
            f.write(f"{rec['ticker']} Prob={rec['prob']:.2f} Precio={rec['price']:.2f} SL={rec['sl_pct']:.2f} TP={rec['tp_pct']:.2f} Regime={rec['regime']}\n")
        f.write("\n--- Recomendaciones de venta ---\n")
        for rec in sell_recommendations:
            f.write(f"{rec['ticker']} {rec['reason']} Precio={rec['price']:.2f} PnL={rec['pnl']:.2f} Pct={rec['pnl_pct']:.2f}\n")
        f.write("\n--- Estado de posiciones ---\n")
        for pos in positions_status:
            f.write(f"{pos['ticker']} Shares={pos['shares']} Buy={pos['buy_price']:.2f} Current={pos['current_price']:.2f} PnL={pos['unrealized_pnl']:.2f} Pct={pos['growth_pct']:.2f} Signal={pos['sell_signal']}\n")
        f.write("\n--- Descartes ---\n")
        for d in discard_log:
            f.write(f"{d['ticker']} {d['motivo']} Prob={d['prob_win']:.2f} ATR={d['ATR_Pct']:.2f}\n")
    print("\n".join(resumen))

if __name__ == "__main__":
    backtest_alerts()
