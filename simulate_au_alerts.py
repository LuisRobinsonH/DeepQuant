
# ALERTA AU STOCK: Solo envía el daily recommendation filtrado por Telegram
import pandas as pd
from core.recommendation import get_recommendations
from telegram import Bot
import os
from datetime import datetime

TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN', '8595554468:AAF_a9CR9zj2_352MLy6p_dHkfq20pKE_Xg')
TELEGRAM_CHAT_ID = int(os.getenv('TELEGRAM_CHAT_ID', '6351372403'))
CAPITAL = 5000.0
MIN_BUY = 500.0

import csv
RECOMMEND_LOG = 'last_recommendations.csv'

def build_alert_message(buys):
    msg = ''
    prev = {}
    prev_date = None
    # Cargar recomendaciones previas si existen
    if os.path.exists(RECOMMEND_LOG):
        prev_date = datetime.fromtimestamp(os.path.getmtime(RECOMMEND_LOG)).strftime('%Y-%m-%d %H:%M')
        with open(RECOMMEND_LOG, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                prev[row['ticker']] = row
    # Ordenar por probabilidad descendente
    buys = sorted(buys, key=lambda x: x.get('prob', 0), reverse=True)
    # Resumen de señales
    prev_set = set(prev.keys())
    curr_set = set(b['ticker'] for b in buys)
    nuevas = curr_set - prev_set
    repetidas = curr_set & prev_set
    eliminadas = prev_set - curr_set
    resumen = f"Nuevas: {len(nuevas)} | Repetidas: {len(repetidas)} | Eliminadas: {len(eliminadas)}"
    if not buys:
        return 'No hay oportunidad de compra para AU Stock hoy.'
    msg += f'🚀 Oportunidad de compra AU Stock ({datetime.now().strftime("%Y-%m-%d")})\n'
    if prev_date:
        msg += f'Última alerta: {prev_date}\n'
    msg += resumen + "\n\n"
    rows_to_save = []
    for b in buys:
        symbol = b['ticker']
        prob = b.get('prob', 0)
        price = b.get('price', 0)
        tp_pct = b.get('tp_pct', 0)
        sl_pct = b.get('sl_pct', 0)
        shares = int(max(MIN_BUY, CAPITAL * 0.02) // price) if price > 0 else 0
        take_profit = price * (1 + tp_pct)
        stop_loss = price * (1 - sl_pct)
        # Cambio respecto a recomendación anterior
        price_prev = prev[symbol]['price'] if symbol in prev else None
        pct_change = ((float(price) - float(price_prev)) / float(price_prev) * 100) if price_prev and float(price_prev) > 0 else None
        # Mostrar solo si cambio >±1% o es nueva
        show_change = pct_change is None or abs(pct_change) >= 1 or symbol in nuevas
        if not show_change:
            continue
        emoji = ''
        if pct_change is not None:
            if pct_change > 1:
                emoji = '🟢'
            elif pct_change < -1:
                emoji = '🔴'
            else:
                emoji = '⚪'
        else:
            emoji = '🆕'
        msg += f"{emoji} {symbol} | Prob: {prob:.0%}\n"
        msg += f"  Precio actual: ${price:.2f}"
        if price_prev:
            msg += f" | Anterior: ${float(price_prev):.2f}"
        if pct_change is not None:
            msg += f" | Cambio: {pct_change:+.2f}%"
        msg += "\n"
        msg += f"  Sugerencia: {shares} shares\n"
        msg += f"  Take Profit: ${take_profit:.2f} | Stop Loss: ${stop_loss:.2f}\n"
        # Guardar para log
        rows_to_save.append({
            'ticker': symbol,
            'prob': prob,
            'price': price,
            'tp_pct': tp_pct,
            'sl_pct': sl_pct
        })
    # Guardar recomendaciones actuales
    with open(RECOMMEND_LOG, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['ticker','prob','price','tp_pct','sl_pct'])
        writer.writeheader()
        writer.writerows(rows_to_save)
    return msg

async def main():
    # Solo recomendaciones de compra para hoy
    buys, _, _, _, _ = get_recommendations()
    msg = build_alert_message(buys)
    bot = Bot(token=TELEGRAM_TOKEN)
    try:
        response = await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=msg)
        print('Mensaje enviado a Telegram.')
        print('Respuesta Telegram:', response)
    except Exception as e:
        print('Error al enviar mensaje a Telegram:', e)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
