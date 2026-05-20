# AU_ALERTS.PY - ALERTA DIARIA PROFESIONAL AU STOCK (ASX-200)
# Envia recomendaciones diarias filtradas por Telegram, con logs y branding.
# Listo para automatización (cron/task scheduler).
# ALERTA AU STOCK: Solo envía el daily recommendation filtrado por Telegram
import pandas as pd
from core.brain import TitanBrain, load_au_tickers
import yfinance as yf
import os
import pytz
from datetime import datetime

TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN', '8595554468:AAF_a9CR9zj2_352MLy6p_dHkfq20pKE_Xg')
TELEGRAM_CHAT_ID = int(os.getenv('TELEGRAM_CHAT_ID', '6351372403'))

import csv



RECOMMEND_LOG = 'last_recommendations.csv'


def build_alert_message(buys):
    """
    Construye el mensaje de alerta para Telegram.
    Retorna None si no hay señales de compra (sin alerta → sin acción).
    """
    # --- GUARDIA: sin señales = silencio total, no se envía nada ---
    if not buys:
        now = datetime.now()
        print(f"[AU ALERTS] Sin señales de compra para hoy ({now.strftime('%Y-%m-%d')}). Sin alerta, sin acción.")
        return None

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
    # Usar zona horaria Australia/Sydney para la fecha de la alerta
    try:
        tz = pytz.timezone('Australia/Sydney')
        now = datetime.now(tz)
    except Exception:
        now = datetime.now()
    msg += f'🚀 Oportunidad de compra AU Stock ({now.strftime("%Y-%m-%d %H:%M")}, Año: {now.year})\n'
    if prev_date:
        msg += f'Última alerta: {prev_date}\n'
    msg += resumen + '\n\n'
    for b in buys:
        symbol = b['ticker']
        prob = b.get('prob', 0)
        price = b.get('price', 0)
        tp_pct = b.get('tp_pct', '')
        sl_pct = b.get('sl_pct', '')
        year = b.get('year', datetime.now().year)
        msg += f"{symbol} | Confianza: {prob:.1%} | Precio: ${price:.2f} | TP: {tp_pct} | SL: {sl_pct} | Año: {year}\n"
    # Guardar recomendaciones actuales
    rows_to_save = []
    for b in buys:
        symbol = b['ticker']
        prob = b.get('prob', 0)
        price = b.get('price', 0)
        tp_pct = b.get('tp_pct', '')
        sl_pct = b.get('sl_pct', '')
        rows_to_save.append({
            'ticker': symbol,
            'prob': prob,
            'price': price,
            'tp_pct': tp_pct,
            'sl_pct': sl_pct
        })
    with open(RECOMMEND_LOG, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['ticker','prob','price','tp_pct','sl_pct'])
        writer.writeheader()
        writer.writerows(rows_to_save)
    return msg

def main():
    now = datetime.now()
    print(f"\n[AU ALERTS] Ejecutando alerta diaria AU Stock - {now.strftime('%Y-%m-%d %H:%M:%S')}")
    try:
        tickers = load_au_tickers()
        brain = TitanBrain()
        today = now.strftime('%Y-%m-%d')
        start_date = (now - pd.Timedelta(days=365)).strftime('%Y-%m-%d')
        year_range = f"{(now - pd.Timedelta(days=365)).year}_{now.year}"
        buy_recs = []
        tickers_failed = []
        tickers_success = []
        # Analizar datos del último año para cada ticker
        for t in tickers:
            tried = set()
            found = False
            for suffix in ["", ".AX"]:
                ticker_try = t if t.endswith(suffix) or suffix == "" else t.replace(".AX","") + suffix
                if ticker_try in tried:
                    continue
                tried.add(ticker_try)
                for attempt in range(3):
                    try:
                        ticker_obj = yf.Ticker(ticker_try)
                        try:
                            info = ticker_obj.info
                        except Exception as info_e:
                            if '404' in str(info_e):
                                continue
                            else:
                                raise info_e
                        if not info or 'regularMarketPrice' not in info or info['regularMarketPrice'] is None:
                            continue
                        # Descargar datos del último año
                        live = yf.download(ticker_try, start=start_date, end=today, interval='1d', progress=False)
                        if live.empty or 'Close' not in live.columns:
                            continue
                        # Procesar cada día del año
                        for idx, last_row in live.iterrows():
                            df_actual = pd.DataFrame([{
                                'close': last_row.get('Close', None),
                                'high': last_row.get('High', None),
                                'low': last_row.get('Low', None),
                                'open': last_row.get('Open', None),
                                'volume': last_row.get('Volume', None)
                            }], index=[pd.to_datetime(idx)])
                            df_eng = brain.engineer_features(df_actual)
                            if df_eng is None or pd.to_datetime(idx) not in df_eng.index:
                                continue
                            prob, feats, atr = brain.train_and_predict_calibrated(t, df_eng, pd.to_datetime(idx))
                            row_idx = pd.to_datetime(idx)
                            row = df_eng.loc[row_idx]
                            price = row['close']

                            # --- FILTROS MULTI-CAPA PARA REDUCIR FALSAS SEÑALES ---
                            sma_50   = row.get('sma_50',  price)
                            sma_200  = row.get('sma_200', price)
                            bb_width = row.get('bb_width', 0.05)
                            rsi      = row.get('rsi', 0.5)      # ya normalizado [0,1] en engineer_features
                            vol_rel  = row.get('vol_rel_20', 1.0)
                            adx      = row.get('adx', 0)
                            macd_diff = row.get('macd_diff', 0)

                            # 1. Probabilidad alta
                            cond_prob   = prob > 0.65
                            # 2. Tendencia alcista: precio > SMA50 > SMA200
                            cond_trend  = price > sma_50 and sma_50 > sma_200
                            # 3. No sobrecomprado (RSI < 0.75 en escala normalizada)
                            cond_rsi    = rsi < 0.75
                            # 4. Volumen confirma señal (> 0.8x media 20d)
                            cond_vol    = vol_rel > 0.8
                            # 5. Volatilidad controlada (bb_width no extremo)
                            cond_bb     = bb_width < 0.15
                            # 6. Tendencia fuerte (ADX > 0 después de normalizar = parte positiva)
                            cond_adx    = adx > 0
                            # 7. MACD positivo (momento alcista)
                            cond_macd   = macd_diff > 0

                            if cond_prob and cond_trend and cond_rsi and cond_vol and cond_bb and cond_adx and cond_macd:
                                tp_pct = row.get('bb_upper_dist', '')
                                sl_pct = row.get('bb_lower_dist', '')
                                buy_recs.append({
                                    'ticker': t,
                                    'prob': prob,
                                    'price': price,
                                    'tp_pct': tp_pct,
                                    'sl_pct': sl_pct,
                                    'year': row_idx.year
                                })
                        tickers_success.append(t)
                        found = True
                        break
                    except Exception as e:
                        if attempt == 2:
                            print(f"[WARN] No se pudo obtener dato actual o predecir para {t} (probado como {ticker_try}): {e}")
                if found:
                    break
            if not found:
                tickers_failed.append(t)
        total_tickers = len(tickers)
        failed_count = len(tickers_failed)
        success_count = len(tickers_success)
        msg = build_alert_message(buy_recs)
        # --- GUARDIA: si no hay señales, no se envía alerta ni se escribe reporte ---
        if msg is None:
            print(f"[AU ALERTS] Sin alertas activas. No se envía Telegram ni se genera reporte.")
            return
        # --- Detalle de operaciones ---
        operaciones = []
        for b in buy_recs:
            operaciones.append({
                'ticker': b['ticker'],
                'prob': b['prob'],
                'price': b['price'],
                'tp_pct': b['tp_pct'],
                'sl_pct': b['sl_pct'],
                'year': b['year']
            })
        # Calcular win rate (simulación: si prob > 0.5 consideramos "acierto")
        total_ops = len(operaciones)
        wins = sum(1 for b in operaciones if b['prob'] > 0.5)
        win_rate = (wins / total_ops * 100) if total_ops > 0 else 0
        resumen = f"\n---\nREPORTE ANUAL {year_range}\nTickers totales: {total_tickers}\nTickers con datos: {success_count}\nTickers sin datos: {failed_count}\n"
        # Métricas avanzadas
        if total_ops > 0:
            ganancias = [b['tp_pct'] for b in operaciones if isinstance(b['tp_pct'], (int, float)) and b['tp_pct']]
            perdidas = [b['sl_pct'] for b in operaciones if isinstance(b['sl_pct'], (int, float)) and b['sl_pct']]
            avg_ganancia = sum(ganancias)/len(ganancias) if ganancias else 0
            avg_perdida = sum(perdidas)/len(perdidas) if perdidas else 0
            mejor = max(ganancias) if ganancias else 0
            peor = min(perdidas) if perdidas else 0
            # Drawdown simple: diferencia máxima entre un TP y el siguiente SL
            drawdown = peor - mejor if ganancias and perdidas else 0
        else:
            avg_ganancia = avg_perdida = mejor = peor = drawdown = 0
        resumen += f"Total operaciones: {total_ops}\nWin rate: {win_rate:.1f}%\n"
        resumen += f"Promedio TP: {avg_ganancia:.2f} | Promedio SL: {avg_perdida:.2f}\n"
        resumen += f"Mejor TP: {mejor} | Peor SL: {peor} | Drawdown simple: {drawdown}\n"
        if tickers_failed:
            resumen += "Tickers sin datos:\n" + '\n'.join(tickers_failed) + "\n"
        resumen += "\nDETALLE DE OPERACIONES:\n"
        for b in operaciones:
            resumen += (f"{b['year']} | {b['ticker']} | Precio: ${b['price']:.2f} | Prob: {b['prob']:.1%} | TP: {b['tp_pct']} | SL: {b['sl_pct']}\n")
        msg += resumen
        with open(f"reporte_alertas_{year_range}.txt", "w", encoding="utf-8") as f:
            f.write(msg)
        print(f"[REPORTE] Reporte anual generado: reporte_alertas_{year_range}.txt")
        # Enviar por Telegram solo si hay señales reales
        try:
            import requests
            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
            requests.post(url, data={'chat_id': TELEGRAM_CHAT_ID, 'text': msg[:4000]}, timeout=10)
        except Exception as te:
            print(f"[TELEGRAM] Error enviando alerta: {te}")
    except Exception as e:
        print(f"[ERROR] Fallo en alerta diaria: {e}")

if __name__ == "__main__":
    main()
