# daily_recommend.py - TITAN AI: DAILY TRADING RECOMMENDATIONS

import pandas as pd
import json
import os
from datetime import datetime
from core.recommendation import get_recommendations

def get_daily_recommendations():
    print("\n🔮 TITAN AI: RECOMENDACIONES DIARIAS DE TRADING")
    print("="*60)
    buy_recommendations, sell_recommendations, positions_status, latest_date = get_recommendations()

    # --- GUARDIA PRINCIPAL: Sin alertas = sin acción ---
    if not buy_recommendations and not sell_recommendations:
        print("\n✅ Sin alertas activas hoy. No se toma ninguna acción.")
        print("   Principio de riesgo: si no hay señal, no operamos.")
        return

    # --- Enviar alertas por Telegram solo si hay recomendaciones ---
    from core.telegram_alert import send_telegram_alert
    if buy_recommendations:
        for rec in buy_recommendations:
            msg = (f"🚨 TITAN ALERT: BUY {rec['ticker']}\n"
                   f"Precio: ${rec['price']:.2f}\n"
                   f"Probabilidad: {rec['prob']:.1%}\n"
                   f"EV: {rec['ev']:.2f}\n"
                   f"Régimen: {rec.get('regime','')}\n")
            send_telegram_alert(msg)
    if sell_recommendations:
        for rec in sell_recommendations:
            msg = (f"🚨 TITAN ALERT: SELL {rec['ticker']}\n"
                   f"Precio: ${rec['price']:.2f}\n"
                   f"Razón: {rec.get('reason','')}\n")
            send_telegram_alert(msg)

    def prompt_add_buys_to_portfolio(buy_recommendations, portfolio):
        print("\n¿Deseas agregar alguna de las siguientes compras a tu portafolio?")
        for rec in buy_recommendations:
            ticker = rec['ticker']
            price = rec['price']
            print(f"  {ticker} a ${price:.2f} (Prob: {rec['prob']:.1%}, EV: {rec['ev']:.2f})")
            resp = input(f"¿Agregar {ticker} al portafolio? (s/n): ").strip().lower()
            if resp == 's':
                shares = input(f"¿Cuántas acciones de {ticker} compraste?: ").strip()
                try:
                    shares = int(shares)
                except:
                    print("Cantidad inválida, se omite.")
                    continue
                sl_pct = rec.get('sl_pct', 0.05)
                tp_pct = rec.get('tp_pct', 0.10)
                stop_price = price * (1 - sl_pct)
                take_profit_price = price * (1 + tp_pct)
                portfolio[ticker] = {
                    'shares': shares,
                    'buy_price': price,
                    'highest_price': price,
                    'stop_price': stop_price,
                    'take_profit_price': take_profit_price,
                    'atr_at_buy': 0.0
                }
                print(f"✔️  {ticker} agregado al portafolio.")
        with open('current_portfolio.json', 'w') as f:
            json.dump(portfolio, f, default=str)

    # Cargar portafolio actual
    portfolio = {}
    if os.path.exists('current_portfolio.json'):
        try:
            with open('current_portfolio.json', 'r') as f:
                portfolio = json.load(f)
                # Convert string keys back to proper types
                for t, pos in portfolio.items():
                    pos['shares'] = int(pos['shares'])
                    pos['buy_price'] = float(pos['buy_price'])
                    pos['highest_price'] = float(pos['highest_price'])
                    pos['stop_price'] = float(pos['stop_price'])
                    pos['take_profit_price'] = float(pos['take_profit_price'])
                    pos['atr_at_buy'] = float(pos['atr_at_buy'])
        except:
            print("⚠️  No se pudo cargar el portafolio actual, asumiendo vacío.")
    
    # ASX Universe
    tickers = [
        'BHP.AX', 'CBA.AX', 'CSL.AX', 'WES.AX', 'NAB.AX', 'WBC.AX', 'ANZ.AX', 
        'MQG.AX', 'FMG.AX', 'TLS.AX', 'RIO.AX', 'GMG.AX', 'STO.AX', 'WDS.AX', 
        'QBE.AX', 'ALL.AX', 'SCG.AX', 'ORG.AX', 'NST.AX', 'SUN.AX', 'MIN.AX', 
        'PLS.AX', 'IGO.AX', 'TCL.AX', 'S32.AX', 'REA.AX', 'QAN.AX', 'RMD.AX',
        'AMC.AX', 'BSL.AX', 'CPU.AX', 'VCX.AX', 'ASX.AX', 'SHL.AX', 'JHX.AX'
    ]
    
    print(f"📡 Obteniendo datos más recientes para {len(tickers)} activos...")
    
    # Get latest data (last 2 years for feature calculation)
    raw_data = brain.get_data(tickers, start_date="2023-01-01")
    
    # Process data
    processed_data = {}
    valid_tickers = []
    
    for t in tickers:
        if t in raw_data.columns:
            df = brain.engineer_features(raw_data[t].copy())
            if df is not None and len(df) > 100:
                processed_data[t] = df
                valid_tickers.append(t)
    
    # Mostrar resultados y recomendaciones
    if latest_date:
        print(f"📅 Fecha de análisis: {latest_date.date()}")
    print("\n" + "="*60)
    print("🎯 RECOMENDACIONES PARA HOY")
    print("="*60)
    # Mostrar posiciones actuales
    if positions_status:
        print(f"\n📊 POSICIONES ACTUALES ({len(positions_status)}):")
        total_portfolio_value = sum(pos['current_value'] for pos in positions_status)
        total_unrealized_pnl = sum(pos['unrealized_pnl'] for pos in positions_status)
        for pos in positions_status:
            emoji = "📈" if pos['unrealized_pnl'] > 0 else "📉"
            signal = f" → {pos['sell_signal']}" if pos['sell_signal'] else ""
            print(f"  {emoji} {pos['ticker']} | {pos['shares']} acciones | Compra: ${pos['buy_price']:.2f} | Actual: ${pos['current_price']:.2f} | Crecimiento: {pos['growth_pct']:+.2%} | Valor: ${pos['current_value']:.0f}{signal}")
        print(f"\n  💰 Valor total portafolio: ${total_portfolio_value:,.0f}")
        print(f"  📊 P&L no realizado: ${total_unrealized_pnl:,.0f} ({(total_unrealized_pnl/total_portfolio_value) if total_portfolio_value else 0:+.2%})")
    # Guardar recomendaciones a Excel
    if latest_date:
        date_str = latest_date.strftime('%Y-%m-%d')
        filename = f'recommendations_{date_str}.xlsx'
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            if positions_status:
                pos_df = pd.DataFrame(positions_status)
                pos_df['Date'] = date_str
                pos_df = pos_df[['ticker', 'shares', 'buy_price', 'current_price', 'current_value', 'unrealized_pnl', 'growth_pct', 'sell_signal', 'stop_price', 'take_profit_price', 'Date']]
                pos_df.columns = ['Ticker', 'Acciones', 'Precio_Compra', 'Precio_Actual', 'Valor_Actual', 'P&L_NoRealizado', 'Crecimiento%', 'Señal_Venta', 'Stop_Loss', 'Take_Profit', 'Fecha']
                pos_df.to_excel(writer, sheet_name='Posiciones_Actuales', index=False)
            if buy_recommendations:
                buy_df = pd.DataFrame(buy_recommendations)
                buy_df['Date'] = date_str
                buy_df = buy_df[['ticker', 'price', 'prob', 'ev', 'sl_pct', 'tp_pct', 'regime', 'Date']]
                buy_df.columns = ['Ticker', 'Precio', 'Probabilidad', 'EV', 'SL%', 'TP%', 'Régimen', 'Fecha']
                buy_df.to_excel(writer, sheet_name='Compras', index=False)
            if sell_recommendations:
                sell_df = pd.DataFrame(sell_recommendations)
                sell_df['Date'] = date_str
                sell_df = sell_df[['ticker', 'action', 'reason', 'price', 'shares', 'pnl', 'pnl_pct', 'Date']]
                sell_df.columns = ['Ticker', 'Acción', 'Razón', 'Precio', 'Acciones', 'P&L', 'P&L%', 'Fecha']
                sell_df.to_excel(writer, sheet_name='Ventas', index=False)
        print(f"💾 Recomendaciones guardadas en: {filename}")
    # Mostrar ventas
    if sell_recommendations:
        print(f"\n📉 VENTAS RECOMENDADAS ({len(sell_recommendations)}):")
        for rec in sell_recommendations:
            emoji = "📈" if rec['pnl'] > 0 else "📉"
            print(f"  {emoji} SELL {rec['ticker']} ({rec['reason']}) | Precio: ${rec['price']:.2f} | P&L: ${rec['pnl']:.2f} ({rec['pnl_pct']:+.2%})")
    else:
        print("\n📉 VENTAS RECOMENDADAS: Ninguna")
    # Mostrar compras
    if buy_recommendations:
        print(f"\n📈 COMPRAS RECOMENDADAS ({len(buy_recommendations)}):")
        for rec in buy_recommendations:
            print(f"  ⚔️  BUY {rec['ticker']} | EV:{rec['ev']:.2f} | Prob:{rec['prob']:.1%} | Precio: ${rec['price']:.2f} | Régimen: {rec['regime']}")
    else:
        print("\n📈 COMPRAS RECOMENDADAS: Ninguna")
    # Resumen
    print(f"\n📊 RESUMEN:")
    print(f"   • Posiciones actuales: {len(positions_status)}")
    print(f"   • Ventas recomendadas: {len(sell_recommendations)}")
    print(f"   • Compras recomendadas: {len(buy_recommendations)}")
    if positions_status:
        total_value = sum(p['current_value'] for p in positions_status)
        total_pnl = sum(p['unrealized_pnl'] for p in positions_status)
        print(f"   • Valor portafolio: ${total_value:,.0f}")
        print(f"   • P&L no realizado: ${total_pnl:,.0f} ({(total_pnl/total_value) if total_value else 0:+.2%})")
    print(f"\n💡 RECOMENDACIÓN PRINCIPAL:")
    if sell_recommendations:
        top_sell = sell_recommendations[0]
        print(f"   👉 Vender {top_sell['ticker']} ({top_sell['reason']}) - P&L potencial: ${top_sell['pnl']:.2f}")
    elif buy_recommendations:
        top_buy = buy_recommendations[0]
        print(f"   👉 Comprar {top_buy['ticker']} (Confianza: {top_buy['prob']:.1%})")
    else:
        print("   👉 Mantener posiciones actuales, esperar mejores oportunidades")