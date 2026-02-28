# view_trades.py
import sqlite3
import pandas as pd

def analyze_performance():
    conn = sqlite3.connect('trading_bot.db')
    
    # 1. Extraer todas las transacciones
    query = """
    SELECT date, ticker, action, price, shares, pnl, cash, equity 
    FROM transactions 
    WHERE action IN ('BUY', 'SELL')
    ORDER BY date ASC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    if df.empty:
        print("❌ No hay operaciones registradas.")
        return

    # Convertir fecha
    df['date'] = pd.to_datetime(df['date']).dt.date
    
    print("\n📜 BITÁCORA DE OPERACIONES (2025)")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)

    # 2. Análisis Forense de Pérdidas Grandes (El caso IGO)
    print("\n🚨 ANÁLISIS FORENSE DE PÉRDIDAS (> $150)")
    losses = df[df['pnl'] < -150]
    
    if not losses.empty:
        for i, row in losses.iterrows():
            ticker = row['ticker']
            sell_date = row['date']
            loss_amount = row['pnl']
            sell_price = row['price']
            
            # Buscar cuándo se compró
            buys = df[(df['ticker'] == ticker) & (df['action'] == 'BUY') & (df['date'] < sell_date)]
            if not buys.empty:
                buy_row = buys.iloc[-1]
                buy_price = buy_row['price']
                buy_date = buy_row['date']
                
                drop_pct = ((sell_price - buy_price) / buy_price) * 100
                
                print(f"\n⚠️ CASO: {ticker}")
                print(f"   📅 Compra: {buy_date} a ${buy_price:.2f}")
                print(f"   📅 Venta:  {sell_date} a ${sell_price:.2f}")
                print(f"   📉 Caída:  {drop_pct:.2f}%")
                print(f"   💸 Pérdida Total: ${loss_amount:.2f}")
                print(f"   🔍 DIAGNÓSTICO: Posible 'Gap de Apertura'. El precio abrió muy por debajo del Stop Loss.")
    else:
        print("✅ No hay pérdidas catastróficas. El sistema de riesgo funcionó bien.")

if __name__ == "__main__":
    analyze_performance()