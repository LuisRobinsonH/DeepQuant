# fetch_realtime_au.py
"""
Obtiene el último precio y datos de acciones australianas en tiempo real usando yfinance.
"""
import yfinance as yf
import pandas as pd

def fetch_realtime_au(symbols):
    data = {}
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            # El historial de 1d incluye el último precio si el mercado está abierto
            df = ticker.history(period="2d", interval="1d")
            if not df.empty:
                last = df.iloc[-1]
                data[symbol] = {
                    'date': last.name.strftime('%Y-%m-%d'),
                    'close': last['Close'],
                    'high': last['High'],
                    'low': last['Low'],
                    'open': last['Open'],
                    'volume': last['Volume']
                }
        except Exception as e:
            print(f"Error obteniendo {symbol}: {e}")
    return data

if __name__ == "__main__":
    # Ejemplo de uso
    symbols = ['BHP.AX', 'WBC.AX']
    print(fetch_realtime_au(symbols))
