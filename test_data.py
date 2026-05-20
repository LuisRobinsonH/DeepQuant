# test_data.py
import yfinance as yf
import pandas as pd

print("🔍 DIAGNÓSTICO DE DATOS (YAHOO FINANCE)")
print("---------------------------------------")

# 1. Intentamos descargar BHP
ticker = "BHP.AX"
print(f"📡 Descargando {ticker}...")

# Forzamos una descarga simple
data = yf.download(ticker, period="1y", progress=False)

# 2. Análisis Forense
print(f"\n📊 RESULTADOS:")
if data.empty:
    print("❌ ERROR: El DataFrame está VACÍO. Yahoo te está bloqueando o no tienes internet.")
else:
    print(f"✅ Datos recibidos. Filas: {len(data)}")
    print("\n🧐 ESTRUCTURA DE COLUMNAS (Aquí está el problema):")
    print(data.columns)
    
    print("\n📄 PRIMERA FILA DE DATOS:")
    print(data.head(1))

    # 3. Prueba de Acceso
    print("\n🧪 PRUEBA DE ACCESO A 'Close':")
    try:
        # Intento 1: Acceso directo (Formato viejo)
        c = data['Close']
        print(f"   -> Acceso directo data['Close']: ✅ FUNCIONA")
    except:
        print(f"   -> Acceso directo data['Close']: ❌ FALLÓ")
        
    try:
        # Intento 2: Acceso MultiIndex (Formato nuevo)
        c = data['Close'][ticker]
        print(f"   -> Acceso MultiIndex data['Close']['{ticker}']: ✅ FUNCIONA")
    except:
        print(f"   -> Acceso MultiIndex data['Close']['{ticker}']: ❌ FALLÓ")
        
    try:
        # Intento 3: Tupla (Formato nuevo alternativo)
        c = data[('Close', ticker)]
        print(f"   -> Acceso Tupla data[('Close', '{ticker}')]: ✅ FUNCIONA")
    except:
        print(f"   -> Acceso Tupla data[('Close', '{ticker}')]: ❌ FALLÓ")

print("\n🏁 Fin del diagnóstico.")