# TitanHedge/reset_system.py
import os

db_path = os.path.join("database", "trading_bot.db")

if os.path.exists(db_path):
    try:
        os.remove(db_path)
        print(f"✅ Base de datos antigua eliminada: {db_path}")
        print("Ahora ejecuta 'python orchestrator.py' para crear la nueva estructura.")
    except Exception as e:
        print(f"❌ Error al borrar: {e}")
        print("Intenta borrar el archivo 'database/trading_bot.db' manualmente.")
else:
    print("⚠️ No se encontró base de datos previa. Todo limpio.")