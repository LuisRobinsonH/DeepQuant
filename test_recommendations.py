# test_recommendations.py
from core.recommendation import get_recommendations

buy, sell, positions, date, discard = get_recommendations()

print(f"Fecha de datos más reciente: {date}")
print("\n=== RECOMENDACIONES DE COMPRA ===")
for rec in buy:
    print(rec)
print("\n=== RECOMENDACIONES DE VENTA ===")
for rec in sell:
    print(rec)
print("\n=== POSICIONES ACTUALES ===")
for pos in positions:
    print(pos)
print("\n=== DESCARTADOS (diagnóstico) ===")
for d in discard:
    print(d)
