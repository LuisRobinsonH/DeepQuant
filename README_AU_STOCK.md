## 🏆 Modelo en producción: V37 (2025-2026)

**Validación walk-forward 17 años (2008-2025) — 16/17 PASS — +290.84% compuesto**

| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| `PROB_MOMENTUM` | 0.52 | Umbral mínimo LightGBM BULL |
| `PROB_SIZE_MAX` | 0.90 | Sizing máximo = 90% capital |
| `PROB_SIZE_MIN` | 0.30 | Sizing mínimo = 30% capital |
| `SL_ATR` | 2.5 | Stop = entrada − 2.5×ATR14 |
| `BE_TRIGGER_ATR` | 1.5 | Break-even a entrada + 1.5×ATR14 |
| `MAX_HOLD_DAYS` | 35 | Máximo días por posición |
| `STOP_COOLDOWN_DAYS` | 5 | Enfriamiento post stop-loss |

**Resultados año a año:**

| Año | P&L % | Estado | Capital (desde $8k acum.) |
|-----|-------|--------|---------------------------|
| 2010 | 0.00% | — | $8,000 |
| 2011 | 0.00% | — | $8,000 |
| 2012 | +7.39% | ✅ | $8,591 |
| 2013 | +23.03% | ✅ | $10,570 |
| 2014 | 0.00% | — | $10,570 |
| 2015 | +3.29% | ✅ | $10,918 |
| 2016 | −3.15% | ❌ Brexit | $10,574 |
| 2017 | +6.17% | ✅ | $11,226 |
| 2018 | +7.77% | ✅ | $12,098 |
| 2019 | +22.25% | ✅ | $14,790 |
| 2020 | 0.00% | — | $14,790 |
| 2021 | +57.02% | ✅ | $23,233 |
| 2022 | 0.00% | — | $23,233 |
| 2023 | 0.00% | — | $23,233 |
| 2024 | +12.56% | ✅ | $26,152 |
| 2025 | +19.62% | ✅ | $31,285 |
| 2026 | 0.00% | — | $31,285 |

> 2016 es un caso de cisne negro (Brexit, irreducible). Todos los demás filtros alternativos
> probados (V38: GC_AGE_MIN=45) destruyeron años ganadores (2012, 2019, 2024). Se acepta como
> coste estructural de mercado.

---

## Cambios y mejoras recientes (2026)

- **V37 en producción**: PROB_SIZE_MAX=0.90, STOP_COOLDOWN=5, ATR-based stops, 16/17 años PASS.
- Sizing dinámico por probabilidad: posición escala con prob/PROB_SIZE_MAX.
- Golden Cross exacto + dist_SMA200 ≥ +2% + Gate 3m (slope 63 días).
- Stop ATR: `stop = entry − 2.5×ATR14` | Break-even: `entry + 1.5×ATR14 → stop a entrada`.
- STOP_COOLDOWN: 5 días sin re-entrar en ticker tras un stop-loss (stop_history.json).
- Eliminado tier REVERSION (solo BULL_V37).
- Validación cruzada temporal (walk-forward) en el entrenamiento del modelo.
- Alertas Telegram con ATR, Gate3m, EV y sizing dinámico.
- Historial de rendimiento de alertas (alert_history.csv).
- Alertas personalizadas por usuario (user_alert_config.json).

# README_AU_STOCK.md

# DeepQuant AU Investment

Sistema de análisis cuantitativo y gestión de portafolio de inversión para acciones australianas (ASX).

## Componentes principales
- **investment/au_stock_data/fetch_au_stock_data.py**: Descarga datos históricos de acciones australianas desde Yahoo Finance.
- **investment/au_stock_data/fetch_fundamentals.py**: Descarga automática de datos fundamentales (EPS, precio) usando yfinance.
- **investment/analytics/portfolio_analysis.py**: Análisis y métricas de portafolio de inversión.
- **investment/analytics/fundamental_analysis.py**: Cálculo de ratios fundamentales (P/E, etc.).
- **investment/analytics/portfolio_scoring.py**: Scoring y ranking de acciones para inversión.
- **investment/analytics/generate_investment_report.py**: Generación de reporte óptimo de portafolio.
- **investment/run_investment_pipeline.py**: Pipeline maestro para automatizar todo el flujo de inversión.

## Uso rápido
1. Edita `investment/au_stock_data/au_symbols.txt` con tus símbolos de interés.
2. Ejecuta `python investment/au_stock_data/fetch_au_stock_data.py` para descargar datos históricos (opcional).
3. Ejecuta `python investment/au_stock_data/fetch_fundamentals.py` para obtener fundamentales actualizados.
4. Edita `investment/analytics/example_holdings.csv` con tus posiciones reales.
5. Ejecuta `python investment/run_investment_pipeline.py` para generar el reporte óptimo (`investment/analytics/investment_report.csv`).

## Personalización
- Modifica o extiende los módulos en `investment/analytics/` y `investment/core/` para agregar criterios de scoring, rebalanceo, o métricas fundamentales.
- Automatiza la ejecución con tareas programadas para mantener el portafolio siempre actualizado.

---

**Nota:**
- Este sistema está enfocado únicamente en inversión cuantitativa y fundamental. No incluye lógica de trading, señales rápidas ni bots automáticos.
