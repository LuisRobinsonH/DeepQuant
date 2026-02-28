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
