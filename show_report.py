# show_report.py  —  Reporte completo de predicciones 2026
import pandas as pd
import glob
import os

# ── cargar archivo más reciente ────────────────────────────────────
files = sorted(glob.glob('predictions_2026_*.xlsx'))
if not files:
    print("❌  No se encontró ningún archivo predictions_2026_*.xlsx")
    print("    Ejecuta primero:  python predict_2026.py")
    raise SystemExit

archivo = files[-1]
print(f"\n📂  Leyendo: {archivo}\n")

signals = pd.read_excel(archivo, sheet_name='Señales_BUY')
todas   = pd.read_excel(archivo, sheet_name='Todas_Predicciones')

# ── limpiar tipos ──────────────────────────────────────────────────
signals['Fecha']  = pd.to_datetime(signals['Fecha'])
signals['Prob']   = signals['Prob'].astype(float)
signals['Precio'] = pd.to_numeric(signals['Precio'], errors='coerce')
signals['Ret5d_%'] = pd.to_numeric(signals['Ret5d_%'], errors='coerce')

# separar confirmados vs pendientes
confirmados = signals[signals['Correcto'].isin(['✅ SUBIÓ', '❌ BAJÓ'])].copy()
pendientes  = signals[~signals['Correcto'].isin(['✅ SUBIÓ', '❌ BAJÓ'])].copy()

W = 80
print("╔" + "═"*(W-2) + "╗")
print("║" + "  TITAN AI — REPORTE PREDICCIONES 2026  (solo inferencia, sin simulación)".center(W-2) + "║")
print("╚" + "═"*(W-2) + "╝")

# ══════════════════════════════════════════════════════════════════
#  RESUMEN GLOBAL
# ══════════════════════════════════════════════════════════════════
print("\n" + "─"*W)
print("  📊  RESUMEN GLOBAL")
print("─"*W)

dias_totales  = todas['Fecha'].nunique()
dias_con_senal = signals['Fecha'].nunique()
prob_media    = signals['Prob'].mean()
prob_max      = signals['Prob'].max()
ticker_top    = signals.groupby('Ticker').size().idxmax()
ticker_top_n  = signals.groupby('Ticker').size().max()

print(f"  Período            : 2026-01-01  →  {signals['Fecha'].max().strftime('%Y-%m-%d')}")
print(f"  Días hábiles       : {dias_totales}")
print(f"  Días con ≥1 señal  : {dias_con_senal}  ({dias_con_senal/dias_totales:.0%} de los días)")
print(f"  Total señales BUY  : {len(signals)}  (umbral prob ≥ 50%)")
print(f"  Prob media         : {prob_media:.1%}")
print(f"  Prob máxima        : {prob_max:.1%}  ({signals.loc[signals['Prob'].idxmax(),'Ticker']}  {signals.loc[signals['Prob'].idxmax(),'Fecha'].strftime('%Y-%m-%d')})")
print(f"  Ticker más activo  : {ticker_top}  ({ticker_top_n} señales)")

# ══════════════════════════════════════════════════════════════════
#  VALIDACIÓN (señales con resultado conocido)
# ══════════════════════════════════════════════════════════════════
print("\n" + "─"*W)
print("  📈  VALIDACIÓN  (señales con resultado a 5 días conocido)")
print("─"*W)

if len(confirmados):
    wins   = (confirmados['Correcto'] == '✅ SUBIÓ').sum()
    losses = (confirmados['Correcto'] == '❌ BAJÓ').sum()
    wr     = wins / len(confirmados)
    avg_ret = confirmados['Ret5d_%'].mean()
    med_ret = confirmados['Ret5d_%'].median()
    avg_win = confirmados.loc[confirmados['Correcto']=='✅ SUBIÓ','Ret5d_%'].mean()
    avg_los = confirmados.loc[confirmados['Correcto']=='❌ BAJÓ','Ret5d_%'].mean()
    best    = confirmados.loc[confirmados['Ret5d_%'].idxmax()]
    worst   = confirmados.loc[confirmados['Ret5d_%'].idxmin()]
    rr      = abs(avg_win / avg_los) if avg_los != 0 else float('nan')

    print(f"  Señales evaluadas  : {len(confirmados)}")
    print(f"  ✅ Subieron (5d)   : {wins}   ({wr:.1%})")
    print(f"  ❌ Bajaron (5d)    : {losses}   ({1-wr:.1%})")
    print(f"  Retorno medio      : {avg_ret:+.2f}%")
    print(f"  Retorno mediano    : {med_ret:+.2f}%")
    print(f"  Avg ganadora       : {avg_win:+.2f}%")
    print(f"  Avg perdedora      : {avg_los:+.2f}%")
    print(f"  Ratio R/R          : {rr:.2f}x")
    print(f"  🏆 Mejor señal     : {best['Ticker']}  {best['Fecha'].strftime('%Y-%m-%d')}  +{best['Ret5d_%']:.2f}%  a ${best['Precio']:.3f}")
    print(f"  💀 Peor señal      : {worst['Ticker']}  {worst['Fecha'].strftime('%Y-%m-%d')}  {worst['Ret5d_%']:+.2f}%  a ${worst['Precio']:.3f}")

    # Por prob bucket
    print(f"\n  Rendimiento por nivel de probabilidad:")
    confirmados['prob_bucket'] = pd.cut(confirmados['Prob'],
                                         bins=[0.49,0.55,0.60,0.70,0.80,1.0],
                                         labels=['50-55%','55-60%','60-70%','70-80%','>80%'])
    bkt = confirmados.groupby('prob_bucket', observed=True).agg(
        Señales=('Ret5d_%','count'),
        WinRate=('Correcto', lambda x: (x=='✅ SUBIÓ').mean()),
        AvgRet=('Ret5d_%','mean')
    )
    print(f"  {'Prob':>8}  {'Señales':>8}  {'Win%':>7}  {'AvgRet%':>9}")
    print("  " + "-"*42)
    for idx, row in bkt.iterrows():
        print(f"  {str(idx):>8}  {row['Señales']:>8.0f}  {row['WinRate']:>7.1%}  {row['AvgRet']:>+8.2f}%")

# ══════════════════════════════════════════════════════════════════
#  RANKING POR TICKER
# ══════════════════════════════════════════════════════════════════
print("\n" + "─"*W)
print("  🏅  RANKING POR TICKER  (señales confirmadas)")
print("─"*W)

if len(confirmados):
    tk = confirmados.groupby('Ticker').agg(
        Señales=('Ret5d_%','count'),
        WinRate=('Correcto', lambda x: (x=='✅ SUBIÓ').mean()),
        AvgRet=('Ret5d_%','mean'),
        BestRet=('Ret5d_%','max'),
        WorstRet=('Ret5d_%','min'),
    ).sort_values('AvgRet', ascending=False).reset_index()

    print(f"  {'Ticker':<10} {'Señales':>7} {'Win%':>7} {'AvgRet%':>9} {'Mejor%':>9} {'Peor%':>9}")
    print("  " + "-"*58)
    for _, r in tk.iterrows():
        flag = "🟢" if r['WinRate'] >= 0.55 else ("🔴" if r['WinRate'] < 0.45 else "🟡")
        print(f"  {flag} {r['Ticker']:<8} {r['Señales']:>7.0f} {r['WinRate']:>7.1%} {r['AvgRet']:>+8.2f}% {r['BestRet']:>+8.2f}% {r['WorstRet']:>+8.2f}%")

# ══════════════════════════════════════════════════════════════════
#  DETALLE COMPLETO — SEÑALES CONFIRMADAS
# ══════════════════════════════════════════════════════════════════
print("\n" + "─"*W)
print("  📋  SEÑALES CONFIRMADAS — detalle")
print("─"*W)
print(f"  {'Fecha':<12} {'Ticker':<10} {'Precio':>8} {'Prob':>6}  {'Ret5d%':>8}  resultado")
print("  " + "-"*65)

for _, r in confirmados.sort_values(['Fecha','Prob'], ascending=[True,False]).iterrows():
    icon = "✅" if r['Correcto'] == '✅ SUBIÓ' else "❌"
    print(f"  {r['Fecha'].strftime('%Y-%m-%d'):<12} {r['Ticker']:<10} "
          f"{r['Precio']:>8.3f}  {r['Prob']:>5.1%}  {r['Ret5d_%']:>+7.2f}%  {icon}")

# ══════════════════════════════════════════════════════════════════
#  SEÑALES PENDIENTES (último periodo, resultado aún desconocido)
# ══════════════════════════════════════════════════════════════════
if len(pendientes):
    print("\n" + "─"*W)
    print(f"  ⏳  SEÑALES EN CURSO / PENDIENTES  ({len(pendientes)} señales, faltan datos futuros)")
    print("─"*W)
    print(f"  {'Fecha':<12} {'Ticker':<10} {'Precio':>8} {'Prob':>6}   estado")
    print("  " + "-"*55)
    for _, r in pendientes.sort_values(['Fecha','Prob'], ascending=[True,False]).iterrows():
        print(f"  {r['Fecha'].strftime('%Y-%m-%d'):<12} {r['Ticker']:<10} "
              f"{r['Precio']:>8.3f}  {r['Prob']:>5.1%}   ⏳ pendiente")

print("\n" + "═"*W)
print(f"  Fuente: {archivo}")
print("═"*W + "\n")
