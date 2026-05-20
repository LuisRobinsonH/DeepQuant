import pandas as pd
import numpy as np

xls = pd.ExcelFile('sim_v7_20260227_1422.xlsx')
print('Sheets:', xls.sheet_names)

for s in xls.sheet_names:
    df = pd.read_excel(xls, sheet_name=s)
    if 'pnl' not in df.columns:
        continue
    print(f"\n{'='*60}")
    print(f"  {s} ({len(df)} trades)")
    print(f"{'='*60}")
    
    if len(df) == 0:
        continue
    
    # Overall stats
    total_gross = df['gross_pnl'].sum()
    total_comm = df['commission'].sum()
    total_net = df['pnl'].sum()
    print(f"  Gross P&L: ${total_gross:+,.2f}")
    print(f"  Commission: ${total_comm:,.2f}")
    print(f"  NET P&L: ${total_net:+,.2f}")
    
    # Commission vs gross analysis
    comm_exceeds = (df['commission'].abs() > df['gross_pnl'].abs()).sum()
    print(f"  Trades where commission > |gross P&L|: {comm_exceeds}/{len(df)} ({comm_exceeds/len(df)*100:.0f}%)")
    
    # Stop loss analysis
    stops = df[df['reason'] == 'STOP']
    if len(stops) > 0:
        days_held = stops['days'].mean()
        avg_gross = stops['gross_pnl'].mean()
        avg_comm = stops['commission'].mean()
        gross_pos = (stops['gross_pnl'] > 0).sum()
        print(f"\n  STOP LOSS Analysis ({len(stops)} trades):")
        print(f"    Avg days held: {days_held:.1f}")
        print(f"    Avg gross P&L: ${avg_gross:.2f}")
        print(f"    Avg commission: ${avg_comm:.2f}")
        print(f"    Gross profitable stops: {gross_pos}/{len(stops)}")
        
        # Distribution of gross P&L on stops
        bins = [-1000, -200, -100, -50, 0, 50, 100, 200, 1000]
        labels = ['<-200', '-200~-100', '-100~-50', '-50~0', '0~50', '50~100', '100~200', '>200']
        stops['pnl_bin'] = pd.cut(stops['gross_pnl'], bins=bins, labels=labels)
        print(f"    Gross P&L distribution on STOPS:")
        for lbl in labels:
            n = (stops['pnl_bin'] == lbl).sum()
            if n > 0:
                print(f"      {lbl:>12s}: {n} trades")
    
    # TP1 analysis
    tp1s = df[df['reason'] == 'TP1']
    if len(tp1s) > 0:
        print(f"\n  TP1 Analysis ({len(tp1s)} trades):")
        print(f"    Avg gross P&L: ${tp1s['gross_pnl'].mean():.2f}")
        print(f"    Avg commission: ${tp1s['commission'].mean():.2f}")
    
    # Per-ticker analysis
    by_ticker = df.groupby('ticker').agg(
        count=('pnl', 'count'),
        total_pnl=('pnl', 'sum'),
        total_gross=('gross_pnl', 'sum'),
        total_comm=('commission', 'sum'),
        wr=('pnl', lambda x: (x > 0).mean() * 100)
    ).sort_values('total_pnl')
    
    print(f"\n  Worst 10 tickers:")
    for idx, row in by_ticker.head(10).iterrows():
        print(f"    {idx:10s} {int(row['count']):2d}× PNL:${row['total_pnl']:>+8.2f} Gross:${row['total_gross']:>+8.2f} Comm:${row['total_comm']:>6.2f} WR:{row['wr']:.0f}%")
    
    print(f"\n  Best 5 tickers:")
    for idx, row in by_ticker.tail(5).iterrows():
        print(f"    {idx:10s} {int(row['count']):2d}× PNL:${row['total_pnl']:>+8.2f} Gross:${row['total_gross']:>+8.2f} Comm:${row['total_comm']:>6.2f} WR:{row['wr']:.0f}%")

    # Yearly breakdown (for P1)
    if 'entry' in df.columns and len(df) > 5:
        df['year'] = pd.to_datetime(df['entry']).dt.year
        by_year = df.groupby('year').agg(
            count=('pnl', 'count'),
            total_pnl=('pnl', 'sum'),
            total_gross=('gross_pnl', 'sum'),
            total_comm=('commission', 'sum'),
            wr=('pnl', lambda x: (x > 0).mean() * 100)
        )
        print(f"\n  By Year:")
        for yr, row in by_year.iterrows():
            print(f"    {yr}: {int(row['count']):3d} trades | NET:${row['total_pnl']:>+8.2f} Gross:${row['total_gross']:>+8.2f} Comm:${row['total_comm']:>6.2f} WR:{row['wr']:.0f}%")
