"""Quick analysis of VAS.AX year-by-year and market breadth to understand
what regime filters would have avoided 2022 losses."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, yfinance as yf
from pathlib import Path

SYM_FILE = Path("au_stock_data/au_symbols.txt")
TICKERS = [s.strip() for s in SYM_FILE.read_text().splitlines() if s.strip()]
TICKERS = [t if t.endswith(".AX") else t+".AX" for t in TICKERS]

print("Downloading VAS + universe...")
all_t = list(set(TICKERS + ["VAS.AX"]))
raw = yf.download(all_t, start="2021-01-01", period="max",
                  group_by="ticker", auto_adjust=True, threads=True)

vas = raw["VAS.AX"].dropna(subset=["Close"])
vas["SMA50"]  = vas["Close"].rolling(50).mean()
vas["SMA200"] = vas["Close"].rolling(200).mean()
vas["ATR14"]  = (vas["High"] - vas["Low"]).rolling(14).mean()
vas["ATR_pct"] = vas["ATR14"] / vas["Close"] * 100
vas["MOM_1m"] = vas["Close"].pct_change(20)
vas["MOM_3m"] = vas["Close"].pct_change(63)

# Market breadth
data = {}
for t in TICKERS:
    try:
        tmp = raw[t].dropna(subset=["Close"])
        if len(tmp) > 100:
            data[t] = tmp
    except:
        pass

print(f"\n{len(data)} tickers loaded")

# Calculate breadth
dates = vas.index[vas.index >= "2022-01-01"]
breadth = []
for d in dates:
    above_sma50 = 0
    above_sma200 = 0
    total = 0
    for t, df in data.items():
        if d not in df.index:
            continue
        c = df.loc[d, "Close"]
        sma50 = df["Close"].rolling(50).mean()
        sma200 = df["Close"].rolling(200).mean()
        if d in sma50.index and pd.notna(sma50.loc[d]):
            total += 1
            if c > sma50.loc[d]:
                above_sma50 += 1
            if d in sma200.index and pd.notna(sma200.loc[d]):
                if c > sma200.loc[d]:
                    above_sma200 += 1
    pct50 = above_sma50 / total * 100 if total > 0 else 0
    pct200 = above_sma200 / total * 100 if total > 0 else 0
    breadth.append((d, pct50, pct200, total))

breadth_df = pd.DataFrame(breadth, columns=["date", "pct_above_sma50", "pct_above_sma200", "total"])
breadth_df.set_index("date", inplace=True)

# Monthly analysis
for year in [2022, 2023, 2024, 2025, 2026]:
    print(f"\n{'='*70}")
    print(f"  {year}")
    print(f"{'='*70}")
    yr_vas = vas[vas.index.year == year]
    yr_brd = breadth_df[breadth_df.index.year == year]
    
    if len(yr_vas) == 0:
        continue
    
    for month in range(1, 13):
        m_vas = yr_vas[yr_vas.index.month == month]
        m_brd = yr_brd[yr_brd.index.month == month]
        if len(m_vas) == 0:
            continue
        
        first = m_vas.iloc[0]
        last = m_vas.iloc[-1]
        ret = (last["Close"] / first["Close"] - 1) * 100
        
        above_200 = "YES" if last["Close"] > last["SMA200"] else "NO"
        above_50 = "YES" if last["Close"] > last["SMA50"] else "NO"
        golden = "YES" if pd.notna(last["SMA50"]) and pd.notna(last["SMA200"]) and last["SMA50"] > last["SMA200"] else "NO"
        
        avg_b50 = m_brd["pct_above_sma50"].mean() if len(m_brd) > 0 else 0
        avg_b200 = m_brd["pct_above_sma200"].mean() if len(m_brd) > 0 else 0
        
        mom1m = last["MOM_1m"] * 100 if pd.notna(last["MOM_1m"]) else 0
        mom3m = last["MOM_3m"] * 100 if pd.notna(last["MOM_3m"]) else 0
        atr_pct = m_vas["ATR_pct"].mean()
        
        # Would we trade this month?
        trade = "TRADE" if (above_200 == "YES" and above_50 == "YES" and 
                           golden == "YES" and avg_b50 > 55 and mom1m > 0) else "SKIP"
        
        print(f"  {year}-{month:02d} | VAS ret:{ret:>+5.1f}% | >200:{above_200} >50:{above_50} Golden:{golden} | "
              f"Breadth50:{avg_b50:>4.0f}% Breadth200:{avg_b200:>4.0f}% | "
              f"MOM1m:{mom1m:>+5.1f}% MOM3m:{mom3m:>+5.1f}% | ATR:{atr_pct:.2f}% | {trade}")

# Summary: how many months would we trade?
print(f"\n\n{'='*70}")
print(f"  SUMMARY: Trading months with strict filter")
print(f"{'='*70}")
print(f"  Filter: VAS>SMA200 + VAS>SMA50 + Golden Cross + Breadth50>55% + MOM1m>0")
