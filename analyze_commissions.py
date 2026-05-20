"""Commission analysis for DeepQuant MODERATE tier"""

cap = 8000
mod_pct = 0.45
mod_pos = cap * mod_pct

def commsec(v):
    if v <= 0: return 0
    if v <= 1000: return 10.0
    if v <= 10000: return 19.95
    if v <= 25000: return 29.95
    return v * 0.0012

print("=" * 60)
print("CURRENT MODERATE position analysis")
print("=" * 60)
print(f"Capital: ${cap:,.0f} x {mod_pct*100:.0f}% = ${mod_pos:,.0f}")
rt = commsec(mod_pos) * 2
print(f"CommSec: ${commsec(mod_pos):.2f} entry + ${commsec(mod_pos):.2f} exit = ${rt:.2f} RT")
print(f"Break-even gross needed: ${rt:.2f} = {rt/mod_pos*100:.2f}% of position")
print()
print("WES.AX 2023: gross $41.52, comm $39.90, net +$1.62")
print("  That's a 1.15% gross move on $3,600 -> barely breaks even")
print("  ARB.AX 2024: gross $75.24, comm $39.90, net +$35.34 -> 2.09% gross")

print()
print("=" * 60)
print("Commission % drag by position size")
print("=" * 60)
for pos in [800, 1000, 1001, 1500, 2000, 3000, 3600, 5000, 6400]:
    c = commsec(pos)
    rt2 = c * 2
    drag_pct = rt2 / pos * 100
    bracket = "<=1K" if pos <= 1000 else "1K-10K"
    print(f"  ${pos:5,} ({bracket}): ${c:.2f}x2 = ${rt2:.2f} = {drag_pct:.2f}% drag")

print()
print("=" * 60)
print("KEY INSIGHT: smaller positions are NOT cheaper %")
print("=" * 60)
print("  $1,000 -> $10x2 = $20   -> 2.00% drag")
print("  $3,600 -> $19.95x2=$39.90 -> 1.11% drag")
print("  $1,000 << $3,600 but HIGHER % drag!")
print("=> The $10 bracket is NOT more efficient per trade")
print()
print("WES.AX with $1,000 position:")
wes_gross_1k = 41.52 * 1000 / 3600
wes_net_1k = wes_gross_1k - 20.0
print(f"  gross ${wes_gross_1k:.2f}, comm $20.00, net ${wes_net_1k:.2f}")
print("  => WORSE! (vs +$1.62 with $3,600 position)")

print()
print("=" * 60)
print("ROOT CAUSE: BE_STOP ignores commission")
print("=" * 60)
print("Current logic: when price >= entry + 1.0xATR, stop = entry")
print("  -> exit AT entry price = $0 gross - $39.90 comm = -$39.90 net")
print("  -> Trade needs +$39.90 EXTRA gain to survive commission")
print()
print("WES.AX: gained $41.52 gross, only $1.62 net -> 96% eaten by comm!")

approx_shares = 68  # ~$3,600 / $52.90 WES price
comm_per_share = 39.90 / approx_shares
print()
print("FIX: commission-aware BE stop")
print(f"  WES: ~{approx_shares} shares, $39.90 total comm = ${comm_per_share:.3f}/share")
print(f"  True BE stop = entry + ${comm_per_share:.3f} (not just entry)")
print(f"  This means position needs to be +{comm_per_share/52.90*100:.1f}% profitable before going 'at BE'")

print()
print("=" * 60)
print("ALTERNATIVE FIXES to get MORE trades")
print("=" * 60)
print("""
1. BULL_MAX_MONTH 3 -> 5, BULL_MAX_YEAR 12 -> 20
   - 2024 BULL months (Aug, Sep) had 3-4 consecutive winners
   - CHC +$911 (Aug) -> ORA +$507 (Aug) -> JHX +$249 (Sep) already chains
   - But there were 45-51 BULL candidates per month! Only trading 3-4/month
   - Allowing more would risk OVER-TRADING in winning months
   
2. TWO concurrent positions in BULL windows (MAX_POS 1 -> 2)
   - Currently: enter 1 stock, wait for exit, enter next
   - BULL months are >85% win rate -> holding 2 at once is fine
   - Risk: 2 losing trades in same bad BULL month
   
3. Commission-aware BE stop (MOST IMPACTFUL for MODERATE)
   - Stop moves to entry + (entry_comm + exit_comm_est) / shares
   - Ensures BE_STOP exits actually cover commission
   - WES would have been: stop at $52.90 + $0.587 = $53.49 instead of $52.90
   - The trade might hold longer and get a real profit, or stop out with $0 net
   
4. Raise MOD_BE_ATR from 1.0 -> 1.3xATR
   - Currently: once price moves up 1.0xATR, lock in breakeven
   - Higher threshold means we only lock in after a bigger move
   - More room for the trade to develop -> potentially larger gains
   - Risk: some winners could become losers if we don't lock in fast enough
   
5. Allow serial MODERATE trades (MOD_MAX_MONTH 1 -> 2)
   - Jul 2023 and Aug 2023 both had MODERATE windows
   - Currently only 1 per month, could allow 2
   - But Aug 2023 already had a losing trade with higher prob
""")
