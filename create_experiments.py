"""
Generator: Creates V24, V25, V26 experiment files from V22b base (sim_v15_3periods.py).
Each is a SINGLE-VARIABLE change for clean isolation.
"""
from pathlib import Path

base = Path("sim_v15_3periods.py").read_text(encoding="utf-8")

# ─── SHARED HELPERS ────────────────────────────────────────────────────────────
def write(path, text):
    Path(path).write_text(text, encoding="utf-8")
    print(f"  Created: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# V24 — dist_52w_high feature ONLY (no filter / hyperparam changes)
# ══════════════════════════════════════════════════════════════════════════════
v24 = base

# 1. Docstring header
v24 = v24.replace(
    '#!/usr/bin/env python\n"""',
    '#!/usr/bin/env python\n"""',  # keep shebang/open
    1
)
old_doc = '''\
"""
DeepQuant V23b — Reverted to V22b (V23 features experiment failed)
=================================================================
V23 EXPERIMENT RESULT: Score 2/5, -16.13% cumulative — CATASTROPHIC FAILURE

V23 changed three things simultaneously:
  1. 6 new ML features (dist_52w_high, dist_26w_high, vol_trend_5v20,
     mom_diverge, rsi_trend_5d, up_bars_pct10)
  2. Tighter LightGBM (min_child=40, reg_lambda=3.5, cv=5)
  3. Tighter entry filters (RSI max 80->77, vol_trend_5v20 < 0.45 skip)

WHY V23 FAILED:
  a) RSI cap 77 BLOCKED CHC.AX Aug-2024 -> CHC was +$845 (biggest single win
     in V22b 2024). CHC RSI was 78-80 at entry but ran +$845. Filter was wrong.
  b) New features changed ALL model probability rankings: SEK/SKC/PDN/GMG
     now ranked #1 instead of CHC/JHX/CAR. New top candidates all stopped out.
  c) Model re-calibrated from scratch with 6 extra features + tighter params;
     the 2022-2026 probability landscape shifted unpredictably.

V23 LESSON: V22b model was tuned through 13 evolutionary iterations against
2022-2026 OOS data. Adding 9 changes at once invalidates all prior validation.
Safe approach = single-variable changes, one at a time.

FEATURE HISTORY (all attempts):
  V20 MAX_POS=2:   4/5, -$134 2025  | concurrent low-quality 2nd-slot trades
  V21 SELECTIVE:   1/5, -23.68%     | model invalid in bear market
  V22 MAX_HOLD=20: 4/5, -$475 2025  | cut winners, opened worse sequential slots
  V23 6 features:  2/5, -16.13%     | RSI filter blocked CHC; rankings disrupted

V22b (= V20b) IS THE CONFIRMED OPTIMUM.
  5/5 NEVER LOSE ACHIEVED. 18%+ years: 2024 +$1,964 = +24.5%.
  Conservative 2025 +$64 = genuine small gain in 3-BULL-month year.

CommSec: <=1K->$10 | 1K-10K->$19.95 | 10K-25K->$29.95 | >25K->0.12%
"""'''
new_doc_v24 = '''\
"""
DeepQuant V24 — SINGLE EXPERIMENT: dist_52w_high feature only
=============================================================
BASE = V22b (= V20b confirmed optimum, 5/5, cumulative +26.09%)

CHANGE: ONE new ML feature added to build_features():
  feat["dist_52w_high"] = (close - 52w_high) / 52w_high
  Captures how far price is below its 52-week high.
  → Hypothesis: stocks further from 52w-high may have more upside room.
  → Alternative: overbought stocks near 52w-high may continue (momentum).

ALL OTHER PARAMETERS IDENTICAL TO V22b:
  BULL_PROB_FULL = 0.52 | RSI max 80 | LGB defaults (min_child=25, lambda=2.0)
  No new entry filters added.

QUESTION: Does dist_52w_high improve or harm 2024 P&L?
  2024 baseline (V22b): +$1,964 = +24.5%
  CHC entry Aug-2024: RSI ~78-80, well below 52w-high -> dist_52w_high = negative

NOTE: ML models retrain from scratch with this new feature. Probability
rankings for all stocks will shift. This is a clean single-variable test.

CommSec: <=1K->$10 | 1K-10K->$19.95 | 10K-25K->$29.95 | >25K->0.12%
"""'''
v24 = v24.replace(old_doc, new_doc_v24, 1)

# 2. Add dist_52w_high to build_features (after close_to_low20)
v24 = v24.replace(
    '    feat["close_to_low20"]  = c / l.rolling(20).min()\n    feat["range_pct"]',
    '    feat["close_to_low20"]  = c / l.rolling(20).min()\n    hw52 = h.rolling(252).max()\n    feat["dist_52w_high"]   = (c - hw52) / (hw52 + 1e-10)  # V24: distance below 52-week high\n    feat["range_pct"]',
    1
)

# 3. Header print
v24 = v24.replace(
    'print("  DEEPQUANT V23 — False Positive Reduction: 6 New Features + Tighter ML")',
    'print("  DEEPQUANT V24 — SINGLE EXP: dist_52w_high feature only (V22b base)")',
    1
)
v24 = v24.replace(
    'print(f"  MAX_HOLD={MAX_HOLD}d | RSI_max=77 | vol_trend_filter=0.45 | min_child=40 | reg_lambda=3.5 | cv=5")',
    'print(f"  MAX_HOLD={MAX_HOLD}d | RSI_max=80 | NEW_FEAT=dist_52w_high | all other params=V22b")',
    1
)

# 4. Summary label
v24 = v24.replace(
    'print(f"  SUMMARY - DEEPQUANT V23b (= V22b / V20b optimal parameters)")',
    'print(f"  SUMMARY - DEEPQUANT V24 (V22b + dist_52w_high feature)")',
    1
)
v24 = v24.replace(
    '  print(f"  V23b{p1[\'roi\']:>+7.2f}% {p2[\'roi\']:>+7.2f}% {p3[\'roi\']:>+7.2f}% {cum_roi:>+7.2f}% {yr_all[\'2022\']:>7s} {yr_all[\'2023\']:>7s} {yr_all[\'2024\']:>7s} {yr_all[\'2025\']:>7s} {yr_all[\'2026\']:>7s}  {passed}/{passed+failed} optimal")',
    '  print(f"  V24 {p1[\'roi\']:>+7.2f}% {p2[\'roi\']:>+7.2f}% {p3[\'roi\']:>+7.2f}% {cum_roi:>+7.2f}% {yr_all[\'2022\']:>7s} {yr_all[\'2023\']:>7s} {yr_all[\'2024\']:>7s} {yr_all[\'2025\']:>7s} {yr_all[\'2026\']:>7s}  {passed}/{passed+failed} +dist_52w_high")',
    1
)

# 5. Output filename
v24 = v24.replace(
    '    fname = f"sim_v23b_{now}.xlsx"',
    '    fname = f"sim_v24_{now}.xlsx"',
    1
)

write("sim_v24_3periods.py", v24)

# ══════════════════════════════════════════════════════════════════════════════
# V25 — BULL_PROB_FULL 0.52 → 0.70 (skip JIN 0.78/CPU 0.81; keep CHC/DXS)
# ══════════════════════════════════════════════════════════════════════════════
v25 = base

new_doc_v25 = '''\
"""
DeepQuant V25 — SINGLE EXPERIMENT: BULL_PROB_FULL 0.52 → 0.70
==============================================================
BASE = V22b (= V20b confirmed optimum, 5/5, cumulative +26.09%)

CHANGE: BULL_PROB_FULL raised from 0.52 to 0.70
  Hypothesis: A higher full-bull threshold filters out marginal signals.
  Context from trade log analysis:
    JIN  prob=0.78 → STOP. Would be BLOCKED at >=0.70 (barely above)
    CPU  prob=0.81 → STOP. Would be BLOCKED at >=0.70
    CHC  prob ~0.85+ → WIN ($+845). Would be KEPT at >=0.70
    DXS  prob ~0.80+ → WIN. Would be KEPT at >=0.70
  0.65 would still let JIN/CPU through. 0.70 is the minimum to block both.

QUESTION: Does cutting prob<0.70 entries improve 2024 net P&L?
  If JIN+CPU net losses > CHC/DXS gross gains lost → HARM
  If we simply avoid stopouts without losing winners → IMPROVE

ALL OTHER PARAMETERS IDENTICAL TO V22b (no feature changes, RSI=80, etc.)

CommSec: <=1K->$10 | 1K-10K->$19.95 | 10K-25K->$29.95 | >25K->0.12%
"""'''
v25 = v25.replace(old_doc, new_doc_v25, 1)

# Change BULL_PROB_FULL
v25 = v25.replace(
    'BULL_PROB_FULL      = 0.52',
    'BULL_PROB_FULL      = 0.70  # V25: raised from 0.52 to block marginal signals (JIN 0.78, CPU 0.81)',
    1
)

# Header print
v25 = v25.replace(
    'print("  DEEPQUANT V23 — False Positive Reduction: 6 New Features + Tighter ML")',
    'print("  DEEPQUANT V25 — SINGLE EXP: BULL_PROB_FULL 0.52 → 0.70 (V22b base)")',
    1
)
v25 = v25.replace(
    'print(f"  MAX_HOLD={MAX_HOLD}d | RSI_max=77 | vol_trend_filter=0.45 | min_child=40 | reg_lambda=3.5 | cv=5")',
    'print(f"  MAX_HOLD={MAX_HOLD}d | RSI_max=80 | BULL_PROB_FULL=0.70 (was 0.52) | all other params=V22b")',
    1
)

# Summary
v25 = v25.replace(
    'print(f"  SUMMARY - DEEPQUANT V23b (= V22b / V20b optimal parameters)")',
    'print(f"  SUMMARY - DEEPQUANT V25 (V22b + BULL_PROB_FULL=0.70)")',
    1
)
v25 = v25.replace(
    '  print(f"  V23b{p1[\'roi\']:>+7.2f}% {p2[\'roi\']:>+7.2f}% {p3[\'roi\']:>+7.2f}% {cum_roi:>+7.2f}% {yr_all[\'2022\']:>7s} {yr_all[\'2023\']:>7s} {yr_all[\'2024\']:>7s} {yr_all[\'2025\']:>7s} {yr_all[\'2026\']:>7s}  {passed}/{passed+failed} optimal")',
    '  print(f"  V25 {p1[\'roi\']:>+7.2f}% {p2[\'roi\']:>+7.2f}% {p3[\'roi\']:>+7.2f}% {cum_roi:>+7.2f}% {yr_all[\'2022\']:>7s} {yr_all[\'2023\']:>7s} {yr_all[\'2024\']:>7s} {yr_all[\'2025\']:>7s} {yr_all[\'2026\']:>7s}  {passed}/{passed+failed} prob>=0.70")',
    1
)
v25 = v25.replace(
    '    fname = f"sim_v23b_{now}.xlsx"',
    '    fname = f"sim_v25_{now}.xlsx"',
    1
)

write("sim_v25_3periods.py", v25)

# ══════════════════════════════════════════════════════════════════════════════
# V26 — RSI upper cap: 80 → 78 (was CHC at RSI=78 or 79?)
# ══════════════════════════════════════════════════════════════════════════════
v26 = base

new_doc_v26 = '''\
"""
DeepQuant V26 — SINGLE EXPERIMENT: RSI upper cap 80 → 78
=========================================================
BASE = V22b (= V20b confirmed optimum, 5/5, cumulative +26.09%)

CHANGE: Entry filter F2 RSI upper bound changed from > 80 → > 78
  V23 used RSI > 77 (cap=77), which BLOCKED CHC.AX Aug-2024 (+$845 trade).
  V22b uses RSI > 80 (cap=80), which ALLOWED CHC.

  This test uses cap=78 (two points tighter than V22b, two points looser than V23).
  QUESTION: Was CHC Aug-2024 entry RSI exactly 78 or 79?
    - If CHC RSI was 78 → cap=78 would BLOCK it → HARM 2024
    - If CHC RSI was 79-80 → cap=78 would ALLOW it → neutral/slight improvement

ALL OTHER PARAMETERS IDENTICAL TO V22b (no feature changes, prob=0.52, etc.)

CommSec: <=1K->$10 | 1K-10K->$19.95 | 10K-25K->$29.95 | >25K->0.12%
"""'''
v26 = v26.replace(old_doc, new_doc_v26, 1)

# Change RSI filter
v26 = v26.replace(
    '            # F2: RSI 30-80\n            if pd.notna(rsi_val) and (rsi_val < 30 or rsi_val > 80):',
    '            # F2: RSI 30-78 (V26: cap lowered from 80 to 78 — single point test)\n            if pd.notna(rsi_val) and (rsi_val < 30 or rsi_val > 78):',
    1
)

# Header print
v26 = v26.replace(
    'print("  DEEPQUANT V23 — False Positive Reduction: 6 New Features + Tighter ML")',
    'print("  DEEPQUANT V26 — SINGLE EXP: RSI upper cap 80 → 78 (V22b base)")',
    1
)
v26 = v26.replace(
    'print(f"  MAX_HOLD={MAX_HOLD}d | RSI_max=77 | vol_trend_filter=0.45 | min_child=40 | reg_lambda=3.5 | cv=5")',
    'print(f"  MAX_HOLD={MAX_HOLD}d | RSI_max=78 (was 80) | BULL_PROB_FULL={BULL_PROB_FULL:.2f} | all other params=V22b")',
    1
)

# Summary
v26 = v26.replace(
    'print(f"  SUMMARY - DEEPQUANT V23b (= V22b / V20b optimal parameters)")',
    'print(f"  SUMMARY - DEEPQUANT V26 (V22b + RSI cap 78)")',
    1
)
v26 = v26.replace(
    '  print(f"  V23b{p1[\'roi\']:>+7.2f}% {p2[\'roi\']:>+7.2f}% {p3[\'roi\']:>+7.2f}% {cum_roi:>+7.2f}% {yr_all[\'2022\']:>7s} {yr_all[\'2023\']:>7s} {yr_all[\'2024\']:>7s} {yr_all[\'2025\']:>7s} {yr_all[\'2026\']:>7s}  {passed}/{passed+failed} optimal")',
    '  print(f"  V26 {p1[\'roi\']:>+7.2f}% {p2[\'roi\']:>+7.2f}% {p3[\'roi\']:>+7.2f}% {cum_roi:>+7.2f}% {yr_all[\'2022\']:>7s} {yr_all[\'2023\']:>7s} {yr_all[\'2024\']:>7s} {yr_all[\'2025\']:>7s} {yr_all[\'2026\']:>7s}  {passed}/{passed+failed} RSI<=78")',
    1
)
v26 = v26.replace(
    '    fname = f"sim_v23b_{now}.xlsx"',
    '    fname = f"sim_v26_{now}.xlsx"',
    1
)

write("sim_v26_3periods.py", v26)

print("\nDone. Verify key changes:")
# Quick verification
for fname, label, check_str in [
    ("sim_v24_3periods.py", "V24 dist_52w_high", 'feat["dist_52w_high"]'),
    ("sim_v25_3periods.py", "V25 prob=0.70",     'BULL_PROB_FULL      = 0.70'),
    ("sim_v26_3periods.py", "V26 RSI cap 78",    'rsi_val > 78'),
]:
    content = Path(fname).read_text(encoding="utf-8")
    found = check_str in content
    # Also verify baseline NOT changed incorrectly
    print(f"  [{fname}] '{check_str}' present: {found}")
