import threading
import time

def monitor_alerts(interval=60):
    print("[MONITOR] Iniciando monitor de alertas en tiempo real...")
    while True:
        print("[MONITOR] Escaneando señales...")
        scan_buy_signals()
        time.sleep(interval)

def start_monitor():
    t = threading.Thread(target=monitor_alerts, args=(60,), daemon=True)
    t.start()
"""
alerts_live.py — TitanBrain Live Alert System (V37)
────────────────────────────────────────────────────
MODELO: V37 — 16/17 años PASS | +290.84% compuesto | $8k→$31,267
VALIDADO: walk-forward 2008-2026 (17 períodos anuales independientes)

MODOS:
  python alerts_live.py          → escanea señales + monitorea portfolio → envía Telegram
  python alerts_live.py bot      → bot continuo (polling) que procesa confirmaciones
  python alerts_live.py scan     → solo escanear, sin polling posterior
  python alerts_live.py bought QBE.AX 21.67   → confirmar compra manual
  python alerts_live.py sold   QBE.AX 23.50   → confirmar venta manual

FLUJO:
  1. Scan detecta BUY signal → Telegram con botón [✅ Compré | ❌ Ignorar]
  2. Usuario pulsa ✅ Compré  → posición entra a current_portfolio.json
  3. Monitor diario detecta EXIT → Telegram con botón [✅ Vendí | 🔄 Mantener]
  4. Usuario pulsa ✅ Vendí   → posición sale del portfolio + log P&L

Filtros V37 (SOLO BULL — sin REVERSION):
  - prob ≥ 0.52 (BULL_PROB_FULL, gate LightGBM)
  - Golden Cross exacto: SMA50 > SMA200
  - dist_SMA200 ≥ +2% (precio bien sobre la media larga)
  - Gate 3m: precio actual > precio hace 63 días (momentum positivo)
  - RSI < 78, MACD > 0, ADX ≥ 15, vol_ratio ≥ 1.20, mom5 ≥ 1.5%
  - Stop ATR: entry − 2.5×ATR14  |  Breakeven: entry + 1.5×ATR14
  - Sizing dinámico: frac = prob / 0.90 (mín 33% capital, máx 100%)
  - MAX_HOLD: 35 días  |  STOP_COOLDOWN: 5 días post stop-loss
"""

import sys, os, json, time, math
import yfinance as yf
import pandas as pd
import numpy as np
import ta as ta_lib
import warnings
import requests
import joblib
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# ── Cargar .env local si existe (para uso local — nunca commitear .env) ────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # En CI no es necesario, las vars vienen de GitHub Secrets

# ── TELEGRAM (requiere env vars — nunca hardcodear tokens) ────────
TELEGRAM_TOKEN   = os.getenv('TELEGRAM_TOKEN') or ''
TELEGRAM_CHAT_ID = int(os.getenv('TELEGRAM_CHAT_ID') or '0')
TG_BASE          = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"

if not TELEGRAM_TOKEN or TELEGRAM_CHAT_ID == 0:
    print("[WARNING] TELEGRAM_TOKEN / TELEGRAM_CHAT_ID no configurados.")
    print("          Local: crea un archivo .env con TELEGRAM_TOKEN=... y TELEGRAM_CHAT_ID=...")
    print("          CI: configura los secretos en GitHub Actions.")

# ── PARÁMETROS V37 (validados walk-forward 2008-2026, 16/17 PASS, +290.84%) ──
# Entrada (solo BULL — sin tier REVERSION)
PROB_MOMENTUM    = 0.52        # BULL_PROB_FULL: umbral mínimo de prob para BULL
PROB_SIZE_MIN    = 0.30        # fracción mínima de capital (30%)
PROB_SIZE_MAX    = 0.90        # fracción máxima de capital (90%) → sizing dinámico
# Stop / Breakeven ATR-based
SL_ATR           = 2.5         # stop = entry − SL_ATR × ATR14
BE_TRIGGER_ATR   = 1.5         # breakeven trigger = entry + BE_TRIGGER_ATR × ATR14
# Targets indicativos (para Telegram — el sistema sale por stop/BE/max-hold)
TP1_PCT          = 0.09
TP2_PCT          = 0.22
# Filtros técnicos de entrada
VOL_MIN_MOM      = 1.20        # vol_ratio mínimo
MOM5_MIN         = 0.015       # mom5d mínimo (+1.5%)
# Gestión de posición
MAX_HOLD_DAYS    = 35          # máximo días en posición
MIN_POSITION     = 3000        # AUD mínimo por operación
STOP_COOLDOWN_DAYS = 5         # días de enfriamiento tras stop-loss por ticker
MAX_POS_PCT      = 0.90        # límite máximo de capital en una posición
CAPITAL_DEFAULT  = 8000.0
COMMISSION_FLAT  = 10.0
COMMISSION_RATE  = 0.0011

# ── ARCHIVOS ──────────────────────────────────────────────────────
PORTFOLIO_FILE    = 'current_portfolio.json'
TRADE_HISTORY_FILE= 'trade_history.json'
MODEL_CACHE_FILE  = 'models_cache.joblib'
FEATURE_CACHE_FILE= 'features_cache.joblib'
UNIVERSE_CACHE_FILE = 'universe_cache.json'

# ── FILTRO DE UNIVERSO ───────────────────────────────────────────
# Elimina small-caps que aumentan ruido sin mejorar retornos
# (JIN ~$2B excluido; CHC/ORA/DXS/MQG >$3B incluidos)
UNIVERSE_MIN_MCAP   = 3_000_000_000   # AUD — mínimo market cap
UNIVERSE_CACHE_DAYS = 7               # refrescar cada 7 días

# ══════════════════════════════════════════════════════════════════
# UNIVERSE FILTER — Market Cap cache (refresh weekly)
# ══════════════════════════════════════════════════════════════════

def fetch_universe(tickers: list[str], min_mcap: float = UNIVERSE_MIN_MCAP) -> list[str]:
    """
    Retorna subconjunto de `tickers` con marketCap >= min_mcap AUD.
    Carga de universe_cache.json si fue generado en los últimos UNIVERSE_CACHE_DAYS días.
    Si el caché expiró o no existe, llama yfinance .fast_info para cada ticker y
    guarda el resultado. Tarda ~60s la primera vez; luego es instantáneo.

    Tickers sin datos de market-cap (ETFs, etc.) se MANTIENEN en el universo
    para no excluir VAS.AX u otros índices de referencia necesarios.
    """
    today_str = datetime.now().strftime('%Y-%m-%d')

    # ── Intentar caché ────────────────────────────────────────────
    if os.path.exists(UNIVERSE_CACHE_FILE):
        try:
            with open(UNIVERSE_CACHE_FILE, 'r', encoding='utf-8') as f:
                cache = json.load(f)
            cache_date = datetime.strptime(cache.get('date', '2000-01-01'), '%Y-%m-%d')
            age_days   = (datetime.now() - cache_date).days
            if age_days < UNIVERSE_CACHE_DAYS:
                mcaps = cache.get('mcaps', {})
                filtered = _apply_mcap_filter(tickers, mcaps, min_mcap)
                print(f"[UNIVERSE] Caché vigente ({age_days}d) → "
                      f"{len(filtered)}/{len(tickers)} tickers pasan filtro mcap ≥${min_mcap/1e9:.0f}B")
                return filtered
        except Exception:
            pass

    # ── Refrescar desde yfinance ──────────────────────────────────
    print(f"[UNIVERSE] Actualizando market caps ({len(tickers)} tickers)...")
    mcaps: dict[str, float] = {}
    for i, t in enumerate(tickers):
        sys.stdout.write(f"\r  [{i+1}/{len(tickers)}] {t:<12}")
        sys.stdout.flush()
        try:
            info = yf.Ticker(t).fast_info
            mc   = getattr(info, 'market_cap', None)
            if mc and mc > 0:
                mcaps[t] = float(mc)
        except Exception:
            pass
    print()

    # Guardar caché
    try:
        with open(UNIVERSE_CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump({'date': today_str, 'mcaps': mcaps}, f, indent=2)
        print(f"[UNIVERSE] Caché guardado ({len(mcaps)} market-caps)")
    except Exception as e:
        print(f"[UNIVERSE] Warning: no se pudo guardar caché: {e}")

    filtered = _apply_mcap_filter(tickers, mcaps, min_mcap)
    print(f"[UNIVERSE] {len(filtered)}/{len(tickers)} tickers pasan filtro mcap ≥${min_mcap/1e9:.0f}B")
    return filtered


def _apply_mcap_filter(tickers: list[str], mcaps: dict, min_mcap: float) -> list[str]:
    """Filtra tickers. Sin dato de mcap → pasa (conservador)."""
    result = []
    excluded = []
    for t in tickers:
        mc = mcaps.get(t)
        if mc is None:
            result.append(t)          # sin dato → incluir
        elif mc >= min_mcap:
            result.append(t)
        else:
            excluded.append(f"{t}(${mc/1e9:.1f}B)")
    if excluded:
        print(f"[UNIVERSE] Excluidos small-caps: {', '.join(excluded)}")
    return result


# ── TICKERS COMPLETO (pool inicial — el filtro de mcap reduce esto en runtime) ─
ASX_TICKERS = [
    'BHP.AX','CBA.AX','CSL.AX','WES.AX','NAB.AX','WBC.AX','ANZ.AX',
    'MQG.AX','FMG.AX','TLS.AX','RIO.AX','GMG.AX','STO.AX','WDS.AX',
    'QBE.AX','ALL.AX','SCG.AX','ORG.AX','NST.AX','SUN.AX','MIN.AX',
    'PLS.AX','IGO.AX','TCL.AX','S32.AX','REA.AX','QAN.AX','RMD.AX',
    'AMC.AX','BSL.AX','CPU.AX','ASX.AX','SHL.AX','JHX.AX','WOW.AX',
    'COH.AX','XRO.AX','TWE.AX','CAR.AX','SEK.AX',
    'APA.AX','ALD.AX','ARB.AX','AZJ.AX','BPT.AX','BRG.AX','BWP.AX',
    'CDA.AX','CHC.AX','CIP.AX','CLW.AX','CMW.AX','CNU.AX','COL.AX',
    'CQR.AX','CWY.AX','DMP.AX','DXS.AX','EDV.AX','ELD.AX','EVN.AX',
    'FLT.AX','GNC.AX','GOZ.AX','GWA.AX','HLS.AX','HVN.AX','IEL.AX',
    'ILU.AX','INA.AX','IRE.AX','JIN.AX','LLC.AX','LYC.AX','MFG.AX',
    'MGR.AX','MPL.AX','NHC.AX','NHF.AX','NSR.AX','NXT.AX','ORA.AX',
    'ORI.AX','PDN.AX','PME.AX','PMV.AX','PNI.AX','QUB.AX','RHC.AX',
    'RRL.AX','RSG.AX','SBM.AX','SCP.AX','SDF.AX','SGM.AX','SGP.AX',
    'SIQ.AX','SKC.AX','SOL.AX','SPK.AX','SPL.AX','SST.AX','SUL.AX',
    'TAH.AX','TPG.AX','VCX.AX','VEA.AX','WOR.AX','WTC.AX','ZIM.AX',
]

# ══════════════════════════════════════════════════════════════════
# TELEGRAM HELPERS
# ══════════════════════════════════════════════════════════════════

def tg_send(text: str, reply_markup=None) -> dict:
    """Envía mensaje Telegram. Retorna la respuesta JSON."""
    if not TELEGRAM_TOKEN or TELEGRAM_CHAT_ID == 0:
        return {
            'ok': False,
            'error': 'telegram_not_configured',
            'description': 'TELEGRAM_TOKEN / TELEGRAM_CHAT_ID no configurados'
        }

    payload = {
        'chat_id':    TELEGRAM_CHAT_ID,
        'text':       text[:4096],
        'parse_mode': 'HTML',
    }
    if reply_markup:
        payload['reply_markup'] = json.dumps(reply_markup)
    try:
        r = requests.post(f"{TG_BASE}/sendMessage", data=payload, timeout=15)
        return r.json()
    except Exception as e:
        print(f"[TG ERROR] {e}")
        return {}


def _tg_ok(resp: dict) -> bool:
    """Normaliza respuesta de Telegram para saber si el envío fue exitoso."""
    return isinstance(resp, dict) and bool(resp.get('ok'))


def tg_edit(chat_id, message_id, text: str) -> None:
    """Edita un mensaje ya enviado (para confirmar acción del botón)."""
    try:
        requests.post(f"{TG_BASE}/editMessageText", data={
            'chat_id':    chat_id,
            'message_id': message_id,
            'text':       text[:4096],
            'parse_mode': 'HTML',
        }, timeout=10)
    except Exception:
        pass


def tg_answer_callback(callback_query_id: str, text: str = '') -> None:
    """Responde el callback para quitar el loading del botón."""
    try:
        requests.post(f"{TG_BASE}/answerCallbackQuery", data={
            'callback_query_id': callback_query_id,
            'text': text,
        }, timeout=10)
    except Exception:
        pass


def tg_get_updates(offset: int = 0, timeout: int = 30) -> list:
    """Obtiene updates del bot (long-polling)."""
    try:
        r = requests.get(f"{TG_BASE}/getUpdates", params={
            'offset':  offset,
            'timeout': timeout,
            'allowed_updates': json.dumps(['callback_query', 'message']),
        }, timeout=timeout + 5)
        data = r.json()
        if data.get('ok'):
            return data.get('result', [])
    except Exception as e:
        print(f"[TG POLL ERROR] {e}")
    return []


# ══════════════════════════════════════════════════════════════════
# PORTFOLIO (current_portfolio.json)
# ══════════════════════════════════════════════════════════════════

def load_portfolio() -> dict:
    if os.path.exists(PORTFOLIO_FILE):
        try:
            with open(PORTFOLIO_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_portfolio(pf: dict) -> None:
    with open(PORTFOLIO_FILE, 'w', encoding='utf-8') as f:
        json.dump(pf, f, indent=2, ensure_ascii=False)


def calc_commission(gross: float) -> float:
    return max(COMMISSION_FLAT, gross * COMMISSION_RATE)


# ── STOP COOLDOWN HISTORY (V37) ───────────────────────────────────
STOP_HISTORY_FILE = 'stop_history.json'

def _load_stop_history() -> dict:
    """Carga historial de stop-loss {ticker: 'YYYY-MM-DD'}."""
    if os.path.exists(STOP_HISTORY_FILE):
        try:
            with open(STOP_HISTORY_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def _save_stop_history(hist: dict) -> None:
    with open(STOP_HISTORY_FILE, 'w') as f:
        json.dump(hist, f, indent=2)

def record_stop_loss(ticker: str) -> None:
    """Registra fecha de hoy como último stop-loss para el ticker."""
    hist = _load_stop_history()
    hist[ticker] = datetime.now().strftime('%Y-%m-%d')
    _save_stop_history(hist)



def calc_shares(price: float, capital: float = CAPITAL_DEFAULT,
                prob: float = PROB_SIZE_MAX) -> int:
    """
    Sizing dinámico V37: posición = capital × (prob / PROB_SIZE_MAX).
    Mínimo MIN_POSITION AUD, máximo capital×MAX_POS_PCT.
    """
    frac         = max(PROB_SIZE_MIN, min(PROB_SIZE_MAX, prob)) / PROB_SIZE_MAX
    position_aud = max(MIN_POSITION, capital * frac)
    position_aud = min(position_aud, capital * MAX_POS_PCT)
    return max(1, int(position_aud / price))


def add_position(ticker: str, price: float, shares: int,
                 stop: float, tp1: float, tp2: float,
                 estrategia: str, indicadores: dict,
                 capital: float = CAPITAL_DEFAULT,
                 atr_entry: float = 0.0,
                 be_target: float = 0.0) -> None:
    """Agrega posición al portfolio."""
    pf = load_portfolio()
    comm = calc_commission(shares * price)
    pf[ticker] = {
        'buy_price':   round(price, 3),
        'buy_date':    datetime.now().strftime('%Y-%m-%d'),
        'shares':      shares,
        'monto_aud':   round(shares * price + comm, 2),
        'stop':        round(stop, 3),
        'tp1':         round(tp1, 3),
        'tp2':         round(tp2, 3),
        'atr_entry':   round(atr_entry, 4),
        'be_target':   round(be_target, 3),
        'estrategia':  estrategia,
        'be_lock':     False,
        'tp1_hit':     False,
        'capital':     capital,
        'indicadores': indicadores,
    }
    save_portfolio(pf)
    print(f"[PORTFOLIO] + {ticker} @ AUD {price:.3f}  x{shares} accs  monto AUD {shares*price+comm:.0f}")


def close_position(ticker: str, sell_price: float, reason: str = 'MANUAL') -> dict | None:
    """Cierra posición, calcula P&L, guarda en historial."""
    pf = load_portfolio()
    if ticker not in pf:
        print(f"[PORTFOLIO] {ticker} no encontrado en portfolio")
        return None

    pos        = pf.pop(ticker)
    shares     = pos['shares']
    buy_price  = pos['buy_price']
    buy_date   = pos['buy_date']
    estrategia = pos.get('estrategia', '?')

    sell_gross  = shares * sell_price
    sell_comm   = calc_commission(sell_gross)
    buy_gross   = shares * buy_price
    buy_comm    = calc_commission(buy_gross)

    pnl_aud = sell_gross - sell_comm - buy_gross - buy_comm
    pnl_pct = (sell_price - buy_price) / buy_price

    hold_days = (datetime.now() - datetime.strptime(buy_date, '%Y-%m-%d')).days
    resultado = 'WIN' if pnl_aud > 0 else 'LOSS'

    trade = {
        'ticker':      ticker,
        'estrategia':  estrategia,
        'resultado':   resultado,
        'buy_date':    buy_date,
        'sell_date':   datetime.now().strftime('%Y-%m-%d'),
        'buy_price':   buy_price,
        'sell_price':  round(sell_price, 3),
        'shares':      shares,
        'pnl_aud':     round(pnl_aud, 2),
        'pnl_pct':     round(pnl_pct * 100, 2),
        'hold_days':   hold_days,
        'razon':       reason,
    }

    # Guardar historial
    history = []
    if os.path.exists(TRADE_HISTORY_FILE):
        try:
            with open(TRADE_HISTORY_FILE, 'r', encoding='utf-8') as f:
                history = json.load(f)
        except Exception:
            history = []
    history.append(trade)
    with open(TRADE_HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    save_portfolio(pf)
    sign = '+' if pnl_aud >= 0 else ''
    print(f"[PORTFOLIO] - {ticker} @ AUD {sell_price:.3f}  P&L: {sign}AUD {pnl_aud:.2f} ({sign}{pnl_pct*100:.2f}%) [{reason}]")
    return trade


# ══════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING (misma función que sim_2k.py)
# ══════════════════════════════════════════════════════════════════

FEATURE_COLS = [
    'dist_sma20','dist_sma50','dist_sma200','ma_cross_20_50','ma_cross_50_200',
    'atr_pct','vol_regime','momentum_3','momentum_5','momentum_10','momentum_20',
    'roc_5','roc_10','rsi','macd_diff','adx',
    'bb_width','bb_upper_dist','bb_lower_dist',
    'stoch_k','stoch_d','williams_r','cci',
    'vol_rel_20','vol_rel_50',
    'close_to_max5','close_to_min5','close_to_max20','close_to_min20',
]


def fetch_live(ticker: str, start: str, end: str = None) -> pd.DataFrame | None:
    try:
        kw = dict(start=start, progress=False, auto_adjust=True)
        if end:
            kw['end'] = end
        df = yf.download(ticker, **kw)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        df.index   = pd.to_datetime(df.index)
        df.index.name = 'date'
        needed = ['close','high','low','open','volume']
        if not all(c in df.columns for c in needed):
            return None
        return df[needed].copy()
    except Exception:
        return None


def fetch_intraday(ticker: str, interval: str = '15m') -> pd.DataFrame | None:
    """
    Descarga velas intraday de las últimas 24h  (yfinance gratis: interval 5m/15m, hasta 60 días).
    Úsalo para detectar si stop o TP fue tocado intradía aunque el precio se haya recuperado,
    y para calcular VWAP de la sesión actual y dar nota de timing de entrada.
    """
    try:
        df = yf.download(ticker, period='2d', interval=interval, progress=False, auto_adjust=True)
        if df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]
        needed = ['close', 'high', 'low', 'open', 'volume']
        if not all(c in df.columns for c in needed):
            return None
        df.index = pd.to_datetime(df.index)
        # Normalizar a UTC y filtrar solo las últimas ~24 horas (sesión de hoy)
        if df.index.tz is None:
            df.index = df.index.tz_localize('UTC')
        cutoff = pd.Timestamp.now(tz='UTC') - pd.Timedelta(hours=24)
        df = df[df.index >= cutoff]
        if df.empty:
            return None
        return df[needed].copy()
    except Exception:
        return None


def engineer(raw_df: pd.DataFrame) -> pd.DataFrame | None:
    df = raw_df.copy()
    if len(df) < 120:
        return None
    c = df['close']

    df['sma_20']  = ta_lib.trend.sma_indicator(c, 20)
    df['sma_50']  = ta_lib.trend.sma_indicator(c, 50)
    df['sma_200'] = ta_lib.trend.sma_indicator(c, 200)
    df['dist_sma20']  = (c / df['sma_20'])  - 1
    df['dist_sma50']  = (c / df['sma_50'])  - 1
    df['dist_sma200'] = (c / df['sma_200']) - 1
    df['ma_cross_20_50']  = (df['sma_20']  > df['sma_50']).astype(int)
    df['ma_cross_50_200'] = (df['sma_50']  > df['sma_200']).astype(int)

    atr = ta_lib.volatility.average_true_range(df['high'], df['low'], c, 14)
    df['atr']     = atr
    df['atr_pct'] = np.where(c > 0, atr / c, 0)
    atr_ma = atr.rolling(50).mean()
    df['vol_regime'] = np.where(atr_ma > 0, atr / atr_ma, 1.0)

    for p in [3, 5, 10, 20]:
        df[f'momentum_{p}'] = c.pct_change(p)
    df['roc_5']     = ta_lib.momentum.roc(c, 5)
    df['roc_10']    = ta_lib.momentum.roc(c, 10)
    df['rsi']       = ta_lib.momentum.rsi(c, 14) / 100.0
    df['macd_diff'] = ta_lib.trend.macd_diff(c)
    df['adx']       = ta_lib.trend.adx(df['high'], df['low'], c, 14)

    bb_h = ta_lib.volatility.bollinger_hband(c, 20)
    bb_l = ta_lib.volatility.bollinger_lband(c, 20)
    df['bb_width']      = (bb_h - bb_l) / c
    df['bb_upper_dist'] = (bb_h - c) / c
    df['bb_lower_dist'] = (c - bb_l) / c

    df['stoch_k']    = ta_lib.momentum.stoch(df['high'], df['low'], c, 14, 3) / 100.0
    df['stoch_d']    = ta_lib.momentum.stoch_signal(df['high'], df['low'], c, 14, 3) / 100.0
    df['williams_r'] = ta_lib.momentum.williams_r(df['high'], df['low'], c, 14)
    df['cci']        = ta_lib.trend.cci(df['high'], df['low'], c, 20)

    vol  = df['volume']
    vm20 = vol.rolling(20).mean()
    vm50 = vol.rolling(50).mean()
    df['vol_rel_20'] = np.where(vm20 > 0, vol / vm20, 1.0)
    df['vol_rel_50'] = np.where(vm50 > 0, vol / vm50, 1.0)

    for w, sfx in [(5, '5'), (20, '20')]:
        df[f'max_{sfx}'] = c.rolling(w).max()
        df[f'min_{sfx}'] = c.rolling(w).min()
    df['close_to_max5']  = (c / df['max_5'])  - 1
    df['close_to_min5']  = (c / df['min_5'])  - 1
    df['close_to_max20'] = (c / df['max_20']) - 1
    df['close_to_min20'] = (c / df['min_20']) - 1

    # NORMALIZACIÓN ROLLING
    skip = {'close','open','high','low','volume',
            'sma_20','sma_50','sma_200','atr','max_5','min_5','max_20','min_20'}
    for col in df.columns:
        if col in skip:
            continue
        if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
            rm = df[col].rolling(252, min_periods=50).mean()
            rs = df[col].rolling(252, min_periods=50).std()
            df[col] = np.where(rs > 0, (df[col] - rm) / rs, 0.0)

    return df.replace([np.inf, -np.inf], np.nan).dropna()


# ══════════════════════════════════════════════════════════════════
# SCAN DE SEÑALES DE COMPRA (lógica v4 exacta de sim_2k.py)
# ══════════════════════════════════════════════════════════════════

def scan_buy_signals(capital: float = CAPITAL_DEFAULT) -> list[dict]:
    """
    Escanea todos los tickers con los mismos filtros V37.
    Retorna lista de señales ordenadas por score.
    """
    # ── Config personalizada (user_alert_config.json override) ───────
    import csv
    user_config_file = "user_alert_config.json"
    def get_user_config():
        if os.path.exists(user_config_file):
            with open(user_config_file, "r") as f:
                return json.load(f)
        return {"prob_min": PROB_MOMENTUM, "rsi_max": 78, "vol_max": 2.0}
    user_config = get_user_config()

    # ── Log de alertas históricas ─────────────────────────────────────
    alert_history_file = "alert_history.csv"
    def log_alert(ticker, prob, price, rsi, adx, macd, vol_ratio, mom5, sma50, sma200):
        with open(alert_history_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([datetime.now().isoformat(), ticker, prob, price,
                             rsi, adx, macd, vol_ratio, mom5, sma50, sma200])

    if not os.path.exists(MODEL_CACHE_FILE) or not os.path.exists(FEATURE_CACHE_FILE):
        print("[ERROR] No se encontró models_cache.joblib. Ejecuta primero train_model_2021_2025.py")
        return []

    print("[SCAN] Cargando modelos desde caché...")
    models = joblib.load(MODEL_CACHE_FILE)
    feats  = joblib.load(FEATURE_CACHE_FILE)
    print(f"[SCAN] {len(models)} modelos cargados")

    today    = datetime.now().strftime('%Y-%m-%d')
    start    = (datetime.now() - timedelta(days=600)).strftime('%Y-%m-%d')

    portfolio  = load_portfolio()
    signals    = []
    near_misses = []   # candidatos con prob ≥ 0.28 aunque fallen filtros técnicos

    # ── Filtro de universo: solo large-caps ≥ UNIVERSE_MIN_MCAP ──────────
    active_tickers = fetch_universe(ASX_TICKERS)

    print(f"[SCAN] Descargando datos y analizando {len(active_tickers)} tickers "
          f"(de {len(ASX_TICKERS)} universo, filtro mcap ≥${UNIVERSE_MIN_MCAP/1e9:.0f}B)...")
    for i, ticker in enumerate(active_tickers):
        sys.stdout.write(f"\r  [{i+1}/{len(ASX_TICKERS)}] {ticker:<10}")
        sys.stdout.flush()

        # Saltar si ya está en portfolio
        if ticker in portfolio:
            continue

        if ticker not in models:
            continue

        raw = fetch_live(ticker, start=start)
        if raw is None or len(raw) < 210:
            continue

        df_eng = engineer(raw)
        if df_eng is None or df_eng.empty:
            continue

        # Obtener última fila para predicción
        feats_t = feats.get(ticker, [])
        avail   = [f for f in feats_t if f in df_eng.columns]
        if not avail:
            continue

        try:
            last_row = df_eng[avail].replace([np.inf, -np.inf], np.nan).dropna()
            if last_row.empty:
                continue
            prob = float(models[ticker].predict_proba(last_row.values[-1:])[:, 1][0])
        except Exception:
            continue

        # ── Indicadores técnicos sobre precios RAW (sin normalizar) ──────
        raw_slice  = raw
        raw_close  = raw_slice['close']
        real_price = float(raw_close.iloc[-1])
        if real_price <= 0:
            continue

        sma50_raw  = float(raw_close.rolling(50).mean().iloc[-1])
        sma200_raw = float(raw_close.rolling(200).mean().iloc[-1])
        rsi_raw    = float(ta_lib.momentum.rsi(raw_close, 14).iloc[-1])
        macd_d_raw = float(ta_lib.trend.macd_diff(raw_close).iloc[-1])
        adx_raw    = float(ta_lib.trend.adx(raw_slice['high'], raw_slice['low'], raw_close, 14).iloc[-1])
        vol_avg20  = float(raw_slice['volume'].rolling(20).mean().iloc[-1])
        vol_ratio  = float(raw_slice['volume'].iloc[-1]) / vol_avg20 if vol_avg20 > 0 else 0.0
        mom5       = float(raw_close.pct_change(5).iloc[-1])
        atr_now    = float(ta_lib.volatility.average_true_range(
            raw_slice['high'], raw_slice['low'], raw_close, 14).iloc[-1])
        # Gate 3m: precio actual > precio hace 63 días (slope positivo)
        gate_3m    = bool(len(raw_close) >= 63 and real_price > float(raw_close.iloc[-63]))

        if any(math.isnan(x) for x in [sma50_raw, sma200_raw, rsi_raw, macd_d_raw, adx_raw]):
            continue

        # ── STOP_COOLDOWN V37: saltar si el ticker tuvo stop-loss reciente ─
        cooldown_hist = _load_stop_history()
        if ticker in cooldown_hist:
            try:
                last_stop_dt = datetime.strptime(cooldown_hist[ticker], '%Y-%m-%d')
                if (datetime.now() - last_stop_dt).days < STOP_COOLDOWN_DAYS:
                    continue
            except Exception:
                pass

        # ── Near-miss tracking (prob >= 0.42, aunque filtros fallen) ─────
        if prob >= 0.42:
            _sh  = calc_shares(real_price, capital, prob)
            _com = calc_commission(_sh * real_price)
            _sl  = round(real_price - SL_ATR * atr_now, 3)
            near_misses.append({
                'ticker':      ticker,
                'prob':        round(prob, 4),
                'price':       real_price,
                'rsi':         round(rsi_raw, 1),
                'adx':         round(adx_raw, 1),
                'macd':        'd>0' if macd_d_raw > 0 else 'd<0',
                'vol_ratio':   round(vol_ratio, 3),
                'mom5pct':     round(mom5 * 100, 2),
                'above_sma50': real_price > sma50_raw,
                'stop':        _sl,
                'tp1':         round(real_price * (1 + TP1_PCT), 3),
                'tp2':         round(real_price * (1 + TP2_PCT), 3),
                'riesgo_aud':  round(_sh * (real_price - _sl), 2),
                'monto_aud':   round(_sh * real_price + _com, 2),
                'shares':      _sh,
            })

        # ── FILTROS BULL V37 (pure MOMENTUM — sin tier REVERSION) ────────
        if prob >= user_config.get("prob_min", PROB_MOMENTUM):
            # Golden Cross exacto: SMA50 debe superar SMA200
            if sma50_raw <= sma200_raw                      : continue
            # Precio bien posicionado: dist_SMA200 ≥ +2%
            if real_price < sma200_raw * 1.02               : continue
            # Gate 3m: slope positivo en 63 días
            if not gate_3m                                   : continue
            # Filtros técnicos estándar
            if rsi_raw >= user_config.get("rsi_max", 78)    : continue
            if macd_d_raw <= 0                               : continue
            if adx_raw < 15                                  : continue
            if vol_ratio < VOL_MIN_MOM                       : continue
            if mom5 < MOM5_MIN                               : continue

            print(f"[ALERTA V37] {ticker} | Prob: {prob:.3f} | Precio: {real_price:.2f} | "
                  f"RSI: {rsi_raw:.1f} | ADX: {adx_raw:.1f} | "
                  f"MACD: {'d>0' if macd_d_raw > 0 else 'd<0'} | "
                  f"VolRatio: {vol_ratio:.3f} | Mom5: {mom5*100:.2f}% | "
                  f"SMA50: {sma50_raw:.2f} | SMA200: {sma200_raw:.2f} | "
                  f"Gate3m: {gate_3m}")
            log_alert(ticker, prob, real_price, rsi_raw, adx_raw,
                      'd>0' if macd_d_raw > 0 else 'd<0',
                      vol_ratio, mom5*100, sma50_raw, sma200_raw)
            estrategia = 'BULL_V37'
            score = prob * (1.0 + max(0.0, min(1.0, mom5 * 15)))

        else:
            continue

        # ── Calcular stops ATR-based V37 ──────────────────────────────────
        stop_atr  = round(real_price - SL_ATR * atr_now, 3)
        be_target = round(real_price + BE_TRIGGER_ATR * atr_now, 3)   # trigger breakeven
        tp1_use   = TP1_PCT
        tp2_use   = max(TP2_PCT, (atr_now / real_price) * 3.5)

        shares    = calc_shares(real_price, capital, prob)
        comm      = calc_commission(shares * real_price)
        monto_aud = round(shares * real_price + comm, 2)
        riesgo_aud = round(shares * (real_price - stop_atr), 2)

        # ── Nota de timing intradía basada en VWAP de la sesión ──────────
        entry_note = ''
        vwap_val   = None
        intraday_s = fetch_intraday(ticker, '15m')
        if intraday_s is not None and not intraday_s.empty and len(intraday_s) >= 3:
            vol_sum = float(intraday_s['volume'].sum())
            if vol_sum > 0:
                typical  = (intraday_s['high'] + intraday_s['low'] + intraday_s['close']) / 3
                vwap_val = float((typical * intraday_s['volume']).sum() / vol_sum)
                dist_v   = (real_price - vwap_val) / vwap_val
                if dist_v > 0.02:
                    entry_note = f'ESPERA PULLBACK (+{dist_v*100:.1f}% sobre VWAP {vwap_val:.3f})'
                elif dist_v < -0.005:
                    entry_note = f'ENTRADA IDEAL ({dist_v*100:.1f}% bajo VWAP {vwap_val:.3f})'
                else:
                    entry_note = f'CERCA DE VWAP ({vwap_val:.3f})'

        signals.append({
            'ticker':     ticker,
            'estrategia': estrategia,
            'score':      round(score, 4),
            'price':      real_price,
            'prob':       round(prob, 4),
            'stop':       stop_atr,
            'be_target':  be_target,
            'tp1':        round(real_price * (1 + tp1_use), 3),
            'tp2':        round(real_price * (1 + tp2_use), 3),
            'atr_entry':  round(atr_now, 4),
            'shares':     shares,
            'monto_aud':  monto_aud,
            'riesgo_aud': riesgo_aud,
            'indicadores': {
                'prob':      round(prob, 4),
                'rsi':       round(rsi_raw, 1),
                'macd':      round(macd_d_raw, 5),
                'adx':       round(adx_raw, 1),
                'vol_ratio': round(vol_ratio, 3),
                'dist_sma50':  round((real_price / sma50_raw - 1) * 100, 2),
                'dist_sma200': round((real_price / sma200_raw - 1) * 100, 2),
                'momentum5': round(mom5 * 100, 2),
                'gate_3m':   gate_3m,
                'atr':       round(atr_now, 4),
            },
            'entry_note': entry_note,
            'vwap':       round(vwap_val, 3) if vwap_val else None,
        })

    print(f"\n[SCAN] Señales encontradas: {len(signals)}")
    near_sorted = sorted(near_misses, key=lambda x: x['prob'], reverse=True)[:10]
    return sorted(signals, key=lambda x: x['score'], reverse=True), near_sorted


# ══════════════════════════════════════════════════════════════════
# MONITOREO DE POSICIONES ABIERTAS (condiciones de salida v4)
# ══════════════════════════════════════════════════════════════════

def check_portfolio_exits() -> list[dict]:
    """
    Revisa cada posición abierta y detecta si se debe vender.
    Retorna lista de alertas de venta pendientes.
    """
    portfolio = load_portfolio()
    if not portfolio:
        return []

    today    = datetime.now().strftime('%Y-%m-%d')
    start    = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
    exits    = []
    updated  = False

    for ticker, pos in list(portfolio.items()):
        raw = fetch_live(ticker, start=start)
        if raw is None or raw.empty:
            continue

        raw_close  = raw['close']
        price_now  = float(raw_close.iloc[-1])

        # ── Intraday OHLC: detecta toque de stop/TP durante el día ────────
        intraday   = fetch_intraday(ticker, '15m')
        today_high = float(intraday['high'].max()) if intraday is not None and not intraday.empty else price_now
        today_low  = float(intraday['low'].min())  if intraday is not None and not intraday.empty else price_now

        buy_price  = float(pos['buy_price'])
        buy_date   = datetime.strptime(pos['buy_date'], '%Y-%m-%d')
        hold_days  = (datetime.now() - buy_date).days
        stop       = float(pos['stop'])
        tp1        = float(pos['tp1'])
        tp2        = float(pos['tp2'])
        be_lock    = pos.get('be_lock', False)
        tp1_hit    = pos.get('tp1_hit', False)
        unrealized = (price_now - buy_price) / buy_price

        # 1. Actualizar Breakeven lock V37 (ATR-based trigger)
        be_trigger = float(pos.get('be_target', 0.0)) or buy_price * 1.035
        if not be_lock and hold_days >= 2 and (price_now >= be_trigger or today_high >= be_trigger):
            new_stop = max(stop, buy_price)    # mover stop a break-even (entrada)
            portfolio[ticker]['stop'] = round(new_stop, 3)
            portfolio[ticker]['be_lock'] = True
            be_lock = True
            updated = True
            print(f"[BE LOCK V37] {ticker}: stop → AUD {new_stop:.3f} (entrada) | trigger era {be_trigger:.3f}")

        # 2. TP1 hit → registrar, stop sube a breakeven (incluyendo toque intradía)
        if not tp1_hit and (price_now >= tp1 or today_high >= tp1):
            portfolio[ticker]['tp1_hit'] = True
            tp1_hit = True
            if not be_lock:
                portfolio[ticker]['stop'] = round(max(stop, buy_price), 3)
                portfolio[ticker]['be_lock'] = True
                be_lock = True
            updated = True

        razon = None
        urgencia = 'normal'

        # 3. Stop Loss hit (precio actual O mínimo intradía tocó el stop)
        if price_now <= stop or today_low <= stop:
            razon    = 'STOP_LOSS'
            urgencia = 'urgente'
            record_stop_loss(ticker)   # V37: cooldown 5 días

        # 4. TP2 hit (precio actual O máximo intradía superó TP2)
        elif price_now >= tp2 or today_high >= tp2:
            razon   = 'TP2_ALCANZADO'
            urgencia = 'normal'

        # 5. MACD invertido post-TP1 o post-BE
        elif hold_days >= 3:
            try:
                macd_now = float(ta_lib.trend.macd_diff(raw_close).iloc[-1])
                min_profit = 0.005 if be_lock else 0.01
                if macd_now < 0 and unrealized > min_profit:
                    razon    = 'MACD_INVERTIDO'
                    urgencia = 'normal'
            except Exception:
                pass

        # 6. Max hold days
        if razon is None and hold_days >= MAX_HOLD_DAYS:
            razon    = 'MAX_DIAS'
            urgencia = 'normal'

        if razon:
            exits.append({
                'ticker':    ticker,
                'razon':     razon,
                'urgencia':  urgencia,
                'price_now': price_now,
                'today_high': round(today_high, 3),
                'today_low':  round(today_low, 3),
                'buy_price': buy_price,
                'hold_days': hold_days,
                'pnl_pct':   round(unrealized * 100, 2),
                'pnl_aud':   round(pos['shares'] * (price_now - buy_price) - COMMISSION_FLAT * 2, 2),
                'shares':    pos['shares'],
                'tp1_hit':   tp1_hit,
                'be_lock':   be_lock,
                'estrategia': pos.get('estrategia', '?'),
            })

    if updated:
        save_portfolio(portfolio)

    return exits


# ══════════════════════════════════════════════════════════════════
# MENSAJES DE ALERTA
# ══════════════════════════════════════════════════════════════════

def send_buy_alert(signal: dict) -> bool:
    """Envía alerta de compra con botones inline. Retorna True si Telegram confirmó entrega."""
    t   = signal['ticker']
    pr  = signal['price']
    ind = signal['indicadores']
    est = signal['estrategia']

    emoji = '🚀' if est == 'MOMENTUM' else '🔄'
    sign_prob = '🟢' if signal['prob'] >= 0.55 else '🟡'

    # ── Cálculo de capital, escenarios y Expected Value ───────────────
    tp1_use_pct  = TP1_PCT
    tp2_use_pct  = signal['tp2'] / pr - 1
    tp1_gain_aud = round(signal['shares'] * pr * tp1_use_pct - COMMISSION_FLAT, 0)
    tp2_gain_aud = round(signal['shares'] * pr * tp2_use_pct - COMMISSION_FLAT, 0)
    risk_aud     = signal['riesgo_aud']
    prob_n       = signal['prob']
    ev_tp1       = round(prob_n * tp1_gain_aud - (1 - prob_n) * risk_aud, 0)
    pct_cap      = round(signal['monto_aud'] / CAPITAL_DEFAULT * 100, 0)
    atr_val      = signal.get('atr_entry', 0.0)
    be_tgt       = signal.get('be_target', round(pr * 1.035, 3))
    sl_pct       = round((pr - signal['stop']) / pr * 100, 1) if pr > 0 else 5.0

    text = (
        f"{emoji} <b>SEÑAL V37 — {t}</b>\n"
        f"{'━'*30}\n"
        f"{sign_prob} <b>Prob IA:</b> {signal['prob']:.1%}  |  Modelo: {est}\n"
        f"📅 {datetime.now().strftime('%d/%m/%Y %H:%M')}\n\n"
        f"💵 <b>Precio entrada:</b>  AUD {pr:.3f}\n"
        f"🛑 <b>Stop ATR (2.5×ATR={atr_val:.3f}):</b>  AUD {signal['stop']:.3f}"
            f"  → riesgo <b>AUD {risk_aud:.0f}</b> (−{sl_pct:.1f}%)\n"
        f"🔒 <b>Break-even (+1.5×ATR):</b> AUD {be_tgt:.3f} → stop → entrada\n"
        f"🎯 <b>TP1 parcial (+{tp1_use_pct*100:.0f}%):</b> AUD {signal['tp1']:.3f}"
            f"  → ganancia <b>+AUD {tp1_gain_aud:.0f}</b>\n"
        f"🏆 <b>TP2 final   (+{tp2_use_pct*100:.0f}%):</b> AUD {signal['tp2']:.3f}"
            f"  → ganancia <b>+AUD {tp2_gain_aud:.0f}</b>\n\n"
        f"💰 <b>Capital de tus AUD {CAPITAL_DEFAULT:.0f}:</b>\n"
        f"   Invertir: <b>AUD {signal['monto_aud']:.0f}</b> ({pct_cap:.0f}%)"
            f"  ×{signal['shares']} acciones\n\n"
        f"🎲 <b>Expected Value (EV):</b> <b>+AUD {ev_tp1:.0f}</b>\n"
        f"   (prob {prob_n:.0%} × +{tp1_gain_aud:.0f} − {1-prob_n:.0%} × {risk_aud:.0f})\n\n"
        f"📊 <b>Indicadores:</b>\n"
        f"   RSI: {ind['rsi']} | ADX: {ind['adx']} | Vol×: {ind['vol_ratio']} | Mom5d: {ind['momentum5']}%\n"
        f"   Dist SMA50: +{ind['dist_sma50']:.1f}% | MACD: {ind['macd']:.4f}"
        f" | Gate3m: {'✅' if ind.get('gate_3m') else '❌'}"
    )
    entry_note = signal.get('entry_note', '')
    vwap_val   = signal.get('vwap')
    if entry_note:
        emoji_note = '⚠️' if 'ESPERA' in entry_note else ('✅' if 'IDEAL' in entry_note else '🔵')
        text += f"\n\n⏱ <b>Timing intradía (VWAP):</b>\n   {emoji_note} {entry_note}"
    else:
        text += "\n\n⏱ <i>Timing intradía: sin datos (mercado cerrado o pre-apertura)</i>"

    # Callback data: "bought:QBE.AX:21.670:103" o "ignore:QBE.AX"
    callback_buy    = f"bought:{t}:{pr:.3f}:{signal['shares']}:{signal['stop']:.3f}:{signal['tp1']:.3f}:{signal['tp2']:.3f}:{est}"
    callback_ignore = f"ignore:{t}"

    markup = {
        'inline_keyboard': [[
            {'text': '✅ Compré', 'callback_data': callback_buy[:60]},
            {'text': '❌ Ignorar', 'callback_data': callback_ignore},
        ]]
    }
    # Guardar datos completos en archivo temporal para recuperar en callback
    _save_pending_buy(t, signal)

    resp = tg_send(text, reply_markup=markup)
    if _tg_ok(resp):
        print(f"[ALERT BUY] {t} @ AUD {pr:.3f}  prob={signal['prob']:.1%}")
        return True

    desc = resp.get('description', 'sin detalle') if isinstance(resp, dict) else 'sin respuesta JSON'
    print(f"[ALERT BUY ERROR] {t} no enviada. Motivo: {desc}")
    return False


def send_sell_alert(exit_info: dict) -> bool:
    """Envía alerta de venta con botones inline. Retorna True si Telegram confirmó entrega."""
    t         = exit_info['ticker']
    razon     = exit_info['razon']
    price_now = exit_info['price_now']
    pnl_pct   = exit_info['pnl_pct']
    pnl_aud   = exit_info['pnl_aud']
    urgencia  = exit_info['urgencia']

    sign_pnl = '✅ +' if pnl_aud >= 0 else '❌ '
    emoji_ur = '🚨' if urgencia == 'urgente' else '💡'

    razon_texto = {
        'STOP_LOSS':      '🛑 Stop Loss alcanzado — SALIR INMEDIATAMENTE',
        'TP2_ALCANZADO':  '🏆 Objetivo TP2 alcanzado — Tomar ganancias',
        'MACD_INVERTIDO': '📉 MACD se invirtió — Señal de agotamiento',
        'MAX_DIAS':       '⏰ Máximo de días alcanzado (15d)',
    }.get(razon, razon)

    be_txt  = ' | 🔒 BE activo' if exit_info.get('be_lock') else ''
    tp1_txt = ' | 🎯 TP1 ya cobrado' if exit_info.get('tp1_hit') else ''

    text = (
        f"{emoji_ur} <b>SEÑAL DE VENTA — {t}</b>\n"
        f"{'━'*30}\n"
        f"⚡ <b>{razon_texto}</b>\n\n"
        f"💵 <b>Precio actual:</b> AUD {price_now:.3f}\n"
        f"� <b>Rango intradía:</b> 🔻 {exit_info.get('today_low', price_now):.3f}  –  🔺 {exit_info.get('today_high', price_now):.3f}\n"
        f"�📥 <b>Precio entrada:</b> AUD {exit_info['buy_price']:.3f}\n"
        f"📦 <b>Acciones:</b> {exit_info['shares']}\n"
        f"💹 <b>P&L estimado:</b> {sign_pnl}AUD {abs(pnl_aud):.2f} ({sign_pnl}{abs(pnl_pct):.2f}%)\n"
        f"📅 <b>Días en trade:</b> {exit_info['hold_days']}d{be_txt}{tp1_txt}\n"
        f"📈 Estrategia: {exit_info.get('estrategia','?')}"
    )

    # Callback data para confirmar venta
    callback_sell  = f"sold:{t}:{price_now:.3f}:{razon}"
    callback_hold  = f"hold:{t}"

    markup = {
        'inline_keyboard': [[
            {'text': '✅ Vendí', 'callback_data': callback_sell[:60]},
            {'text': '🔄 Mantener', 'callback_data': callback_hold},
        ]]
    }
    resp = tg_send(text, reply_markup=markup)
    if _tg_ok(resp):
        print(f"[ALERT SELL] {t} @ AUD {price_now:.3f}  {razon}  P&L {pnl_aud:+.2f}")
        return True

    desc = resp.get('description', 'sin detalle') if isinstance(resp, dict) else 'sin respuesta JSON'
    print(f"[ALERT SELL ERROR] {t} no enviada. Motivo: {desc}")
    return False


# ── Pending buy cache (para recuperar datos completos en callback) ─

PENDING_FILE = '.pending_buys.json'

def _save_pending_buy(ticker: str, signal: dict) -> None:
    pending = {}
    if os.path.exists(PENDING_FILE):
        try:
            with open(PENDING_FILE, 'r') as f:
                pending = json.load(f)
        except Exception:
            pass
    pending[ticker] = signal
    with open(PENDING_FILE, 'w') as f:
        json.dump(pending, f)


def _load_pending_buy(ticker: str) -> dict | None:
    if not os.path.exists(PENDING_FILE):
        return None
    try:
        with open(PENDING_FILE, 'r') as f:
            return json.load(f).get(ticker)
    except Exception:
        return None


def _remove_pending_buy(ticker: str) -> None:
    if not os.path.exists(PENDING_FILE):
        return
    try:
        with open(PENDING_FILE, 'r') as f:
            pending = json.load(f)
        pending.pop(ticker, None)
        with open(PENDING_FILE, 'w') as f:
            json.dump(pending, f)
    except Exception:
        pass


# ══════════════════════════════════════════════════════════════════
# PROCESAMIENTO DE CALLBACKS (respuestas a botones)
# ══════════════════════════════════════════════════════════════════

def process_callback(update: dict) -> None:
    """Procesa un callback_query (presión de botón inline)."""
    cq      = update.get('callback_query', {})
    cq_id   = cq.get('id', '')
    data    = cq.get('data', '')
    chat_id = cq.get('message', {}).get('chat', {}).get('id')
    msg_id  = cq.get('message', {}).get('message_id')

    parts = data.split(':')
    action = parts[0] if parts else ''

    if action == 'bought':
        # bought:QBE.AX:21.670:103:stop:tp1:tp2:EST  (se trunca, usamos pending)
        ticker = parts[1] if len(parts) > 1 else ''
        signal = _load_pending_buy(ticker)

        if signal:
            add_position(
                ticker      = ticker,
                price       = signal['price'],
                shares      = signal['shares'],
                stop        = signal['stop'],
                tp1         = signal['tp1'],
                tp2         = signal['tp2'],
                estrategia  = signal['estrategia'],
                indicadores = signal['indicadores'],
                atr_entry   = signal.get('atr_entry', 0.0),
                be_target   = signal.get('be_target', 0.0),
            )
            _remove_pending_buy(ticker)
            tg_answer_callback(cq_id, f"✅ {ticker} agregado al portfolio")
            tg_edit(chat_id, msg_id,
                f"✅ <b>COMPRA CONFIRMADA — {ticker}</b>\n"
                f"Entrada: AUD {signal['price']:.3f}  x{signal['shares']} acciones\n"
                f"Stop: AUD {signal['stop']:.3f}  |  TP1: AUD {signal['tp1']:.3f}  |  TP2: AUD {signal['tp2']:.3f}\n"
                f"El sistema ahora monitorea la posición.")
        else:
            tg_answer_callback(cq_id, f"⚠️ No se encontraron datos de {ticker}")

    elif action == 'ignore':
        ticker = parts[1] if len(parts) > 1 else ''
        _remove_pending_buy(ticker)
        tg_answer_callback(cq_id, f"Señal de {ticker} ignorada")
        tg_edit(chat_id, msg_id, f"❌ Señal de {ticker} ignorada.")

    elif action == 'sold':
        # sold:QBE.AX:23.500:STOP_LOSS
        ticker    = parts[1] if len(parts) > 1 else ''
        price_str = parts[2] if len(parts) > 2 else '0'
        razon     = parts[3] if len(parts) > 3 else 'MANUAL'
        try:
            sell_price = float(price_str)
        except ValueError:
            sell_price = 0.0

        trade = close_position(ticker, sell_price, razon)
        if trade:
            sign = '+' if trade['pnl_aud'] >= 0 else ''
            emoji = '✅' if trade['pnl_aud'] >= 0 else '🔴'
            tg_answer_callback(cq_id, f"{emoji} Posición {ticker} cerrada")
            tg_edit(chat_id, msg_id,
                f"{emoji} <b>VENTA CONFIRMADA — {ticker}</b>\n"
                f"Entrada: AUD {trade['buy_price']:.3f}  →  Salida: AUD {trade['sell_price']:.3f}\n"
                f"P&amp;L: {sign}AUD {abs(trade['pnl_aud']):.2f} ({sign}{abs(trade['pnl_pct']):.2f}%)\n"
                f"Días en trade: {trade['hold_days']}d  |  {razon}\n"
                f"Posición eliminada del portfolio.")
        else:
            tg_answer_callback(cq_id, f"⚠️ No se encontró {ticker} en portfolio")

    elif action == 'hold':
        ticker = parts[1] if len(parts) > 1 else ''
        tg_answer_callback(cq_id, f"Manteniendo posición en {ticker}")
        tg_edit(chat_id, msg_id, f"🔄 <b>{ticker}</b> — Posición mantenida. Próximo chequeo mañana.")

    else:
        tg_answer_callback(cq_id, "Acción no reconocida")


def process_message(update: dict) -> None:
    """Procesa comandos de texto (/estado, /portfolio, /help)."""
    msg     = update.get('message', {})
    text    = msg.get('text', '').strip()
    chat_id = msg.get('chat', {}).get('id')

    if not text.startswith('/') or chat_id != TELEGRAM_CHAT_ID:
        return

    cmd = text.split()[0].lower()

    if cmd == '/portfolio' or cmd == '/estado':
        pf = load_portfolio()
        if not pf:
            tg_send("📂 Portfolio vacío. Sin posiciones abiertas.")
            return
        lines = ["<b>📂 PORTFOLIO ACTUAL</b>\n"]
        for ticker, pos in pf.items():
            bp   = pos['buy_price']
            stop = pos['stop']
            tp1  = pos['tp1']
            tp2  = pos['tp2']
            date = pos.get('buy_date', '?')
            be   = '🔒' if pos.get('be_lock') else '  '
            tp_  = '🎯' if pos.get('tp1_hit') else '  '
            hold = (datetime.now() - datetime.strptime(date, '%Y-%m-%d')).days
            lines.append(
                f"{be}{tp_} <b>{ticker}</b> ({pos.get('estrategia','?')})\n"
                f"   Entrada: AUD {bp:.3f}  |  {pos['shares']} acc  |  {hold}d\n"
                f"   Stop: {stop:.3f}  TP1: {tp1:.3f}  TP2: {tp2:.3f}"
            )
        tg_send('\n'.join(lines))

    elif cmd == '/historial':
        if not os.path.exists(TRADE_HISTORY_FILE):
            tg_send("📋 Sin historial de trades aún.")
            return
        with open(TRADE_HISTORY_FILE, 'r') as f:
            history = json.load(f)
        if not history:
            tg_send("📋 Sin historial de trades aún.")
            return
        wins   = [t for t in history if t['resultado'] == 'WIN']
        losses = [t for t in history if t['resultado'] == 'LOSS']
        total_pnl = sum(t['pnl_aud'] for t in history)
        wr = len(wins) / len(history) * 100 if history else 0
        lines = [
            f"<b>📋 HISTORIAL — {len(history)} trades</b>",
            f"Win Rate: {wr:.1f}% ({len(wins)}W / {len(losses)}L)",
            f"P&amp;L Total: {'+'if total_pnl>=0 else ''}AUD {total_pnl:.2f}\n",
        ]
        for t in history[-10:]:  # últimos 10
            sign = '+' if t['pnl_aud'] >= 0 else ''
            emoji = '✅' if t['pnl_aud'] >= 0 else '❌'
            lines.append(
                f"{emoji} {t['ticker']} {t['sell_date']}  "
                f"{sign}AUD {t['pnl_aud']:.2f} ({sign}{t['pnl_pct']:.1f}%) | {t['razon']}"
            )
        tg_send('\n'.join(lines))

    elif cmd == '/help' or cmd == '/ayuda':
        tg_send(
            "<b>🤖 TitanBrain Bot — Comandos</b>\n\n"
            "/portfolio  — Ver posiciones abiertas\n"
            "/estado     — Igual que /portfolio\n"
            "/historial  — Ver últimos 10 trades cerrados\n"
            "/help       — Esta ayuda\n\n"
            "<i>Las alertas se envían automáticamente al correr alerts_live.py</i>"
        )


# ══════════════════════════════════════════════════════════════════
# POLLING LOOP
# ══════════════════════════════════════════════════════════════════

def bot_poll(max_seconds: int | None = None) -> None:
    """
    Loop de polling Telegram.
    max_seconds=None → infinito (modo 'bot' permanente)
    max_seconds=N    → para después de N segundos (modo scan post-alerta)
    """
    print(f"\n[BOT] Escuchando respuestas Telegram{' (mode permanente)' if max_seconds is None else f' ({max_seconds}s)'}...")
    offset    = 0
    start_ts  = time.time()

    while True:
        if max_seconds is not None and (time.time() - start_ts) > max_seconds:
            print("[BOT] Timeout, saliendo del loop.")
            break

        updates = tg_get_updates(offset=offset, timeout=20)
        for upd in updates:
            offset = upd['update_id'] + 1
            if 'callback_query' in upd:
                process_callback(upd)
            elif 'message' in upd:
                process_message(upd)

        time.sleep(1)


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def is_market_relevant() -> bool:
    """
    Retorna True si tiene sentido correr el scan ahora.
    ASX opera Lunes-Viernes 10:00-16:00 AEST (UTC+10/+11).
    Para simplificar: descartamos fin de semana y horario nocturno local
    (fuera de 06:00-22:00 hora local del servidor).
    """
    now = datetime.now()
    if now.weekday() >= 5:     # sábado=5, domingo=6
        return False
    if now.hour < 6 or now.hour >= 22:
        return False
    return True


def run_exit_monitor() -> None:
    """
    Monitor LIVIANO — solo descarga los tickers del portfolio activo.
    Diseñado para correr cada 15 min durante el mercado sin consumir mucho tiempo.
    """
    portfolio = load_portfolio()
    if not portfolio:
        print("[MONITOR] Portfolio vacío. Nada que monitorear.")
        return

    now = datetime.now()
    print(f"\n[MONITOR] {now.strftime('%Y-%m-%d %H:%M')} — {len(portfolio)} posición(es) activa(s)")
    if not TELEGRAM_TOKEN or TELEGRAM_CHAT_ID == 0:
        print("[MONITOR WARN] Telegram no configurado: se evaluarán salidas pero no se podrán entregar alertas.")

    exits = check_portfolio_exits()
    sent_sell = 0
    if exits:
        for ex in exits:
            sent_sell += 1 if send_sell_alert(ex) else 0
        print(f"[MONITOR] {sent_sell}/{len(exits)} alerta(s) de venta entregada(s) por Telegram")
    else:
        # Mostrar estado breve de cada posición
        start = (datetime.now() - timedelta(days=10)).strftime('%Y-%m-%d')
        for ticker, pos in portfolio.items():
            raw = fetch_live(ticker, start=start)
            if raw is None or raw.empty:
                continue
            price_now  = float(raw['close'].iloc[-1])
            buy_price  = float(pos['buy_price'])
            unrealized = (price_now - buy_price) / buy_price
            stop       = pos['stop']
            tp1        = pos['tp1']
            dist_stop  = (price_now - stop) / buy_price * 100
            sign       = '+' if unrealized >= 0 else ''
            be         = ' BE🔒' if pos.get('be_lock') else ''
            print(f"  {ticker:<10} {sign}{unrealized*100:.2f}%  precio={price_now:.3f}  dist_stop={dist_stop:.1f}%{be}")
        print("[MONITOR] Sin señales de salida — posiciones dentro de parámetros")


def run_scan(capital: float = CAPITAL_DEFAULT, force: bool = False) -> None:
    """Escanea señales + chequea portfolio → envía alertas."""
    now = datetime.now()
    print(f"\n{'='*55}")
    print(f"  TitanBrain LIVE ALERTS — {now.strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*55}")
    if not TELEGRAM_TOKEN or TELEGRAM_CHAT_ID == 0:
        print("[WARN] Telegram no configurado: se detectarán señales, pero no se entregarán alertas.")

    if not force and not is_market_relevant():
        print(f"[INFO] Fuera de horario de mercado ({now.strftime('%A %H:%M')}). Saltando scan.")
        print("[INFO] Pasa '--force' para correr igual.")
        return

    # 1. Chequear posiciones abiertas (prioridad)
    print("\n[1/2] Chequeando portfolio abierto...")
    exits = check_portfolio_exits()
    sent_sell = 0
    for ex in exits:
        sent_sell += 1 if send_sell_alert(ex) else 0
    if not exits:
        pf = load_portfolio()
        if pf:
            print(f"  {len(pf)} posición(es) abierta(s) — sin señales de salida hoy")
        else:
            print("  Portfolio vacío")

    # 2. Scan nuevas señales de compra
    print("\n[2/2] Escaneando señales de compra...")
    signals, near_misses = scan_buy_signals(capital=capital)
    sent_buy = 0
    for sig in signals:
        sent_buy += 1 if send_buy_alert(sig) else 0

    if not signals and not exits:
        print("\n[INFO] Sin señales hoy — top candidatos por probabilidad:")
        if near_misses:
            hdr = f"  {'Ticker':<10} {'Prob':>6} {'Precio':>8} {'Monto':>8} {'TP1':>8} {'Stop':>8} {'EV':>7}  {'RSI':>5} {'Vol×':>5} {'Mom5%':>6}  Filtros"
            print(hdr)
            print('  ' + '-' * (len(hdr) - 2))
            for nm in near_misses:
                tp1_gain = round(calc_shares(nm['price'], capital, nm['prob']) * nm['price'] * TP1_PCT - COMMISSION_FLAT, 0)
                ev_est   = round(nm['prob'] * tp1_gain - (1 - nm['prob']) * nm['riesgo_aud'], 0)
                sma_ok   = '✅' if nm['above_sma50'] else '❌ SMA'
                rsi_ok   = '✅' if nm['rsi'] < 78 else '❌ RSI'
                macd_ok  = '✅' if nm['macd'] == 'd>0' else '❌ MACD'
                vol_ok   = '✅' if nm['vol_ratio'] >= VOL_MIN_MOM else '❌ Vol'
                mom_ok   = '✅' if nm['mom5pct'] >= MOM5_MIN * 100 else '❌ Mom'
                flags    = ' '.join(f for f in [sma_ok, rsi_ok, macd_ok, vol_ok, mom_ok] if '❌' in f) or 'todos✅'
                ev_str   = f'+{ev_est:.0f}' if ev_est >= 0 else f'{ev_est:.0f}'
                print(f"  {nm['ticker']:<10} {nm['prob']:>6.1%} {nm['price']:>8.3f}"
                      f" {nm['monto_aud']:>7.0f} {nm['tp1']:>8.3f} {nm['stop']:>8.3f}"
                      f" {ev_str:>7}  {nm['rsi']:>5.1f} {nm['vol_ratio']:>5.2f} {nm['mom5pct']:>6.1f}%"
                      f"  Falla: {flags}")
        else:
            print("  Sin candidatos con prob ≥ 28% en el universo de hoy.")
        return

    total_candidates = len(signals) + len(exits)
    total_sent = sent_buy + sent_sell
    if total_sent == total_candidates:
        print(f"\n[OK] {total_sent} alerta(s) entregada(s): {sent_buy} compra(s), {sent_sell} venta(s)")
    else:
        print(f"\n[WARN] Entrega parcial: {total_sent}/{total_candidates} alerta(s) entregada(s)")
        print(f"       Compras: {sent_buy}/{len(signals)} | Ventas: {sent_sell}/{len(exits)}")


if __name__ == '__main__':
    args = sys.argv[1:]

    # ── Alerta manual de compra (envía botones, NO agrega al portfolio) ──
    if len(args) >= 3 and args[0].lower() == 'bought':
        ticker     = args[1].upper()
        price      = float(args[2])
        shares     = int(args[3]) if len(args) > 3 else calc_shares(price)
        estrategia = args[4].upper() if len(args) > 4 else 'BULL_V37'
        # Stop manual: 5% fijo (sin ATR disponible en CLI)
        stop       = round(price * 0.95, 3)
        tp1        = round(price * (1 + TP1_PCT), 3)
        tp2        = round(price * (1 + TP2_PCT), 3)
        comm       = calc_commission(shares * price)
        signal = {
            'ticker':     ticker,
            'estrategia': estrategia,
            'score':      0,
            'price':      price,
            'prob':       0,
            'stop':       stop,
            'be_target':  round(price * 1.035, 3),
            'tp1':        tp1,
            'tp2':        tp2,
            'atr_entry':  0.0,
            'shares':     shares,
            'monto_aud':  round(shares * price + comm, 2),
            'riesgo_aud': round(shares * price * 0.05, 2),
            'indicadores': {},
        }
        send_buy_alert(signal)
        print(f"[ALERT] Alerta enviada a Telegram. Esperando confirmación via botón...")
        bot_poll(max_seconds=300)

    # ── Alerta manual de venta (envía botones, NO cierra posición directo) ──
    elif len(args) >= 3 and args[0].lower() == 'sold':
        ticker     = args[1].upper()
        sell_price = float(args[2])
        razon      = args[3].upper() if len(args) > 3 else 'MANUAL'
        pf = load_portfolio()
        if ticker not in pf:
            print(f"[ERROR] {ticker} no está en el portfolio")
        else:
            pos        = pf[ticker]
            buy_price  = pos['buy_price']
            shares     = pos['shares']
            hold_days  = (datetime.now() - datetime.strptime(pos['buy_date'], '%Y-%m-%d')).days
            pnl_aud    = round(shares * (sell_price - buy_price) - COMMISSION_FLAT * 2, 2)
            pnl_pct    = round((sell_price - buy_price) / buy_price * 100, 2)
            exit_info  = {
                'ticker':     ticker,
                'razon':      razon,
                'urgencia':   'normal',
                'price_now':  sell_price,
                'buy_price':  buy_price,
                'hold_days':  hold_days,
                'pnl_pct':    pnl_pct,
                'pnl_aud':    pnl_aud,
                'shares':     shares,
                'tp1_hit':    pos.get('tp1_hit', False),
                'be_lock':    pos.get('be_lock', False),
                'estrategia': pos.get('estrategia', '?'),
            }
            send_sell_alert(exit_info)
            print(f"[ALERT] Alerta de venta enviada a Telegram. Esperando confirmación via botón...")
            bot_poll(max_seconds=300)

    # ── Monitor liviano de portfolio (para correr cada 15min) ───────
    elif args and args[0].lower() == 'exit-monitor':
        run_exit_monitor()
        print("\n[BOT] Esperando respuestas por 60 segundos...")
        bot_poll(max_seconds=60)

    # ── Modo bot permanente ───────────────────────────────────────
    elif args and args[0].lower() == 'bot':
        bot_poll(max_seconds=None)

    # ── Solo scan (sin polling) ───────────────────────────────────
    elif args and args[0].lower() == 'scan':
        force = '--force' in args
        run_scan(force=force)

    # ── Default: scan + polling 90s para capturar respuestas rápidas ─
    else:
        force = '--force' in args
        run_scan(force=force)
        print("\n[BOT] Esperando respuestas por 90 segundos...")
        bot_poll(max_seconds=90)
