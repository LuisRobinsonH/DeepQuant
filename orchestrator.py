# orchestrator.py - TITAN V5: QUANTUM FORTRESS (FIXED)
import pandas as pd
import numpy as np
from tqdm import tqdm
from core.brain import TitanBrain
from core.risk_engine import RiskEngine
from core.db_manager import (
    clear_db, save_transaction, save_intelligence, 
    update_portfolio, initialize_db
)
import warnings
import os

# Silence unnecessary warnings
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# --- QUANTUM CONFIGURATION ---
MIN_EV_RATIO = 0.05          # EV ratio mínimo: permite más oportunidades de calidad razonable
BASE_RISK_PER_TRADE = 0.02   # Riesgo 2% del capital por operación (conservador pero activo)
MIN_TRADE_SIZE = 300         # Minimum position size in dollars
MAX_DAILY_TRADES = 5         # Máximo 5 nuevas posiciones por día
MIN_PROB_THRESHOLD = 0.48    # Probabilidad mínima: ligeramente por encima de azar (genera más señales)
MAX_DRAWDOWN_FREEZE = 0.20   # Si el drawdown supera este nivel, se congelan todas las compras


def run_quantum_fortress():
    """
    Main orchestration engine for Titan trading system.
    
    Features:
    - Walk-forward validation with temporal purging
    - Expected Value-based position selection
    - Multi-layer risk management
    - Dynamic position sizing based on system health
    """
    
    # Initialize database
    initialize_db()
    clear_db()
    
    # Initialize core engines
    brain = TitanBrain()
    risk_engine = RiskEngine()
    
    # --- ASX TOP 40 UNIVERSE ---
    tickers = [
        'BHP.AX', 'CBA.AX', 'CSL.AX', 'WES.AX', 'NAB.AX', 'WBC.AX', 'ANZ.AX', 
        'MQG.AX', 'FMG.AX', 'TLS.AX', 'RIO.AX', 'GMG.AX', 'STO.AX', 'WDS.AX', 
        'QBE.AX', 'ALL.AX', 'SCG.AX', 'ORG.AX', 'NST.AX', 'SUN.AX', 'MIN.AX', 
        'PLS.AX', 'IGO.AX', 'TCL.AX', 'S32.AX', 'REA.AX', 'QAN.AX', 'RMD.AX',
        'AMC.AX', 'BSL.AX', 'CPU.AX', 'VCX.AX', 'ASX.AX', 'SHL.AX', 'JHX.AX'
    ]

    # --- DATA PREPARATION ---
    print("\n" + "="*60)
    print("🏰 TITAN V5: QUANTUM FORTRESS - ENHANCED")
    print("="*60)
    
    raw_data = brain.get_data(tickers, start_date="2018-01-01")
    processed_data = {}
    valid_tickers = []
    
    print("\n🧠 Feature Engineering & Data Validation...")
    for t in tqdm(tickers):
        if t in raw_data.columns:
            df = brain.engineer_features(raw_data[t].copy())
            if df is not None and len(df) > 300:
                processed_data[t] = df
                valid_tickers.append(t)

    if not valid_tickers:
        print("❌ Error: No se procesaron datos válidos.")
        return

    print(f"✅ Procesados: {len(valid_tickers)} activos válidos")

    # --- PRE-TRAIN MODELS FOR PERFORMANCE ---
    # (Omitido: TitanBrain entrena modelos on-the-fly por cada predicción)
    start_sim = pd.Timestamp("2023-01-01")

    # --- SIMULATION SETUP ---
    start_sim = pd.Timestamp("2023-01-01")
    
    # Get common date range
    all_dates = sorted(list(processed_data[valid_tickers[0]].index))
    sim_dates = [d for d in all_dates if d >= start_sim]

    # Initialize portfolio
    cash = 10000.0
    portfolio = {}
    equity_curve = []
    daily_trades = 0
    
    print(f"\n📊 Configuración de Simulación:")
    print(f"   • Período: {sim_dates[0].date()} → {sim_dates[-1].date()}")
    print(f"   • Días: {len(sim_dates)}")
    print(f"   • Capital Inicial: ${cash:,.2f}")
    print(f"   • Min EV Ratio: {MIN_EV_RATIO:.2f}")
    print(f"   • Risk per Trade: {BASE_RISK_PER_TRADE:.1%}")

    # --- MAIN SIMULATION LOOP ---
    print("\n🚀 Iniciando simulación...\n")
    
    for current_date in tqdm(sim_dates, desc="Trading"):
        
        # Reset daily trade counter
        daily_trades = 0
        
        # --- A. MARK-TO-MARKET VALUATION ---
        equity = cash
        current_prices = {}
        day_data = {}
        
        for t in valid_tickers:
            try:
                row = processed_data[t].loc[current_date]
                current_prices[t] = row['Close']
                day_data[t] = row
            except KeyError:
                continue

        # Value current positions
        for t, pos in portfolio.items():
            price = current_prices.get(t, pos['buy_price'])
            equity += pos['shares'] * price
        
        equity_curve.append(equity)

        # --- B. SYSTEM HEALTH CHECK ---
        sizing_scalar = risk_engine.check_system_health(equity)
        regime = risk_engine.get_market_regime(current_date)

        # --- C. EXIT MANAGEMENT (Process Sells) ---
        to_sell = []
        
        for t, pos in list(portfolio.items()):
            if t not in day_data:
                continue
            
            row = day_data[t]
            price_now = row['Close']
            
            # Update trailing high
            if row['High'] > pos['highest_price']:
                pos['highest_price'] = row['High']
            
            sell_reason = ""
            sell_price = 0
            
            # Check exit conditions
            # 1. Stop Loss
            if row['Low'] <= pos['stop_price']:
                sell_price = pos['stop_price']
                sell_reason = "STOP_LOSS"
            
            # 2. Take Profit
            elif row['High'] >= pos['take_profit_price']:
                sell_price = pos['take_profit_price']
                sell_reason = "TAKE_PROFIT"
            
            # Execute sell if triggered
            if sell_price > 0:
                # Apply slippage (0.1%)
                sell_price = sell_price * 0.999
                
                # Calculate P&L
                cash += pos['shares'] * sell_price
                pnl = (sell_price - pos['buy_price']) * pos['shares']
                pnl_pct = (sell_price / pos['buy_price']) - 1
                
                # Record transaction
                save_transaction({
                    'date': current_date,
                    'ticker': t,
                    'action': 'SELL',
                    'price': sell_price,
                    'shares': pos['shares'],
                    'pnl': pnl,
                    'cash': cash,
                    'equity': equity,
                    'net_worth': cash + sum(
                        portfolio[ticker]['shares'] * current_prices.get(ticker, portfolio[ticker]['buy_price'])
                        for ticker in portfolio if ticker != t
                    )
                })
                
                update_portfolio(t, 0, 0, 'SELL')
                to_sell.append(t)
                
                # Log result
                emoji = "📈" if pnl > 0 else "📉"
                print(f"  {emoji} SELL {t} ({sell_reason}) | P&L: ${pnl:.2f} ({pnl_pct:+.2%})")

        # Remove sold positions
        for t in to_sell:
            del portfolio[t]

        # --- D. ENTRY MANAGEMENT (Expected Value Engine) ---
        # CIRCUIT BREAKER: si el drawdown supera el límite, congelamos todas las compras
        current_dd = (10000.0 - equity) / 10000.0 if equity < 10000.0 else 0.0
        if current_dd >= MAX_DRAWDOWN_FREEZE:
            print(f"🛑 CIRCUIT BREAKER ACTIVADO: Drawdown {current_dd:.1%} ≥ {MAX_DRAWDOWN_FREEZE:.1%}. Sin nuevas compras.")
            continue
        if cash > MIN_TRADE_SIZE and daily_trades < MAX_DAILY_TRADES:
            
            candidates = []
            available_tickers = [
                t for t in valid_tickers 
                if t in day_data and t not in portfolio
            ]
            
            # Scan for opportunities
            for t in available_tickers:
                
                # 1. LIQUIDITY FILTER
                if day_data[t]['vol_rel_20'] < 0.1:
                    continue
                
                # 2. BRAIN: Get AI prediction
                prob_win, features, atr_pct = brain.train_and_predict_calibrated(
                    t, processed_data[t], current_date
                )

                # GUARDIA DE SEÑAL: solo operamos si la probabilidad supera el umbral mínimo
                if prob_win < MIN_PROB_THRESHOLD:
                    continue
                
                # 3. RISK ENGINE: Define trade structure
                sl_pct, tp_pct = risk_engine.get_dynamic_stops(regime, atr_pct)
                prob_loss = 1.0 - prob_win
                
                # 4. EXPECTED VALUE CALCULATION
                # Simulate $1000 position
                risk_unitary = 1000 * sl_pct
                reward_unitary = 1000 * tp_pct
                
                expected_value = (prob_win * reward_unitary) - (prob_loss * risk_unitary)
                
                # EV Ratio (Expected Value per unit of risk)
                ev_ratio = expected_value / risk_unitary if risk_unitary > 0 else 0
                
                # Filter by minimum EV threshold
                if ev_ratio > MIN_EV_RATIO:
                    candidates.append({
                        't': t,
                        'prob': prob_win,
                        'ev': ev_ratio,
                        'sl_pct': sl_pct,
                        'tp_pct': tp_pct,
                        'atr': atr_pct,
                        'features': features,
                        'price': day_data[t]['Close']
                    })

            # Sort by best EV
            candidates.sort(key=lambda x: x['ev'], reverse=True)
            
            # Process top candidates
            for cand in candidates:
                if cash < MIN_TRADE_SIZE or daily_trades >= MAX_DAILY_TRADES:
                    break
                
                t = cand['t']
                
                # 5. RISK ENGINE: Sector exposure check
                if not risk_engine.check_factor_exposure(
                    t, portfolio, equity, current_prices
                ):
                    continue
                
                # 6. POSITION SIZING
                shares = risk_engine.calculate_optimal_position_size(
                    equity=equity,
                    price=cand['price'],
                    stop_pct=cand['sl_pct'],
                    base_risk_pct=BASE_RISK_PER_TRADE,
                    sizing_scalar=sizing_scalar
                )
                
                cost = shares * cand['price']
                
                # Adjust if exceeds available cash
                if cost > cash:
                    shares = int(cash / cand['price'])
                    cost = shares * cand['price']
                
                # Validate minimum size and position limit
                if shares > 0 and cost >= MIN_TRADE_SIZE:
                    
                    if not risk_engine.check_position_size_limit(cost, equity):
                        continue
                    
                    # --- EXECUTE BUY ---
                    cash -= cost
                    daily_trades += 1
                    
                    portfolio[t] = {
                        'shares': shares,
                        'buy_price': cand['price'],
                        'highest_price': cand['price'],
                        'stop_price': cand['price'] * (1 - cand['sl_pct']),
                        'take_profit_price': cand['price'] * (1 + cand['tp_pct']),
                        'atr_at_buy': cand['atr']
                    }
                    
                    # Record transaction
                    save_transaction({
                        'date': current_date,
                        'ticker': t,
                        'action': 'BUY',
                        'price': cand['price'],
                        'shares': shares,
                        'pnl': 0,
                        'cash': cash,
                        'equity': equity,
                        'net_worth': equity
                    })
                    
                    update_portfolio(t, shares, cand['price'], 'BUY')
                    
                    # Record AI intelligence
                    save_intelligence({
                        'date': current_date,
                        'ticker': t,
                        'raw_prob': cand['prob'],
                        'kelly_score': int(cand['ev'] * 10),
                        'features_used': f"EV:{cand['ev']:.2f}|{','.join(cand['features'][:3])}",
                        'forecast_type': 'QUANTUM_V5',
                        'close_price': cand['price'],
                        'volatility': cand['atr']
                    })
                    
                    print(f" ⚔️  BUY {t} | EV:{cand['ev']:.2f} | Prob:{cand['prob']:.1%} | ${cost:.0f} | Regime:{regime}")

    import json

    # --- FINAL REPORT ---
    print("\n" + "="*60)
    print("🏁 SIMULACIÓN COMPLETADA")
    print("="*60)

    # Save current portfolio for daily recommendations
    with open('current_portfolio.json', 'w') as f:
        json.dump(portfolio, f, default=str)

    if len(equity_curve) > 0:
        final_equity = equity_curve[-1]
        roi = ((final_equity - 10000) / 10000) * 100

        # Calculate metrics
        returns = pd.Series(equity_curve).pct_change().dropna()
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        max_dd = ((pd.Series(equity_curve).cummax() - pd.Series(equity_curve)) / pd.Series(equity_curve).cummax()).max()

        print(f"\n📊 Resultados:")
        print(f"   • Capital Inicial:  ${10000:,.2f}")
        print(f"   • Capital Final:    ${final_equity:,.2f}")
        print(f"   • Retorno Neto:     {roi:+.2f}%")
        print(f"   • Sharpe Ratio:     {sharpe:.2f}")
        print(f"   • Max Drawdown:     {max_dd:.2%}")
        print(f"   • Estado Sistema:   {'🟢 Saludable' if sizing_scalar == 1.0 else '🔴 En Drawdown'}")
        print(f"   • Régimen Final:    {regime}")
    else:
        print("⚠️  No se realizaron operaciones o datos insuficientes.")


if __name__ == "__main__":
    run_quantum_fortress()