# core/risk_engine.py
import yfinance as yf
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')


class RiskEngine:
    def __init__(self):
        self.market_index = None
        
        # Sector mapping for Australian market (ASX)
        self.sector_map = {
            # Materials
            'BHP.AX': 'Materials', 'RIO.AX': 'Materials', 'FMG.AX': 'Materials',
            'S32.AX': 'Materials', 'MIN.AX': 'Materials', 'NST.AX': 'Materials',
            'PLS.AX': 'Materials', 'IGO.AX': 'Materials', 'AMC.AX': 'Materials',
            
            # Financials
            'CBA.AX': 'Financials', 'NAB.AX': 'Financials', 'WBC.AX': 'Financials',
            'ANZ.AX': 'Financials', 'MQG.AX': 'Financials', 'QBE.AX': 'Financials',
            
            # Healthcare
            'CSL.AX': 'Healthcare', 'COH.AX': 'Healthcare', 'RMD.AX': 'Healthcare',
            
            # Consumer
            'WES.AX': 'Consumer', 'WOW.AX': 'Consumer', 'TWE.AX': 'Consumer',
            
            # Communication
            'TLS.AX': 'Communication',
            
            # Energy
            'STO.AX': 'Energy', 'WDS.AX': 'Energy', 'ORG.AX': 'Energy',
            
            # Technology
            'XRO.AX': 'Technology', 'REA.AX': 'Technology', 'CAR.AX': 'Technology',
            
            # Industrials
            'QAN.AX': 'Industrials', 'TCL.AX': 'Industrials', 'ALL.AX': 'Industrials'
        }
        
        self.max_sector_exposure = 0.25  # Max 25% en un solo sector (reducido para diversificación)
        self.max_single_position = 0.10  # Max 10% en una sola posición (reducido)
        self.initial_equity = 10000.0

    def get_market_regime(self, current_date):
        """
        Determine market regime (BULL/BEAR/NEUTRAL) based on ASX200 index.
        Uses 200-day SMA as regime filter.
        """
        # Download and cache market index data
        if self.market_index is None:
            try:
                self.market_index = yf.download(
                    "^AXJO",  # ASX 200 Index
                    start="2018-01-01",
                    progress=False,
                    auto_adjust=True
                )
                self.market_index['SMA_200'] = self.market_index['Close'].rolling(200).mean()
            except Exception:
                return "NEUTRAL"
        
        try:
            # Get most recent data up to current_date
            recent_data = self.market_index.loc[:current_date]
            if len(recent_data) == 0:
                return "NEUTRAL"
            
            row = recent_data.iloc[-1]
            
            # Regime determination
            if pd.isna(row['SMA_200']):
                return "NEUTRAL"
            
            # BULL: Price > 200 SMA
            if row['Close'] > row['SMA_200']:
                return "BULL"
            # BEAR: Price < 200 SMA
            else:
                return "BEAR"
                
        except Exception:
            return "NEUTRAL"

    def get_dynamic_stops(self, regime, atr_pct):
        """
        Define stop-loss y take-profit basados en régimen y ATR.
        R:R mínimo 2:1 para asegurar expectativa positiva.
        """
        if atr_pct <= 0 or np.isnan(atr_pct):
            atr_pct = 0.02

        if regime == "BULL":
            sl_mult = 1.5   # 1.5x ATR stop
            tp_mult = 4.5   # 3:1 R:R
        elif regime == "BEAR":
            sl_mult = 1.0   # muy conservador en bear
            tp_mult = 2.0   # objetivo rápido
        else:  # NEUTRAL
            sl_mult = 1.3
            tp_mult = 3.0   # 2.3:1 R:R

        sl_pct = np.clip(atr_pct * sl_mult, 0.01, 0.06)   # Máx 6% stop-loss
        tp_pct = np.clip(atr_pct * tp_mult, 0.03, 0.25)   # Mínimo 3% objetivo

        return sl_pct, tp_pct

    def update_trailing_stop(self, position, current_price):
        """
        Actualiza el stop-loss con trailing stop automático.
        Protege beneficios: el stop sube cuando el precio sube, nunca baja.
        Se activa cuando el precio gana >= 1 ATR desde entrada.

        Args:
            position (dict): debe contener 'buy_price', 'stop_price',
                             'highest_price', 'atr_at_buy'
            current_price (float): precio actual del activo

        Returns:
            dict: posición actualizada con nuevo stop_price y highest_price
        """
        pos = position.copy()
        atr = pos.get('atr_at_buy', pos['buy_price'] * 0.02)

        # Actualizar precio más alto alcanzado
        if current_price > pos.get('highest_price', pos['buy_price']):
            pos['highest_price'] = current_price

        gain_from_entry = current_price - pos['buy_price']

        # Activar trailing cuando ganancia >= 1x ATR
        if gain_from_entry >= atr:
            # Trailing stop: highest_price - 1.5 ATR (nunca menor al stop original)
            trailing = pos['highest_price'] - (1.5 * atr)
            pos['stop_price'] = max(pos['stop_price'], trailing)

        return pos

    def check_system_health(self, current_equity):
        """
        Monitor system drawdown and adjust position sizing accordingly.
        
        Implements defensive scaling when system is in drawdown:
        - 0-5% DD: Full size (1.0x)
        - 5-15% DD: Reduced size (0.8x)
        - 15%+ DD: Conservative size (0.5x)
        
        Args:
            current_equity (float): Current portfolio value
            
        Returns:
            float: Sizing multiplier [0.5, 1.0]
        """
        # Calculate drawdown percentage
        dd_pct = (self.initial_equity - current_equity) / self.initial_equity

        if dd_pct > 0.20:
            # Drawdown crítico: congelar nuevas operaciones (señal al orchestrator)
            print("🛑 SYSTEM DRAWDOWN > 20%: FREEZE — tamaño al 0% (circuit breaker)")
            return 0.0
        elif dd_pct > 0.15:
            # Drawdown severo: tamaño al 40%
            print("⚠️  SYSTEM DRAWDOWN > 15%: Reduciendo tamaño al 40%")
            return 0.4
        elif dd_pct > 0.08:
            # Drawdown moderado: tamaño al 70%
            print("⚠️  SYSTEM DRAWDOWN > 8%: Reduciendo tamaño al 70%")
            return 0.7
        elif dd_pct > 0.04:
            # Drawdown leve: tamaño al 90%
            return 0.9

        # Normal operation: tamaño completo
        return 1.0

    def check_factor_exposure(self, candidate, portfolio, current_equity, current_prices):
        """
        Verify sector diversification constraints.
        
        FIXED: Now uses current market prices instead of buy prices
        for accurate exposure calculation.
        
        Args:
            candidate (str): Ticker being considered
            portfolio (dict): Current positions
            current_equity (float): Total portfolio value
            current_prices (dict): Current market prices
            
        Returns:
            bool: True if position is allowed, False if sector is saturated
        """
        candidate_sector = self.sector_map.get(candidate, 'Unknown')
        
        # Allow unknown sectors (but log warning)
        if candidate_sector == 'Unknown':
            print(f"   ⚠️  {candidate}: Sector desconocido")
            return True
        
        # Calculate current exposure in candidate's sector
        sector_exposure = 0.0
        
        for ticker, pos in portfolio.items():
            ticker_sector = self.sector_map.get(ticker, 'Unknown')
            
            if ticker_sector == candidate_sector:
                # FIXED: Use current price, not buy price
                current_price = current_prices.get(ticker, pos['buy_price'])
                position_value = pos['shares'] * current_price
                sector_exposure += position_value
        
        # Calculate exposure as percentage of equity
        exposure_pct = sector_exposure / current_equity if current_equity > 0 else 0
        
        # Check if adding this position would exceed limit
        if exposure_pct >= self.max_sector_exposure:
            print(f"   🚫 Factor Risk: {candidate_sector} saturado ({exposure_pct:.1%})")
            return False
        
        return True

    def check_position_size_limit(self, position_value, current_equity):
        """
        Ensure single position doesn't exceed maximum allocation.
        
        Args:
            position_value (float): Value of proposed position
            current_equity (float): Total portfolio value
            
        Returns:
            bool: True if within limits, False otherwise
        """
        position_pct = position_value / current_equity if current_equity > 0 else 0
        
        if position_pct > self.max_single_position:
            print(f"   🚫 Position too large: {position_pct:.1%} > {self.max_single_position:.1%}")
            return False
        
        return True

    def calculate_optimal_position_size(self, 
                                       equity, 
                                       price, 
                                       stop_pct, 
                                       base_risk_pct=0.01,
                                       sizing_scalar=1.0):
        """
        Calculate position size based on risk per trade.
        
        Formula: Shares = (Equity * Risk%) / (Price * Stop%)
        
        Args:
            equity (float): Current portfolio value
            price (float): Entry price
            stop_pct (float): Stop-loss distance as percentage
            base_risk_pct (float): Base risk per trade (default 1%)
            sizing_scalar (float): Adjustment factor from system health
            
        Returns:
            int: Number of shares to buy
        """
        # Calculate risk amount in dollars
        risk_dollars = equity * base_risk_pct * sizing_scalar
        
        # Calculate stop distance in dollars
        stop_distance = price * stop_pct
        
        # Avoid division by zero
        if stop_distance <= 0:
            return 0
        
        # Calculate shares
        shares = int(risk_dollars / stop_distance)
        
        return max(0, shares)  # Ensure non-negative