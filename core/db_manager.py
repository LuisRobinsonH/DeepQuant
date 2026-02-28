# TitanHedge/core/db_manager.py
import sqlite3
import pandas as pd
import os

DB_NAME = 'trading_bot.db'

def get_db_connection():
    conn = sqlite3.connect(DB_NAME)
    return conn

def initialize_db():
    """Crea las tablas si no existen."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Tabla de Transacciones
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS transactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT,
        ticker TEXT,
        action TEXT,
        price REAL,
        shares REAL,
        pnl REAL,
        cash REAL,
        equity REAL,
        net_worth REAL
    )
    ''')
    
    # Tabla de Inteligencia (Para auditar a la IA)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS intelligence (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT,
        ticker TEXT,
        raw_prob REAL,
        kelly_score REAL,
        features_used TEXT,
        forecast_type TEXT,
        close_price REAL,
        volatility REAL
    )
    ''')
    
    # Tabla de Portafolio Actual
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS portfolio (
        ticker TEXT PRIMARY KEY,
        shares REAL,
        buy_price REAL,
        current_value REAL
    )
    ''')
    
    conn.commit()
    conn.close()

def clear_db():
    """Borra datos viejos e inicializa tablas limpias."""
    if os.path.exists(DB_NAME):
        os.remove(DB_NAME)
    initialize_db()

def save_transaction(data):
    """Guarda una operación de compra/venta/snapshot."""
    # Asegurar que la DB existe antes de guardar
    if not os.path.exists(DB_NAME):
        initialize_db()
        
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
    INSERT INTO transactions (date, ticker, action, price, shares, pnl, cash, equity, net_worth)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        str(data['date']).split(' ')[0], # Limpiar fecha
        data['ticker'],
        data['action'],
        data['price'],
        data['shares'],
        data['pnl'],
        data['cash'],
        data['equity'],
        data['net_worth']
    ))
    
    conn.commit()
    conn.close()

def save_intelligence(data):
    """Guarda el 'pensamiento' de la IA."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
    INSERT INTO intelligence (date, ticker, raw_prob, kelly_score, features_used, forecast_type, close_price, volatility)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        str(data['date']).split(' ')[0],
        data['ticker'],
        data['raw_prob'],
        data['kelly_score'],
        data['features_used'],
        data['forecast_type'],
        data['close_price'],
        data['volatility']
    ))
    
    conn.commit()
    conn.close()

def update_portfolio(ticker, shares, price, action):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    if action == 'BUY':
        cursor.execute('''
        INSERT OR REPLACE INTO portfolio (ticker, shares, buy_price, current_value)
        VALUES (?, ?, ?, ?)
        ''', (ticker, shares, price, shares * price))
    elif action == 'SELL':
        cursor.execute('DELETE FROM portfolio WHERE ticker = ?', (ticker,))
        
    conn.commit()
    conn.close()