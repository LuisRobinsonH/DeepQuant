# core/brain.py
import yfinance as yf
import pandas as pd
import numpy as np
import ta
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
import warnings
import joblib
import os

warnings.filterwarnings('ignore')


def load_au_tickers(symbols_path="au_stock_data/au_symbols.txt"):
    tickers = []
    with open(symbols_path, "r") as f:
        for line in f:
            symbol = line.strip()
            if symbol and not symbol.startswith("#"):
                tickers.append(symbol)
    return tickers

class TitanBrain:
    def calibrate_model(self, model, X, y):
        """
        Calibrate a model using CalibratedClassifierCV (sigmoid method).
        Returns a calibrated model ready for probability prediction.
        """
        from sklearn.calibration import CalibratedClassifierCV
        try:
            calibrated = CalibratedClassifierCV(model, method='sigmoid', cv=3)
            calibrated.fit(X, y)
            return calibrated
        except Exception as e:
            import warnings
            warnings.warn(f"[TitanBrain] Model calibration failed: {e}. Using uncalibrated model.")
            return model

    def __init__(self):
        self.trained_models = {}  # Cache for pre-trained models
        self.feature_cache = {}   # Cache for selected features per ticker
        self.model_file = 'models_cache.joblib'
        self.features_file = 'features_cache.joblib'
        self._load_cached_models()

    def _load_cached_models(self):
        """Load cached models and features if they exist."""
        try:
            if os.path.exists(self.model_file):
                self.trained_models = joblib.load(self.model_file)
                print(f"🧠 Loaded {len(self.trained_models)} cached models")
            if os.path.exists(self.features_file):
                self.feature_cache = joblib.load(self.features_file)
                print(f"🧠 Loaded {len(self.feature_cache)} cached features")
        except Exception as e:
            print(f"⚠️ Failed to load cached models: {e}")
            self.trained_models = {}
            self.feature_cache = {}

    def _save_cached_models(self):
        """Save trained models and features to disk."""
        try:
            joblib.dump(self.trained_models, self.model_file)
            joblib.dump(self.feature_cache, self.features_file)
            print(f"🧠 Saved {len(self.trained_models)} models and features to cache")
        except Exception as e:
            print(f"⚠️ Failed to save models: {e}")
    def get_data(self, tickers, start_date="2018-01-01"):
        """
        Carga datos históricos de archivos locales si existen, si no, intenta descargar de Yahoo Finance y guarda el CSV.
        Si falla el scraping, loguea el error y sigue con los datos disponibles.
        """
        print(f"🧠 TitanBrain: Descargando universo desde {start_date}...")
        data = {}
        for t in tickers:
            fname = f"au_stock_data/{t.replace('.AX','')}.csv"
            df = None
            if os.path.exists(fname):
                try:
                    df = pd.read_csv(fname, skiprows=1, header=0)
                    if df.columns[0].strip().lower() != 'date':
                        df.columns = ['date', 'close', 'high', 'low', 'open', 'volume']
                    col_map = {c: c.capitalize() for c in df.columns}
                    if 'close' in df.columns: col_map['close'] = 'Close'
                    if 'high' in df.columns: col_map['high'] = 'High'
                    if 'low' in df.columns: col_map['low'] = 'Low'
                    if 'open' in df.columns: col_map['open'] = 'Open'
                    if 'volume' in df.columns: col_map['volume'] = 'Volume'
                    df = df.rename(columns=col_map)
                    df.columns = [c.lower() for c in df.columns]
                    if 'date' not in df.columns:
                        raise ValueError("No 'date' column after loading CSV")
                    df['date'] = pd.to_datetime(df['date'])
                    df = df.set_index('date')
                    col_map = {}
                    for c in df.columns:
                        if c.startswith('close'): col_map[c] = 'close'
                        elif c.startswith('high'): col_map[c] = 'high'
                        elif c.startswith('low'): col_map[c] = 'low'
                        elif c.startswith('open'): col_map[c] = 'open'
                        elif c.startswith('volume'): col_map[c] = 'volume'
                    df = df.rename(columns=col_map)
                    data[t] = df
                    continue
                except Exception as e:
                    print(f"⚠️ Error cargando {fname}: {e}")
            # Si no existe el CSV o falló la carga, intentar descargar de Yahoo Finance
            try:
                print(f"[INFO] Descargando {t} desde Yahoo Finance...")
                import yfinance as yf
                df = yf.download(t, start=start_date, progress=False, auto_adjust=True)
                if not df.empty:
                    df = df.rename(columns={
                        'Close': 'close', 'High': 'high', 'Low': 'low', 'Open': 'open', 'Volume': 'volume'
                    })
                    df = df[['close', 'high', 'low', 'open', 'volume']]
                    # Guardar CSV para uso futuro
                    df_reset = df.reset_index()
                    df_reset.to_csv(fname, index=False, encoding='utf-8')
                    df.index.name = 'date'
                    data[t] = df
                else:
                    print(f"⚠️ No se pudo descargar datos para {t} de Yahoo Finance.")
            except Exception as e:
                print(f"⚠️ Error descargando {t} de Yahoo Finance: {e}")
        return data

    # CRITERIO DE CALIDAD: Solo se generan señales cuando hay alta probabilidad y condiciones técnicas sólidas.
    # El feature engineering y el target están configurados para priorizar calidad sobre cantidad.
    def engineer_features(self, df, horizon=5):
        """
        Enhanced feature engineering for AU Stock signals. Adds more indicators, normalization, and robust feature selection.
        """
        df = df.copy()
        if len(df) < 100:
            return None

        # Trend and moving averages
        df['sma_20'] = ta.trend.sma_indicator(df['close'], 20)
        df['sma_50'] = ta.trend.sma_indicator(df['close'], 50)
        df['sma_200'] = ta.trend.sma_indicator(df['close'], 200)
        df['dist_sma20'] = (df['close'] / df['sma_20']) - 1
        df['dist_sma50'] = (df['close'] / df['sma_50']) - 1
        df['dist_sma200'] = (df['close'] / df['sma_200']) - 1
        df['ma_cross_20_50'] = (df['sma_20'] > df['sma_50']).astype(int)
        df['ma_cross_50_200'] = (df['sma_50'] > df['sma_200']).astype(int)

        # Volatility and ATR
        atr = ta.volatility.average_true_range(df['high'], df['low'], df['close'], 14)
        df['atr'] = atr
        df['atr_pct'] = np.where(df['close'] > 0, atr / df['close'], 0)
        atr_ma = atr.rolling(50).mean()
        df['vol_regime'] = np.where(atr_ma > 0, atr / atr_ma, 1.0)

        # Momentum
        df['momentum_3'] = df['close'].pct_change(3)
        df['momentum_5'] = df['close'].pct_change(5)
        df['momentum_10'] = df['close'].pct_change(10)
        df['momentum_20'] = df['close'].pct_change(20)
        df['roc_5'] = ta.momentum.roc(df['close'], 5)
        df['roc_10'] = ta.momentum.roc(df['close'], 10)

        # RSI, MACD, ADX
        df['rsi'] = ta.momentum.rsi(df['close'], 14) / 100.0
        df['macd_diff'] = ta.trend.macd_diff(df['close'])
        df['adx'] = ta.trend.adx(df['high'], df['low'], df['close'], 14)

        # Bollinger Bands
        bb_high = ta.volatility.bollinger_hband(df['close'], 20)
        bb_low = ta.volatility.bollinger_lband(df['close'], 20)
        df['bb_width'] = (bb_high - bb_low) / df['close']
        df['bb_upper_dist'] = (bb_high - df['close']) / df['close']
        df['bb_lower_dist'] = (df['close'] - bb_low) / df['close']

        # Stochastic Oscillator
        df['stoch_k'] = ta.momentum.stoch(df['high'], df['low'], df['close'], 14, 3) / 100.0
        df['stoch_d'] = ta.momentum.stoch_signal(df['high'], df['low'], df['close'], 14, 3) / 100.0

        # Williams %R
        df['williams_r'] = ta.momentum.williams_r(df['high'], df['low'], df['close'], lbp=14)

        # CCI
        df['cci'] = ta.trend.cci(df['high'], df['low'], df['close'], 20)

        # Relative volume and averages
        if 'volume' in df.columns and df['volume'].sum() > 0:
            vol_ma20 = df['volume'].rolling(20).mean()
            vol_ma50 = df['volume'].rolling(50).mean()
            df['vol_rel_20'] = np.where(vol_ma20 > 0, df['volume'] / vol_ma20, 1.0)
            df['vol_rel_50'] = np.where(vol_ma50 > 0, df['volume'] / vol_ma50, 1.0)
        else:
            df['vol_rel_20'] = 1.0
            df['vol_rel_50'] = 1.0

        # Recent patterns
        df['max_5'] = df['close'].rolling(5).max()
        df['min_5'] = df['close'].rolling(5).min()
        df['max_20'] = df['close'].rolling(20).max()
        df['min_20'] = df['close'].rolling(20).min()
        df['close_to_max5'] = (df['close'] / df['max_5']) - 1
        df['close_to_min5'] = (df['close'] / df['min_5']) - 1
        df['close_to_max20'] = (df['close'] / df['max_20']) - 1
        df['close_to_min20'] = (df['close'] / df['min_20']) - 1

        df = df.replace([np.inf, -np.inf], np.nan).dropna()

        # TARGET ROBUSTO: ganancia futura > 1.0x ATR en 'horizon' días
        # Esto representa una señal real con significado económico, no micro-ruido.
        future_ret = df['close'].shift(-horizon) / df['close'] - 1
        # atr_pct sin normalizar (se guardó antes de normalizar) para el threshold
        raw_atr_pct = atr / df['close']  # noqa — usa ATR calculado arriba
        raw_atr_pct = raw_atr_pct.reindex(df.index).fillna(method='ffill').fillna(0.02)
        # Win = retorno futuro > 1.0x ATR (perfil riesgo/recompensa mínimo 1:1)
        df['target'] = (future_ret > raw_atr_pct * 1.0).astype(int)
        # Si menos del 5% son positivos, relajar a 0.5x ATR
        if df['target'].mean() < 0.05:
            df['target'] = (future_ret > raw_atr_pct * 0.5).astype(int)
        # Si sigue sin positivos, usar umbral fijo del 1%
        if df['target'].sum() == 0:
            df['target'] = (future_ret > 0.01).astype(int)

        # NORMALIZACIÓN ROLLING para evitar data leakage
        # Solo se normalizan las features, NO el target ni precios base
        skip_cols = {'target', 'close', 'open', 'high', 'low', 'volume',
                     'sma_20', 'sma_50', 'sma_200', 'atr', 'max_5', 'min_5', 'max_20', 'min_20'}
        for col in df.columns:
            if col in skip_cols:
                continue
            if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                roll_mean = df[col].rolling(252, min_periods=50).mean()
                roll_std  = df[col].rolling(252, min_periods=50).std()
                df[col] = np.where(roll_std > 0, (df[col] - roll_mean) / roll_std, 0.0)

        return df.replace([np.inf, -np.inf], np.nan).dropna()

    def select_stable_features(self, X, y):
        """
        Feature selection based on temporal stability and permutation importance.
        Returns features that maintain consistent importance across time folds and are robust to noise.
        """
        if len(X) < 200:
            return list(X.columns)
        from sklearn.inspection import permutation_importance
        rf = RandomForestClassifier(
            n_estimators=50, 
            max_depth=3, 
            random_state=42,
            n_jobs=-1
        )
        splits = TimeSeriesSplit(n_splits=3)
        feature_scores = {col: [] for col in X.columns}
        for train_index, _ in splits.split(X):
            if len(train_index) < 100:
                continue
            X_fold = X.iloc[train_index]
            y_fold = y.iloc[train_index]
            if y_fold.sum() < 5:
                continue
            try:
                rf.fit(X_fold, y_fold)
                perm = permutation_importance(rf, X_fold, y_fold, n_repeats=5, random_state=42)
                for i, name in enumerate(X.columns):
                    feature_scores[name].append(perm.importances_mean[i])
            except Exception:
                continue
        # Seleccionar features estables: mediana de importancia > umbral y varianza baja entre folds
        stable_feats = [
            f for f, scores in feature_scores.items()
            if len(scores) > 0 and np.median(scores) > 0.001 and np.std(scores) < 0.50
        ]
        # Garantizar mínimo de 8 features; si hay muy pocas, usar todas
        if len(stable_feats) < 8:
            stable_feats = list(X.columns)
        # Limitar a máximo 20 features para evitar curse of dimensionality
        if len(stable_feats) > 20:
            scores_median = {f: np.median(feature_scores[f]) for f in stable_feats}
            stable_feats = sorted(stable_feats, key=lambda f: scores_median[f], reverse=True)[:20]
        return stable_feats


    def train_and_predict_calibrated(self, ticker, df, predict_date, calibration_method='auto', telegram_token=None, telegram_chat_id=None):
        """
        Fast prediction using pre-trained model.
        Falls back to training if no pre-trained model available.
        Returns:
            prob (float): Calibrated probability of success [0,1]
            features (list): Active features used
            atr_pct (float): Current ATR percentage
        """
        # Try to use pre-trained model first
        if ticker in self.trained_models and ticker in self.feature_cache:
            model = self.trained_models[ticker]
            active_features = self.feature_cache[ticker]
            # Get data for prediction
            if predict_date in df.index:
                last_row = df.loc[predict_date]
            else:
                last_row = df.iloc[-1]
            current_X = last_row[active_features].values.reshape(1, -1)
            prob = model.predict_proba(current_X)[0][1]
            atr_pct = last_row['atr_pct'] if 'atr_pct' in last_row.index else 0.02
            return prob, active_features, atr_pct
        # Si no hay modelo cacheado, usar fallback
        return self._train_and_predict_fallback(ticker, df, predict_date, calibration_method=calibration_method, telegram_token=telegram_token, telegram_chat_id=telegram_chat_id)

    def _train_and_predict_fallback(self, ticker, df, predict_date, calibration_method='auto', telegram_token=None, telegram_chat_id=None):
        """
        Fallback training method - original implementation for robustness.
        Only used when pre-trained model is unavailable.
        """
        # 1. TEMPORAL PURGING
        # Remove last 7 days before prediction to avoid leakage
        purge_days = 7
        # Asegura que predict_date sea un Timestamp para operaciones de fecha
        predict_date_ts = pd.Timestamp(predict_date)
        cutoff_date = predict_date_ts - pd.Timedelta(days=purge_days)
        
        train_data = df.loc[df.index < cutoff_date].copy()
        
        # Insufficient training data
        if len(train_data) < 300:
            return 0.0, [], 0.0
        

        # Usar todas las features generadas por engineer_features (filtradas por disponibilidad)
        all_candidate_features = [
            'dist_sma20', 'dist_sma50', 'dist_sma200', 'ma_cross_20_50', 'ma_cross_50_200',
            'atr_pct', 'vol_regime',
            'momentum_3', 'momentum_5', 'momentum_10', 'momentum_20',
            'roc_5', 'roc_10',
            'rsi', 'macd_diff', 'adx',
            'bb_width', 'bb_upper_dist', 'bb_lower_dist',
            'stoch_k', 'stoch_d', 'williams_r', 'cci',
            'vol_rel_20', 'vol_rel_50',
            'close_to_max5', 'close_to_min5', 'close_to_max20', 'close_to_min20',
        ]
        all_features = [f for f in all_candidate_features if f in train_data.columns]
        X = train_data[all_features]
        y = train_data['target']

        # 2. FEATURE SELECTION (Stability-based)
        active_features = self.select_stable_features(X, y)
        # Validate active_features (must be lowercase and present in X)
        if not active_features or any(f is None or f not in X.columns for f in active_features):
            import warnings
            warnings.warn(f"[TitanBrain] Feature selection failed for {ticker} {predict_date}, using default features.")
            active_features = [f for f in all_features if f in X.columns]
        # Only use lowercase features
        X = X[active_features]

        # Permitir entrenamiento aunque haya pocos positivos (no artificiales)
        # Si no hay ningún positivo, retornar probabilidad 0
        if y.sum() == 0:
            return 0.0, active_features, 0.0
        
        # 3. ENSEMBLE MODEL (Orthogonal learners, mejorado)
        # Puedes ajustar hiperparámetros aquí para tuning futuro
        # BALANCEO DE CLASES: class_weight='balanced' compensa que los wins son minoría
        lr_pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('lr', LogisticRegression(
                C=0.3,
                penalty='l2',
                solver='liblinear',
                class_weight='balanced',  # FIX: balancea clases
                random_state=42
            ))
        ])
        rf = RandomForestClassifier(
            n_estimators=150,
            max_depth=6,
            min_samples_leaf=8,
            class_weight='balanced',  # FIX: balancea clases
            random_state=42,
            n_jobs=-1
        )
        gb = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.05,  # más bajo = menos overfit
            max_depth=3,
            subsample=0.8,       # stochastic boosting reduce overfit
            random_state=42
        )
        ensemble = VotingClassifier(
            estimators=[('rf', rf), ('lr', lr_pipe), ('gb', gb)],
            voting='soft',
            weights=[2, 1, 2]  # árboles pesan más que LR
        )
        # Probability calibration method selection
        from sklearn.calibration import CalibratedClassifierCV
        if calibration_method == 'isotonic':
            try:
                calibrated_model = CalibratedClassifierCV(ensemble, method='isotonic', cv=3)
                calibrated_model.fit(X, y)
            except Exception:
                calibrated_model = CalibratedClassifierCV(ensemble, method='sigmoid', cv=3)
                calibrated_model.fit(X, y)
        elif calibration_method == 'sigmoid':
            calibrated_model = CalibratedClassifierCV(ensemble, method='sigmoid', cv=3)
            calibrated_model.fit(X, y)
        else:  # auto
            try:
                calibrated_model = CalibratedClassifierCV(ensemble, method='isotonic', cv=3)
                calibrated_model.fit(X, y)
            except Exception:
                calibrated_model = CalibratedClassifierCV(ensemble, method='sigmoid', cv=3)
                calibrated_model.fit(X, y)

        # 5. PREDICTION
        try:
            last_row = df.loc[predict_date]
        except KeyError:
            # Fallback to last available date
            last_row = df.iloc[-1]
        
        # Prepare features for prediction
        current_X = last_row[active_features].values.reshape(1, -1)
        
        # Get calibrated probability
        try:
            prob = calibrated_model.predict_proba(current_X)[0][1]
        except Exception:
            prob = 0.5  # Neutral probability if prediction fails
        atr_pct = last_row['atr_pct'] if 'atr_pct' in last_row.index else 0.02
        return prob, active_features, atr_pct