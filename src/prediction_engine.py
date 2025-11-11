
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
import sys
import os

# Fix Windows console
if sys.platform == 'win32':
    os.system('chcp 65001 > nul')

warnings.filterwarnings('ignore')


class PredictionEngine:
    """Production forex prediction engine"""
    
    def __init__(self):
        """Initialize engine"""
        print("[OK] Prediction Engine initialized")
        self.cache = {}
    
    def get_forex_data(self, symbol, days=90):
        """Get live forex data from yfinance"""
        try:
            print(f"[->] Fetching live data for {symbol}...")
            
            # Check cache
            cache_key = f"{symbol}_data"
            if cache_key in self.cache:
                print(f"[OK] Using cached data")
                return self.cache[cache_key]
            
            # Download data
            data = yf.download(
                symbol,
                period=f'{days}d',
                interval='1d',
                progress=False,
                auto_adjust=False
            )
            
            if len(data) == 0:
                print(f"[ERROR] No data for {symbol}")
                return None
            
            # Keep needed columns
            data = data[['Open', 'High', 'Low', 'Close']].copy()
            
            # Remove timezone
            if data.index.tz is not None:
                data.index = data.index.tz_localize(None)
            
            # Cache it
            self.cache[cache_key] = data
            
            # Get live price (latest close)
            live_price = float(data['Close'].iloc[-1])
            print(f"[OK] Downloaded {len(data)} records | Live Price: ${live_price:.5f}")
            
            return data
            
        except Exception as e:
            print(f"[ERROR] Failed to fetch data: {e}")
            return None
    
    def calculate_indicators(self, df):
        """Calculate technical indicators"""
        try:
            print("[->] Calculating indicators...")
            
            df = df.copy()
            
            if len(df) < 50:
                print(f"[ERROR] Not enough data: {len(df)}")
                return None
            
            # === MOVING AVERAGES ===
            df['SMA_5'] = df['Close'].rolling(window=5).mean()
            df['SMA_10'] = df['Close'].rolling(window=10).mean()
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            
            # === RSI ===
            delta = df['Close'].diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.rolling(window=14).mean()
            avg_loss = loss.rolling(window=14).mean()
            rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
            rsi = 100 - (100 / (1 + rs))
            df['RSI'] = rsi
            
            # === MACD ===
            exp1 = df['Close'].ewm(span=12, adjust=False).mean()
            exp2 = df['Close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = exp1 - exp2
            df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
            df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
            
            # === BOLLINGER BANDS ===
            bb_sma = df['Close'].rolling(window=20).mean()
            bb_std = df['Close'].rolling(window=20).std()
            df['BB_Upper'] = bb_sma + (bb_std * 2)
            df['BB_Lower'] = bb_sma - (bb_std * 2)
            df['BB_Middle'] = bb_sma
            
            # === ATR ===
            high_low = df['High'] - df['Low']
            high_close = abs(df['High'] - df['Close'].shift())
            low_close = abs(df['Low'] - df['Close'].shift())
            tr_list = [high_low, high_close, low_close]
            tr = pd.concat(tr_list, axis=1).max(axis=1)
            df['ATR'] = tr.rolling(window=14).mean()
            
            # === VOLATILITY ===
            df['Daily_Return'] = df['Close'].pct_change() * 100
            df['Volatility'] = df['Daily_Return'].rolling(window=20).std()
            
            # === INTRADAY RANGE ===
            df['Intraday_Range'] = ((df['High'] - df['Low']) / df['Close']) * 100
            
            # Fill NaN
            df = df.fillna(method='ffill').fillna(method='bfill')
            df = df.dropna()
            
            if len(df) == 0:
                print("[ERROR] All data became NaN")
                return None
            
            print(f"[OK] Calculated all indicators")
            return df
            
        except Exception as e:
            print(f"[ERROR] Calculation failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def analyze_signals(self, df):
        """Analyze 4 trading signals"""
        try:
            latest = df.iloc[-1]
            
            rsi = float(latest['RSI'])
            macd = float(latest['MACD'])
            macd_signal = float(latest['MACD_Signal'])
            sma_20 = float(latest['SMA_20'])
            sma_50 = float(latest['SMA_50'])
            close = float(latest['Close'])
            
            signals = []
            
            # Signal 1: MA Trend
            if sma_20 > sma_50:
                signals.append(1)
            else:
                signals.append(0)
            
            # Signal 2: RSI
            if rsi > 70:
                signals.append(0)
            elif rsi < 30:
                signals.append(1)
            else:
                signals.append(1 if rsi > 50 else 0)
            
            # Signal 3: MACD
            if macd > macd_signal:
                signals.append(1)
            else:
                signals.append(0)
            
            # Signal 4: Price vs SMA20
            if close > sma_20:
                signals.append(1)
            else:
                signals.append(0)
            
            up_votes = sum(signals)
            confidence = (up_votes / 4) * 100
            direction = "UP" if up_votes >= 2 else "DOWN"
            
            return {
                'direction': direction,
                'confidence': confidence,
                'rsi': rsi,
                'macd': macd,
                'macd_signal': macd_signal,
                'sma_20': sma_20,
                'sma_50': sma_50
            }
            
        except Exception as e:
            print(f"[ERROR] Signal analysis failed: {e}")
            return None
    
    def get_indicators_dict(self, df):
        """Get all indicators"""
        try:
            latest = df.iloc[-1]
            
            return {
                'price': float(latest['Close']),
                'sma_5': float(latest['SMA_5']),
                'sma_10': float(latest['SMA_10']),
                'sma_20': float(latest['SMA_20']),
                'sma_50': float(latest['SMA_50']),
                'rsi': float(latest['RSI']),
                'macd': float(latest['MACD']),
                'macd_signal': float(latest['MACD_Signal']),
                'macd_hist': float(latest['MACD_Hist']),
                'bb_upper': float(latest['BB_Upper']),
                'bb_lower': float(latest['BB_Lower']),
                'bb_middle': float(latest['BB_Middle']),
                'atr': float(latest['ATR']),
                'daily_return': float(latest['Daily_Return']),
                'volatility': float(latest['Volatility']),
                'intraday_range': float(latest['Intraday_Range'])
            }
            
        except Exception as e:
            print(f"[ERROR] Getting indicators failed: {e}")
            return None
    
    def get_prediction(self, symbol):
        """Complete prediction pipeline"""
        
        print(f"\n{'='*70}")
        print(f"[PREDICTING] {symbol}")
        print(f"{'='*70}")
        
        try:
            # Step 1: Get data
            data = self.get_forex_data(symbol, days=90)
            if data is None:
                print("[ERROR] ABORT: Failed to get data")
                return None
            
            # Step 2: Calculate indicators
            indicators_data = self.calculate_indicators(data)
            if indicators_data is None:
                print("[ERROR] ABORT: Failed to calculate indicators")
                return None
            
            # Step 3: Analyze signals
            signals = self.analyze_signals(indicators_data)
            if signals is None:
                print("[ERROR] ABORT: Failed to analyze signals")
                return None
            
            # Step 4: Get all indicators
            indicators = self.get_indicators_dict(indicators_data)
            if indicators is None:
                print("[ERROR] ABORT: Failed to get indicators")
                return None
            
            # Step 5: Build result
            result = {
                'symbol': symbol,
                'prediction': {
                    'direction': signals['direction'],
                    'confidence': signals['confidence'],
                    'probability_up': signals['confidence'] if signals['direction'] == 'UP' else (100 - signals['confidence']),
                    'probability_down': (100 - signals['confidence']) if signals['direction'] == 'UP' else signals['confidence']
                },
                'indicators': indicators,
                'price': indicators['price'],
                'timestamp': datetime.now().isoformat(),
                'status': 'SUCCESS'
            }
            
            print(f"[OK] SUCCESS: {signals['direction']} with {signals['confidence']:.1f}% confidence")
            print(f"[OK] Live Price: ${indicators['price']:.5f}")
            print(f"{'='*70}\n")
            
            return result
        
        except Exception as e:
            print(f"[ERROR] FATAL: {e}")
            import traceback
            traceback.print_exc()
            return None


# ==================== TEST ====================

if __name__ == "__main__":
    
    print("\n" + "="*70)
    print("ForexAI - LIVE FOREX PREDICTION ENGINE")
    print("Using yfinance - Accurate & Reliable")
    print("="*70 + "\n")
    
    engine = PredictionEngine()
    
    pairs = [
        'EURUSD=X',
        'GBPUSD=X',
        'USDJPY=X',
        'USDCAD=X',
        'AUDUSD=X'
    ]
    
    for symbol in pairs:
        result = engine.get_prediction(symbol)
        
        if result:
            pred = result['prediction']
            inds = result['indicators']
            
            print(f"\n{'='*70}")
            print(f"[RESULT] {symbol.replace('=X', '')}")
            print(f"{'='*70}")
            print(f"   LIVE Price: ${inds['price']:.5f}")
            print(f"   Prediction: {pred['direction']}")
            print(f"   Confidence: {pred['confidence']:.1f}%")
            print(f"   RSI: {inds['rsi']:.1f}")
            print(f"   SMA20: ${inds['sma_20']:.5f}")
            print(f"   SMA50: ${inds['sma_50']:.5f}")
            print(f"   Volatility: {inds['volatility']:.4f}%")
            print(f"{'='*70}\n")