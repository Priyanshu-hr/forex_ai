import pandas as pd
import numpy as np
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Creates technical indicators for ML models"""
    
    def __init__(self, data):
        self.data = data.copy()
        logger.info("[OK] FeatureEngineer initialized")
    
    def moving_averages(self):
        """Create moving averages"""
        logger.info("[CREATING] Moving averages...")
        
        self.data['SMA_10'] = self.data['Close'].rolling(10).mean()
        self.data['SMA_20'] = self.data['Close'].rolling(20).mean()
        self.data['SMA_50'] = self.data['Close'].rolling(50).mean()
        self.data['SMA_200'] = self.data['Close'].rolling(200).mean()
        
        logger.info("[OK] Moving averages created")
        return self
    
    def rsi(self, period=14):
        """Calculate RSI (Relative Strength Index)"""
        logger.info("[CREATING] RSI...")
        
        delta = self.data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        self.data['RSI'] = 100 - (100 / (1 + rs))
        
        logger.info("[OK] RSI created")
        return self
    
    def macd(self, fast=12, slow=26, signal=9):
        """Calculate MACD"""
        logger.info("[CREATING] MACD...")
        
        ema_fast = self.data['Close'].ewm(span=fast, adjust=False).mean()
        ema_slow = self.data['Close'].ewm(span=slow, adjust=False).mean()
        self.data['MACD'] = ema_fast - ema_slow
        self.data['MACD_Signal'] = self.data['MACD'].ewm(span=signal, adjust=False).mean()
        self.data['MACD_Hist'] = self.data['MACD'] - self.data['MACD_Signal']
        
        logger.info("[OK] MACD created")
        return self
    
    def bollinger_bands(self, period=20):
        """Calculate Bollinger Bands"""
        logger.info("[CREATING] Bollinger Bands...")
        
        middle = self.data['Close'].rolling(period).mean()
        std = self.data['Close'].rolling(period).std()
        self.data['BB_Upper'] = middle + (std * 2)
        self.data['BB_Lower'] = middle - (std * 2)
        self.data['BB_Middle'] = middle
        self.data['BB_Width'] = (self.data['BB_Upper'] - self.data['BB_Lower']) / middle * 100
        
        logger.info("[OK] Bollinger Bands created")
        return self
    
    def price_features(self):
        """Create price-based features"""
        logger.info("[CREATING] Price features...")
        
        self.data['Daily_Return'] = self.data['Close'].pct_change() * 100
        self.data['Intraday_Range'] = ((self.data['High'] - self.data['Low']) / self.data['Close']) * 100
        self.data['Gap'] = ((self.data['Open'] - self.data['Close'].shift(1)) / self.data['Close'].shift(1)) * 100
        
        logger.info("[OK] Price features created")
        return self
    
    def volatility(self, period=20):
        """Calculate volatility"""
        logger.info("[CREATING] Volatility...")
        
        self.data['Volatility'] = self.data['Daily_Return'].rolling(period).std()
        
        # Average True Range
        high_low = self.data['High'] - self.data['Low']
        high_close = abs(self.data['High'] - self.data['Close'].shift())
        low_close = abs(self.data['Low'] - self.data['Close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        self.data['ATR'] = true_range.rolling(period).mean()
        
        logger.info("[OK] Volatility created")
        return self
    
    def momentum(self):
        """Calculate momentum indicators"""
        logger.info("[CREATING] Momentum...")
        
        self.data['ROC_5'] = ((self.data['Close'] - self.data['Close'].shift(5)) / self.data['Close'].shift(5)) * 100
        self.data['ROC_10'] = ((self.data['Close'] - self.data['Close'].shift(10)) / self.data['Close'].shift(10)) * 100
        
        logger.info("[OK] Momentum created")
        return self
    
    def remove_nan(self):
        """Remove rows with NaN values"""
        logger.info("[PROCESSING] Removing NaN rows...")
        
        before = len(self.data)
        self.data = self.data.dropna()
        removed = before - len(self.data)
        
        logger.info(f"[OK] Rows removed: {removed}")
        return self
    
    def save(self, filename):
        """Save features"""
        os.makedirs('data/processed', exist_ok=True)
        self.data.to_csv(f"data/processed/{filename}")
        logger.info(f"[SAVED] data/processed/{filename}")
        return self
    
    def get_data(self):
        """Return data"""
        return self.data


if __name__ == "__main__":
    
    print("\n" + "="*70)
    print("PHASE 3: FEATURE ENGINEERING")
    print("="*70 + "\n")
    
    pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCAD', 'AUDUSD']
    
    for pair in pairs:
        print(f"[PROCESSING] {pair}")
        print("-" * 70)
        
        try:
            data = pd.read_csv(f'data/processed/{pair}_processed.csv',
                              index_col=0, parse_dates=True)
            
            engineer = FeatureEngineer(data)
            
            features = (engineer
                       .moving_averages()
                       .rsi()
                       .macd()
                       .bollinger_bands()
                       .price_features()
                       .volatility()
                       .momentum()
                       .remove_nan()
                       .save(f'{pair}_features.csv')
                       .get_data())
            
            print(f"[OK] {pair}: {len(features.columns)} features, {len(features)} records\n")
            
        except Exception as e:
            print(f"[ERROR] Failed to process {pair}: {str(e)}\n")
    
    print("="*70)
    print("Phase 3 Complete! All features engineered.")
    print("="*70)