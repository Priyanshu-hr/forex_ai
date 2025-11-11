import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLDataPreparation:
    """Prepares data for machine learning models"""
    
    def __init__(self, features_data):
        self.data = features_data.copy()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = MinMaxScaler()
        logger.info("[OK] MLDataPreparation initialized")
    
    def create_target(self):
        """Create binary target: 1=UP, 0=DOWN"""
        logger.info("[CREATING] Target variable...")
        
        # Target: 1 if tomorrow's close > today's close, else 0
        self.data['Target'] = (self.data['Close'].shift(-1) > self.data['Close']).astype(int)
        self.data = self.data[:-1]  # Remove last row (no target)
        
        up_count = (self.data['Target'] == 1).sum()
        down_count = (self.data['Target'] == 0).sum()
        
        logger.info(f"[OK] Target created: UP={up_count}, DOWN={down_count}")
        return self
    
    def select_features(self):
        """Select features for model"""
        logger.info("[SELECTING] Features...")
        
        self.feature_columns = [
            'SMA_10', 'SMA_20', 'SMA_50', 'SMA_200',
            'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
            'BB_Upper', 'BB_Lower', 'BB_Middle', 'BB_Width',
            'Daily_Return', 'Intraday_Range', 'Gap',
            'Volatility', 'ATR', 'ROC_5', 'ROC_10'
        ]
        
        self.X = self.data[self.feature_columns].copy()
        self.y = self.data['Target'].copy()
        
        logger.info(f"[OK] Selected {len(self.feature_columns)} features")
        return self
    
    def scale_features(self):
        """Normalize features to 0-1 range"""
        logger.info("[SCALING] Features to 0-1 range...")
        
        self.X_scaled = self.scaler.fit_transform(self.X)
        self.X_scaled = pd.DataFrame(self.X_scaled, columns=self.feature_columns)
        
        logger.info("[OK] Features scaled")
        return self
    
    def split_data(self, test_size=0.2):
        """Split into train/test sets"""
        logger.info(f"[SPLITTING] Data (80/20 split)...")
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_scaled, self.y,
            test_size=test_size,
            random_state=42,
            stratify=self.y
        )
        
        logger.info(f"[OK] Train: {len(self.X_train)}, Test: {len(self.X_test)}")
        return self
    
    def save_data(self):
        """Save train/test data and scaler"""
        logger.info("[SAVING] Data and scaler...")
        
        os.makedirs('data/processed', exist_ok=True)
        os.makedirs('data/models', exist_ok=True)
        
        self.X_train.to_csv('data/processed/X_train.csv', index=False)
        self.X_test.to_csv('data/processed/X_test.csv', index=False)
        self.y_train.to_csv('data/processed/y_train.csv', index=False)
        self.y_test.to_csv('data/processed/y_test.csv', index=False)
        
        joblib.dump(self.scaler, 'data/models/scaler.pkl')
        joblib.dump(self.feature_columns, 'data/models/feature_columns.pkl')
        
        logger.info("[OK] Data and scaler saved")
        return self
    
    def get_data(self):
        """Return prepared data"""
        return {
            'X_train': self.X_train,
            'X_test': self.X_test,
            'y_train': self.y_train,
            'y_test': self.y_test,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns
        }


if __name__ == "__main__":
    
    print("\n" + "="*70)
    print("PHASE 4: ML DATA PREPARATION")
    print("="*70 + "\n")
    
    pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCAD', 'AUDUSD']
    
    for pair in pairs:
        print(f"[PREPARING] {pair}")
        print("-" * 70)
        
        try:
            features = pd.read_csv(f'data/processed/{pair}_features.csv', index_col=0, parse_dates=True)
            
            prep = MLDataPreparation(features)
            ml_data = (prep
                      .create_target()
                      .select_features()
                      .scale_features()
                      .split_data()
                      .save_data()
                      .get_data())
            
            print(f"[OK] {pair} prepared\n")
            
        except Exception as e:
            print(f"[ERROR] Failed to prepare {pair}: {str(e)}\n")
    
    print("="*70)
    print("Phase 4 Complete! Data ready for ML model training.")
    print("="*70)