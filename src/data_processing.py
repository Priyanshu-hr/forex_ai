
import pandas as pd
import numpy as np
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProcessor:
    """Cleans and validates forex data"""
    
    def __init__(self, data):
        self.raw_data = data.copy()
        self.processed_data = None
        logger.info("[OK] DataProcessor initialized")
    
    def remove_missing_values(self):
        """Handle missing values using forward fill"""
        logger.info("[PROCESSING] Removing missing values...")
        
        missing_before = self.raw_data.isnull().sum().sum()
        self.processed_data = self.raw_data.ffill().bfill()
        missing_after = self.processed_data.isnull().sum().sum()
        
        logger.info(f"[OK] Missing values: {missing_before} -> {missing_after}")
        return self
    
    def remove_duplicates(self):
        """Remove duplicate rows"""
        logger.info("[PROCESSING] Removing duplicates...")
        
        duplicates = self.processed_data.duplicated().sum()
        self.processed_data = self.processed_data[~self.processed_data.duplicated()]
        
        logger.info(f"[OK] Duplicates removed: {duplicates}")
        return self
    
    def ensure_data_types(self):
        """Ensure correct data types"""
        logger.info("[PROCESSING] Ensuring correct data types...")
        
        for col in ['Open', 'High', 'Low', 'Close']:
            if col in self.processed_data.columns:
                self.processed_data[col] = pd.to_numeric(self.processed_data[col], errors='coerce')
        
        logger.info("[OK] Data types corrected")
        return self
    
    def sort_by_date(self):
        """Sort data chronologically"""
        logger.info("[PROCESSING] Sorting by date...")
        
        if not self.processed_data.index.is_monotonic_increasing:
            self.processed_data = self.processed_data.sort_index()
        
        start_date = str(self.processed_data.index[0])[:10]
        end_date = str(self.processed_data.index[-1])[:10]
        
        logger.info(f"[OK] Sorted: {start_date} to {end_date}")
        return self
    
    def display_report(self):
        """Display data quality report"""
        print("\n" + "="*70)
        print("DATA QUALITY REPORT")
        print("="*70)
        print(f"Records: {len(self.processed_data)}")
        
        start_date = str(self.processed_data.index[0])[:10]
        end_date = str(self.processed_data.index[-1])[:10]
        print(f"Date Range: {start_date} to {end_date}")
        
        print(f"Missing Values: {self.processed_data.isnull().sum().sum()}")
        print(f"Duplicates: {self.processed_data.duplicated().sum()}")
        print("\nPrice Statistics (Close):")
        print(self.processed_data['Close'].describe())
        print("="*70)
        return self
    
    def save(self, filename):
        """Save processed data"""
        os.makedirs('data/processed', exist_ok=True)
        filepath = f"data/processed/{filename}"
        self.processed_data.to_csv(filepath)
        logger.info(f"[SAVED] {filepath}")
        return self
    
    def get_data(self):
        """Return processed dataset"""
        return self.processed_data


if __name__ == "__main__":
    
    print("\n" + "="*70)
    print("DATA PROCESSING & CLEANING")
    print("="*70 + "\n")
    
    pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCAD', 'AUDUSD']
    
    for pair in pairs:
        print(f"[PROCESSING] {pair}")
        print("-" * 70)
        
        try:
            raw_data = pd.read_csv(
                f'data/historical/{pair}_historical.csv',
                index_col=0,
                parse_dates=True
            )
            
            processor = DataProcessor(raw_data)
            
            processed = (processor
                        .remove_missing_values()
                        .remove_duplicates()
                        .ensure_data_types()
                        .sort_by_date()
                        .display_report()
                        .save(f'{pair}_processed.csv')
                        .get_data())
            
            print(f"[OK] {pair} processing complete!\n")
            
        except Exception as e:
            print(f"[ERROR] Failed to process {pair}: {str(e)}\n")
    
    print("="*70)
    print("Phase 2 Complete! Data cleaned and ready.")
    print("="*70)