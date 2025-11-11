import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ForexDataCollector:
    """Downloads historical forex data"""
    
    def __init__(self):
        self.currency_pairs = {
            'EURUSD': 'EURUSD=X',
            'GBPUSD': 'GBPUSD=X',
            'USDJPY': 'USDJPY=X',
            'USDCAD': 'USDCAD=X',
            'AUDUSD': 'AUDUSD=X'
        }
        os.makedirs('data/historical', exist_ok=True)
        logger.info("[OK] ForexDataCollector ready")
    
    def download_data(self, pair_name, pair_code, years=5):
        """Download historical data for a currency pair"""
        
        logger.info(f"[DOWNLOADING] {pair_name} ({years} years)...")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years * 365)
        
        try:
            # Download data
            data = yf.download(
                pair_code,
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                interval='1d',
                progress=False,
                auto_adjust=False  # Don't auto-adjust
            )
            
            # Remove timezone info
            if data.index.tz is not None:
                data.index = data.index.tz_localize(None)
            
            # Keep only needed columns
            columns_to_keep = ['Open', 'High', 'Low', 'Close']
            data = data[columns_to_keep]
            
            logger.info(f"[SUCCESS] Downloaded {len(data)} records for {pair_name}")
            return data
            
        except Exception as e:
            logger.error(f"[ERROR] Error downloading {pair_name}: {str(e)}")
            return None
    
    def save_data(self, data, pair_name):
        """Save data to CSV"""
        filename = f"data/historical/{pair_name}_historical.csv"
        data.to_csv(filename)
        logger.info(f"[SAVED] {filename}")
    
    def download_all(self, years=5):
        """Download all currency pairs"""
        
        print("\n" + "="*70)
        print("FOREX DATA COLLECTION")
        print("="*70 + "\n")
        
        successful = 0
        
        for pair_name, pair_code in self.currency_pairs.items():
            data = self.download_data(pair_name, pair_code, years)
            
            if data is not None and len(data) > 0:
                self.save_data(data, pair_name)
                successful += 1
            
            print()
        
        print("="*70)
        print(f"[COMPLETE] Downloaded {successful}/{len(self.currency_pairs)} pairs")
        print("="*70)


if __name__ == "__main__":
    collector = ForexDataCollector()
    collector.download_all(years=5)
    print("\nPhase 1 Complete! Data ready for processing.\n")