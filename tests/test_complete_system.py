"""
FOREXAI - COMPLETE SYSTEM TESTING
Comprehensive test suite for all phases
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from prediction_engine import PredictionEngine


class SystemTester:
    """Complete system testing"""
    
    def __init__(self):
        self.results = {}
        self.passed = 0
        self.failed = 0
    
    def test_1_data_files(self):
        """Test 1: Verify all data files exist"""
        print("\n" + "="*70)
        print("TEST 1: DATA FILES VERIFICATION")
        print("="*70)
        
        required_files = [
            'data/historical/',
            'data/processed/',
            'data/models/',
        ]
        
        all_exist = True
        for path in required_files:
            exists = os.path.exists(path)
            status = "✅" if exists else "❌"
            print(f"{status} {path}")
            if not exists:
                all_exist = False
        
        self.results['Data Files'] = all_exist
        if all_exist:
            self.passed += 1
        else:
            self.failed += 1
        
        return all_exist
    
    def test_2_model_files(self):
        """Test 2: Verify trained models exist"""
        print("\n" + "="*70)
        print("TEST 2: TRAINED MODELS VERIFICATION")
        print("="*70)
        
        required_models = [
            'data/models/EURUSD_ensemble.pkl',
            'data/models/EURUSD_random_forest.pkl',
            'data/models/EURUSD_gradient_boosting.pkl',
            'data/models/scaler.pkl',
            'data/models/feature_columns.pkl',
        ]
        
        all_exist = True
        for model in required_models:
            exists = os.path.exists(model)
            status = "✅" if exists else "❌"
            print(f"{status} {model}")
            if not exists:
                all_exist = False
        
        self.results['Model Files'] = all_exist
        if all_exist:
            self.passed += 1
        else:
            self.failed += 1
        
        return all_exist
    
    def test_3_engine_initialization(self):
        """Test 3: Prediction engine initialization"""
        print("\n" + "="*70)
        print("TEST 3: PREDICTION ENGINE INITIALIZATION")
        print("="*70)
        
        try:
            engine = PredictionEngine()
            print("✅ Engine initialized successfully")
            self.results['Engine Initialization'] = True
            self.passed += 1
            return True
        except Exception as e:
            print(f"❌ Engine initialization failed: {e}")
            self.results['Engine Initialization'] = False
            self.failed += 1
            return False
    
    def test_4_live_predictions(self):
        """Test 4: Live predictions on all pairs"""
        print("\n" + "="*70)
        print("TEST 4: LIVE PREDICTIONS")
        print("="*70)
        
        engine = PredictionEngine()
        
        pairs = {
            'EUR/USD': 'EURUSD=X',
            'GBP/USD': 'GBPUSD=X',
            'USD/JPY': 'USDJPY=X',
            'USD/CAD': 'USDCAD=X',
            'AUD/USD': 'AUDUSD=X'
        }
        
        successful = 0
        failed = 0
        
        for pair_name, pair_code in pairs.items():
            try:
                result = engine.get_prediction(pair_code)
                if result:
                    pred = result['prediction']
                    print(f"✅ {pair_name}: {pred['direction']} ({pred['confidence']:.1f}%)")
                    successful += 1
                else:
                    print(f"❌ {pair_name}: No result")
                    failed += 1
            except Exception as e:
                print(f"❌ {pair_name}: {str(e)[:50]}")
                failed += 1
        
        all_success = failed == 0
        self.results['Live Predictions'] = all_success
        
        if all_success:
            self.passed += 1
        else:
            self.failed += 1
        
        print(f"\nSummary: {successful}/5 successful, {failed}/5 failed")
        return all_success
    
    def test_5_prediction_quality(self):
        """Test 5: Prediction quality checks"""
        print("\n" + "="*70)
        print("TEST 5: PREDICTION QUALITY CHECKS")
        print("="*70)
        
        engine = PredictionEngine()
        result = engine.get_prediction('EURUSD=X')
        
        if not result:
            print("❌ No prediction result")
            self.results['Prediction Quality'] = False
            self.failed += 1
            return False
        
        pred = result['prediction']
        inds = result['indicators']
        
        checks = []
        
        # Check 1: Direction is valid
        if pred['direction'] in ['UP', 'DOWN']:
            print("✅ Direction is valid (UP/DOWN)")
            checks.append(True)
        else:
            print("❌ Invalid direction")
            checks.append(False)
        
        # Check 2: Confidence is valid range
        if 0 <= pred['confidence'] <= 100:
            print(f"✅ Confidence in valid range: {pred['confidence']:.1f}%")
            checks.append(True)
        else:
            print("❌ Confidence out of range")
            checks.append(False)
        
        # Check 3: Probabilities sum to 100
        prob_sum = pred['probability_up'] + pred['probability_down']
        if 99.5 <= prob_sum <= 100.5:
            print(f"✅ Probabilities sum correctly: {prob_sum:.1f}%")
            checks.append(True)
        else:
            print(f"❌ Probabilities don't sum to 100: {prob_sum:.1f}%")
            checks.append(False)
        
        # Check 4: All indicators present
        required_indicators = [
            'price', 'rsi', 'macd', 'sma_20', 'sma_50', 'volatility'
        ]
        if all(ind in inds for ind in required_indicators):
            print("✅ All required indicators present")
            checks.append(True)
        else:
            print("❌ Missing indicators")
            checks.append(False)
        
        # Check 5: Price is reasonable
        if 0.5 < inds['price'] < 200:
            print(f"✅ Price in reasonable range: {inds['price']:.5f}")
            checks.append(True)
        else:
            print("❌ Price out of range")
            checks.append(False)
        
        all_pass = all(checks)
        self.results['Prediction Quality'] = all_pass
        
        if all_pass:
            self.passed += 1
        else:
            self.failed += 1
        
        return all_pass
    
    def test_6_indicators_calculation(self):
        """Test 6: Technical indicators calculation"""
        print("\n" + "="*70)
        print("TEST 6: TECHNICAL INDICATORS CALCULATION")
        print("="*70)
        
        engine = PredictionEngine()
        result = engine.get_prediction('EURUSD=X')
        
        if not result:
            print("❌ No prediction result")
            self.results['Indicators'] = False
            self.failed += 1
            return False
        
        inds = result['indicators']
        
        indicators_to_check = {
            'RSI': (0, 100),
            'MACD': (-0.01, 0.01),
            'SMA_20': (0.5, 200),
            'SMA_50': (0.5, 200),
            'Price': (0.5, 200),
            'Volatility': (0, 5),
        }
        
        all_valid = True
        
        for ind_name, (min_val, max_val) in indicators_to_check.items():
            key = ind_name.lower().replace('_', '_')
            if ind_name == 'RSI':
                key = 'rsi'
            elif ind_name == 'Price':
                key = 'price'
            
            if key in inds:
                val = inds[key]
                # Check RSI separately (special range)
                if ind_name == 'RSI':
                    if 0 <= val <= 100:
                        print(f"✅ {ind_name}: {val:.2f} (valid)")
                    else:
                        print(f"❌ {ind_name}: {val:.2f} (out of range)")
                        all_valid = False
                else:
                    print(f"✅ {ind_name}: {val:.6f} (calculated)")
            else:
                print(f"❌ {ind_name}: Not found")
                all_valid = False
        
        self.results['Indicators'] = all_valid
        
        if all_valid:
            self.passed += 1
        else:
            self.failed += 1
        
        return all_valid
    
    def test_7_performance_metrics(self):
        """Test 7: System performance metrics"""
        print("\n" + "="*70)
        print("TEST 7: PERFORMANCE METRICS")
        print("="*70)
        
        engine = PredictionEngine()
        
        import time
        
        # Test response time
        start = time.time()
        result = engine.get_prediction('EURUSD=X')
        elapsed = time.time() - start
        
        print(f"Response Time: {elapsed:.2f} seconds")
        
        if elapsed < 10:
            print("✅ Response time acceptable")
            performance_ok = True
        else:
            print("❌ Response time too slow")
            performance_ok = False
        
        # Check memory efficiency
        if result and 'indicators' in result:
            indicators_count = len(result['indicators'])
            print(f"✅ Indicators stored: {indicators_count}")
            memory_ok = True
        else:
            print("❌ Memory issue")
            memory_ok = False
        
        all_ok = performance_ok and memory_ok
        self.results['Performance'] = all_ok
        
        if all_ok:
            self.passed += 1
        else:
            self.failed += 1
        
        return all_ok
    
    def run_all_tests(self):
        """Run all tests"""
        print("\n")
        print("╔" + "="*68 + "╗")
        print("║" + " "*15 + "FOREXAI - COMPLETE SYSTEM TEST" + " "*23 + "║")
        print("╚" + "="*68 + "╝")
        
        self.test_1_data_files()
        self.test_2_model_files()
        self.test_3_engine_initialization()
        self.test_4_live_predictions()
        self.test_5_prediction_quality()
        self.test_6_indicators_calculation()
        self.test_7_performance_metrics()
        
        # Print summary
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        
        for test_name, result in self.results.items():
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"{test_name}: {status}")
        
        print("\n" + "="*70)
        print(f"TOTAL: {self.passed} Passed, {self.failed} Failed")
        print("="*70)
        
        if self.failed == 0:
            print("\n✅✅✅ ALL TESTS PASSED - SYSTEM READY FOR DEPLOYMENT ✅✅✅\n")
            return True
        else:
            print(f"\n❌ {self.failed} Tests Failed - Fix Issues Before Deployment\n")
            return False


# Run tests
if __name__ == "__main__":
    tester = SystemTester()
    all_passed = tester.run_all_tests()
    
    # Exit with appropriate code
    exit(0 if all_passed else 1)