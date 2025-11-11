"""Quick installation test - FIXED"""

import sys
print(f"Python Version: {sys.version}\n")

libraries = {
    'pandas': '📊 Data processing',
    'numpy': '🔢 Numerical computing',
    'sklearn': '🤖 Machine learning',  # Changed from scikit-learn
    'streamlit': '🌐 Web app framework',
    'plotly': '📈 Interactive charts',
    'yfinance': '💱 Forex data',
    'joblib': '💾 Save/load models'
}

print("Checking installations...\n")

all_good = True
for lib, description in libraries.items():
    try:
        __import__(lib)
        print(f"✅ {lib:20} {description}")
    except ImportError:
        print(f"❌ {lib:20} {description}")
        all_good = False

if all_good:
    print("\n" + "="*60)
    print("🎉 ALL INSTALLATIONS SUCCESSFUL!")
    print("="*60)
    print("\n✨ You're ready to start the Forex project!")
    print("\n📋 Next: Run Phase 1 - Data Collection")
    print("   Command: python src/data_collection.py")
else:
    print("\n❌ Some libraries failed. Please check error messages above.")