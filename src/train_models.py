"""
MACHINE LEARNING MODEL TRAINING

Trains multiple ML models:
1. Random Forest Classifier
2. Gradient Boosting Classifier
3. Logistic Regression
4. Voting Ensemble (combines all models)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import joblib
import logging
import os
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Trains and evaluates ML models for forex prediction"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        logger.info("[OK] ModelTrainer initialized")
    
    def load_data(self, pair='EURUSD'):
        """Load prepared data"""
        logger.info(f"[LOADING] Data for {pair}...")
        
        X_train = pd.read_csv('data/processed/X_train.csv')
        X_test = pd.read_csv('data/processed/X_test.csv')
        y_train = pd.read_csv('data/processed/y_train.csv').values.ravel()
        y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()
        
        logger.info(f"[OK] Data loaded: Train={len(X_train)}, Test={len(X_test)}")
        
        return X_train, X_test, y_train, y_test
    
    def train_random_forest(self, X_train, y_train):
        """Train Random Forest model"""
        logger.info("[TRAINING] Random Forest...")
        
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1,
            verbose=0
        )
        
        model.fit(X_train, y_train)
        logger.info("[OK] Random Forest trained")
        
        return model
    
    def train_gradient_boosting(self, X_train, y_train):
        """Train Gradient Boosting model"""
        logger.info("[TRAINING] Gradient Boosting...")
        
        model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=7,
            random_state=42,
            verbose=0
        )
        
        model.fit(X_train, y_train)
        logger.info("[OK] Gradient Boosting trained")
        
        return model
    
    def train_logistic_regression(self, X_train, y_train):
        """Train Logistic Regression model"""
        logger.info("[TRAINING] Logistic Regression...")
        
        model = LogisticRegression(
            max_iter=1000,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        logger.info("[OK] Logistic Regression trained")
        
        return model
    
    def create_ensemble(self, rf_model, gb_model, lr_model):
        """Create voting ensemble"""
        logger.info("[CREATING] Voting Ensemble...")
        
        ensemble = VotingClassifier(
            estimators=[
                ('rf', rf_model),
                ('gb', gb_model),
                ('lr', lr_model)
            ],
            voting='soft'
        )
        
        logger.info("[OK] Ensemble created")
        return ensemble
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """Evaluate model performance"""
        logger.info(f"[EVALUATING] {model_name}...")
        
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        try:
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_pred_proba)
        except:
            roc_auc = 0
        
        results = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'ROC-AUC': roc_auc
        }
        
        logger.info(f"[RESULTS] {model_name}:")
        for metric, value in results.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        return results
    
    def save_models(self, models_dict, pair='EURUSD'):
        """Save trained models"""
        logger.info("[SAVING] Models...")
        
        os.makedirs('data/models', exist_ok=True)
        
        for model_name, model in models_dict.items():
            filename = f'data/models/{pair}_{model_name}.pkl'
            joblib.dump(model, filename)
            logger.info(f"[SAVED] {filename}")
    
    def train_all(self, pair='EURUSD'):
        """Train all models"""
        
        # Load data
        X_train, X_test, y_train, y_test = self.load_data(pair)
        
        # Train individual models
        rf_model = self.train_random_forest(X_train, y_train)
        gb_model = self.train_gradient_boosting(X_train, y_train)
        lr_model = self.train_logistic_regression(X_train, y_train)
        
        # Create ensemble
        ensemble = self.create_ensemble(rf_model, gb_model, lr_model)
        ensemble.fit(X_train, y_train)
        
        # Evaluate all models
        models_to_eval = {
            'Random Forest': rf_model,
            'Gradient Boosting': gb_model,
            'Logistic Regression': lr_model,
            'Voting Ensemble': ensemble
        }
        
        results = {}
        for model_name, model in models_to_eval.items():
            results[model_name] = self.evaluate_model(model, X_test, y_test, model_name)
        
        # Save all models
        models_to_save = {
            'random_forest': rf_model,
            'gradient_boosting': gb_model,
            'logistic_regression': lr_model,
            'ensemble': ensemble
        }
        self.save_models(models_to_save, pair)
        
        return results


if __name__ == "__main__":
    
    print("\n" + "="*70)
    print("PHASE 5: MACHINE LEARNING MODEL TRAINING")
    print("="*70 + "\n")
    
    trainer = ModelTrainer()
    
    # Train models for EURUSD (we'll use this for the web app)
    print("[TRAINING MODELS FOR EURUSD]")
    print("-" * 70)
    
    results = trainer.train_all('EURUSD')
    
    # Display results summary
    print("\n" + "="*70)
    print("MODEL PERFORMANCE SUMMARY")
    print("="*70)
    
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    print("\n" + "="*70)
    print("Phase 5 Complete! Models trained and saved.")
    print("="*70)