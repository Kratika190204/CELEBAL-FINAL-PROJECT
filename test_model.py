#!/usr/bin/env python3
"""
Test script for CLV Model to verify integration fixes
"""

import pandas as pd
import numpy as np
from clv_model import EnhancedCLVPredictor
from utils import generate_sample_data
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_clv_model():
    """Test the CLV model with sample data"""
    print("🧪 Testing CLV Model Integration...")
    
    try:
        # Initialize the model
        print("1. Initializing CLV Predictor...")
        clv_predictor = EnhancedCLVPredictor()
        
        # Generate sample data
        print("2. Generating sample data...")
        sample_data = generate_sample_data(n_customers=50)
        print(f"   Sample data shape: {sample_data.shape}")
        print(f"   Columns: {list(sample_data.columns)}")
        
        # Test data cleaning
        print("3. Testing data cleaning...")
        cleaned_data = clv_predictor._clean_data(sample_data.copy())
        print(f"   Cleaned data shape: {cleaned_data.shape}")
        print(f"   Cleaned columns: {list(cleaned_data.columns)}")
        
        # Test temporal split
        print("4. Testing temporal split...")
        feature_data, target_data = clv_predictor.create_temporal_split(cleaned_data)
        print(f"   Feature data shape: {feature_data.shape}")
        print(f"   Target data shape: {target_data.shape}")
        
        # Test target preparation
        print("5. Testing target preparation...")
        clv_data = clv_predictor.prepare_target_variable(feature_data, target_data)
        print(f"   CLV data shape: {clv_data.shape}")
        print(f"   CLV columns: {list(clv_data.columns)}")
        
        # Test feature preparation
        print("6. Testing feature preparation...")
        X, y = clv_predictor.prepare_features_for_modeling(clv_data)
        print(f"   Feature matrix shape: {X.shape}")
        print(f"   Target shape: {y.shape}")
        print(f"   Feature names: {clv_predictor.feature_names}")
        
        # Test model training
        print("7. Testing model training...")
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        results = clv_predictor.train_and_evaluate_models(X_train, X_test, y_train, y_test)
        print(f"   Training completed successfully!")
        print(f"   Best model R2: {results['Test_R2'].max():.4f}")
        
        # Test predictions
        print("8. Testing predictions...")
        predictions = clv_predictor.predict_clv(X_test)
        print(f"   Predictions shape: {predictions.shape}")
        print(f"   Prediction range: ${predictions.min():.2f} - ${predictions.max():.2f}")
        
        print("\n✅ All tests passed! CLV Model integration is working correctly.")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_clv_model()
    if success:
        print("\n🎉 CLV Model is ready to use!")
    else:
        print("\n🔧 Please check the error messages above and fix any issues.") 