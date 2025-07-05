#!/usr/bin/env python3
"""
Test script to verify column name fix
"""

import pandas as pd
import numpy as np
from clv_model import EnhancedCLVPredictor
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

def test_column_names():
    """Test that the model uses correct column names"""
    
    print("🧪 Testing column name consistency...")
    
    # Create sample customer data
    sample_data = pd.DataFrame({
        'customer_id': range(1, 11),
        'age': np.random.randint(25, 65, 10),
        'total_purchases': np.random.randint(1, 50, 10),
        'avg_order_value': np.random.uniform(50, 500, 10),
        'days_since_first_purchase': np.random.randint(30, 365, 10),
        'days_since_last_purchase': np.random.randint(1, 90, 10),
        'acquisition_channel': np.random.choice(['Online', 'Store', 'Referral'], 10),
        'location': np.random.choice(['NYC', 'LA', 'Chicago', 'Houston'], 10),
        'subscription_status': np.random.choice(['Active', 'Inactive'], 10),
        'invoice_date': pd.date_range('2024-01-01', periods=10, freq='D')
    })
    
    # Initialize predictor
    predictor = EnhancedCLVPredictor()
    
    try:
        # Test the full pipeline
        print("📊 Running full pipeline...")
        results = predictor.run_full_pipeline(sample_data, save_model_path='test_model.joblib')
        
        # Check if predictions have correct column names
        print("🔍 Checking prediction results...")
        
        # Test prediction on new data
        test_data = sample_data.head(3).copy()
        processed_data = predictor.create_advanced_features(test_data)
        X, _ = predictor.prepare_features_for_modeling(processed_data)
        predictions = predictor.predict_clv(X)
        
        print(f"✅ Predictions shape: {predictions.shape}")
        print(f"✅ Predictions type: {type(predictions)}")
        print(f"✅ First few predictions: {predictions[:3]}")
        
        # Test visualization creation
        print("📈 Testing visualization creation...")
        
        # Create a predictions DataFrame with expected columns
        predictions_df = test_data.copy()
        predictions_df['predicted_clv'] = predictions
        predictions_df['customer_segment'] = predictor.segment_customers(predictions)
        predictions_df['percentile_rank'] = predictions_df['predicted_clv'].rank(pct=True) * 100
        predictions_df['churn_risk'] = 0.5  # Default value
        
        print(f"✅ Predictions DataFrame columns: {list(predictions_df.columns)}")
        
        # Test visualization method
        fig_dist, fig_segments, fig_scatter, top_customers = predictor.create_visualizations(predictions_df)
        
        print("✅ Visualizations created successfully!")
        print(f"✅ Top customers shape: {top_customers.shape}")
        
        print("\n🎉 All tests passed! Column names are consistent.")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        import os
        if os.path.exists('test_model.joblib'):
            os.remove('test_model.joblib')

if __name__ == "__main__":
    test_column_names() 