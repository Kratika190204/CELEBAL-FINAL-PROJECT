import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import pickle
import joblib
from scipy.stats import spearmanr
from scipy import stats
import logging
import os
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings('ignore')

# Machine Learning libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.cluster import KMeans
from sklearn.inspection import permutation_importance
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CLVConfig:
    """Configuration class for CLV model parameters"""
    
    # Data cleaning parameters
    MIN_PURCHASE_AMOUNT = 0
    MIN_QUANTITY = 0
    
    # Feature engineering parameters
    TRAINING_PERIOD_RATIO = 0.8
    RFM_QUANTILES = 5
    LOOKBACK_MONTHS = 12  # Months to look back for features
    PREDICTION_MONTHS = 6  # Months to predict forward
    
    # Model training parameters
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    CV_FOLDS = 5
    N_JOBS = -1
    
    # Business parameters
    HIGH_VALUE_THRESHOLD_PERCENTILE = 90
    CHURN_RISK_MULTIPLIER = 2.0

class EnhancedCLVPredictor:
    """Production-ready CLV prediction model with advanced features"""
    
    def __init__(self, config=None):
        self.config = config or CLVConfig()
        self.models = {}
        self.best_model = None
        self.scaler = None
        self.feature_names = []
        self.model_performance = {}
        self.is_fitted = False
        
        logging.info("CLV Predictor initialized")
    
    def load_and_clean_data(self, file_path):
        """Load and clean the retail dataset with enhanced error handling"""
        logging.info(f"Loading data from {file_path}")
        
        try:
            # Check current directory
            current_dir = os.getcwd()
            print(f"Current directory: {current_dir}")
            print(f"Files in current directory: {os.listdir('.')}")
            
            # Try different possible filenames
            possible_files = [
                file_path,
                file_path.replace(' ', '_'),
                file_path.replace(' ', ''),
                'online_retail.xlsx',
                'online retail.xlsx',
                'OnlineRetail.xlsx',
                'Online Retail.xlsx'
            ]
            
            df = None
            loaded_file = None
            
            for filename in possible_files:
                if os.path.exists(filename):
                    loaded_file = filename
                    break
            
            if loaded_file is None:
                # List all Excel files in directory
                excel_files = [f for f in os.listdir('.') if f.endswith(('.xlsx', '.xls'))]
                print(f"Available Excel files: {excel_files}")
                
                if excel_files:
                    print(f"Using the first available Excel file: {excel_files[0]}")
                    loaded_file = excel_files[0]
                else:
                    raise FileNotFoundError("No Excel files found in the current directory")
            
            # Try loading the file
            print(f"Attempting to load: {loaded_file}")
            
            if loaded_file.endswith('.xlsx') or loaded_file.endswith('.xls'):
                try:
                    # Try reading Excel file
                    df = pd.read_excel(loaded_file)
                    logging.info(f"Successfully loaded Excel file: {loaded_file}")
                except Exception as e:
                    print(f"Error reading Excel file: {e}")
                    # Try reading as CSV
                    csv_file = loaded_file.replace('.xlsx', '.csv').replace('.xls', '.csv')
                    if os.path.exists(csv_file):
                        df = pd.read_csv(csv_file)
                        logging.info(f"Loaded CSV version: {csv_file}")
                    else:
                        raise
            else:
                df = pd.read_csv(loaded_file)
            
            if df is None:
                raise ValueError("Failed to load any data file")
            
            logging.info(f"Original dataset shape: {df.shape}")
            logging.info(f"Columns: {list(df.columns)}")
            
            # Data cleaning
            df = self._clean_data(df)
            logging.info(f"Cleaned dataset shape: {df.shape}")
            
            return df
            
        except Exception as e:
            logging.error(f"Error loading data: {e}")
            print(f"Error details: {str(e)}")
            raise
    
   
    
    def create_temporal_split(self, df):
        """Create proper temporal split to prevent data leakage"""
        logging.info("Creating temporal split to prevent data leakage")
        
        # Check if this is customer-level data (no transaction-level columns)
        transaction_columns = ['total_amount', 'quantity', 'unit_price', 'stock_code', 'invoice']
        has_transaction_data = any(col in df.columns for col in transaction_columns)
        
        if not has_transaction_data:
            logging.info("Customer-level data detected - skipping temporal split")
            # For customer-level data, return the same data for both features and targets
            # The target will be calculated from the existing features
            return df.copy(), pd.DataFrame()
        
        # Transaction data processing - only if we have actual transaction data
        if 'invoice_date' not in df.columns:
            logging.warning("No invoice_date found in transaction data - treating as customer-level")
            return df.copy(), pd.DataFrame()
        
        df_sorted = df.sort_values('invoice_date')
        
        # Calculate split date based on configuration
        min_date = df_sorted['invoice_date'].min()
        max_date = df_sorted['invoice_date'].max()
        total_days = (max_date - min_date).days
        
        # Use lookback period for features and prediction period for targets
        feature_end_date = max_date - timedelta(days=self.config.PREDICTION_MONTHS * 30)
        feature_start_date = feature_end_date - timedelta(days=self.config.LOOKBACK_MONTHS * 30)
        
        logging.info(f"Feature period: {feature_start_date} to {feature_end_date}")
        logging.info(f"Prediction period: {feature_end_date} to {max_date}")
        
        # Split data
        feature_data = df_sorted[
            (df_sorted['invoice_date'] >= feature_start_date) & 
            (df_sorted['invoice_date'] <= feature_end_date)
        ]
        
        target_data = df_sorted[
            df_sorted['invoice_date'] > feature_end_date
        ]
        
        logging.info(f"Feature data shape: {feature_data.shape}")
        logging.info(f"Target data shape: {target_data.shape}")
        
        return feature_data, target_data
            
    def _clean_data(self, df):
        """Enhanced data cleaning with comprehensive column name handling
        
        Args:
            df: Input DataFrame (either transaction or customer-level data)
            
        Returns:
            Cleaned DataFrame with standardized column names
            
        Raises:
            ValueError: If required columns are missing after standardization
        """
        # Convert all column names to lowercase with underscores
        df.columns = [str(col).strip().lower().replace(' ', '_') for col in df.columns]
        
        # Comprehensive column name mapping
        column_mapping = {
            # Customer ID variations
            'customerid': 'customer_id',
            'clientid': 'customer_id',
            'cust_id': 'customer_id',
            
            # Transaction metrics
            'total_transactions': 'total_purchases',
            'transaction_count': 'total_purchases',
            'purchase_count': 'total_purchases',
            'avg_spend': 'avg_order_value',
            'average_spend': 'avg_order_value',
            'avg_transaction_value': 'avg_order_value',
            
            # Temporal metrics
            'customer_age_days': 'days_since_first_purchase',
            'first_purchase_days': 'days_since_first_purchase',
            'recency_days': 'days_since_last_purchase',
            'last_purchase_days': 'days_since_last_purchase',
            'days_since_first': 'days_since_first_purchase',
            'days_since_last': 'days_since_last_purchase',
            
            # Date fields
            'invoicedate': 'invoice_date',
            'transactiondate': 'invoice_date',
            'date': 'invoice_date',
            
            # Other common variations
            'channel': 'acquisition_channel',
            'source': 'acquisition_channel',
            'region': 'location',
            'sub_status': 'subscription_status'
        }
        
        # Apply column name standardization
        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
        
        # Validate we have either transaction or customer-level data
        has_transaction_data = 'invoice_date' in df.columns
        has_customer_data = all(col in df.columns for col in [
            'customer_id', 'total_purchases', 'avg_order_value',
            'days_since_first_purchase', 'days_since_last_purchase'
        ])
        
        if not (has_transaction_data or has_customer_data):
            missing_customer_cols = [
                col for col in [
                    'customer_id', 'total_purchases', 'avg_order_value',
                    'days_since_first_purchase', 'days_since_last_purchase'
                ] if col not in df.columns
            ]
            raise ValueError(
                "Data must contain either:\n"
                "1. Transaction data with 'invoice_date'\n"
                "2. Customer-level data with all of: "
                f"{missing_customer_cols}\n"
                f"Actual columns: {list(df.columns)}"
            )
        
        # Convert date fields if present
        if 'invoice_date' in df.columns:
            df['invoice_date'] = pd.to_datetime(df['invoice_date'], errors='coerce')
            df = df.dropna(subset=['invoice_date'])
        
        # Ensure numeric fields are properly typed
        numeric_cols = [
            'total_purchases', 'avg_order_value',
            'days_since_first_purchase', 'days_since_last_purchase'
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].fillna(0)
        
        # Validate temporal relationships
        if all(col in df.columns for col in ['days_since_first_purchase', 'days_since_last_purchase']):
            invalid = df['days_since_last_purchase'] > df['days_since_first_purchase']
            if invalid.any():
                df.loc[invalid, 'days_since_last_purchase'] = df.loc[invalid, 'days_since_first_purchase']
        
        # Add default values for optional columns if missing
        optional_cols = {
            'age': 35,
            'acquisition_channel': 'Unknown',
            'location': 'Unknown',
            'subscription_status': 'None'
        }
        for col, default in optional_cols.items():
            if col not in df.columns:
                df[col] = default
        
        return df

    def create_advanced_features(self, df, reference_date=None):
        """Create comprehensive features for CLV prediction"""
        logging.info("Creating advanced features")
        
        # Check if this is customer-level data (no transaction columns)
        transaction_columns = ['total_amount', 'quantity', 'unit_price', 'stock_code', 'invoice']
        has_transaction_data = any(col in df.columns for col in transaction_columns)
        
        if not has_transaction_data:
            logging.info("Customer-level data detected - returning data as-is")
            # For customer-level data, return the data directly
            return df.copy()
        
        # Transaction data processing
        if reference_date is None:
            if 'invoice_date' in df.columns:
                reference_date = df['invoice_date'].max()
            else:
                reference_date = pd.Timestamp.now()
        
        # Basic aggregations using lowercase column names
        customer_features = df.groupby('customer_id').agg({
            'invoice_date': ['min', 'max', 'count'],
            'invoice': 'nunique',
            'total_amount': ['sum', 'mean', 'std', 'min', 'max', 'median'],
            'quantity': ['sum', 'mean', 'std'],
            'stock_code': 'nunique',
            'unit_price': ['mean', 'std'],
            'year': 'nunique',
            'month': 'nunique',
            'quarter': 'nunique',
            'day_of_week': lambda x: x.mode().iloc[0] if not x.empty else 0,
            'hour': lambda x: x.mode().iloc[0] if not x.empty else 0,
            'is_weekend': 'mean',
            'season': lambda x: x.mode().iloc[0] if not x.empty else 'Spring'
        }).reset_index()
        
        # Flatten column names
        customer_features.columns = [
            'customer_id', 'first_purchase', 'last_purchase', 'total_transactions',
            'unique_invoices', 'total_revenue', 'avg_order_value', 'std_order_value',
            'min_order_value', 'max_order_value', 'median_order_value', 'total_quantity', 
            'avg_quantity', 'std_quantity', 'unique_products', 'avg_unit_price', 
            'std_unit_price', 'years_active', 'months_active', 'quarters_active',
            'preferred_day_of_week', 'preferred_hour', 'weekend_purchase_rate',
            'preferred_season'
        ]
        
        # Calculate derived features
        customer_features['customer_lifespan'] = (
            customer_features['last_purchase'] - customer_features['first_purchase']
        ).dt.days
        
        customer_features['days_since_last_purchase'] = (
            reference_date - customer_features['last_purchase']
        ).dt.days
        
        customer_features['avg_days_between_purchases'] = (
            customer_features['customer_lifespan'] / 
            (customer_features['total_transactions'] - 1).clip(lower=1)
        )
        
        # Advanced behavioral features
        customer_features['purchase_frequency'] = (
            customer_features['total_transactions'] / 
            (customer_features['customer_lifespan'] + 1).clip(lower=1)
        )
        
        customer_features['purchase_velocity'] = (
            customer_features['total_transactions'] / 
            customer_features['customer_lifespan'].clip(lower=1)
        )
        
        customer_features['product_diversity_ratio'] = (
            customer_features['unique_products'] / customer_features['total_transactions']
        )
        
        customer_features['spending_consistency'] = (
            1 / (1 + customer_features['std_order_value'] / 
                 customer_features['avg_order_value'].clip(lower=0.01))
        )
        
        customer_features['revenue_per_transaction'] = (
            customer_features['total_revenue'] / customer_features['total_transactions']
        )
        
        customer_features['revenue_growth_potential'] = (
            customer_features['max_order_value'] / customer_features['avg_order_value']
        )
        
        # RFM Analysis
        customer_features['recency'] = customer_features['days_since_last_purchase']
        customer_features['frequency'] = customer_features['total_transactions']
        customer_features['monetary'] = customer_features['total_revenue']
        
        # RFM Scores with better handling of edge cases
        try:
            customer_features['recency_score'] = pd.qcut(
                customer_features['recency'], self.config.RFM_QUANTILES, 
                labels=[5,4,3,2,1], duplicates='drop'
            )
        except ValueError:
            # Fallback for when there aren't enough unique values
            customer_features['recency_score'] = pd.cut(
                customer_features['recency'], self.config.RFM_QUANTILES, 
                labels=[5,4,3,2,1], duplicates='drop'
            )
        
        try:
            customer_features['frequency_score'] = pd.qcut(
                customer_features['frequency'].rank(method='first'), 
                self.config.RFM_QUANTILES, labels=[1,2,3,4,5], duplicates='drop'
            )
        except ValueError:
            customer_features['frequency_score'] = pd.cut(
                customer_features['frequency'], self.config.RFM_QUANTILES, 
                labels=[1,2,3,4,5], duplicates='drop'
            )
        
        try:
            customer_features['monetary_score'] = pd.qcut(
                customer_features['monetary'], self.config.RFM_QUANTILES, 
                labels=[1,2,3,4,5], duplicates='drop'
            )
        except ValueError:
            customer_features['monetary_score'] = pd.cut(
                customer_features['monetary'], self.config.RFM_QUANTILES, 
                labels=[1,2,3,4,5], duplicates='drop'
            )
        
        # Convert to numeric
        customer_features['recency_score'] = pd.to_numeric(customer_features['recency_score'], errors='coerce')
        customer_features['frequency_score'] = pd.to_numeric(customer_features['frequency_score'], errors='coerce')
        customer_features['monetary_score'] = pd.to_numeric(customer_features['monetary_score'], errors='coerce')
        
        # Churn risk analysis
        churn_threshold = customer_features['avg_days_between_purchases'] * self.config.CHURN_RISK_MULTIPLIER
        customer_features['churn_risk'] = (
            customer_features['days_since_last_purchase'] > churn_threshold
        ).astype(int)
        
        customer_features['churn_probability'] = np.minimum(
            customer_features['days_since_last_purchase'] / churn_threshold.clip(lower=1), 1.0
        )
        
        # Seasonality features
        season_encoder = LabelEncoder()
        customer_features['season_encoded'] = season_encoder.fit_transform(
            customer_features['preferred_season'].astype(str)
        )
        
        # Fill missing values
        numeric_columns = customer_features.select_dtypes(include=[np.number]).columns
        customer_features[numeric_columns] = customer_features[numeric_columns].fillna(0)
        
        logging.info(f"Created features for {len(customer_features)} customers")
        return customer_features
    
    def prepare_target_variable(self, feature_data, target_data):
        """Prepare CLV target variable, handling both temporal and customer-level data
        
        Args:
            feature_data: DataFrame from create_temporal_split (either temporal features or full customer data)
            target_data: DataFrame from create_temporal_split (empty for customer-level data)
        
        Returns:
            DataFrame: Contains features and target variable (predicted_clv)
        """
        # Case 1: Transaction-level data with temporal split and total_amount column
        if (not target_data.empty and 'invoice_date' in feature_data.columns and 
            'total_amount' in feature_data.columns):
            logging.info("Preparing CLV target from temporal split")
            
            # Calculate future CLV from target period
            future_clv = target_data.groupby('customer_id')['total_amount'].sum().reset_index()
            future_clv.columns = ['customer_id', 'predicted_clv']
            
            # Create features from historical data only
            features = self.create_advanced_features(feature_data)
            
            # Merge with future CLV
            clv_data = features.merge(future_clv, on='customer_id', how='left')
            clv_data['predicted_clv'] = clv_data['predicted_clv'].fillna(0)
            
            logging.info(f"Customers with future purchases: {(clv_data['predicted_clv'] > 0).sum()}")
            logging.info(f"Customers with no future purchases: {(clv_data['predicted_clv'] == 0).sum()}")

        # Case 2: Customer-level data (with or without temporal split)
        else:
            logging.info("Customer-level data detected - using existing features directly")
            
            # For customer-level data, use the data directly as features
            # Skip create_advanced_features since it's designed for transaction data
            clv_data = feature_data.copy()
            
            # Estimate predicted_clv using available customer data
            if 'total_purchases' in clv_data.columns and 'avg_order_value' in clv_data.columns:
                clv_data['predicted_clv'] = clv_data['total_purchases'] * clv_data['avg_order_value']
                logging.info("Estimated CLV using: total_purchases * avg_order_value")
            elif 'total_revenue' in clv_data.columns:
                clv_data['predicted_clv'] = clv_data['total_revenue']
                logging.info("Estimated CLV using: total_revenue")
            else:
                # Fallback: create a simple CLV estimate
                clv_data['predicted_clv'] = clv_data.get('total_purchases', 1) * clv_data.get('avg_order_value', 50)
                logging.warning("Using fallback CLV estimation")

        # Add percentile ranks for comparison
        clv_data['CLV_Percentile'] = clv_data['predicted_clv'].rank(pct=True) * 100
        
        logging.info(f"Final CLV dataset shape: {clv_data.shape}")
        return clv_data
    
    def prepare_features_for_prediction(self, df):
        """Prepare features for prediction (without target variable)
        
        Args:
            df: DataFrame with customer data
            
        Returns:
            X: Feature matrix for prediction
        """
        # Define feature columns for both transaction-level and customer-level data
        transaction_features = [
            'total_transactions', 'unique_invoices', 'total_revenue', 'avg_order_value',
            'std_order_value', 'min_order_value', 'max_order_value', 'median_order_value',
            'total_quantity', 'avg_quantity', 'std_quantity', 'unique_products',
            'avg_unit_price', 'std_unit_price', 'customer_lifespan', 'days_since_last_purchase',
            'avg_days_between_purchases', 'purchase_frequency', 'purchase_velocity',
            'product_diversity_ratio', 'spending_consistency', 'revenue_per_transaction',
            'revenue_growth_potential', 'years_active', 'months_active', 'quarters_active',
            'preferred_day_of_week', 'preferred_hour', 'weekend_purchase_rate',
            'recency_score', 'frequency_score', 'monetary_score', 'churn_risk',
            'churn_probability', 'season_encoded'
        ]
        
        customer_features = [
            'age', 'total_purchases', 'avg_order_value', 'days_since_first_purchase',
            'days_since_last_purchase', 'acquisition_channel', 'location', 'subscription_status'
        ]
        
        # Check which type of data we have and select appropriate features
        if any(col in df.columns for col in transaction_features):
            # Transaction-level data
            feature_cols = transaction_features
            logging.info("Using transaction-level features for prediction")
        else:
            # Customer-level data
            feature_cols = customer_features
            logging.info("Using customer-level features for prediction")
        
        # Ensure all features exist
        available_features = [col for col in feature_cols if col in df.columns]
        
        if not available_features:
            logging.warning("No expected features found. Using all numeric columns.")
            # Fallback: use all numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            available_features = numeric_cols
        
        X = df[available_features].copy()
        
        # Handle categorical features
        categorical_features = ['acquisition_channel', 'location', 'subscription_status']
        for cat_feature in categorical_features:
            if cat_feature in X.columns:
                # Create dummy variables for categorical features
                dummies = pd.get_dummies(X[cat_feature], prefix=cat_feature, drop_first=True)
                X = pd.concat([X.drop(columns=[cat_feature]), dummies], axis=1)
                logging.info(f"Encoded categorical feature: {cat_feature}")
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Replace infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        logging.info(f"Prediction feature matrix shape: {X.shape}")
        logging.info(f"Available features for prediction: {list(X.columns)}")
        
        return X

    def prepare_features_for_modeling(self, df):
        """Prepare features for machine learning"""
        
        # Define feature columns for both transaction-level and customer-level data
        transaction_features = [
            'total_transactions', 'unique_invoices', 'total_revenue', 'avg_order_value',
            'std_order_value', 'min_order_value', 'max_order_value', 'median_order_value',
            'total_quantity', 'avg_quantity', 'std_quantity', 'unique_products',
            'avg_unit_price', 'std_unit_price', 'customer_lifespan', 'days_since_last_purchase',
            'avg_days_between_purchases', 'purchase_frequency', 'purchase_velocity',
            'product_diversity_ratio', 'spending_consistency', 'revenue_per_transaction',
            'revenue_growth_potential', 'years_active', 'months_active', 'quarters_active',
            'preferred_day_of_week', 'preferred_hour', 'weekend_purchase_rate',
            'recency_score', 'frequency_score', 'monetary_score', 'churn_risk',
            'churn_probability', 'season_encoded'
        ]
        
        customer_features = [
            'age', 'total_purchases', 'avg_order_value', 'days_since_first_purchase',
            'days_since_last_purchase', 'acquisition_channel', 'location', 'subscription_status'
        ]
        
        # Check which type of data we have and select appropriate features
        if any(col in df.columns for col in transaction_features):
            # Transaction-level data
            feature_cols = transaction_features
            logging.info("Using transaction-level features")
        else:
            # Customer-level data
            feature_cols = customer_features
            logging.info("Using customer-level features")
        
        # Ensure all features exist
        available_features = [col for col in feature_cols if col in df.columns]
        self.feature_names = available_features
        
        if not available_features:
            logging.warning("No expected features found. Using all numeric columns except target.")
            # Fallback: use all numeric columns except the target
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            available_features = [col for col in numeric_cols if col != 'predicted_clv' and col != 'CLV_Percentile']
            self.feature_names = available_features
        
        X = df[available_features].copy()
        y = df['predicted_clv'].copy()
        
        # Handle categorical features
        categorical_features = ['acquisition_channel', 'location', 'subscription_status']
        for cat_feature in categorical_features:
            if cat_feature in X.columns:
                # Create dummy variables for categorical features
                dummies = pd.get_dummies(X[cat_feature], prefix=cat_feature, drop_first=True)
                X = pd.concat([X.drop(columns=[cat_feature]), dummies], axis=1)
                logging.info(f"Encoded categorical feature: {cat_feature}")
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Replace infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        # Update feature names after encoding
        self.feature_names = X.columns.tolist()
        
        logging.info(f"Feature matrix shape: {X.shape}")
        logging.info(f"Target variable shape: {y.shape}")
        logging.info(f"Available features: {self.feature_names}")
        
        return X, y
    
    def calculate_business_metrics(self, y_true, y_pred, customer_ids=None):
        """Calculate business-focused evaluation metrics"""
        
        # Standard regression metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Business-focused metrics
        ranking_corr = spearmanr(y_true, y_pred)[0] if len(y_true) > 1 else 0
        
        # Top customer identification precision
        if len(y_true) > 10:
            top_10_pct = max(1, int(len(y_true) * 0.1))
            true_top_indices = y_true.argsort()[-top_10_pct:]
            pred_top_indices = y_pred.argsort()[-top_10_pct:]
            top_10_precision = len(set(true_top_indices) & set(pred_top_indices)) / top_10_pct
        else:
            top_10_precision = 0
        
        # MAPE (Mean Absolute Percentage Error) for non-zero values
        non_zero_mask = y_true > 0
        if non_zero_mask.sum() > 0:
            mape = mean_absolute_percentage_error(y_true[non_zero_mask], y_pred[non_zero_mask])
        else:
            mape = np.inf
        
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape,
            'Ranking_Correlation': ranking_corr,
            'Top_10_Precision': top_10_precision
        }
    
    def create_model_pipeline(self):
        """Create advanced model pipeline with preprocessing"""
        
        # Define models with simplified hyperparameter grids for faster training
        models = {
            'Linear_Regression': {
                'model': LinearRegression(),
                'params': {},
                'scale': True
            },
            'Ridge_Regression': {
                'model': Ridge(),
                'params': {'alpha': [1.0]},  # Reduced from [0.1, 1.0, 10.0]
                'scale': True
            },
            'Random_Forest': {
                'model': RandomForestRegressor(random_state=self.config.RANDOM_STATE, n_jobs=1),
                'params': {
                    'n_estimators': [100],  # Reduced from [100, 200]
                    'max_depth': [10],      # Reduced from [10, 20, None]
                    'min_samples_split': [2]  # Reduced from [2, 5]
                },
                'scale': False
            },
            'Gradient_Boosting': {
                'model': GradientBoostingRegressor(random_state=self.config.RANDOM_STATE),
                'params': {
                    'n_estimators': [100],  # Reduced from [100, 200]
                    'learning_rate': [0.1], # Reduced from [0.05, 0.1]
                    'max_depth': [3]        # Reduced from [3, 5]
                },
                'scale': False
            }
        }
        
        return models
    
    def train_and_evaluate_models(self, X_train, X_test, y_train, y_test):
        """Train multiple models with hyperparameter tuning"""
        logging.info("Training and evaluating models with hyperparameter tuning")
        
        models = self.create_model_pipeline()
        results = []
        
        # Initialize scalers
        standard_scaler = StandardScaler()
        
        for name, model_config in models.items():
            logging.info(f"Training {name}...")
            
            try:
                model = model_config['model']
                params = model_config['params']
                needs_scaling = model_config['scale']
                
                # Prepare data
                if needs_scaling:
                    X_train_processed = standard_scaler.fit_transform(X_train)
                    X_test_processed = standard_scaler.transform(X_test)
                else:
                    X_train_processed = X_train
                    X_test_processed = X_test
                
                # Hyperparameter tuning
                if params:
                    grid_search = GridSearchCV(
                        model, params, cv=2,  # Reduced from 3 to 2 for speed
                        scoring='r2', n_jobs=1, verbose=0  # Reduced n_jobs from 2 to 1
                    )
                    grid_search.fit(X_train_processed, y_train)
                    best_model = grid_search.best_estimator_
                    best_params = grid_search.best_params_
                else:
                    best_model = model
                    best_model.fit(X_train_processed, y_train)
                    best_params = {}
                
                # Make predictions
                y_pred_train = best_model.predict(X_train_processed)
                y_pred_test = best_model.predict(X_test_processed)
                
                # Calculate metrics
                train_metrics = self.calculate_business_metrics(y_train, y_pred_train)
                test_metrics = self.calculate_business_metrics(y_test, y_pred_test)
                
                # Cross-validation (simplified for speed)
                cv_scores = cross_val_score(
                    best_model, X_train_processed, y_train, 
                    cv=2, scoring='r2'  # Reduced from 3 to 2
                )
                
                # Store results
                result = {
                    'Model': name,
                    'Best_Params': str(best_params),
                    'Train_R2': train_metrics['R2'],
                    'Test_R2': test_metrics['R2'],
                    'Train_RMSE': train_metrics['RMSE'],
                    'Test_RMSE': test_metrics['RMSE'],
                    'Test_MAE': test_metrics['MAE'],
                    'Test_MAPE': test_metrics['MAPE'] if test_metrics['MAPE'] != np.inf else 999,
                    'Ranking_Correlation': test_metrics['Ranking_Correlation'],
                    'Top_10_Precision': test_metrics['Top_10_Precision'],
                    'CV_R2_Mean': cv_scores.mean(),
                    'CV_R2_Std': cv_scores.std(),
                    'Needs_Scaling': needs_scaling
                }
                
                results.append(result)
                
                # Store trained model
                self.models[name] = {
                    'model': best_model,
                    'scaler': standard_scaler if needs_scaling else None,
                    'params': best_params
                }
                
                logging.info(f"Completed {name} - R2: {test_metrics['R2']:.4f}")
                
            except Exception as e:
                logging.error(f"Error training {name}: {e}")
                continue
        
        # Convert to DataFrame and find best model
        results_df = pd.DataFrame(results)
        if not results_df.empty:
            best_idx = results_df['Test_R2'].idxmax()
            best_model_name = results_df.loc[best_idx, 'Model']
            self.best_model = self.models[best_model_name]
            self.is_fitted = True
            
            logging.info(f"Best model: {best_model_name} with R2: {results_df.loc[best_idx, 'Test_R2']:.4f}")
        
        return results_df
    
    def analyze_feature_importance(self, model_name=None):
        """Analyze feature importance for tree-based models"""
        
        if model_name is None and self.models:
            model_name = list(self.models.keys())[0]
        
        model_info = self.models.get(model_name)
        if not model_info:
            return None
        
        model = model_info['model']
        
        # Feature importance for tree-based models
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            return importance_df
        
        return None
    
    def predict_clv(self, X):
        """Make CLV predictions using the best model"""
        
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        model_info = self.best_model
        model = model_info['model']
        scaler = model_info['scaler']
        
        # Prepare data
        if scaler is not None:
            X_processed = scaler.transform(X)
        else:
            X_processed = X
        
        predictions = model.predict(X_processed)
        return predictions
    
    def segment_customers(self, predictions, percentiles=[20, 40, 60, 80]):
        """Segment customers based on predicted CLV"""
        
        segments = pd.cut(predictions, 
                         bins=np.percentile(predictions, [0] + percentiles + [100]),
                         labels=['Low Value', 'Low-Medium', 'Medium', 'Medium-High', 'High Value'],
                         include_lowest=True)
        
        return segments
    
    def generate_business_insights(self, X, y_true, y_pred, segments):
        """Generate comprehensive business insights"""
        
        insights = {}
        
        # Overall model performance
        metrics = self.calculate_business_metrics(y_true, y_pred)
        insights['model_performance'] = metrics
        
        # Segment analysis
        segment_stats = pd.DataFrame({
            'Segment': segments,
            'Predicted_CLV': y_pred,
            'Actual_CLV': y_true
        }).groupby('Segment').agg({
            'Predicted_CLV': ['count', 'mean', 'median', 'std'],
            'Actual_CLV': ['mean', 'median', 'std']
        }).round(2)
        
        insights['segment_analysis'] = segment_stats
        
        # Revenue impact
        total_predicted_clv = y_pred.sum()
        high_value_mask = segments == 'High Value'
        high_value_clv = y_pred[high_value_mask].sum() if high_value_mask.sum() > 0 else 0
        
        insights['revenue_impact'] = {
            'total_predicted_clv': total_predicted_clv,
            'high_value_contribution': high_value_clv,
            'high_value_percentage': (high_value_clv / total_predicted_clv * 100) if total_predicted_clv > 0 else 0
        }
        
        return insights
    
    def save_model(self, filepath):
        """Save the trained model and preprocessing components"""
        
        if not self.is_fitted:
            logging.warning("Model hasn't been fitted yet. Saving current state anyway.")
        
        model_package = {
            'models': self.models,
            'best_model': self.best_model,
            'feature_names': self.feature_names,
            'model_performance': self.model_performance,
            'config': self.config,
            'is_fitted': self.is_fitted,
            'scaler': self.scaler
        }
        
        try:
            # Save using joblib for better sklearn model serialization
            joblib.dump(model_package, filepath)
            logging.info(f"Model saved successfully to {filepath}")
        except Exception as e:
            logging.error(f"Error saving model: {e}")
            raise
    
    def load_model(self, filepath):
        """Load a pre-trained model and preprocessing components"""
        
        try:
            model_package = joblib.load(filepath)
            
            self.models = model_package['models']
            self.best_model = model_package['best_model']
            self.feature_names = model_package['feature_names']
            self.model_performance = model_package['model_performance']
            self.config = model_package['config']
            self.is_fitted = model_package['is_fitted']
            self.scaler = model_package.get('scaler', None)
            
            logging.info(f"Model loaded successfully from {filepath}")
            
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise
    
    def run_full_pipeline(self, input_data, save_model_path=None):
        """Run the complete CLV prediction pipeline
    
        Args:
            input_data: Either a file path (str) or pandas DataFrame containing customer data
            save_model_path: Optional path to save the trained model
        """
        try:
            # Handle both file paths and DataFrames
            if isinstance(input_data, str):
                # Input is a file path
                df = self.load_and_clean_data(input_data)
            elif isinstance(input_data, pd.DataFrame):
                # Input is already a DataFrame
                df = self._clean_data(input_data.copy())
            else:
                raise ValueError("input_data must be either a file path string or pandas DataFrame")
            
            # Create temporal split
            feature_data, target_data = self.create_temporal_split(df)
            
            # Prepare features and target
            clv_data = self.prepare_target_variable(feature_data, target_data)
            X, y = self.prepare_features_for_modeling(clv_data)
            
            # Split data for training
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.config.TEST_SIZE, 
                random_state=self.config.RANDOM_STATE
            )
            
            # Train models
            results = self.train_and_evaluate_models(X_train, X_test, y_train, y_test)
            
            # Make predictions
            y_pred = self.predict_clv(X_test)
            
            # Customer segmentation
            segments = self.segment_customers(y_pred)
            
            # Generate insights
            insights = self.generate_business_insights(X_test, y_test, y_pred, segments)
            
            # Save model if path provided
            if save_model_path:
                self.save_model(save_model_path)
            
            return {
                'model_results': results,
                'predictions': y_pred,
                'segments': segments,
                'insights': insights,
                'feature_importance': self.analyze_feature_importance()
            }
            
        except Exception as e:
            logging.error(f"Error in pipeline: {e}")
            raise

    def create_visualizations(self, predictions_df):
        """Create visualizations for CLV predictions"""
        
        # 1. CLV Distribution Histogram
        fig_dist = px.histogram(
            predictions_df, 
            x='predicted_clv',
            nbins=20,
            title='CLV Distribution',
            labels={'predicted_clv': 'Predicted CLV ($)', 'count': 'Number of Customers'}
        )
        fig_dist.update_layout(
            xaxis_title="Predicted CLV ($)",
            yaxis_title="Number of Customers",
            showlegend=False
        )
        
        # 2. Customer Segments Pie Chart
        segment_counts = predictions_df['customer_segment'].value_counts()
        fig_segments = px.pie(
            values=segment_counts.values,
            names=segment_counts.index,
            title='Customer Segments Distribution'
        )
        fig_segments.update_layout(showlegend=True)
        
        # 3. CLV vs Churn Risk Scatter Plot
        fig_scatter = px.scatter(
            predictions_df,
            x='churn_risk',
            y='predicted_clv',
            color='customer_segment',
            title='CLV vs Churn Risk',
            labels={'churn_risk': 'Churn Risk', 'predicted_clv': 'Predicted CLV ($)'}
        )
        fig_scatter.update_layout(
            xaxis_title="Churn Risk",
            yaxis_title="Predicted CLV ($)"
        )
        
        # 4. Top 10 Customers Table
        top_customers = predictions_df.nlargest(10, 'predicted_clv')[
            ['customer_id', 'predicted_clv', 'customer_segment', 'churn_risk', 'percentile_rank']
        ].copy()
        top_customers['predicted_clv'] = top_customers['predicted_clv'].round(2)
        top_customers['churn_risk'] = (top_customers['churn_risk'] * 100).round(1)
        top_customers['percentile_rank'] = top_customers['percentile_rank'].round(1)
        top_customers.columns = ['Customer ID', 'Predicted CLV ($)', 'Segment', 'Churn Risk (%)', 'Percentile Rank (%)']
        
        return fig_dist, fig_segments, fig_scatter, top_customers