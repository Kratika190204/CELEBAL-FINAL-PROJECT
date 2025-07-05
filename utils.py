import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
import uuid
import streamlit as st

# File paths for data storage
FEEDBACK_FILE = 'data/feedback.json'
FEATURE_REQUESTS_FILE = 'data/feature_requests.json'
DATA_DIR = 'data'

def ensure_data_directory():
    """Ensure data directory exists"""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

def save_feedback(feedback_data: Dict[str, Any]) -> bool:
    """Save feedback to JSON file"""
    try:
        ensure_data_directory()
        
        # Add metadata
        feedback_data['id'] = str(uuid.uuid4())
        feedback_data['timestamp'] = datetime.now().isoformat()
        
        # Load existing feedback
        existing_feedback = load_feedback()
        existing_feedback.append(feedback_data)
        
        # Save updated feedback
        with open(FEEDBACK_FILE, 'w', encoding='utf-8') as f:
            json.dump(existing_feedback, f, indent=2, ensure_ascii=False)
        
        return True
    except Exception as e:
        st.error(f"Error saving feedback: {str(e)}")
        return False

def load_feedback() -> List[Dict[str, Any]]:
    """Load feedback from JSON file"""
    try:
        if not os.path.exists(FEEDBACK_FILE):
            return []
        
        with open(FEEDBACK_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading feedback: {str(e)}")
        return []

def save_feature_request(request_data: Dict[str, Any]) -> bool:
    """Save feature request to JSON file"""
    try:
        ensure_data_directory()
        
        # Add metadata
        request_data['id'] = str(uuid.uuid4())
        request_data['timestamp'] = datetime.now().isoformat()
        
        # Load existing requests
        existing_requests = load_feature_requests()
        existing_requests.append(request_data)
        
        # Save updated requests
        with open(FEATURE_REQUESTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(existing_requests, f, indent=2, ensure_ascii=False)
        
        return True
    except Exception as e:
        st.error(f"Error saving feature request: {str(e)}")
        return False

def load_feature_requests() -> List[Dict[str, Any]]:
    """Load feature requests from JSON file"""
    try:
        if not os.path.exists(FEATURE_REQUESTS_FILE):
            return []
        
        with open(FEATURE_REQUESTS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading feature requests: {str(e)}")
        return []

def validate_customer_data(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Validate customer data and return errors/warnings"""
    errors = []
    warnings = []
    
    required_columns = [
        'customer_id', 'age', 'total_purchases', 'avg_order_value',
        'days_since_first_purchase', 'days_since_last_purchase',
        'acquisition_channel', 'location', 'subscription_status'
    ]
    
    # Check for required columns
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        errors.append(f"Missing required columns: {', '.join(missing_columns)}")
    
    if not errors:  # Only validate data if columns exist
        # Check for duplicate customer IDs
        if df['customer_id'].duplicated().any():
            errors.append("Duplicate customer IDs found")
        
        # Check for negative values
        numeric_columns = ['age', 'total_purchases', 'avg_order_value', 
                          'days_since_first_purchase', 'days_since_last_purchase']
        
        for col in numeric_columns:
            if col in df.columns:
                if (df[col] < 0).any():
                    errors.append(f"Negative values found in {col}")
        
        # Check for unrealistic values
        if 'age' in df.columns:
            if (df['age'] < 13).any() or (df['age'] > 120).any():
                warnings.append("Some age values seem unrealistic (< 13 or > 120)")
        
        if 'days_since_last_purchase' in df.columns and 'days_since_first_purchase' in df.columns:
            if (df['days_since_last_purchase'] > df['days_since_first_purchase']).any():
                errors.append("Days since last purchase cannot be greater than days since first purchase")
    
    return {'errors': errors, 'warnings': warnings}

def format_currency(value: float) -> str:
    """Format value as currency"""
    return f"${value:,.2f}"

def format_percentage(value: float) -> str:
    """Format value as percentage"""
    return f"{value:.2%}"

def generate_sample_data(n_customers: int = 100, data_type: str = 'customer') -> pd.DataFrame:
    """
    Generate bulletproof sample data with:
    - 540-day (18-month) minimum analysis period
    - Consistent lowercase column names
    - Robust temporal relationships
    """
    np.random.seed(42)
    analysis_period = 540  # 18 months in days (FIXED: Maintains minimum requirement)
    base_date = pd.to_datetime('2022-01-01')
    current_date = pd.to_datetime('now')
    
    # Ensure generated data covers required period
    min_last_purchase = current_date - pd.to_timedelta(analysis_period, 'd')
    
    # Common properties (all lowercase)
    customer_ids = [f"cust_{i:04d}" for i in range(1, n_customers + 1)]
    channels = ['online', 'store', 'social', 'referral']
    locations = ['urban', 'suburban', 'rural']
    subs = ['active', 'inactive', 'none']

    if data_type == 'customer':
        # Customer-level data with guaranteed 540-day coverage
        df = pd.DataFrame({
            'customer_id': customer_ids,
            'age': np.clip(np.random.normal(40, 15, n_customers), 18, 80).astype(int),
            'total_purchases': np.clip(np.random.poisson(15, n_customers), 5, 100),
            'avg_order_value': np.clip(np.random.lognormal(3.7, 0.7, n_customers), 15, 1000).round(2),
            'days_since_first_purchase': np.clip(
                np.random.exponential(400, n_customers) + 180,  # Minimum 180 + exp
                180,  # 6 month absolute minimum
                1095  # 3 year maximum
            ).astype(int),
            'days_since_last_purchase': np.clip(
                np.random.exponential(90, n_customers), 
                7,  # Minimum 1 week
                540  # Matches analysis_period
            ).astype(int),
            'acquisition_channel': np.random.choice(channels, n_customers, p=[0.4, 0.3, 0.2, 0.1]),
            'location': np.random.choice(locations, n_customers, p=[0.5, 0.35, 0.15]),
            'subscription_status': np.random.choice(subs, n_customers, p=[0.3, 0.2, 0.5]),
            'invoice_date': [  # Ensures coverage of analysis_period
                current_date - pd.to_timedelta(days, 'd') 
                for days in np.clip(np.random.exponential(90, n_customers), 7, 540)
            ]
        })
        
        # Final validation
        assert (df['days_since_first_purchase'] >= 180).all()
        assert (df['days_since_last_purchase'] <= analysis_period).all()
        
    elif data_type == 'transaction':
        # Transaction data with guaranteed temporal coverage
        transactions = []
        for cust_id in customer_ids:
            # Customer lifetime (540-1095 days)
            lifespan = np.random.randint(analysis_period, 1095)
            first_purchase = base_date + pd.to_timedelta(np.random.randint(0, 180), 'd')
            
            # Generate transactions spanning the analysis period
            n_trans = max(5, int(np.random.poisson(lifespan/60)))
            dates = sorted([
                first_purchase + pd.to_timedelta(int(lifespan*i/n_trans), 'd')
                for i in range(1, n_trans+1)
                if first_purchase + pd.to_timedelta(int(lifespan*i/n_trans), 'd') <= current_date
            ])
            
            # Realistic purchase amounts
            base_spend = np.random.lognormal(3.7, 0.7)
            for i, date in enumerate(dates):
                transactions.append({
                    'customer_id': cust_id,
                    'invoice_date': date,
                    'total_amount': round(base_spend * np.random.uniform(0.7, 1.5), 2),
                    'quantity': max(1, int(np.random.poisson(2))),
                    'product_id': f"prod_{np.random.randint(1000,9999)}",
                    'country': np.random.choice(['us', 'uk', 'de', 'fr']),
                    'acquisition_channel': np.random.choice(channels),
                    'location': np.random.choice(locations),
                    'subscription_status': np.random.choice(subs)
                })
        
        df = pd.DataFrame(transactions)
        
        # Final validation
        date_range = (df['invoice_date'].max() - df['invoice_date'].min()).days
        assert date_range >= analysis_period, f"Needs {analysis_period} days, got {date_range}"
    
    else:
        raise ValueError("data_type must be 'customer' or 'transaction'")

    return df

def export_data_to_csv(df: pd.DataFrame, filename: str) -> str:
    """Export DataFrame to CSV and return the data as string"""
    return df.to_csv(index=False)

def create_email_link(email: str, subject: str = "", body: str = "") -> str:
    """Create a mailto link for email"""
    import urllib.parse
    
    subject_encoded = urllib.parse.quote(subject)
    body_encoded = urllib.parse.quote(body)
    
    return f"mailto:{email}?subject={subject_encoded}&body={body_encoded}"

def get_system_info() -> Dict[str, Any]:
    """Get system information for support purposes"""
    return {
        'timestamp': datetime.now().isoformat(),
        'python_version': f"{pd.__version__}",
        'pandas_version': pd.__version__,
        'numpy_version': np.__version__
    }

def clean_text_input(text: str) -> str:
    """Clean and validate text input"""
    if not text:
        return ""
    
    # Remove excessive whitespace
    cleaned = ' '.join(text.split())
    
    # Remove potentially harmful characters
    cleaned = cleaned.replace('<', '&lt;').replace('>', '&gt;')
    
    return cleaned.strip()

def calculate_data_quality_score(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate data quality metrics"""
    total_cells = len(df) * len(df.columns)
    missing_cells = df.isnull().sum().sum()
    
    completeness = (total_cells - missing_cells) / total_cells * 100
    
    # Check for duplicates
    duplicate_rate = df.duplicated().sum() / len(df) * 100
    
    # Check for outliers in numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outlier_rate = 0
    
    if len(numeric_cols) > 0:
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
            outlier_rate += outliers / len(df)
        
        outlier_rate = (outlier_rate / len(numeric_cols)) * 100
    
    return {
        'completeness': completeness,
        'duplicate_rate': duplicate_rate,
        'outlier_rate': outlier_rate,
        'overall_score': (completeness * 0.6) - (duplicate_rate * 0.2) - (outlier_rate * 0.2)
    }

def log_user_action(action: str, details: Dict[str, Any] = None):
    """Log user actions for analytics"""
    try:
        ensure_data_directory()
        log_file = os.path.join(DATA_DIR, 'user_actions.json')
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'details': details or {},
            'session_id': st.session_state.get('session_id', 'unknown')
        }
        
        # Load existing logs
        logs = []
        if os.path.exists(log_file):
            with open(log_file, 'r', encoding='utf-8') as f:
                logs = json.load(f)
        
        logs.append(log_entry)
        
        # Keep only last 1000 entries to prevent file from growing too large
        if len(logs) > 1000:
            logs = logs[-1000:]
        
        # Save logs
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(logs, f, indent=2, ensure_ascii=False)
            
    except Exception as e:
        # Don't raise error for logging failures
        pass

def get_user_analytics() -> Dict[str, Any]:
    """Get user analytics for admin dashboard"""
    try:
        log_file = os.path.join(DATA_DIR, 'user_actions.json')
        
        if not os.path.exists(log_file):
            return {'total_actions': 0, 'recent_actions': 0, 'popular_actions': []}
        
        with open(log_file, 'r', encoding='utf-8') as f:
            logs = json.load(f)
        
        # Calculate metrics
        total_actions = len(logs)
        
        # Recent actions (last 24 hours)
        recent_cutoff = datetime.now() - pd.Timedelta(hours=24)
        recent_actions = len([
            log for log in logs 
            if datetime.fromisoformat(log['timestamp']) > recent_cutoff
        ])
        
        # Popular actions
        action_counts = {}
        for log in logs:
            action = log['action']
            action_counts[action] = action_counts.get(action, 0) + 1
        
        popular_actions = sorted(action_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'total_actions': total_actions,
            'recent_actions': recent_actions,
            'popular_actions': popular_actions
        }
        
    except Exception as e:
        return {'total_actions': 0, 'recent_actions': 0, 'popular_actions': []}