import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import io
import base64
from typing import Dict, List, Tuple
import warnings
import os
warnings.filterwarnings('ignore')

# Import custom modules
from clv_model import EnhancedCLVPredictor
from utils import (
    save_feedback, save_feature_request, load_feedback, load_feature_requests,
    validate_customer_data, format_currency, format_percentage,
    generate_sample_data, export_data_to_csv, create_email_link
)
from feedback_manager import FeedbackManager
from documentation_handler import DocumentationHandler

# Set page configuration
st.set_page_config(
    page_title="CLV Model Interface",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load external CSS
def load_css():
    with open('styles.css', 'r') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

try:
    load_css()
except FileNotFoundError:
    st.warning("CSS file not found. Using default styling.")

# Initialize session state
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'customer_data' not in st.session_state:
    st.session_state.customer_data = None
if 'manual_customers' not in st.session_state:
    st.session_state.manual_customers = pd.DataFrame()
if 'show_all_customers' not in st.session_state:
    st.session_state.show_all_customers = False
if 'trigger_prediction' not in st.session_state:
    st.session_state.trigger_prediction = False
if 'show_feedback_dashboard' not in st.session_state:
    st.session_state.show_feedback_dashboard = False
if 'show_feature_requests' not in st.session_state:
    st.session_state.show_feature_requests = False
if 'show_feedback_form' not in st.session_state:
    st.session_state.show_feedback_form = False
if 'show_feature_request_form' not in st.session_state:
    st.session_state.show_feature_request_form = False


# Initialize components
clv_predictor = EnhancedCLVPredictor()
feedback_manager = FeedbackManager()
doc_handler = DocumentationHandler()

# Check if we have a saved model
MODEL_FILE = 'clv_model.joblib'

# Add a simple way to force retraining (for development/testing)
FORCE_RETRAIN = False  # Set to True if you want to retrain the model

# Only initialize model if not already fitted (avoid retraining on every load)
if not clv_predictor.is_fitted and not FORCE_RETRAIN:
    # Try to load existing model first
    if os.path.exists(MODEL_FILE):
        try:
            with st.spinner("Loading existing CLV model..."):
                clv_predictor.load_model(MODEL_FILE)
                st.success("✅ CLV Model loaded from saved file!")
        except Exception as e:
            st.warning(f"Could not load saved model: {str(e)}")
            FORCE_RETRAIN = True  # Fall back to training new model

# Train new model if needed
if not clv_predictor.is_fitted or FORCE_RETRAIN:
    with st.spinner("Training new CLV model (this may take a moment)..."):
        try:
            # Save sample data to temp file first
            sample_data = generate_sample_data()
            sample_data.to_csv('temp_sample_data.csv', index=False)
            
            # Run the pipeline
            clv_predictor.run_full_pipeline('temp_sample_data.csv')
            
            # Save the trained model
            clv_predictor.save_model(MODEL_FILE)
            
            # Clean up temp file only after successful initialization
            if os.path.exists('temp_sample_data.csv'):
                os.remove('temp_sample_data.csv')
                
            st.success("✅ CLV Model trained and saved successfully!")
            
        except Exception as e:
            st.error(f"❌ Error training model: {str(e)}")
            # Clean up temp file even if there's an error
            if os.path.exists('temp_sample_data.csv'):
                os.remove('temp_sample_data.csv')
else:
    st.success("✅ CLV Model ready!")

def handle_manual_entry():
    """Handle manual customer data entry"""
    st.sidebar.write("### Enter customer data:")
    
    with st.sidebar.form("manual_entry"):
        customer_id = st.text_input("Customer ID", f"CUST_{len(st.session_state.manual_customers) + 1:04d}")
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        total_purchases = st.number_input("Total Purchases", min_value=0, value=5)
        avg_order_value = st.number_input("Average Order Value ($)", min_value=0.0, value=50.0)
        days_since_first = st.number_input("Days Since First Purchase", min_value=1, value=180)
        days_since_last = st.number_input("Days Since Last Purchase", min_value=0, value=30)
        acquisition_channel = st.selectbox("Acquisition Channel", 
                                         ['Online', 'Store', 'Social Media', 'Referral'], 
                                         index=0)
        location = st.selectbox("Location", ['Urban', 'Suburban', 'Rural'], index=0)
        subscription_status = st.selectbox("Subscription Status", 
                                         ['Active', 'Inactive', 'None'], 
                                         index=0)
        
        col1, col2 = st.columns(2)
        with col1:
            add_customer = st.form_submit_button("➕ Add Customer", type="secondary")
        with col2:
            add_and_predict = st.form_submit_button("🔮 Add & Predict", type="primary")
        
        if add_customer or add_and_predict:
            # Validation logic (same as original)
            validation_errors = []
            
            if not customer_id.strip():
                validation_errors.append("Customer ID cannot be empty!")
            else:
                if not st.session_state.manual_customers.empty and 'customer_id' in st.session_state.manual_customers.columns:
                    existing_ids = st.session_state.manual_customers['customer_id'].tolist()
                    if customer_id.strip() in existing_ids:
                        validation_errors.append(f"Customer ID '{customer_id}' already exists!")
            
            if age < 18 or age > 100:
                validation_errors.append("Age must be between 18 and 100!")
            if total_purchases < 0:
                validation_errors.append("Total purchases cannot be negative!")
            if avg_order_value <= 0:
                validation_errors.append("Average order value must be greater than 0!")
            if days_since_first < 1:
                validation_errors.append("Days since first purchase must be at least 1!")
            if days_since_last < 0:
                validation_errors.append("Days since last purchase cannot be negative!")
            if days_since_last > days_since_first:
                validation_errors.append("Days since last purchase cannot be greater than days since first purchase!")
            
            if validation_errors:
                for error in validation_errors:
                    st.sidebar.error(error)
            else:
                new_customer = pd.DataFrame({
                    'customer_id': [customer_id.strip()], 
                    'age': [age], 
                    'total_purchases': [total_purchases],
                    'avg_order_value': [avg_order_value], 
                    'days_since_first_purchase': [days_since_first],
                    'days_since_last_purchase': [days_since_last], 
                    'acquisition_channel': [acquisition_channel],
                    'location': [location], 
                    'subscription_status': [subscription_status]
                })
                
                if st.session_state.manual_customers.empty:
                    st.session_state.manual_customers = new_customer
                else:
                    st.session_state.manual_customers = pd.concat([st.session_state.manual_customers, new_customer], 
                                                                ignore_index=True)
                
                st.session_state.customer_data = st.session_state.manual_customers.copy()
                st.sidebar.success(f"✅ Customer {customer_id} added! Total: {len(st.session_state.manual_customers)}")
                
                if add_and_predict:
                    st.session_state.trigger_prediction = True
    
    return not st.session_state.manual_customers.empty

def display_manual_customers():
    """Display and manage manually entered customers"""
    if not st.session_state.manual_customers.empty:
        st.sidebar.write("### Added Customers")
        
        for idx, row in st.session_state.manual_customers.iterrows():
            with st.sidebar.expander(f"👤 {row['customer_id']}", expanded=False):
                st.write(f"*Age:* {row['age']}")
                st.write(f"*Purchases:* {row['total_purchases']}")
                st.write(f"*Avg Order:* ${row['avg_order_value']:.2f}")
                st.write(f"*Channel:* {row['acquisition_channel']}")
                
                if st.button(f"🗑 Remove", key=f"remove_{idx}"):
                    st.session_state.manual_customers = st.session_state.manual_customers.drop(idx).reset_index(drop=True)
                    if not st.session_state.manual_customers.empty:
                        st.session_state.customer_data = st.session_state.manual_customers.copy()
                    else:
                        st.session_state.customer_data = None
                    st.rerun()
        
        st.sidebar.write("### Bulk Actions")
        col1, col2 = st.sidebar.columns(2)
        
        with col1:
            if st.button("📋 View All", type="secondary"):
                st.session_state.show_all_customers = True
        
        with col2:
            if st.button("🗑 Clear All", type="secondary"):
                st.session_state.manual_customers = pd.DataFrame()
                st.session_state.customer_data = None
                st.session_state.predictions = None
                st.sidebar.success("All customers cleared!")
                st.rerun()
        
        if st.sidebar.button("💾 Export Customer List", type="primary"):
            csv = st.session_state.manual_customers.to_csv(index=False)
            st.sidebar.download_button(
                label="📄 Download CSV",
                data=csv,
                file_name=f"manual_customers_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                key="export_manual"
            )

def display_feature_request_form():
    """Display feature request form"""
    st.subheader("💡 Request a Feature")
    
    with st.form("feature_request_form"):
        st.write("*Help us improve the CLV Model Interface by suggesting new features:*")
        
        col1, col2 = st.columns(2)
        with col1:
            feature_title = st.text_input("Feature Title", placeholder="Brief title for your feature request")
            priority = st.selectbox("Priority", ["Low", "Medium", "High"], index=1)
        
        with col2:
            category = st.selectbox("Category", 
                                  ["UI/UX Improvement", "New Analysis Feature", "Data Export", 
                                   "Performance", "Integration", "Other"])
            user_email = st.text_input("Your Email (Optional)", placeholder="your.email@example.com")
        
        feature_description = st.text_area("Feature Description", 
                                         height=150,
                                         placeholder="Please describe the feature you'd like to see. Include:\n- What problem it would solve\n- How you envision it working\n- Any specific requirements")
        
        use_case = st.text_area("Use Case", 
                               height=100,
                               placeholder="Describe a specific scenario where this feature would be helpful")
        
        additional_notes = st.text_area("Additional Notes (Optional)", 
                                      height=80,
                                      placeholder="Any additional information, mockups, or references")
        
        submitted = st.form_submit_button("🚀 Submit Feature Request", type="primary")
        
        if submitted:
            if not feature_title.strip():
                st.error("Please provide a feature title!")
            elif not feature_description.strip():
                st.error("Please provide a feature description!")
            else:
                request_data = {
                    'feature_title': feature_title.strip(),
                    'feature_description': feature_description.strip(),
                    'use_case': use_case.strip() if use_case.strip() else "Not provided",
                    'additional_notes': additional_notes.strip() if additional_notes.strip() else "None",
                    'priority': priority,
                    'category': category,
                    'user_email': user_email.strip() if user_email.strip() else "Anonymous",
                    'status': 'Pending'
                }
                
                if save_feature_request(request_data):
                    st.success("🎉 Thank you! Your feature request has been submitted successfully.")
                    st.info("💡 Our development team will review your request and prioritize it accordingly.")
                else:
                    st.error("❌ Sorry, there was an error submitting your request. Please try again.")

def display_feedback_dashboard():
    """Display feedback dashboard for administrators"""
    st.write("DEBUG: display_feedback_dashboard function called")
    
    # Use the feedback manager's admin dashboard
    try:
        feedback_manager.display_admin_feedback_dashboard()
        st.write("DEBUG: feedback_manager.display_admin_feedback_dashboard() completed successfully")
    except Exception as e:
        st.error(f"DEBUG: Error in feedback dashboard: {str(e)}")
        st.write("DEBUG: Showing fallback feedback display")
        
        # Fallback: simple feedback display
        st.subheader("📊 Feedback Dashboard (Fallback)")
        
        # Load feedback directly
        feedbacks = load_feedback()
        
        if not feedbacks:
            st.info("No feedback received yet.")
            return
        
        st.write(f"DEBUG: Found {len(feedbacks)} feedback entries")
        
        # Show simple list
        for i, feedback in enumerate(feedbacks):
            st.write(f"**Feedback {i+1}:** {feedback.get('feedback_text', 'No text')}")
            st.write(f"**Type:** {feedback.get('feedback_type', 'Unknown')}")
            st.write(f"**Rating:** {feedback.get('rating', 'N/A')}/5")
            st.write(f"**Date:** {feedback.get('timestamp', 'Unknown')}")
            st.write("---")

def display_feature_requests_admin():
    """Display feature requests admin dashboard"""
    st.subheader("💡 Feature Requests Dashboard")
    
    # Load feature requests
    feature_requests = load_feature_requests()
    
    if not feature_requests:
        st.info("No feature requests received yet.")
        return
    
    # Show summary metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Requests", len(feature_requests))
    
    with col2:
        pending_requests = len([r for r in feature_requests if r.get('status') == 'Pending'])
        st.metric("Pending", pending_requests)
    
    with col3:
        high_priority = len([r for r in feature_requests if r.get('priority') == 'High'])
        st.metric("High Priority", high_priority)
    
    # Display feature requests in a simple table
    st.subheader("📋 All Feature Requests")
    
    # Convert to DataFrame for display
    df = pd.DataFrame(feature_requests)
    
    # Format the data for display
    display_df = df.copy()
    display_df['timestamp'] = pd.to_datetime(display_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
    
    # Reorder columns for better display
    display_df = display_df[['timestamp', 'feature_title', 'priority', 'category', 'status', 'user_email']]
    display_df.columns = ['Date', 'Title', 'Priority', 'Category', 'Status', 'Email']
    
    # Display the table
    st.dataframe(display_df, use_container_width=True)
    
    # Show individual feature request details in expanders
    st.subheader("📝 Detailed Feature Requests")
    for i, request in enumerate(reversed(feature_requests)):
        priority_color = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}.get(request.get('priority', 'Medium'), "⚪")
        status_color = {"Pending": "🟡", "Approved": "🟢", "Rejected": "🔴", "In Progress": "🔵", "Completed": "🟣"}.get(request.get('status', 'Pending'), "⚪")
        
        with st.expander(f"{priority_color} {status_color} Request #{request['id']} - {request.get('feature_title', 'Untitled')}"):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write(f"**Description:** {request.get('feature_description', 'No description')}")
                if request.get('use_case') and request.get('use_case') != "Not provided":
                    st.write(f"**Use Case:** {request.get('use_case')}")
                if request.get('additional_notes') and request.get('additional_notes') != "None":
                    st.write(f"**Additional Notes:** {request.get('additional_notes')}")
            
            with col2:
                st.write(f"**Priority:** {request.get('priority', 'Medium')}")
                st.write(f"**Category:** {request.get('category', 'Other')}")
                st.write(f"**Status:** {request.get('status', 'Pending')}")
                st.write(f"**Date:** {request.get('timestamp', 'Unknown')[:19]}")
                if request.get('user_email') and request.get('user_email') != "Anonymous":
                    st.write(f"**Email:** {request.get('user_email')}")

def main():
    # Header with text and instructions on the left, image on the right
    col_left, col_right = st.columns([2, 1])
    with col_left:
        st.markdown("""
        <h1 class="main-header">
            <span style="background: linear-gradient(90deg, #0a1f44, #4a6fa5);
                       -webkit-background-clip: text;
                       -webkit-text-fill-color: transparent;">
                Customer Lifetime Value (CLV) Predictor
            </span>
        </h1>
        """, unsafe_allow_html=True)
        st.markdown("""
        ### 🚀 Welcome to the CLV Predictor!
        This tool helps you predict Customer Lifetime Value (CLV) for your customers using advanced machine learning models.

        **Getting Started:**
        1. **Choose a data source** from the sidebar:
           - Upload your own CSV file with customer data
           - Generate sample data to explore the tool
           - Manually enter customer information
        2. **Configure prediction parameters** in the sidebar
        3. **Generate predictions** and explore the results

        **Required Data Fields:**
        - `customer_id`: Unique identifier for each customer
        - `age`: Customer age
        - `total_purchases`: Total number of purchases made
        - `avg_order_value`: Average order value in dollars
        - `days_since_first_purchase`: Days since first purchase
        - `days_since_last_purchase`: Days since last purchase
        - `acquisition_channel`: How the customer was acquired
        - `location`: Customer location (Urban, Suburban, Rural)
        - `subscription_status`: Current subscription status
        """)
    with col_right:
        st.image("img1.jpg", caption="CLV Model Overview", use_container_width=True)
    
    # Check for admin access
    admin_access = st.sidebar.checkbox("🔐 Admin Access", help="For developers only")
    
    if admin_access:
        admin_password = st.sidebar.text_input("Admin Password", type="password")
        if admin_password == "clv_admin_2024":  # Change this password
            st.sidebar.success("✅ Admin access granted")
            
            # Admin navigation
            admin_action = st.sidebar.selectbox("Admin Actions", 
                                              ["Main Interface", "View Feedback Dashboard", "View Feature Requests"])
            
            if admin_action == "View Feedback Dashboard":
                st.session_state.show_feedback_dashboard = True
            elif admin_action == "View Feature Requests":
                st.session_state.show_feature_requests = True
            else:
                st.session_state.show_feedback_dashboard = False
                st.session_state.show_feature_requests = False
        else:
            if admin_password:
                st.sidebar.error("❌ Invalid admin password")
    
    # Show admin dashboards if requested
    if st.session_state.show_feedback_dashboard:
        display_feedback_dashboard()
        return
    
    if st.session_state.show_feature_requests:
        display_feature_requests_admin()
        return
    
    # Show feedback form if requested
    if st.session_state.show_feedback_form:
        feedback_manager.display_feedback_form()
        if st.button("← Back to Main Interface", type="secondary"):
            st.session_state.show_feedback_form = False
            st.rerun()
        return
    
    # Sidebar configuration
    st.sidebar.markdown('<div class="sidebar-header">📊 Model Configuration</div>', unsafe_allow_html=True)
    st.sidebar.markdown('<div class="sidebar-header">Data Input</div>', unsafe_allow_html=True)
    
    data_source = st.sidebar.radio(
        "Choose data source:",
        ["Upload CSV File", "Use Sample Data", "Manual Entry"],
        label_visibility="collapsed"
    )
    
    customer_data = None
    
    # Data source handling (same as original logic)
    if data_source == "Upload CSV File":
        uploaded_file = st.sidebar.file_uploader(
            "Choose a CSV file",
            type="csv",
            help="Upload your customer data CSV file"
        )
        
        if uploaded_file is not None:
            try:
                customer_data = pd.read_csv(uploaded_file)
                st.sidebar.success(f"✅ File uploaded successfully! ({len(customer_data)} customers)")
            except Exception as e:
                st.sidebar.error(f"❌ Error reading file: {str(e)}")
    
    elif data_source == "Use Sample Data":
        if st.sidebar.button("Generate Sample Data", type="primary"):
            customer_data = generate_sample_data()
            st.session_state.customer_data = customer_data
            st.sidebar.success(f"✅ Sample data generated! ({len(customer_data)} customers)")
        
        if st.session_state.customer_data is not None:
            customer_data = st.session_state.customer_data

    elif data_source == "Manual Entry":
        has_manual_data = handle_manual_entry()
        display_manual_customers()
        
        if has_manual_data:
            customer_data = st.session_state.manual_customers.copy()
        
        if st.session_state.show_all_customers:
            st.subheader("👥 All Manually Added Customers")
            if not st.session_state.manual_customers.empty:
                st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
                st.dataframe(st.session_state.manual_customers, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.info("No customers added yet.")
            if st.button("❌ Close View", type="primary"):
                st.session_state.show_all_customers = False
                st.rerun()
    
    # Model Parameters
    st.sidebar.markdown('<div class="sidebar-header">🎛 Prediction Parameters</div>', unsafe_allow_html=True)
    
    time_horizon = st.sidebar.selectbox(
        "Time Horizon",
        [6, 12, 24, 36],
        index=1,
        help="Prediction period in months"
    )
    
    discount_rate = st.sidebar.slider(
        "Discount Rate",
        min_value=0.0,
        max_value=0.2,
        value=0.05,
        step=0.01,
        help="Annual discount rate for future cash flows"
    )
    
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.5,
        max_value=0.95,
        value=0.8,
        step=0.05,
        help="Minimum confidence level for predictions"
    )
    
    # Main content area
    st.markdown('<div class="main-content">', unsafe_allow_html=True)
    
    if customer_data is not None:
        # Display data preview and metrics (same as original)
        st.subheader("📋 Data Preview")
        st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
        st.dataframe(customer_data.head(), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Data quality indicators
        st.markdown('<div class="metric-grid">', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>Total Customers</h3>
                <p style="font-size: 2rem; font-weight: bold; color: var(--primary);">{len(customer_data)}</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            missing_pct = (customer_data.isnull().sum().sum() / (len(customer_data) * len(customer_data.columns))) * 100
            st.markdown(f"""
            <div class="metric-card">
                <h3>Data Completeness</h3>
                <p style="font-size: 2rem; font-weight: bold; color: var(--primary);">{100-missing_pct:.1f}%</p>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            avg_purchases = customer_data['total_purchases'].mean() if 'total_purchases' in customer_data.columns else 0
            st.markdown(f"""
            <div class="metric-card">
                <h3>Avg Purchases</h3>
                <p style="font-size: 2rem; font-weight: bold; color: var(--primary);">{avg_purchases:.1f}</p>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            avg_order_value = customer_data['avg_order_value'].mean() if 'avg_order_value' in customer_data.columns else 0
            st.markdown(f"""
            <div class="metric-card">
                <h3>Avg Order Value</h3>
                <p style="font-size: 2rem; font-weight: bold; color: var(--primary);">${avg_order_value:.2f}</p>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Prediction logic (same as original)
        predict_button_clicked = st.button("🔮 Generate CLV Predictions", type="primary", use_container_width=True)
        
        if predict_button_clicked or st.session_state.trigger_prediction:
            with st.spinner("Calculating CLV predictions..."):
                try:
                    # First ensure the data is in correct format for prediction
                    if 'customer_id' in customer_data.columns:
                        # Create features from the input data
                        processed_data = clv_predictor.create_advanced_features(customer_data)
                        # Prepare features for prediction (not modeling)
                        X = clv_predictor.prepare_features_for_prediction(processed_data)
                        # Make predictions
                        raw_predictions = clv_predictor.predict_clv(X)
                        
                        # Create a proper predictions DataFrame with all required columns
                        predictions_df = customer_data.copy()
                        predictions_df['predicted_clv'] = raw_predictions
                        
                        # Add customer segments
                        segments = clv_predictor.segment_customers(raw_predictions)
                        predictions_df['customer_segment'] = segments
                        
                        # Add percentile rank
                        predictions_df['percentile_rank'] = predictions_df['predicted_clv'].rank(pct=True) * 100
                        
                        # Add confidence intervals (simple estimation)
                        predictions_df['confidence_lower'] = predictions_df['predicted_clv'] * 0.8
                        predictions_df['confidence_upper'] = predictions_df['predicted_clv'] * 1.2
                        
                        # Add churn risk (simple estimation based on days since last purchase)
                        if 'days_since_last_purchase' in predictions_df.columns:
                            max_days = predictions_df['days_since_last_purchase'].max()
                            predictions_df['churn_risk'] = predictions_df['days_since_last_purchase'] / max_days
                        else:
                            predictions_df['churn_risk'] = 0.5  # Default value
                        
                        st.session_state.predictions = predictions_df
                        st.session_state.trigger_prediction = False
                        st.success("✅ Predictions generated successfully!")
                    else:
                        st.error("❌ Invalid customer data format. Please check your input data.")
                except Exception as e:
                    st.error(f"❌ Prediction failed: {str(e)}")
        
        # Display predictions (same visualization logic as original)
        if st.session_state.predictions is not None:
            predictions_df = st.session_state.predictions
            
            st.markdown('<div class="prediction-results">', unsafe_allow_html=True)
            st.subheader("📈 Prediction Results")
            
            # Summary metrics with responsive grid
            st.markdown('<div class="metric-grid">', unsafe_allow_html=True)
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_clv = predictions_df['predicted_clv'].mean()
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Average CLV</h3>
                    <p style="font-size: 2rem; font-weight: bold; color: var(--primary);">${avg_clv:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                total_clv = predictions_df['predicted_clv'].sum()
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Total CLV</h3>
                    <p style="font-size: 2rem; font-weight: bold; color: var(--primary);">${total_clv:,.2f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                high_value_customers = len(predictions_df[predictions_df['customer_segment'] == 'High Value'])
                st.markdown(f"""
                <div class="metric-card">
                    <h3>High Value Customers</h3>
                    <p style="font-size: 2rem; font-weight: bold; color: var(--primary);">{high_value_customers}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                avg_churn_risk = predictions_df['churn_risk'].mean()
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Avg Churn Risk</h3>
                    <p style="font-size: 2rem; font-weight: bold; color: var(--primary);">{avg_churn_risk:.2%}</p>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Visualizations
            st.subheader("📊 Analytics Dashboard")
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            
            fig_dist, fig_segments, fig_scatter, top_customers = clv_predictor.create_visualizations(predictions_df)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.plotly_chart(fig_dist, use_container_width=True, config={'displayModeBar': False})
                st.plotly_chart(fig_scatter, use_container_width=True, config={'displayModeBar': False})
            
            with col2:
                st.plotly_chart(fig_segments, use_container_width=True, config={'displayModeBar': False})
                
                st.subheader("🏆 Top 10 Customers by CLV")
                st.dataframe(top_customers, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Detailed results table with filters
            st.subheader("📋 Detailed Predictions")
            st.markdown('<div class="filter-controls">', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                segment_filter = st.multiselect(
                    "Filter by Segment",
                    predictions_df['customer_segment'].unique(),
                    default=predictions_df['customer_segment'].unique()
                )
            
            with col2:
                min_clv = st.number_input("Minimum CLV", value=0.0)
            
            with col3:
                max_churn_risk = st.slider("Max Churn Risk", 0.0, 1.0, 1.0)
            st.markdown('</div>', unsafe_allow_html=True)
            
            filtered_df = predictions_df[
                (predictions_df['customer_segment'].isin(segment_filter)) &
                (predictions_df['predicted_clv'] >= min_clv) &
                (predictions_df['churn_risk'] <= max_churn_risk)
            ]
            
            st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
            st.dataframe(filtered_df.style.format({
                'predicted_clv': '${:,.2f}',
                'confidence_lower': '${:,.2f}',
                'confidence_upper': '${:,.2f}',
                'avg_order_value': '${:,.2f}',
                'churn_risk': '{:.2%}',
                'percentile_rank': '{:.1f}%'
            }), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Export functionality
            st.subheader("📥 Export Results")
            st.markdown('<div class="export-section">', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    label="📄 Download as CSV",
                    data=csv,
                    file_name=f"clv_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    type="primary"
                )
            
            with col2:
                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                    filtered_df.to_excel(writer, sheet_name='CLV_Predictions', index=False)
                excel_data = excel_buffer.getvalue()
                
                st.download_button(
                    label="📊 Download as Excel",
                    data=excel_data,
                    file_name=f"clv_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    type="secondary"
                )
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Email support section
            st.markdown("---")
            st.markdown("### 📧 Need Help?")
            support_email = "kratikasoni73@gmail.com"
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            padding: 20px; border-radius: 10px; text-align: center;">
                    <h4 style="color: white; margin-bottom: 15px;">Contact Support</h4>
                    <p style="color: white; margin-bottom: 15px;">Have questions or need assistance?</p>
                    <a href="mailto:{support_email}?subject=CLV Model Support Request&body=Hello,%0D%0A%0D%0AI need help with the CLV Model Interface.%0D%0A%0D%0APlease describe your issue:%0D%0A" 
                       style="display: inline-block; background: white; color: #667eea; padding: 12px 24px; 
                              border-radius: 25px; text-decoration: none; font-weight: bold; 
                              transition: all 0.3s ease;">
                        📧 Email Support: {support_email}
                    </a>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                if st.button("💡 Request Feature", type="primary", use_container_width=True):
                    st.session_state.show_feature_request_form = True
                    st.session_state.show_feature_requests = False  # Ensure admin dashboard doesn't open
                    st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    else:
        st.info("👆 Please select a data source and load customer data to begin CLV analysis.")
    
    # Close main content container
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer with documentation and feedback
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📚 View Documentation", type="secondary", use_container_width=True):
            doc_handler.display_documentation()
    
    with col2:
        if st.button("💬 Provide Feedback", type="secondary", use_container_width=True):
            st.session_state.show_feedback_form = True
    
    with col3:
        st.markdown("""
        <div style="text-align: center; padding: 10px;">
            <p style="margin: 0; color: #666;">Version 1.0</p>
            <p style="margin: 0; color: #666;">© 2024 CLV Model Interface</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Add logic to display the user feature request form
    if st.session_state.show_feature_request_form:
        display_feature_request_form()
        if st.button("← Back to Main Interface", type="secondary"):
            st.session_state.show_feature_request_form = False
            st.rerun()
        return
    
if __name__ == "__main__":
    main()