import streamlit as st
import json
import os
from datetime import datetime
import pandas as pd

class FeedbackManager:
    def __init__(self):
        self.feedback_file = "data/feedback.json"
        self._ensure_data_directory()
    
    def _ensure_data_directory(self):
        """Ensure data directory exists"""
        os.makedirs("data", exist_ok=True)
    
    def display_feedback_form(self):
        """Display feedback form for users"""
        st.subheader("💬 Provide Feedback")
        
        st.markdown("""
        Your feedback helps us improve the CLV Model Interface. Please share your thoughts, 
        suggestions, or report any issues you've encountered.
        """)
        
        with st.form("feedback_form"):
            feedback_type = st.selectbox(
                "Feedback Type",
                ["General Feedback", "Bug Report", "Feature Suggestion", "Performance Issue", "UI/UX Improvement"]
            )
            
            rating = st.slider(
                "Overall Rating",
                min_value=1,
                max_value=5,
                value=4
            )
            
            user_email = st.text_input(
                "Your Email (Optional)",
                placeholder="your.email@example.com"
            )
            
            feedback_text = st.text_area(
                "Your Feedback",
                height=150,
                placeholder="Please share your detailed feedback here..."
            )
            
            submitted = st.form_submit_button("📤 Submit Feedback", type="primary")
            
            if submitted:
                if not feedback_text.strip():
                    st.error("Please provide your feedback before submitting!")
                else:
                    # Create feedback data
                    feedback_data = {
                        'id': f"FB_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        'timestamp': datetime.now().isoformat(),
                        'feedback_type': feedback_type,
                        'rating': rating,
                        'feedback_text': feedback_text.strip(),
                        'user_email': user_email.strip() if user_email.strip() else "Anonymous"
                    }
                    
                    # Save feedback
                    if self._save_feedback(feedback_data):
                        st.success("🎉 Thank you for your feedback! We appreciate your input.")
                        st.balloons()
                    else:
                        st.error("❌ Sorry, there was an error submitting your feedback. Please try again.")
    
    def _save_feedback(self, feedback_data):
        """Save feedback to JSON file"""
        try:
            # Load existing feedback or create empty list
            feedback_list = []
            if os.path.exists(self.feedback_file):
                try:
                    with open(self.feedback_file, 'r', encoding='utf-8') as f:
                        feedback_list = json.load(f)
                except (json.JSONDecodeError, FileNotFoundError):
                    feedback_list = []
            # Add new feedback
            feedback_list.append(feedback_data)
            # Save back to file
            with open(self.feedback_file, 'w', encoding='utf-8') as f:
                json.dump(feedback_list, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            st.error(f"Error saving feedback: {str(e)}")
            return False
    
    def load_feedback(self):
        """Load all feedback from JSON file"""
        try:
            if not os.path.exists(self.feedback_file):
                return []
            with open(self.feedback_file, 'r', encoding='utf-8') as f:
                feedback_list = json.load(f)
            return feedback_list
        except Exception as e:
            st.error(f"Error loading feedback: {str(e)}")
            return []
    
    def display_admin_feedback_dashboard(self):
        """Display feedback dashboard for administrators"""
        st.subheader("📊 Feedback Dashboard")
        
        # Load feedback
        feedback_list = self.load_feedback()
        
        if not feedback_list:
            st.info("No feedback received yet.")
            return
        
        # Show summary metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Feedback", len(feedback_list))
        
        with col2:
            avg_rating = sum([f.get('rating', 0) for f in feedback_list]) / len(feedback_list)
            st.metric("Average Rating", f"{avg_rating:.1f}/5")
        
        with col3:
            recent_feedback = len([f for f in feedback_list if 
                                 datetime.fromisoformat(f.get('timestamp', '2000-01-01T00:00:00')) > 
                                 datetime.now() - pd.Timedelta(days=7)])
            st.metric("This Week", recent_feedback)
        
        # Display feedback in a simple table
        st.subheader("📋 All Submitted Feedback")
        
        # Convert to DataFrame for display
        df = pd.DataFrame(feedback_list)
        
        # Format the data for display
        display_df = df.copy()
        display_df['timestamp'] = pd.to_datetime(display_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
        display_df['rating'] = display_df['rating'].astype(str) + '/5'
        
        # Reorder columns for better display
        display_df = display_df[['timestamp', 'feedback_type', 'rating', 'user_email', 'feedback_text']]
        display_df.columns = ['Date', 'Type', 'Rating', 'Email', 'Feedback']
        
        # Display the table
        st.dataframe(display_df, use_container_width=True)
        
        # Show individual feedback details in expanders
        st.subheader("📝 Detailed Feedback")
        for i, feedback in enumerate(reversed(feedback_list)):
            with st.expander(f"Feedback #{feedback['id']} - {feedback['feedback_type']} - Rating: {feedback['rating']}/5"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"**Feedback:** {feedback['feedback_text']}")
                    st.write(f"**Type:** {feedback['feedback_type']}")
                    st.write(f"**Rating:** {feedback['rating']}/5")
                
                with col2:
                    st.write(f"**Date:** {feedback['timestamp'][:19]}")
                    st.write(f"**Email:** {feedback['user_email']}")