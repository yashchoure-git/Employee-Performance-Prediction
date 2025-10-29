import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Employee Performance Predictor",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Load the model and preprocessors
@st.cache_resource
def load_model_artifacts():
    try:
        model = joblib.load('employee_performance_model.pkl')
        scaler = joblib.load('scaler.pkl')
        le_dept = joblib.load('label_encoder_dept.pkl')
        le_salary = joblib.load('label_encoder_salary.pkl')
        le_performance = joblib.load('label_encoder_performance.pkl')
        feature_names = joblib.load('feature_names.pkl')
        return model, scaler, le_dept, le_salary, le_performance, feature_names
    except FileNotFoundError as e:
        st.error(f"Error loading model files: {e}")
        st.info("Please run the Jupyter notebook first to train and save the model.")
        st.stop()

# Load artifacts
model, scaler, le_dept, le_salary, le_performance, feature_names = load_model_artifacts()

# Title and description
st.title("🎯 Employee Performance Prediction System")
st.markdown("""
This application uses a **Machine Learning model** to predict employee performance ratings based on various factors.
Fill in the employee details below to get a prediction.
""")

# Sidebar for information
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    ### Model Information
    - **Algorithm**: Random Forest Classifier
    - **Performance Classes**: Poor, Average, Good, Excellent
    - **Features**: 9 key employee attributes
    
    ### How to Use
    1. Enter employee details in the form
    2. Click "Predict Performance"
    3. View the prediction and insights
    """)
    
    st.markdown("---")
    st.markdown("### Model Metrics")
    st.info("Trained on 1000+ employee records")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("📝 Employee Information")
    
    # Create input form
    with st.form("prediction_form"):
        # Row 1
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Age", min_value=18, max_value=70, value=30, step=1)
        with c2:
            department = st.selectbox("Department", le_dept.classes_)
        with c3:
            salary_level = st.selectbox("Salary Level", le_salary.classes_)
        
        # Row 2
        c1, c2, c3 = st.columns(3)
        with c1:
            years_at_company = st.number_input("Years at Company", min_value=0, max_value=40, value=3, step=1)
        with c2:
            projects_completed = st.number_input("Projects Completed", min_value=0, max_value=100, value=15, step=1)
        with c3:
            training_hours = st.number_input("Training Hours", min_value=0, max_value=200, value=30, step=1)
        
        # Row 3
        c1, c2, c3 = st.columns(3)
        with c1:
            avg_monthly_hours = st.number_input("Avg Monthly Hours", min_value=80, max_value=300, value=180, step=5)
        with c2:
            last_promotion_years = st.number_input("Years Since Last Promotion", min_value=0, max_value=20, value=2, step=1)
        with c3:
            satisfaction_score = st.slider("Satisfaction Score", min_value=0.0, max_value=1.0, value=0.7, step=0.05)
        
        # Submit button
        submit_button = st.form_submit_button("🔮 Predict Performance")

with col2:
    st.header("📊 Quick Stats")
    st.metric("Department", department)
    st.metric("Experience", f"{years_at_company} years")
    st.metric("Satisfaction", f"{satisfaction_score:.0%}")

# Make prediction when form is submitted
if submit_button:
    # Prepare input data
    input_data = pd.DataFrame({
        'age': [age],
        'department': [department],
        'years_at_company': [years_at_company],
        'projects_completed': [projects_completed],
        'avg_monthly_hours': [avg_monthly_hours],
        'training_hours': [training_hours],
        'last_promotion_years': [last_promotion_years],
        'salary_level': [salary_level],
        'satisfaction_score': [satisfaction_score]
    })
    
    # Transform categorical variables
    input_data['department'] = le_dept.transform(input_data['department'])
    input_data['salary_level'] = le_salary.transform(input_data['salary_level'])
    
    # Scale features
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)[0]
    
    # Decode prediction
    performance_rating = le_performance.inverse_transform(prediction)[0]
    
    # Display results
    st.markdown("---")
    st.header("🎯 Prediction Results")
    
    # Main prediction
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # Color coding
        color_map = {
            'Poor': '#FF4444',
            'Average': '#FFA500',
            'Good': '#4CAF50',
            'Excellent': '#2196F3'
        }
        
        st.markdown(f"""
        <div style='text-align: center; padding: 2rem; background-color: {color_map.get(performance_rating, '#808080')}; 
        border-radius: 15px; color: white;'>
            <h1 style='margin: 0; font-size: 3rem;'>{performance_rating}</h1>
            <p style='margin: 0; font-size: 1.2rem;'>Performance Rating</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Probability distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Confidence Distribution")
        
        # Create probability chart
        prob_df = pd.DataFrame({
            'Performance': le_performance.classes_,
            'Probability': prediction_proba * 100
        })
        
        fig = px.bar(prob_df, x='Performance', y='Probability',
                     color='Probability',
                     color_continuous_scale='viridis',
                     labels={'Probability': 'Probability (%)'},
                     text=prob_df['Probability'].apply(lambda x: f'{x:.1f}%'))
        
        fig.update_traces(textposition='outside')
        fig.update_layout(
            showlegend=False,
            height=400,
            xaxis_title="Performance Rating",
            yaxis_title="Probability (%)"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("💡 Key Insights")
        
        # Calculate confidence
        max_prob = max(prediction_proba) * 100
        
        st.markdown(f"""
        <div class='metric-card'>
        <strong>Confidence Level:</strong> {max_prob:.1f}%
        </div>
        """, unsafe_allow_html=True)
        
        # Provide insights
        insights = []
        
        if projects_completed > 20:
            insights.append("✅ High project completion rate")
        elif projects_completed < 10:
            insights.append("⚠️ Low project completion rate")
        
        if training_hours > 40:
            insights.append("✅ Strong training commitment")
        elif training_hours < 20:
            insights.append("⚠️ Limited training participation")
        
        if satisfaction_score > 0.7:
            insights.append("✅ High job satisfaction")
        elif satisfaction_score < 0.4:
            insights.append("⚠️ Low job satisfaction")
        
        if last_promotion_years > 5:
            insights.append("⚠️ Long time since last promotion")
        elif last_promotion_years <= 2:
            insights.append("✅ Recently promoted")
        
        if avg_monthly_hours > 200:
            insights.append("⚠️ High workload - risk of burnout")
        elif avg_monthly_hours < 140:
            insights.append("⚠️ Low engagement hours")
        
        for insight in insights:
            st.markdown(f"<div class='metric-card'>{insight}</div>", unsafe_allow_html=True)
        
        # Recommendations
        st.subheader("🎯 Recommendations")
        
        if performance_rating in ['Poor', 'Average']:
            st.markdown("""
            - 📚 Increase training opportunities
            - 🎯 Set clear performance goals
            - 💬 Schedule regular feedback sessions
            - 🌟 Consider skill development programs
            """)
        else:
            st.markdown("""
            - 🏆 Recognize achievements
            - 📈 Consider for leadership roles
            - 🎓 Provide mentorship opportunities
            - 💼 Discuss career advancement
            """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>Employee Performance Prediction System | Powered by Machine Learning</p>
</div>
""", unsafe_allow_html=True)
