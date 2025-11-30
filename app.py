import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from model import get_predictor
from ai_recommendations import generate_health_recommendations
from database import init_db
from auth import render_auth_ui, is_logged_in, get_current_user_id, get_current_username
from history import save_prediction, get_user_predictions, get_trend_data, get_stats_summary, save_health_log, get_health_logs
from pdf_report import generate_pdf_report
from csv_parser import parse_csv_file, convert_to_prediction_format, get_sample_csv_template, validate_health_values
import os

init_db()

st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A5F;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #5A6C7D;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
    }
    .risk-low { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); }
    .risk-moderate { background: linear-gradient(135deg, #F2994A 0%, #F2C94C 100%); }
    .risk-high { background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%); }
    .risk-very-high { background: linear-gradient(135deg, #8E2DE2 0%, #4A00E0 100%); }
    .recommendation-card {
        background: #f8f9fa !important;
        padding: 1.5rem !important;
        border-radius: 0.5rem !important;
        margin-bottom: 1.5rem !important;
        margin-top: 0.5rem !important;
        border-left: 4px solid #667eea !important;
        display: block !important;
        overflow: visible !important;
        box-sizing: border-box !important;
    }
    .priority-high { border-left-color: #eb3349 !important; }
    .priority-medium { border-left-color: #F2994A !important; }
    .priority-low { border-left-color: #11998e !important; }
    .info-box {
        background: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #17a2b8;
        margin: 1rem 0;
    }
    .warning-box {
        background: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .history-card {
        background: white;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin-bottom: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .stat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

def create_gauge_chart(probability, title="Diabetes Risk"):
    if probability < 0.3:
        color = "#38ef7d"
    elif probability < 0.5:
        color = "#F2C94C"
    elif probability < 0.7:
        color = "#f45c43"
    else:
        color = "#8E2DE2"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 20, 'color': '#1E3A5F'}},
        number={'suffix': '%', 'font': {'size': 40, 'color': '#1E3A5F'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#1E3A5F"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#ccc",
            'steps': [
                {'range': [0, 30], 'color': '#d4edda'},
                {'range': [30, 50], 'color': '#fff3cd'},
                {'range': [50, 70], 'color': '#f8d7da'},
                {'range': [70, 100], 'color': '#e2d5f1'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': probability * 100
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': "#1E3A5F"}
    )
    
    return fig

def create_feature_importance_chart(importance_dict):
    df = pd.DataFrame({
        'Feature': list(importance_dict.keys()),
        'Importance': list(importance_dict.values())
    })
    df = df.sort_values('Importance', ascending=True)
    
    feature_labels = {
        'Pregnancies': 'Pregnancies',
        'Glucose': 'Blood Glucose',
        'BloodPressure': 'Blood Pressure',
        'SkinThickness': 'Skin Thickness',
        'Insulin': 'Insulin Level',
        'BMI': 'Body Mass Index',
        'DiabetesPedigreeFunction': 'Family History',
        'Age': 'Age'
    }
    df['Feature'] = df['Feature'].map(feature_labels)
    
    fig = px.bar(
        df, 
        x='Importance', 
        y='Feature',
        orientation='h',
        title='Factor Importance in Prediction',
        color='Importance',
        color_continuous_scale='Blues'
    )
    
    fig.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=False,
        xaxis_title="Relative Importance",
        yaxis_title=""
    )
    
    return fig

def create_risk_comparison_chart(user_values):
    normal_ranges = {
        'Glucose': {'min': 70, 'max': 100, 'user': user_values.get('Glucose', 100)},
        'Blood Pressure': {'min': 60, 'max': 80, 'user': user_values.get('BloodPressure', 70)},
        'BMI': {'min': 18.5, 'max': 24.9, 'user': user_values.get('BMI', 25)},
        'Insulin': {'min': 16, 'max': 166, 'user': user_values.get('Insulin', 80)}
    }
    
    fig = go.Figure()
    
    for i, (cat, vals) in enumerate(normal_ranges.items()):
        normalized_user = (vals['user'] - vals['min']) / (vals['max'] - vals['min']) * 100
        normalized_user = max(0, min(150, normalized_user))
        
        color = '#38ef7d' if 0 <= normalized_user <= 100 else '#f45c43'
        
        fig.add_trace(go.Bar(
            name=cat,
            x=[cat],
            y=[normalized_user],
            marker_color=color,
            text=[f"{vals['user']:.1f}"],
            textposition='outside'
        ))
        
        fig.add_shape(
            type="rect",
            x0=i-0.4, x1=i+0.4,
            y0=0, y1=100,
            fillcolor="rgba(0,255,0,0.1)",
            line=dict(color="green", width=1, dash="dash")
        )
    
    fig.update_layout(
        title="Your Values vs Normal Range",
        yaxis_title="Percentage of Normal Range",
        showlegend=False,
        height=350,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    fig.add_hline(y=100, line_dash="dash", line_color="green", 
                  annotation_text="Upper Normal", annotation_position="right")
    fig.add_hline(y=0, line_dash="dash", line_color="green",
                  annotation_text="Lower Normal", annotation_position="right")
    
    return fig

def create_trend_chart(trend_data):
    if not trend_data['dates']:
        return None
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=trend_data['dates'],
        y=trend_data['risk_scores'],
        mode='lines+markers',
        name='Risk Score',
        line=dict(color='#667eea', width=3),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title='Risk Score Trend Over Time',
        xaxis_title='Date',
        yaxis_title='Risk Score (%)',
        height=350,
        yaxis=dict(range=[0, 100]),
        hovermode='x unified'
    )
    
    return fig

def display_recommendations(recommendations):
    if not recommendations:
        st.warning("Unable to generate recommendations. Please try again.")
        return
    
    st.markdown("### Summary")
    st.info(recommendations.get('summary', 'No summary available.'))
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🍎 Diet Recommendations")
        diet_recs = recommendations.get('diet_recommendations', [])
        if diet_recs:
            for rec in diet_recs:
                with st.container(border=True):
                    st.markdown(f"**{rec.get('title', 'Recommendation')}**")
                    st.write(rec.get('description', ''))
        else:
            st.info("No diet recommendations available")
        
        st.markdown("#### 🧘 Lifestyle Recommendations")
        lifestyle_recs = recommendations.get('lifestyle_recommendations', [])
        if lifestyle_recs:
            for rec in lifestyle_recs:
                with st.container(border=True):
                    st.markdown(f"**{rec.get('title', 'Recommendation')}**")
                    st.write(rec.get('description', ''))
        else:
            st.info("No lifestyle recommendations available")
    
    with col2:
        st.markdown("#### 🏃 Exercise Recommendations")
        exercise_recs = recommendations.get('exercise_recommendations', [])
        if exercise_recs:
            for rec in exercise_recs:
                with st.container(border=True):
                    st.markdown(f"**{rec.get('title', 'Recommendation')}**")
                    st.write(rec.get('description', ''))
        else:
            st.info("No exercise recommendations available")
        
        st.markdown("#### 💊 Medical Advice")
        medical_recs = recommendations.get('medical_advice', [])
        if medical_recs:
            for rec in medical_recs:
                with st.container(border=True):
                    st.markdown(f"**{rec.get('title', 'Recommendation')}**")
                    st.write(rec.get('description', ''))
        else:
            st.info("No medical advice available")
    
    col3, col4 = st.columns(2)
    
    with col3:
        if recommendations.get('warning_signs'):
            st.markdown("#### ⚠️ Warning Signs to Watch")
            for sign in recommendations['warning_signs']:
                st.warning(f"• {sign}")
    
    with col4:
        if recommendations.get('positive_factors'):
            st.markdown("#### ✅ Positive Health Factors")
            for factor in recommendations['positive_factors']:
                st.success(f"• {factor}")


def render_health_tools_tab():
    st.markdown("## Health Tools")
    
    tool_tabs = st.tabs(["BMI Calculator", "Calorie Tracker", "Exercise Log"])
    
    with tool_tabs[0]:
        st.markdown("### BMI Calculator")
        
        calc_col1, calc_col2 = st.columns(2)
        
        with calc_col1:
            calc_height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170, key="bmi_calc_height")
            calc_weight = st.number_input("Weight (kg)", min_value=30.0, max_value=300.0, value=70.0, step=0.1, key="bmi_calc_weight")
            
            if st.button("Calculate BMI", type="primary"):
                bmi = calc_weight / ((calc_height / 100) ** 2)
                st.session_state['calculated_bmi'] = bmi
        
        with calc_col2:
            if 'calculated_bmi' in st.session_state:
                bmi = st.session_state['calculated_bmi']
                
                if bmi < 18.5:
                    category = "Underweight"
                    color = "#F2C94C"
                    advice = "Consider consulting a healthcare provider about healthy weight gain."
                elif bmi < 25:
                    category = "Normal Weight"
                    color = "#38ef7d"
                    advice = "Great! Maintain your healthy lifestyle."
                elif bmi < 30:
                    category = "Overweight"
                    color = "#F2C94C"
                    advice = "Consider lifestyle changes to reduce diabetes risk."
                else:
                    category = "Obese"
                    color = "#f45c43"
                    advice = "Consult a healthcare provider for personalized weight management advice."
                
                st.markdown(f"""
                <div style="background: {color}; padding: 1.5rem; border-radius: 1rem; text-align: center; color: white;">
                    <h2 style="margin: 0;">Your BMI: {bmi:.1f}</h2>
                    <h3 style="margin: 0.5rem 0;">{category}</h3>
                </div>
                """, unsafe_allow_html=True)
                
                st.info(advice)
                
                if is_logged_in():
                    if st.button("Save to Health Log"):
                        save_health_log(
                            get_current_user_id(),
                            'bmi',
                            weight=calc_weight,
                            height=calc_height,
                            bmi=bmi
                        )
                        st.success("BMI saved to your health log!")
        
        st.markdown("### BMI Categories Reference")
        bmi_ref = pd.DataFrame({
            'Category': ['Underweight', 'Normal', 'Overweight', 'Obese Class I', 'Obese Class II', 'Obese Class III'],
            'BMI Range': ['< 18.5', '18.5 - 24.9', '25 - 29.9', '30 - 34.9', '35 - 39.9', '≥ 40'],
            'Health Risk': ['Moderate', 'Low', 'Moderate', 'High', 'Very High', 'Extremely High']
        })
        st.dataframe(bmi_ref, use_container_width=True, hide_index=True)
    
    with tool_tabs[1]:
        st.markdown("### Calorie Tracker")
        
        if not is_logged_in():
            st.info("Log in to track your daily calories and save your records.")
        
        cal_col1, cal_col2 = st.columns(2)
        
        with cal_col1:
            st.markdown("#### Log Calories Consumed")
            calories_consumed = st.number_input("Calories Consumed", min_value=0, max_value=10000, value=0, key="cal_consumed")
            meal_notes = st.text_input("Meal Description (optional)", key="meal_notes")
            
            if st.button("Log Consumed Calories"):
                if is_logged_in():
                    save_health_log(
                        get_current_user_id(),
                        'calories',
                        calories_consumed=calories_consumed,
                        notes=meal_notes
                    )
                    st.success("Calories logged!")
                else:
                    st.warning("Please log in to save your calorie records.")
        
        with cal_col2:
            st.markdown("#### Log Calories Burned")
            calories_burned = st.number_input("Calories Burned", min_value=0, max_value=5000, value=0, key="cal_burned")
            exercise_notes = st.text_input("Exercise Description (optional)", key="exercise_notes")
            
            if st.button("Log Burned Calories"):
                if is_logged_in():
                    save_health_log(
                        get_current_user_id(),
                        'calories',
                        calories_burned=calories_burned,
                        notes=exercise_notes
                    )
                    st.success("Calories burned logged!")
                else:
                    st.warning("Please log in to save your calorie records.")
        
        st.markdown("---")
        st.markdown("### Daily Calorie Guidelines")
        
        calorie_ref = pd.DataFrame({
            'Activity Level': ['Sedentary', 'Lightly Active', 'Moderately Active', 'Very Active'],
            'Women (cal/day)': ['1,600-2,000', '1,800-2,200', '2,000-2,400', '2,200-2,600'],
            'Men (cal/day)': ['2,000-2,400', '2,200-2,600', '2,400-2,800', '2,600-3,200']
        })
        st.dataframe(calorie_ref, use_container_width=True, hide_index=True)
    
    with tool_tabs[2]:
        st.markdown("### Exercise Log")
        
        if not is_logged_in():
            st.info("Log in to track your exercise and save your records.")
        
        ex_col1, ex_col2 = st.columns(2)
        
        with ex_col1:
            exercise_type = st.selectbox("Exercise Type", [
                "Walking", "Running", "Cycling", "Swimming", "Strength Training",
                "Yoga", "HIIT", "Dancing", "Sports", "Other"
            ])
            exercise_duration = st.number_input("Duration (minutes)", min_value=1, max_value=480, value=30)
            exercise_intensity = st.select_slider("Intensity", options=["Light", "Moderate", "Vigorous"])
            
            intensity_multiplier = {"Light": 3, "Moderate": 5, "Vigorous": 8}
            estimated_calories = exercise_duration * intensity_multiplier[exercise_intensity]
            
            st.metric("Estimated Calories Burned", f"~{estimated_calories}")
        
        with ex_col2:
            exercise_notes_log = st.text_area("Notes (optional)", key="exercise_log_notes")
            
            if st.button("Log Exercise", type="primary"):
                if is_logged_in():
                    save_health_log(
                        get_current_user_id(),
                        'exercise',
                        exercise_type=exercise_type,
                        exercise_minutes=exercise_duration,
                        calories_burned=estimated_calories,
                        notes=f"{exercise_intensity} intensity. {exercise_notes_log}"
                    )
                    st.success("Exercise logged!")
                else:
                    st.warning("Please log in to save your exercise records.")
        
        st.markdown("---")
        st.markdown("### Exercise Recommendations for Diabetes Prevention")
        
        exercise_recs = pd.DataFrame({
            'Exercise': ['Brisk Walking', 'Swimming', 'Cycling', 'Strength Training', 'Yoga'],
            'Weekly Target': ['150 min', '150 min', '150 min', '2-3 sessions', '2-3 sessions'],
            'Benefit': ['Improves insulin sensitivity', 'Low impact cardio', 'Burns calories efficiently', 'Builds muscle mass', 'Reduces stress']
        })
        st.dataframe(exercise_recs, use_container_width=True, hide_index=True)


def render_history_tab():
    st.markdown("## Your Health History")
    
    if not is_logged_in():
        st.info("Please log in to view your prediction history and health trends.")
        return
    
    user_id = get_current_user_id()
    
    stats = get_stats_summary(user_id)
    
    if stats.get('total_predictions', 0) > 0:
        stat_cols = st.columns(4)
        
        with stat_cols[0]:
            st.markdown(f"""
            <div class="stat-card">
                <h3 style="margin: 0;">{stats['total_predictions']}</h3>
                <p style="margin: 0.5rem 0 0 0;">Total Assessments</p>
            </div>
            """, unsafe_allow_html=True)
        
        with stat_cols[1]:
            latest_risk = stats.get('latest_risk')
            st.markdown(f"""
            <div class="stat-card" style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);">
                <h3 style="margin: 0;">{latest_risk:.1f}%</h3>
                <p style="margin: 0.5rem 0 0 0;">Latest Risk Score</p>
            </div>
            """, unsafe_allow_html=True)
        
        with stat_cols[2]:
            risk_level = stats.get('latest_risk_level', 'N/A')
            st.markdown(f"""
            <div class="stat-card" style="background: linear-gradient(135deg, #F2994A 0%, #F2C94C 100%);">
                <h3 style="margin: 0;">{risk_level}</h3>
                <p style="margin: 0.5rem 0 0 0;">Current Risk Level</p>
            </div>
            """, unsafe_allow_html=True)
        
        with stat_cols[3]:
            risk_change = stats.get('risk_change')
            if risk_change is not None:
                change_text = f"{risk_change:+.1f}%"
                change_color = "#38ef7d" if risk_change < 0 else "#f45c43"
            else:
                change_text = "N/A"
                change_color = "#667eea"
            st.markdown(f"""
            <div class="stat-card" style="background: {change_color};">
                <h3 style="margin: 0;">{change_text}</h3>
                <p style="margin: 0.5rem 0 0 0;">Risk Change</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        trend_col1, trend_col2 = st.columns(2)
        
        with trend_col1:
            trend_data = get_trend_data(user_id, days=90)
            trend_chart = create_trend_chart(trend_data)
            if trend_chart:
                st.plotly_chart(trend_chart, use_container_width=True)
        
        with trend_col2:
            if trend_data['dates']:
                metrics_df = pd.DataFrame({
                    'Date': trend_data['dates'],
                    'Glucose': trend_data['glucose'],
                    'BMI': trend_data['bmi'],
                    'Blood Pressure': trend_data['blood_pressure']
                })
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=metrics_df['Date'], y=metrics_df['Glucose'], name='Glucose', mode='lines+markers'))
                fig.add_trace(go.Scatter(x=metrics_df['Date'], y=metrics_df['BMI'] * 4, name='BMI (scaled)', mode='lines+markers'))
                fig.add_trace(go.Scatter(x=metrics_df['Date'], y=metrics_df['Blood Pressure'], name='Blood Pressure', mode='lines+markers'))
                
                fig.update_layout(
                    title='Health Metrics Over Time',
                    xaxis_title='Date',
                    yaxis_title='Value',
                    height=350,
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        st.markdown("### Recent Assessments")
        
        predictions = get_user_predictions(user_id, limit=10)
        
        for idx, pred in enumerate(predictions):
            risk_colors = {
                'Low': '#38ef7d',
                'Moderate': '#F2C94C',
                'High': '#f45c43',
                'Very High': '#8E2DE2'
            }
            color = risk_colors.get(pred['risk_level'], '#667eea')
            
            st.markdown(f"""
            <div class="history-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <strong>{pred['date'].strftime('%B %d, %Y at %I:%M %p')}</strong><br>
                        <span style="color: #5A6C7D;">Glucose: {pred['glucose']:.0f} mg/dL | BMI: {pred['bmi']:.1f} | BP: {pred['blood_pressure']:.0f} mmHg</span>
                    </div>
                    <div style="text-align: right;">
                        <span style="background: {color}; color: white; padding: 0.25rem 0.75rem; border-radius: 1rem; font-weight: bold;">
                            {pred['risk_level']}
                        </span><br>
                        <span style="color: #5A6C7D; font-size: 0.9rem;">{pred['risk_probability']*100:.1f}% risk</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True, key=f"history_card_{idx}")
    else:
        st.markdown("""
        <div class="info-box">
            <h4>No Assessment History Yet</h4>
            <p>Complete your first risk assessment to start tracking your health progress over time!</p>
            <p>Go to the <strong>Risk Assessment</strong> tab to get started.</p>
        </div>
        """, unsafe_allow_html=True)


def render_upload_tab():
    st.markdown("## Upload Medical Test Results")
    
    st.markdown("""
    Upload your medical test results in CSV format to automatically populate your health parameters.
    This makes it easy to enter data from lab reports or health tracking apps.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Upload CSV File")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
        
        if uploaded_file is not None:
            file_content = uploaded_file.read()
            result = parse_csv_file(file_content)
            
            if result['success']:
                st.success(result['message'])
                
                warnings = validate_health_values(result['data'])
                for warning in warnings:
                    st.warning(warning)
                
                st.markdown("#### Extracted Values")
                
                extracted_df = pd.DataFrame([
                    {'Parameter': k.replace('_', ' ').title(), 'Value': f"{v:.1f}" if isinstance(v, float) else str(v)}
                    for k, v in result['data'].items()
                ])
                st.dataframe(extracted_df, use_container_width=True, hide_index=True)
                
                if st.button("Use These Values for Assessment", type="primary"):
                    prediction_data = convert_to_prediction_format(result)
                    st.session_state['uploaded_data'] = prediction_data
                    st.success("Values loaded! Go to Risk Assessment tab to complete your assessment.")
            else:
                st.error(result['message'])
    
    with col2:
        st.markdown("### Expected CSV Format")
        
        st.markdown("""
        Your CSV file should contain columns with health measurements. 
        The system will automatically recognize common column names.
        
        **Supported parameters:**
        - Glucose / Blood Glucose
        - Blood Pressure / BP
        - BMI / Body Mass Index
        - Insulin
        - Age
        - Weight & Height (BMI calculated automatically)
        - Skin Thickness
        - Pregnancies
        """)
        
        st.markdown("#### Sample Template")
        sample_csv = get_sample_csv_template()
        st.code(sample_csv, language='csv')
        
        st.download_button(
            label="Download Sample Template",
            data=sample_csv,
            file_name="health_data_template.csv",
            mime="text/csv"
        )


def main():
    st.markdown('<h1 class="main-header">Diabetes Risk Prediction System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Health Assessment & Personalized Recommendations</p>', unsafe_allow_html=True)
    
    predictor = get_predictor()
    
    with st.sidebar:
        logged_in = render_auth_ui()
        
        st.markdown("---")
        st.markdown("## Health Parameters")
        st.markdown("Enter your health information below for a personalized risk assessment.")
        
        uploaded_data = st.session_state.get('uploaded_data', {})
        
        st.markdown("### Personal Information")
        age = st.slider("Age (years)", min_value=18, max_value=100, 
                       value=int(uploaded_data.get('Age', 35)), help="Your current age")
        pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, 
                                       value=uploaded_data.get('Pregnancies', 0), 
                                       help="Number of times pregnant (enter 0 if not applicable)")
        
        st.markdown("### Body Measurements")
        height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
        weight = st.number_input("Weight (kg)", min_value=30, max_value=250, value=70)
        bmi = weight / ((height/100) ** 2)
        if uploaded_data.get('BMI'):
            bmi = uploaded_data['BMI']
        st.metric("Calculated BMI", f"{bmi:.1f}")
        
        skin_thickness = st.slider("Skin Fold Thickness (mm)", min_value=0, max_value=100, 
                                    value=int(uploaded_data.get('SkinThickness', 20)),
                                    help="Triceps skin fold thickness in mm")
        
        st.markdown("### Blood Tests")
        glucose = st.slider("Blood Glucose (mg/dL)", min_value=50, max_value=300, 
                            value=int(uploaded_data.get('Glucose', 100)),
                            help="Fasting blood glucose level")
        blood_pressure = st.slider("Blood Pressure - Diastolic (mmHg)", min_value=40, max_value=140, 
                                    value=int(uploaded_data.get('BloodPressure', 70)),
                                    help="Diastolic blood pressure")
        insulin = st.slider("Insulin Level (μU/mL)", min_value=0, max_value=500, 
                            value=int(uploaded_data.get('Insulin', 80)),
                            help="2-Hour serum insulin")
        
        st.markdown("### Family History")
        family_history = st.select_slider(
            "Diabetes in Family",
            options=["None", "Distant Relative", "Grandparent", "Parent/Sibling", "Multiple Close Relatives"],
            value="None",
            help="Family history of diabetes"
        )
        
        family_history_map = {
            "None": 0.2,
            "Distant Relative": 0.4,
            "Grandparent": 0.6,
            "Parent/Sibling": 0.8,
            "Multiple Close Relatives": 1.2
        }
        dpf = family_history_map[family_history]
        
        st.markdown("---")
        predict_button = st.button("Analyze My Risk", type="primary", use_container_width=True)
    
    if is_logged_in():
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "Risk Assessment", "Health Recommendations", "History & Trends", 
            "Health Tools", "Upload Data", "Educational Resources"
        ])
    else:
        tab1, tab2, tab4, tab5, tab6 = st.tabs([
            "Risk Assessment", "Health Recommendations", 
            "Health Tools", "Upload Data", "Educational Resources"
        ])
        tab3 = None
    
    with tab1:
        if predict_button or 'prediction_result' in st.session_state:
            user_data = {
                'Pregnancies': pregnancies,
                'Glucose': glucose,
                'BloodPressure': blood_pressure,
                'SkinThickness': skin_thickness,
                'Insulin': insulin,
                'BMI': bmi,
                'DiabetesPedigreeFunction': dpf,
                'Age': age
            }
            
            with st.spinner("Analyzing your health data..."):
                result = predictor.predict(user_data)
                st.session_state['prediction_result'] = result
                st.session_state['user_data'] = user_data
            
            st.markdown("## Your Assessment Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("### Diabetes Risk")
                st.plotly_chart(create_gauge_chart(result['probability_diabetes']), use_container_width=True, key="risk_gauge")
            
            with col2:
                st.markdown("### Risk Level")
                risk_level = result['risk_level']
                risk_colors = {
                    'Low': '#38ef7d',
                    'Moderate': '#F2C94C',
                    'High': '#f45c43',
                    'Very High': '#8E2DE2'
                }
                
                with st.container(border=True):
                    st.markdown(f"<h1 style='text-align: center; color: {risk_colors[risk_level]};'>{risk_level}</h1>", unsafe_allow_html=True)
                    st.markdown("<p style='text-align: center; color: #666;'>Based on your health parameters</p>", unsafe_allow_html=True)
            
            with col3:
                st.markdown("### Health Insights")
                bmi_val = user_data['BMI']
                if bmi_val < 18.5:
                    bmi_category = "Underweight"
                    bmi_color = "#F2C94C"
                elif bmi_val < 25:
                    bmi_category = "Normal"
                    bmi_color = "#38ef7d"
                elif bmi_val < 30:
                    bmi_category = "Overweight"
                    bmi_color = "#F2C94C"
                else:
                    bmi_category = "Obese"
                    bmi_color = "#f45c43"
                
                glucose_val = user_data['Glucose']
                if glucose_val < 100:
                    glucose_status = "Normal"
                    glucose_color = "#38ef7d"
                elif glucose_val < 126:
                    glucose_status = "Pre-diabetic"
                    glucose_color = "#F2C94C"
                else:
                    glucose_status = "Diabetic Range"
                    glucose_color = "#f45c43"
                
                with st.container(border=True):
                    with st.container(border=True):
                        st.markdown(f"**BMI Status**")
                        st.markdown(f"<span style='color: {bmi_color}; font-size: 1.1rem;'>{bmi_category}</span> - {bmi_val:.1f}", unsafe_allow_html=True)
                    with st.container(border=True):
                        st.markdown(f"**Glucose Status**")
                        st.markdown(f"<span style='color: {glucose_color}; font-size: 1.1rem;'>{glucose_status}</span> - {glucose_val} mg/dL", unsafe_allow_html=True)
                    with st.container(border=True):
                        st.markdown(f"**Risk Score**")
                        st.markdown(f"<strong style='font-size: 1.3rem;'>{result['probability_diabetes']*100:.0f}/100</strong>", unsafe_allow_html=True)
            
            if is_logged_in():
                save_col1, save_col2 = st.columns([3, 1])
                with save_col2:
                    if st.button("Save This Assessment"):
                        risk_factors = predictor.get_risk_factors(user_data)
                        recommendations = generate_health_recommendations(user_data, result, risk_factors)
                        if save_prediction(get_current_user_id(), user_data, result, recommendations):
                            st.success("Assessment saved to your history!")
                        else:
                            st.error("Failed to save assessment.")
            
            st.divider()
            st.markdown("## Analysis Charts")
            
            col4, col5 = st.columns(2)
            
            with col4:
                st.markdown("### Risk Factors")
                importance = predictor.get_feature_importance()
                st.plotly_chart(create_feature_importance_chart(importance), use_container_width=True, key="feature_importance")
            
            with col5:
                st.markdown("### Your Values vs Normal Range")
                st.plotly_chart(create_risk_comparison_chart(user_data), use_container_width=True, key="risk_comparison")
            
            st.divider()
            risk_factors = predictor.get_risk_factors(user_data)
            if risk_factors:
                st.markdown("## Identified Risk Factors")
                
                cols = st.columns(len(risk_factors) if len(risk_factors) <= 4 else 4)
                for i, factor in enumerate(risk_factors[:4]):
                    with cols[i % 4]:
                        with st.container(border=True):
                            st.markdown(f"**{factor['factor']}**")
                            severity_colors = {
                                'high': '#f45c43',
                                'moderate': '#F2C94C',
                                'low': '#38ef7d'
                            }
                            color = severity_colors.get(factor['severity'], '#667eea')
                            st.markdown(f"<div style='font-size: 1.5rem; color: {color}; text-align: center;'>{factor['value']}</div>", unsafe_allow_html=True)
                            st.caption(f"Normal: {factor['normal_range']}")
        else:
            st.markdown("""
            <div class="info-box">
                <h3>Welcome to the Diabetes Risk Prediction System</h3>
                <p>This tool uses machine learning to assess your risk of developing Type 2 diabetes based on key health indicators.</p>
                <p><strong>How to use:</strong></p>
                <ol>
                    <li>Enter your health parameters in the sidebar</li>
                    <li>Click "Analyze My Risk" to get your assessment</li>
                    <li>Review your personalized recommendations</li>
                </ol>
                <p><em>Note: This tool is for educational purposes only and should not replace professional medical advice.</em></p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### Understanding Diabetes Risk Factors")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown("""
                **Blood Glucose**
                - Normal: < 100 mg/dL
                - Pre-diabetes: 100-125
                - Diabetes: > 126
                """)
            
            with col2:
                st.markdown("""
                **BMI Categories**
                - Underweight: < 18.5
                - Normal: 18.5-24.9
                - Overweight: 25-29.9
                - Obese: > 30
                """)
            
            with col3:
                st.markdown("""
                **Blood Pressure**
                - Normal: < 80 mmHg
                - Elevated: 80-89
                - High: > 90
                """)
            
            with col4:
                st.markdown("""
                **Age Factor**
                - Risk increases after 45
                - Higher risk after 65
                - Family history matters
                """)
    
    with tab2:
        if 'prediction_result' in st.session_state and 'user_data' in st.session_state:
            st.markdown("## Personalized Health Recommendations")
            
            has_api_key = bool(os.environ.get("OPENAI_API_KEY"))
            if has_api_key:
                st.success("AI-powered recommendations enabled")
            else:
                st.info("Using standard recommendations. Add OpenAI API key for AI-powered personalized advice.")
            
            with st.spinner("Generating personalized recommendations..."):
                risk_factors = predictor.get_risk_factors(st.session_state['user_data'])
                recommendations = generate_health_recommendations(
                    st.session_state['user_data'],
                    st.session_state['prediction_result'],
                    risk_factors
                )
                st.session_state['recommendations'] = recommendations
            
            display_recommendations(recommendations)
            
            st.markdown("---")
            
            if 'recommendations' in st.session_state:
                pdf_col1, pdf_col2 = st.columns([3, 1])
                with pdf_col2:
                    pdf_bytes = generate_pdf_report(
                        st.session_state['user_data'],
                        st.session_state['prediction_result'],
                        st.session_state['recommendations'],
                        get_current_username() if is_logged_in() else "User"
                    )
                    st.download_button(
                        label="Download PDF Report",
                        data=pdf_bytes,
                        file_name="diabetes_risk_report.pdf",
                        mime="application/pdf",
                        type="primary"
                    )
            
            st.markdown("""
            <div class="warning-box">
                <strong>Important Disclaimer:</strong> These recommendations are for educational purposes only 
                and should not replace professional medical advice. Always consult with a healthcare provider 
                before making significant changes to your diet, exercise routine, or medication.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Please complete the risk assessment first to receive personalized recommendations.")
    
    if tab3 is not None:
        with tab3:
            render_history_tab()
    
    with tab4:
        render_health_tools_tab()
    
    with tab5:
        render_upload_tab()
    
    with tab6:
        st.markdown("## Understanding Diabetes")
        
        st.markdown("""
        ### What is Type 2 Diabetes?
        
        Type 2 diabetes is a chronic condition that affects the way your body metabolizes sugar (glucose). 
        With Type 2 diabetes, your body either resists the effects of insulin or doesn't produce enough 
        insulin to maintain normal glucose levels.
        """)
        
        st.markdown("---")
        st.markdown("### Global Diabetes Statistics")
        
        stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
        with stats_col1:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 1rem; text-align: center; color: white;">
                <h2 style="margin: 0; font-size: 2rem;">537M</h2>
                <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem;">Adults with Diabetes Worldwide</p>
            </div>
            """, unsafe_allow_html=True)
        with stats_col2:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); padding: 1.5rem; border-radius: 1rem; text-align: center; color: white;">
                <h2 style="margin: 0; font-size: 2rem;">90%</h2>
                <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem;">Have Type 2 Diabetes</p>
            </div>
            """, unsafe_allow_html=True)
        with stats_col3:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #F2994A 0%, #F2C94C 100%); padding: 1.5rem; border-radius: 1rem; text-align: center; color: white;">
                <h2 style="margin: 0; font-size: 2rem;">1 in 10</h2>
                <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem;">Adults Affected</p>
            </div>
            """, unsafe_allow_html=True)
        with stats_col4:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%); padding: 1.5rem; border-radius: 1rem; text-align: center; color: white;">
                <h2 style="margin: 0; font-size: 2rem;">50%</h2>
                <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem;">Remain Undiagnosed</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### Risk Factors
            
            **Modifiable Risk Factors:**
            - Being overweight or obese
            - Physical inactivity
            - Poor diet high in processed foods
            - High blood pressure
            - Abnormal cholesterol levels
            
            **Non-Modifiable Risk Factors:**
            - Age (45 years or older)
            - Family history of diabetes
            - Ethnicity
            - History of gestational diabetes
            """)
            
            risk_data = pd.DataFrame({
                'Factor': ['Obesity', 'Sedentary Lifestyle', 'Family History', 'Age > 45', 'High Blood Pressure', 'Poor Diet'],
                'Risk Increase': [7.0, 2.0, 3.0, 2.5, 1.5, 2.0]
            })
            
            fig_risk = px.bar(
                risk_data,
                x='Factor',
                y='Risk Increase',
                title='Relative Risk Increase by Factor',
                color='Risk Increase',
                color_continuous_scale='Reds'
            )
            fig_risk.update_layout(
                height=300,
                showlegend=False,
                xaxis_title="",
                yaxis_title="Times Higher Risk"
            )
            st.plotly_chart(fig_risk, use_container_width=True, key="risk_factors_chart")
        
        with col2:
            st.markdown("""
            ### Prevention Strategies
            
            **Lifestyle Changes:**
            - Maintain a healthy weight
            - Exercise regularly (150+ min/week)
            - Eat a balanced diet
            - Monitor blood sugar levels
            - Get regular health check-ups
            
            **Diet Tips:**
            - Choose whole grains
            - Eat plenty of vegetables
            - Limit sugary drinks
            - Control portion sizes
            """)
            
            prevention_data = pd.DataFrame({
                'Strategy': ['Weight Loss (7%)', 'Regular Exercise', 'Healthy Diet', 'Medication'],
                'Risk Reduction': [58, 30, 25, 31]
            })
            
            fig_prevention = px.bar(
                prevention_data,
                x='Strategy',
                y='Risk Reduction',
                title='Risk Reduction by Prevention Strategy',
                color='Risk Reduction',
                color_continuous_scale='Greens'
            )
            fig_prevention.update_layout(
                height=300,
                showlegend=False,
                xaxis_title="",
                yaxis_title="% Risk Reduction"
            )
            st.plotly_chart(fig_prevention, use_container_width=True, key="prevention_strategies_chart")
        
        st.markdown("---")
        
        st.markdown("### Understanding Blood Glucose Levels")
        
        glucose_col1, glucose_col2 = st.columns([2, 1])
        
        with glucose_col1:
            glucose_ranges = pd.DataFrame({
                'Category': ['Normal', 'Pre-Diabetes', 'Diabetes'],
                'Fasting (mg/dL)': ['70-99', '100-125', '126+'],
                'After Meal (mg/dL)': ['< 140', '140-199', '200+'],
                'HbA1c (%)': ['< 5.7', '5.7-6.4', '6.5+']
            })
            st.dataframe(glucose_ranges, use_container_width=True, hide_index=True)
        
        with glucose_col2:
            fig_glucose = go.Figure(go.Indicator(
                mode="gauge",
                gauge={
                    'axis': {'range': [70, 200], 'tickwidth': 1},
                    'bar': {'color': "rgba(0,0,0,0)"},
                    'steps': [
                        {'range': [70, 100], 'color': '#38ef7d', 'name': 'Normal'},
                        {'range': [100, 126], 'color': '#F2C94C', 'name': 'Pre-Diabetes'},
                        {'range': [126, 200], 'color': '#f45c43', 'name': 'Diabetes'}
                    ],
                    'threshold': {
                        'line': {'color': "black", 'width': 2},
                        'thickness': 0.75,
                        'value': 100
                    }
                },
                title={'text': "Glucose Scale (mg/dL)"}
            ))
            fig_glucose.update_layout(height=200, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig_glucose, use_container_width=True, key="glucose_scale_gauge")
        
        st.markdown("---")
        
        st.markdown("### How Our Prediction Model Works")
        
        importance = predictor.get_feature_importance()
        
        model_col1, model_col2 = st.columns(2)
        
        with model_col1:
            fig = create_feature_importance_chart(importance)
            st.plotly_chart(fig, use_container_width=True, key="model_feature_importance")
        
        with model_col2:
            st.markdown("""
            Our prediction model uses a **Random Forest classifier** trained on diabetes health data. 
            
            **How It Works:**
            1. **Data Input**: You provide 8 key health parameters
            2. **Feature Scaling**: Values are normalized for accurate comparison
            3. **Ensemble Prediction**: 100 decision trees vote on your risk
            4. **Probability Calculation**: Results are converted to a risk percentage
            
            **Key Features Analyzed:**
            - **Blood Glucose**: Primary indicator (highest weight)
            - **BMI**: Body composition assessment
            - **Age**: Accounts for age-related risk
            - **Blood Pressure**: Cardiovascular correlation
            - **Insulin Levels**: Insulin resistance indicator
            - **Family History**: Genetic predisposition
            """)
        
        st.markdown("---")
        
        st.markdown("### Frequently Asked Questions")
        
        with st.expander("How accurate is this prediction?"):
            st.write("""
            This tool provides an estimate based on common risk factors used in clinical research. 
            It's designed for educational purposes and should not replace professional medical diagnosis. 
            Our model is trained on patterns similar to the Pima Indians Diabetes Dataset, which is 
            widely used in diabetes research. For an official diagnosis, please consult a healthcare provider.
            """)
        
        with st.expander("What should I do if my risk is high?"):
            st.write("""
            If your assessment shows high risk:
            1. **Schedule an appointment** with your doctor for proper blood tests (fasting glucose, HbA1c)
            2. **Start making lifestyle changes** - even small improvements in diet and exercise help
            3. **Monitor your health** - keep track of weight, blood pressure, and any symptoms
            4. **Consider genetic testing** if you have strong family history
            5. **Don't panic** - many people with high risk never develop diabetes with proper prevention
            """)
        
        with st.expander("Can Type 2 diabetes be prevented?"):
            st.write("""
            Yes! Research shows that Type 2 diabetes can often be prevented or delayed:
            - **Diabetes Prevention Program** study showed 58% risk reduction with lifestyle changes
            - Losing just **5-7% of body weight** significantly reduces risk
            - **150 minutes of moderate exercise** per week (like brisk walking) is highly effective
            - **Mediterranean diet** has shown strong protective effects
            - Even if you have pre-diabetes, you can reverse it with lifestyle modifications
            """)
        
        with st.expander("How often should I get tested?"):
            st.write("""
            Testing recommendations vary based on risk:
            - **Adults 45+**: Test every 3 years if normal results
            - **Overweight adults of any age**: Test if you have additional risk factors
            - **Pre-diabetes diagnosis**: Test annually
            - **High risk individuals**: Your doctor may recommend more frequent testing
            - **Gestational diabetes history**: Test every 1-3 years
            """)
        
        with st.expander("What are early warning signs of diabetes?"):
            st.write("""
            Watch for these symptoms, especially if you have risk factors:
            - Increased thirst and frequent urination
            - Unexplained weight loss despite eating more
            - Fatigue and irritability
            - Blurred vision
            - Slow-healing cuts or frequent infections
            - Tingling or numbness in hands/feet
            - Areas of darkened skin (acanthosis nigricans)
            
            Note: Many people with Type 2 diabetes have no symptoms initially, 
            which is why regular screening is important.
            """)
    
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #5A6C7D; padding: 1rem;">
        <small>This application is for educational purposes only. Always consult healthcare professionals for medical advice.</small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
