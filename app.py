import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from model import get_predictor
from ai_recommendations import generate_health_recommendations
import os

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
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
        border-left: 4px solid #667eea;
    }
    .priority-high { border-left-color: #eb3349; }
    .priority-medium { border-left-color: #F2994A; }
    .priority-low { border-left-color: #11998e; }
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
    
    categories = list(normal_ranges.keys())
    
    for i, (cat, vals) in enumerate(normal_ranges.items()):
        range_mid = (vals['max'] + vals['min']) / 2
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

def display_recommendations(recommendations):
    if not recommendations:
        st.warning("Unable to generate recommendations. Please try again.")
        return
    
    st.markdown("### Summary")
    st.info(recommendations.get('summary', 'No summary available.'))
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Diet Recommendations")
        for rec in recommendations.get('diet_recommendations', []):
            priority_class = f"priority-{rec.get('priority', 'medium')}"
            st.markdown(f"""
            <div class="recommendation-card {priority_class}">
                <strong>{rec.get('title', 'Recommendation')}</strong><br>
                <small>{rec.get('description', '')}</small>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("#### Lifestyle Recommendations")
        for rec in recommendations.get('lifestyle_recommendations', []):
            priority_class = f"priority-{rec.get('priority', 'medium')}"
            st.markdown(f"""
            <div class="recommendation-card {priority_class}">
                <strong>{rec.get('title', 'Recommendation')}</strong><br>
                <small>{rec.get('description', '')}</small>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("#### Exercise Recommendations")
        for rec in recommendations.get('exercise_recommendations', []):
            priority_class = f"priority-{rec.get('priority', 'medium')}"
            st.markdown(f"""
            <div class="recommendation-card {priority_class}">
                <strong>{rec.get('title', 'Recommendation')}</strong><br>
                <small>{rec.get('description', '')}</small>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("#### Medical Advice")
        for rec in recommendations.get('medical_advice', []):
            priority_class = f"priority-{rec.get('priority', 'medium')}"
            st.markdown(f"""
            <div class="recommendation-card {priority_class}">
                <strong>{rec.get('title', 'Recommendation')}</strong><br>
                <small>{rec.get('description', '')}</small>
            </div>
            """, unsafe_allow_html=True)
    
    col3, col4 = st.columns(2)
    
    with col3:
        if recommendations.get('warning_signs'):
            st.markdown("#### Warning Signs to Watch")
            st.markdown("""<div class="warning-box">""", unsafe_allow_html=True)
            for sign in recommendations['warning_signs']:
                st.markdown(f"- {sign}")
            st.markdown("</div>", unsafe_allow_html=True)
    
    with col4:
        if recommendations.get('positive_factors'):
            st.markdown("#### Positive Health Factors")
            st.markdown("""<div class="info-box">""", unsafe_allow_html=True)
            for factor in recommendations['positive_factors']:
                st.markdown(f"- {factor}")
            st.markdown("</div>", unsafe_allow_html=True)

def main():
    st.markdown('<h1 class="main-header">Diabetes Risk Prediction System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Health Assessment & Personalized Recommendations</p>', unsafe_allow_html=True)
    
    predictor = get_predictor()
    
    with st.sidebar:
        st.markdown("## Health Parameters")
        st.markdown("Enter your health information below for a personalized risk assessment.")
        
        st.markdown("---")
        
        st.markdown("### Personal Information")
        age = st.slider("Age (years)", min_value=18, max_value=100, value=35, help="Your current age")
        pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=0, 
                                       help="Number of times pregnant (enter 0 if not applicable)")
        
        st.markdown("### Body Measurements")
        height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
        weight = st.number_input("Weight (kg)", min_value=30, max_value=250, value=70)
        bmi = weight / ((height/100) ** 2)
        st.metric("Calculated BMI", f"{bmi:.1f}")
        
        skin_thickness = st.slider("Skin Fold Thickness (mm)", min_value=0, max_value=100, value=20,
                                    help="Triceps skin fold thickness in mm")
        
        st.markdown("### Blood Tests")
        glucose = st.slider("Blood Glucose (mg/dL)", min_value=50, max_value=300, value=100,
                            help="Fasting blood glucose level")
        blood_pressure = st.slider("Blood Pressure - Diastolic (mmHg)", min_value=40, max_value=140, value=70,
                                    help="Diastolic blood pressure")
        insulin = st.slider("Insulin Level (μU/mL)", min_value=0, max_value=500, value=80,
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
    
    tab1, tab2, tab3 = st.tabs(["Risk Assessment", "Health Recommendations", "Educational Resources"])
    
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
            
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                st.plotly_chart(create_gauge_chart(result['probability_diabetes']), use_container_width=True)
            
            with col2:
                risk_level = result['risk_level']
                risk_colors = {
                    'Low': '#38ef7d',
                    'Moderate': '#F2C94C',
                    'High': '#f45c43',
                    'Very High': '#8E2DE2'
                }
                
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, {risk_colors[risk_level]}80 0%, {risk_colors[risk_level]} 100%); 
                            padding: 2rem; border-radius: 1rem; text-align: center; color: white; height: 200px;
                            display: flex; flex-direction: column; justify-content: center;">
                    <h2 style="margin: 0; font-size: 1.5rem;">Risk Level</h2>
                    <h1 style="margin: 0.5rem 0; font-size: 2.5rem;">{risk_level}</h1>
                    <p style="margin: 0;">Based on your health parameters</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
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
                
                st.markdown(f"""
                <div style="background: #f8f9fa; padding: 1rem; border-radius: 1rem;">
                    <h4 style="color: #1E3A5F; margin-bottom: 0.5rem; text-align: center;">Your Health Insights</h4>
                    <div style="display: flex; flex-direction: column; gap: 0.5rem;">
                        <div style="background: white; padding: 0.5rem; border-radius: 0.5rem; border-left: 3px solid {bmi_color};">
                            <small style="color: #5A6C7D;">BMI Status</small><br>
                            <strong style="color: {bmi_color};">{bmi_category}</strong> ({bmi_val:.1f})
                        </div>
                        <div style="background: white; padding: 0.5rem; border-radius: 0.5rem; border-left: 3px solid {glucose_color};">
                            <small style="color: #5A6C7D;">Glucose Status</small><br>
                            <strong style="color: {glucose_color};">{glucose_status}</strong> ({glucose_val} mg/dL)
                        </div>
                        <div style="background: white; padding: 0.5rem; border-radius: 0.5rem; border-left: 3px solid #667eea;">
                            <small style="color: #5A6C7D;">Risk Score</small><br>
                            <strong style="color: #1E3A5F;">{result['probability_diabetes']*100:.0f}/100</strong>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            col4, col5 = st.columns(2)
            
            with col4:
                importance = predictor.get_feature_importance()
                st.plotly_chart(create_feature_importance_chart(importance), use_container_width=True)
            
            with col5:
                st.plotly_chart(create_risk_comparison_chart(user_data), use_container_width=True)
            
            risk_factors = predictor.get_risk_factors(user_data)
            if risk_factors:
                st.markdown("### Identified Risk Factors")
                
                cols = st.columns(len(risk_factors) if len(risk_factors) <= 4 else 4)
                for i, factor in enumerate(risk_factors[:4]):
                    with cols[i % 4]:
                        severity_colors = {
                            'high': '#f45c43',
                            'moderate': '#F2C94C',
                            'low': '#38ef7d'
                        }
                        color = severity_colors.get(factor['severity'], '#667eea')
                        st.markdown(f"""
                        <div style="background: white; padding: 1rem; border-radius: 0.5rem; 
                                    border-left: 4px solid {color}; margin-bottom: 1rem;
                                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                            <strong style="color: #1E3A5F;">{factor['factor']}</strong><br>
                            <span style="font-size: 1.5rem; color: {color};">{factor['value']}</span><br>
                            <small style="color: #5A6C7D;">Normal: {factor['normal_range']}</small>
                        </div>
                        """, unsafe_allow_html=True)
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
            
            display_recommendations(recommendations)
            
            st.markdown("---")
            st.markdown("""
            <div class="warning-box">
                <strong>Important Disclaimer:</strong> These recommendations are for educational purposes only 
                and should not replace professional medical advice. Always consult with a healthcare provider 
                before making significant changes to your diet, exercise routine, or medication.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Please complete the risk assessment first to receive personalized recommendations.")
    
    with tab3:
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
            st.plotly_chart(fig_risk, use_container_width=True)
        
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
            st.plotly_chart(fig_prevention, use_container_width=True)
        
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
            st.plotly_chart(fig_glucose, use_container_width=True)
        
        st.markdown("---")
        
        st.markdown("### How Our Prediction Model Works")
        
        importance = predictor.get_feature_importance()
        
        model_col1, model_col2 = st.columns(2)
        
        with model_col1:
            fig = create_feature_importance_chart(importance)
            st.plotly_chart(fig, use_container_width=True)
        
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
