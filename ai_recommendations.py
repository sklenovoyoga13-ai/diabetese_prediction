import os
import json
from openai import OpenAI

# the newest OpenAI model is "gpt-5" which was released August 7, 2025.
# do not change this unless explicitly requested by the user

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

def get_openai_client():
    if not OPENAI_API_KEY:
        return None
    return OpenAI(api_key=OPENAI_API_KEY)

def generate_health_recommendations(user_data, prediction_result, risk_factors):
    client = get_openai_client()
    
    if not client:
        return get_fallback_recommendations(user_data, prediction_result, risk_factors)
    
    prompt = f"""You are a health advisor providing personalized recommendations for diabetes prevention and management.

Based on the following health assessment, provide detailed, actionable recommendations:

**Patient Profile:**
- Age: {user_data.get('Age', 'N/A')} years
- BMI: {user_data.get('BMI', 'N/A')}
- Blood Glucose Level: {user_data.get('Glucose', 'N/A')} mg/dL
- Blood Pressure (Diastolic): {user_data.get('BloodPressure', 'N/A')} mmHg
- Insulin Level: {user_data.get('Insulin', 'N/A')} μU/mL
- Family History Score: {user_data.get('DiabetesPedigreeFunction', 'N/A')}

**Risk Assessment:**
- Diabetes Risk Level: {prediction_result['risk_level']}
- Probability of Diabetes: {round(prediction_result['probability_diabetes'] * 100, 1)}%

**Identified Risk Factors:**
{json.dumps(risk_factors, indent=2) if risk_factors else 'None identified'}

Please provide recommendations in the following JSON format:
{{
    "summary": "A brief 2-3 sentence summary of the overall health status and main concerns",
    "diet_recommendations": [
        {{"title": "Recommendation title", "description": "Detailed description", "priority": "high/medium/low"}}
    ],
    "exercise_recommendations": [
        {{"title": "Recommendation title", "description": "Detailed description", "priority": "high/medium/low"}}
    ],
    "lifestyle_recommendations": [
        {{"title": "Recommendation title", "description": "Detailed description", "priority": "high/medium/low"}}
    ],
    "medical_advice": [
        {{"title": "Recommendation title", "description": "Detailed description", "priority": "high/medium/low"}}
    ],
    "warning_signs": ["List of symptoms to watch for"],
    "positive_factors": ["List of positive health indicators if any"]
}}

Provide 3-4 recommendations per category. Be specific and actionable. Consider the individual's specific risk factors."""

    try:
        response = client.chat.completions.create(
            model="gpt-5",
            messages=[
                {
                    "role": "system",
                    "content": "You are a knowledgeable health advisor specializing in diabetes prevention and metabolic health. Provide evidence-based, personalized recommendations. Always remind users to consult healthcare professionals for medical decisions."
                },
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            max_completion_tokens=2048
        )
        
        result = json.loads(response.choices[0].message.content)
        return result
    except Exception as e:
        print(f"OpenAI API error: {e}")
        return get_fallback_recommendations(user_data, prediction_result, risk_factors)


def get_fallback_recommendations(user_data, prediction_result, risk_factors):
    risk_level = prediction_result['risk_level']
    bmi = user_data.get('BMI', 25)
    glucose = user_data.get('Glucose', 100)
    age = user_data.get('Age', 30)
    
    recommendations = {
        "summary": f"Based on your health profile, you have a {risk_level.lower()} risk of developing diabetes. ",
        "diet_recommendations": [],
        "exercise_recommendations": [],
        "lifestyle_recommendations": [],
        "medical_advice": [],
        "warning_signs": [
            "Increased thirst and frequent urination",
            "Unexplained weight loss",
            "Fatigue and weakness",
            "Blurred vision",
            "Slow-healing cuts or frequent infections"
        ],
        "positive_factors": []
    }
    
    if risk_level in ['High', 'Very High']:
        recommendations["summary"] += "It's important to take immediate action to reduce your risk factors and consult with a healthcare provider."
    elif risk_level == 'Moderate':
        recommendations["summary"] += "With some lifestyle modifications, you can significantly reduce your risk."
    else:
        recommendations["summary"] += "Continue maintaining your healthy habits to keep your risk low."
    
    if bmi > 25:
        recommendations["diet_recommendations"].append({
            "title": "Calorie Control",
            "description": "Aim to reduce daily caloric intake by 500 calories to achieve gradual weight loss of 1-2 pounds per week.",
            "priority": "high"
        })
    
    recommendations["diet_recommendations"].extend([
        {
            "title": "Increase Fiber Intake",
            "description": "Consume 25-30 grams of fiber daily from vegetables, whole grains, and legumes to help control blood sugar.",
            "priority": "high"
        },
        {
            "title": "Choose Complex Carbohydrates",
            "description": "Replace refined carbs with whole grains like brown rice, quinoa, and whole wheat bread.",
            "priority": "medium"
        },
        {
            "title": "Limit Sugary Beverages",
            "description": "Replace sodas and fruit juices with water, unsweetened tea, or sparkling water with lemon.",
            "priority": "high"
        }
    ])
    
    recommendations["exercise_recommendations"].extend([
        {
            "title": "Regular Aerobic Exercise",
            "description": "Aim for 150 minutes of moderate-intensity exercise per week, such as brisk walking, swimming, or cycling.",
            "priority": "high"
        },
        {
            "title": "Strength Training",
            "description": "Include resistance exercises 2-3 times per week to improve insulin sensitivity and build muscle mass.",
            "priority": "medium"
        },
        {
            "title": "Daily Movement",
            "description": "Take short walks after meals (10-15 minutes) to help regulate post-meal blood sugar levels.",
            "priority": "medium"
        }
    ])
    
    recommendations["lifestyle_recommendations"].extend([
        {
            "title": "Quality Sleep",
            "description": "Aim for 7-9 hours of quality sleep per night. Poor sleep can affect insulin sensitivity.",
            "priority": "high"
        },
        {
            "title": "Stress Management",
            "description": "Practice stress-reduction techniques like meditation, deep breathing, or yoga, as stress can elevate blood sugar.",
            "priority": "medium"
        },
        {
            "title": "Regular Monitoring",
            "description": "Keep track of your weight, blood pressure, and if possible, blood glucose levels regularly.",
            "priority": "medium"
        }
    ])
    
    if risk_level in ['High', 'Very High']:
        recommendations["medical_advice"].extend([
            {
                "title": "Schedule a Doctor's Appointment",
                "description": "Consult with a healthcare provider for a comprehensive diabetes screening and personalized medical advice.",
                "priority": "high"
            },
            {
                "title": "Request HbA1c Test",
                "description": "Ask your doctor about an HbA1c test, which shows average blood sugar levels over the past 2-3 months.",
                "priority": "high"
            }
        ])
    else:
        recommendations["medical_advice"].append({
            "title": "Annual Health Check-up",
            "description": "Schedule an annual physical examination including blood glucose and lipid panel tests.",
            "priority": "medium"
        })
    
    recommendations["medical_advice"].append({
        "title": "Know Your Numbers",
        "description": "Keep track of key health metrics: blood pressure (< 120/80 mmHg), fasting glucose (< 100 mg/dL), and BMI (18.5-24.9).",
        "priority": "medium"
    })
    
    if bmi < 25:
        recommendations["positive_factors"].append("Healthy BMI range")
    if glucose < 100:
        recommendations["positive_factors"].append("Normal fasting glucose level")
    if age < 45:
        recommendations["positive_factors"].append("Age is a protective factor")
    if user_data.get('BloodPressure', 80) < 80:
        recommendations["positive_factors"].append("Healthy blood pressure")
    
    return recommendations
