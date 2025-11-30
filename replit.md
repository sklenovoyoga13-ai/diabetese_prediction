# Diabetes Risk Prediction & Health Recommendation System

## Overview
An AI-powered web application for predicting diabetes risk and providing personalized health recommendations. Built for college project purposes using machine learning and natural language processing.

## Current State
- **Status**: Fully functional MVP
- **Last Updated**: November 30, 2025

## Features
1. **Diabetes Risk Prediction**: Uses Random Forest classifier trained on diabetes health data
2. **Interactive Input Form**: Collects health parameters (age, BMI, glucose, blood pressure, etc.)
3. **Risk Visualization**: Gauge charts, probability indicators, and comparison charts
4. **Personalized Recommendations**: AI-powered health advice using OpenAI GPT-5 (with fallback)
5. **Educational Dashboard**: Information about diabetes risk factors and prevention

## Project Structure
```
├── app.py                    # Main Streamlit application
├── model.py                  # Diabetes prediction ML model
├── ai_recommendations.py     # OpenAI-powered health recommendations
├── .streamlit/config.toml    # Streamlit configuration
└── replit.md                 # Project documentation
```

## Technical Stack
- **Frontend**: Streamlit
- **ML Model**: scikit-learn (Random Forest Classifier)
- **Visualization**: Plotly
- **AI Recommendations**: OpenAI GPT-5
- **Data Processing**: pandas, numpy

## Key Health Parameters
- Age
- BMI (calculated from height/weight)
- Blood Glucose Level
- Blood Pressure (Diastolic)
- Insulin Level
- Family History (Diabetes Pedigree Function)
- Pregnancies
- Skin Fold Thickness

## How to Run
```bash
streamlit run app.py --server.port 5000
```

## Environment Variables
- `OPENAI_API_KEY`: Required for AI-powered recommendations (optional - fallback available)

## Recent Changes
- Initial MVP implementation with ML model and health recommendations
- Added interactive visualizations for risk assessment
- Implemented educational resources section
