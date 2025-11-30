import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle
import os

class DiabetesPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = [
            'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
        ]
        self._train_model()
    
    def _generate_training_data(self):
        np.random.seed(42)
        n_samples = 768
        
        diabetic_samples = int(n_samples * 0.35)
        non_diabetic_samples = n_samples - diabetic_samples
        
        non_diabetic_data = {
            'Pregnancies': np.random.randint(0, 6, non_diabetic_samples),
            'Glucose': np.random.normal(100, 15, non_diabetic_samples).clip(70, 140),
            'BloodPressure': np.random.normal(70, 10, non_diabetic_samples).clip(50, 90),
            'SkinThickness': np.random.normal(20, 8, non_diabetic_samples).clip(0, 50),
            'Insulin': np.random.normal(80, 40, non_diabetic_samples).clip(0, 200),
            'BMI': np.random.normal(25, 4, non_diabetic_samples).clip(18, 35),
            'DiabetesPedigreeFunction': np.random.exponential(0.3, non_diabetic_samples).clip(0.08, 1.0),
            'Age': np.random.randint(21, 50, non_diabetic_samples),
            'Outcome': np.zeros(non_diabetic_samples)
        }
        
        diabetic_data = {
            'Pregnancies': np.random.randint(2, 12, diabetic_samples),
            'Glucose': np.random.normal(155, 30, diabetic_samples).clip(100, 200),
            'BloodPressure': np.random.normal(78, 12, diabetic_samples).clip(60, 110),
            'SkinThickness': np.random.normal(32, 10, diabetic_samples).clip(10, 60),
            'Insulin': np.random.normal(180, 80, diabetic_samples).clip(50, 400),
            'BMI': np.random.normal(34, 6, diabetic_samples).clip(25, 50),
            'DiabetesPedigreeFunction': np.random.exponential(0.5, diabetic_samples).clip(0.1, 2.0),
            'Age': np.random.randint(30, 70, diabetic_samples),
            'Outcome': np.ones(diabetic_samples)
        }
        
        non_diabetic_df = pd.DataFrame(non_diabetic_data)
        diabetic_df = pd.DataFrame(diabetic_data)
        
        df = pd.concat([non_diabetic_df, diabetic_df], ignore_index=True)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        return df
    
    def _train_model(self):
        df = self._generate_training_data()
        
        X = df[self.feature_names]
        y = df['Outcome']
        
        X_scaled = self.scaler.fit_transform(X)
        
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        self.model.fit(X_scaled, y)
    
    def predict(self, features):
        if isinstance(features, dict):
            features_array = np.array([[
                features.get('Pregnancies', 0),
                features.get('Glucose', 100),
                features.get('BloodPressure', 70),
                features.get('SkinThickness', 20),
                features.get('Insulin', 80),
                features.get('BMI', 25),
                features.get('DiabetesPedigreeFunction', 0.5),
                features.get('Age', 30)
            ]])
        else:
            features_array = np.array(features).reshape(1, -1)
        
        features_scaled = self.scaler.transform(features_array)
        
        prediction = self.model.predict(features_scaled)[0]
        probability = self.model.predict_proba(features_scaled)[0]
        
        return {
            'prediction': int(prediction),
            'probability_no_diabetes': float(probability[0]),
            'probability_diabetes': float(probability[1]),
            'risk_level': self._get_risk_level(probability[1])
        }
    
    def _get_risk_level(self, probability):
        if probability < 0.3:
            return 'Low'
        elif probability < 0.5:
            return 'Moderate'
        elif probability < 0.7:
            return 'High'
        else:
            return 'Very High'
    
    def get_feature_importance(self):
        importance = self.model.feature_importances_
        return dict(zip(self.feature_names, importance))
    
    def get_risk_factors(self, features):
        importance = self.get_feature_importance()
        
        risk_factors = []
        
        if features.get('Glucose', 100) > 140:
            risk_factors.append({
                'factor': 'High Glucose Level',
                'value': features['Glucose'],
                'normal_range': '70-140 mg/dL',
                'importance': importance['Glucose'],
                'severity': 'high'
            })
        elif features.get('Glucose', 100) > 100:
            risk_factors.append({
                'factor': 'Elevated Glucose Level',
                'value': features['Glucose'],
                'normal_range': '70-100 mg/dL (fasting)',
                'importance': importance['Glucose'],
                'severity': 'moderate'
            })
        
        if features.get('BMI', 25) > 30:
            risk_factors.append({
                'factor': 'Obesity (High BMI)',
                'value': features['BMI'],
                'normal_range': '18.5-24.9',
                'importance': importance['BMI'],
                'severity': 'high'
            })
        elif features.get('BMI', 25) > 25:
            risk_factors.append({
                'factor': 'Overweight (Elevated BMI)',
                'value': features['BMI'],
                'normal_range': '18.5-24.9',
                'importance': importance['BMI'],
                'severity': 'moderate'
            })
        
        if features.get('BloodPressure', 70) > 90:
            risk_factors.append({
                'factor': 'High Blood Pressure',
                'value': features['BloodPressure'],
                'normal_range': '60-80 mmHg (diastolic)',
                'importance': importance['BloodPressure'],
                'severity': 'high'
            })
        
        if features.get('Age', 30) > 45:
            risk_factors.append({
                'factor': 'Age Factor',
                'value': features['Age'],
                'normal_range': 'Risk increases after 45',
                'importance': importance['Age'],
                'severity': 'moderate'
            })
        
        if features.get('DiabetesPedigreeFunction', 0.5) > 0.8:
            risk_factors.append({
                'factor': 'Strong Family History',
                'value': round(features['DiabetesPedigreeFunction'], 2),
                'normal_range': '< 0.5',
                'importance': importance['DiabetesPedigreeFunction'],
                'severity': 'high'
            })
        elif features.get('DiabetesPedigreeFunction', 0.5) > 0.5:
            risk_factors.append({
                'factor': 'Family History Present',
                'value': round(features['DiabetesPedigreeFunction'], 2),
                'normal_range': '< 0.5',
                'importance': importance['DiabetesPedigreeFunction'],
                'severity': 'moderate'
            })
        
        if features.get('Insulin', 80) > 166:
            risk_factors.append({
                'factor': 'High Insulin Level',
                'value': features['Insulin'],
                'normal_range': '16-166 μU/mL',
                'importance': importance['Insulin'],
                'severity': 'moderate'
            })
        
        risk_factors.sort(key=lambda x: x['importance'], reverse=True)
        
        return risk_factors


def get_predictor():
    if not hasattr(get_predictor, 'instance'):
        get_predictor.instance = DiabetesPredictor()
    return get_predictor.instance
