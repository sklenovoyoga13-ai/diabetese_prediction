import pandas as pd
import io


FIELD_MAPPINGS = {
    'glucose': ['glucose', 'blood_glucose', 'fasting_glucose', 'blood glucose', 'fasting glucose', 'glu', 'bg'],
    'blood_pressure': ['blood_pressure', 'bp', 'diastolic', 'blood pressure', 'bp_diastolic', 'diastolic_bp'],
    'bmi': ['bmi', 'body_mass_index', 'body mass index'],
    'insulin': ['insulin', 'insulin_level', 'insulin level', 'serum_insulin', 'serum insulin'],
    'age': ['age', 'patient_age', 'patient age'],
    'weight': ['weight', 'wt', 'body_weight', 'body weight'],
    'height': ['height', 'ht', 'body_height', 'body height'],
    'skin_thickness': ['skin_thickness', 'skinfold', 'triceps', 'skin thickness', 'skinfold_thickness'],
    'pregnancies': ['pregnancies', 'pregnancy', 'num_pregnancies', 'pregnancy_count'],
    'cholesterol': ['cholesterol', 'total_cholesterol', 'total cholesterol', 'tc'],
    'hdl': ['hdl', 'hdl_cholesterol', 'hdl cholesterol', 'good_cholesterol'],
    'ldl': ['ldl', 'ldl_cholesterol', 'ldl cholesterol', 'bad_cholesterol'],
    'hba1c': ['hba1c', 'a1c', 'glycated_hemoglobin', 'glycated hemoglobin', 'hemoglobin_a1c']
}


def parse_csv_file(file_content: bytes) -> dict:
    try:
        df = pd.read_csv(io.BytesIO(file_content))
        
        df.columns = df.columns.str.lower().str.strip()
        
        extracted_data = {}
        
        for standard_field, possible_names in FIELD_MAPPINGS.items():
            for col in df.columns:
                if col in possible_names or any(name in col for name in possible_names):
                    value = df[col].iloc[-1] if len(df) > 0 else None
                    if pd.notna(value):
                        try:
                            extracted_data[standard_field] = float(value)
                        except (ValueError, TypeError):
                            pass
                    break
        
        if 'weight' in extracted_data and 'height' in extracted_data and 'bmi' not in extracted_data:
            weight_kg = extracted_data['weight']
            height_m = extracted_data['height'] / 100 if extracted_data['height'] > 3 else extracted_data['height']
            if height_m > 0:
                extracted_data['bmi'] = weight_kg / (height_m ** 2)
        
        return {
            'success': True,
            'data': extracted_data,
            'columns_found': list(df.columns),
            'rows_processed': len(df),
            'message': f"Successfully parsed {len(extracted_data)} health parameters from {len(df)} rows"
        }
        
    except pd.errors.EmptyDataError:
        return {
            'success': False,
            'data': {},
            'columns_found': [],
            'rows_processed': 0,
            'message': "The CSV file appears to be empty"
        }
    except Exception as e:
        return {
            'success': False,
            'data': {},
            'columns_found': [],
            'rows_processed': 0,
            'message': f"Error parsing CSV: {str(e)}"
        }


def convert_to_prediction_format(parsed_data: dict) -> dict:
    data = parsed_data.get('data', {})
    
    result = {
        'Glucose': data.get('glucose'),
        'BloodPressure': data.get('blood_pressure'),
        'BMI': data.get('bmi'),
        'Insulin': data.get('insulin'),
        'Age': int(data.get('age')) if data.get('age') else None,
        'SkinThickness': data.get('skin_thickness'),
        'Pregnancies': int(data.get('pregnancies', 0)) if data.get('pregnancies') is not None else 0,
        'DiabetesPedigreeFunction': 0.5
    }
    
    result = {k: v for k, v in result.items() if v is not None}
    
    return result


def get_sample_csv_template() -> str:
    return """date,glucose,blood_pressure,bmi,insulin,age,weight,height
2024-01-15,105,72,26.5,85,42,75,168
2024-02-15,102,70,26.2,80,42,74,168
2024-03-15,98,68,25.8,78,42,73,168"""


def validate_health_values(data: dict) -> list:
    warnings = []
    
    if 'glucose' in data:
        if data['glucose'] < 50 or data['glucose'] > 500:
            warnings.append(f"Glucose value ({data['glucose']} mg/dL) seems unusual. Normal fasting range is 70-100 mg/dL.")
    
    if 'blood_pressure' in data:
        if data['blood_pressure'] < 40 or data['blood_pressure'] > 150:
            warnings.append(f"Blood pressure ({data['blood_pressure']} mmHg) seems unusual. Normal diastolic range is 60-80 mmHg.")
    
    if 'bmi' in data:
        if data['bmi'] < 12 or data['bmi'] > 60:
            warnings.append(f"BMI ({data['bmi']:.1f}) seems unusual. Normal range is 18.5-24.9.")
    
    if 'insulin' in data:
        if data['insulin'] < 0 or data['insulin'] > 600:
            warnings.append(f"Insulin level ({data['insulin']} μU/mL) seems unusual. Normal fasting range is 2-25 μU/mL.")
    
    if 'age' in data:
        if data['age'] < 1 or data['age'] > 120:
            warnings.append(f"Age ({data['age']}) seems unusual.")
    
    return warnings
