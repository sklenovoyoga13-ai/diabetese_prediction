import json
from datetime import datetime, timedelta
from database import PredictionHistory, HealthLog, get_db_session
from sqlalchemy import desc


def save_prediction(user_id: int, user_data: dict, prediction_result: dict, recommendations: dict = None) -> bool:
    db = get_db_session()
    if not db or not user_id:
        return False
    
    try:
        prediction = PredictionHistory(
            user_id=user_id,
            pregnancies=user_data.get('Pregnancies', 0),
            glucose=user_data.get('Glucose', 100),
            blood_pressure=user_data.get('BloodPressure', 70),
            skin_thickness=user_data.get('SkinThickness', 20),
            insulin=user_data.get('Insulin', 80),
            bmi=user_data.get('BMI', 25),
            diabetes_pedigree=user_data.get('DiabetesPedigreeFunction', 0.5),
            age=user_data.get('Age', 30),
            risk_probability=prediction_result.get('probability_diabetes', 0),
            risk_level=prediction_result.get('risk_level', 'Unknown'),
            recommendations=json.dumps(recommendations) if recommendations else None
        )
        db.add(prediction)
        db.commit()
        db.close()
        return True
    except Exception as e:
        print(f"Error saving prediction: {e}")
        db.rollback()
        db.close()
        return False


def get_user_predictions(user_id: int, limit: int = 50) -> list:
    db = get_db_session()
    if not db or not user_id:
        return []
    
    try:
        predictions = db.query(PredictionHistory).filter(
            PredictionHistory.user_id == user_id
        ).order_by(desc(PredictionHistory.created_at)).limit(limit).all()
        
        result = []
        for p in predictions:
            result.append({
                'id': p.id,
                'date': p.created_at,
                'glucose': p.glucose,
                'blood_pressure': p.blood_pressure,
                'bmi': p.bmi,
                'age': p.age,
                'insulin': p.insulin,
                'risk_probability': p.risk_probability,
                'risk_level': p.risk_level,
                'recommendations': json.loads(p.recommendations) if p.recommendations else None
            })
        db.close()
        return result
    except Exception as e:
        print(f"Error getting predictions: {e}")
        db.close()
        return []


def get_prediction_by_id(prediction_id: int, user_id: int) -> dict | None:
    db = get_db_session()
    if not db:
        return None
    
    try:
        prediction = db.query(PredictionHistory).filter(
            PredictionHistory.id == prediction_id,
            PredictionHistory.user_id == user_id
        ).first()
        
        if prediction:
            result = {
                'id': prediction.id,
                'date': prediction.created_at,
                'pregnancies': prediction.pregnancies,
                'glucose': prediction.glucose,
                'blood_pressure': prediction.blood_pressure,
                'skin_thickness': prediction.skin_thickness,
                'insulin': prediction.insulin,
                'bmi': prediction.bmi,
                'diabetes_pedigree': prediction.diabetes_pedigree,
                'age': prediction.age,
                'risk_probability': prediction.risk_probability,
                'risk_level': prediction.risk_level,
                'recommendations': json.loads(prediction.recommendations) if prediction.recommendations else None
            }
            db.close()
            return result
        db.close()
        return None
    except Exception as e:
        print(f"Error getting prediction: {e}")
        db.close()
        return None


def get_trend_data(user_id: int, days: int = 90) -> dict:
    db = get_db_session()
    if not db or not user_id:
        return {'dates': [], 'risk_scores': [], 'glucose': [], 'bmi': [], 'blood_pressure': []}
    
    try:
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        predictions = db.query(PredictionHistory).filter(
            PredictionHistory.user_id == user_id,
            PredictionHistory.created_at >= cutoff_date
        ).order_by(PredictionHistory.created_at).all()
        
        result = {
            'dates': [p.created_at for p in predictions],
            'risk_scores': [p.risk_probability * 100 for p in predictions],
            'glucose': [p.glucose for p in predictions],
            'bmi': [p.bmi for p in predictions],
            'blood_pressure': [p.blood_pressure for p in predictions]
        }
        db.close()
        return result
    except Exception as e:
        print(f"Error getting trend data: {e}")
        db.close()
        return {'dates': [], 'risk_scores': [], 'glucose': [], 'bmi': [], 'blood_pressure': []}


def save_health_log(user_id: int, log_type: str, **kwargs) -> bool:
    db = get_db_session()
    if not db or not user_id:
        return False
    
    try:
        log = HealthLog(
            user_id=user_id,
            log_type=log_type,
            weight=kwargs.get('weight'),
            height=kwargs.get('height'),
            bmi=kwargs.get('bmi'),
            calories_consumed=kwargs.get('calories_consumed'),
            calories_burned=kwargs.get('calories_burned'),
            exercise_minutes=kwargs.get('exercise_minutes'),
            exercise_type=kwargs.get('exercise_type'),
            notes=kwargs.get('notes')
        )
        db.add(log)
        db.commit()
        db.close()
        return True
    except Exception as e:
        print(f"Error saving health log: {e}")
        db.rollback()
        db.close()
        return False


def get_health_logs(user_id: int, log_type: str = None, days: int = 30) -> list:
    db = get_db_session()
    if not db or not user_id:
        return []
    
    try:
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        query = db.query(HealthLog).filter(
            HealthLog.user_id == user_id,
            HealthLog.created_at >= cutoff_date
        )
        
        if log_type:
            query = query.filter(HealthLog.log_type == log_type)
        
        logs = query.order_by(desc(HealthLog.created_at)).all()
        
        result = []
        for log in logs:
            result.append({
                'id': log.id,
                'date': log.created_at,
                'log_type': log.log_type,
                'weight': log.weight,
                'height': log.height,
                'bmi': log.bmi,
                'calories_consumed': log.calories_consumed,
                'calories_burned': log.calories_burned,
                'exercise_minutes': log.exercise_minutes,
                'exercise_type': log.exercise_type,
                'notes': log.notes
            })
        db.close()
        return result
    except Exception as e:
        print(f"Error getting health logs: {e}")
        db.close()
        return []


def get_stats_summary(user_id: int) -> dict:
    db = get_db_session()
    if not db or not user_id:
        return {}
    
    try:
        total_predictions = db.query(PredictionHistory).filter(
            PredictionHistory.user_id == user_id
        ).count()
        
        latest_prediction = db.query(PredictionHistory).filter(
            PredictionHistory.user_id == user_id
        ).order_by(desc(PredictionHistory.created_at)).first()
        
        first_prediction = db.query(PredictionHistory).filter(
            PredictionHistory.user_id == user_id
        ).order_by(PredictionHistory.created_at).first()
        
        result = {
            'total_predictions': total_predictions,
            'latest_risk': latest_prediction.risk_probability * 100 if latest_prediction else None,
            'latest_risk_level': latest_prediction.risk_level if latest_prediction else None,
            'latest_date': latest_prediction.created_at if latest_prediction else None,
            'first_risk': first_prediction.risk_probability * 100 if first_prediction else None,
            'first_date': first_prediction.created_at if first_prediction else None
        }
        
        if latest_prediction and first_prediction and total_predictions > 1:
            result['risk_change'] = result['latest_risk'] - result['first_risk']
        else:
            result['risk_change'] = None
        
        db.close()
        return result
    except Exception as e:
        print(f"Error getting stats: {e}")
        db.close()
        return {}
