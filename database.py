import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, ForeignKey, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship

DATABASE_URL = os.environ.get("DATABASE_URL")

engine = create_engine(DATABASE_URL) if DATABASE_URL else None
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine) if engine else None
Base = declarative_base()


class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(100), unique=True, index=True, nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=True)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    predictions = relationship("PredictionHistory", back_populates="user", cascade="all, delete-orphan")
    health_logs = relationship("HealthLog", back_populates="user", cascade="all, delete-orphan")


class PredictionHistory(Base):
    __tablename__ = "prediction_history"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    pregnancies = Column(Integer, default=0)
    glucose = Column(Float, nullable=False)
    blood_pressure = Column(Float, nullable=False)
    skin_thickness = Column(Float, default=0)
    insulin = Column(Float, default=0)
    bmi = Column(Float, nullable=False)
    diabetes_pedigree = Column(Float, default=0.5)
    age = Column(Integer, nullable=False)
    
    risk_probability = Column(Float, nullable=False)
    risk_level = Column(String(20), nullable=False)
    
    recommendations = Column(Text, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    user = relationship("User", back_populates="predictions")


class HealthLog(Base):
    __tablename__ = "health_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    
    log_type = Column(String(50), nullable=False)
    weight = Column(Float, nullable=True)
    height = Column(Float, nullable=True)
    bmi = Column(Float, nullable=True)
    calories_consumed = Column(Integer, nullable=True)
    calories_burned = Column(Integer, nullable=True)
    exercise_minutes = Column(Integer, nullable=True)
    exercise_type = Column(String(100), nullable=True)
    notes = Column(Text, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    user = relationship("User", back_populates="health_logs")


def init_db():
    if engine:
        Base.metadata.create_all(bind=engine)


def get_db():
    if SessionLocal:
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()
    else:
        yield None


def get_db_session():
    if SessionLocal:
        return SessionLocal()
    return None
