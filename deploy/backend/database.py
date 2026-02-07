from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel
import json

Base = declarative_base()
engine = create_engine("sqlite:///./real_estate.db", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# Модели БД
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    features = Column(JSON)  # Храним входные данные
    predicted_price = Column(Float)
    confidence_low = Column(Float)
    confidence_high = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


# Создание таблиц
Base.metadata.create_all(bind=engine)


# Pydantic модели
class PredictionCreate(BaseModel):
    features: dict
    predicted_price: float
    confidence_low: float
    confidence_high: float


class PredictionUpdate(BaseModel):
    features: Optional[dict] = None


class PredictionResponse(BaseModel):
    id: int
    features: dict
    predicted_price: float
    confidence_low: float
    confidence_high: float
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class DatabaseManager:
    @staticmethod
    def create_prediction(db: Session, user_id: int, data: PredictionCreate):
        db_pred = Prediction(
            user_id=user_id,
            features=data.features,
            predicted_price=data.predicted_price,
            confidence_low=data.confidence_low,
            confidence_high=data.confidence_high
        )
        db.add(db_pred)
        db.commit()
        db.refresh(db_pred)
        return db_pred

    @staticmethod
    def get_predictions(db: Session, user_id: int, skip: int = 0, limit: int = 100):
        return db.query(Prediction).filter(Prediction.user_id == user_id) \
            .order_by(Prediction.created_at.desc()).offset(skip).limit(limit).all()

    @staticmethod
    def get_prediction(db: Session, pred_id: int, user_id: int):
        return db.query(Prediction).filter(
            Prediction.id == pred_id,
            Prediction.user_id == user_id
        ).first()

    @staticmethod
    def update_prediction(db: Session, pred_id: int, user_id: int, data: PredictionUpdate):
        pred = DatabaseManager.get_prediction(db, pred_id, user_id)
        if not pred:
            return None

        if data.features:
            pred.features = data.features
            pred.updated_at = datetime.utcnow()

        db.commit()
        db.refresh(pred)
        return pred

    @staticmethod
    def delete_prediction(db: Session, pred_id: int, user_id: int):
        pred = DatabaseManager.get_prediction(db, pred_id, user_id)
        if not pred:
            return False

        db.delete(pred)
        db.commit()
        return True

    @staticmethod
    def filter_predictions(db: Session, user_id: int, **filters):
        query = db.query(Prediction).filter(Prediction.user_id == user_id)

        if 'date_from' in filters:
            query = query.filter(Prediction.created_at >= filters['date_from'])
        if 'date_to' in filters:
            query = query.filter(Prediction.created_at <= filters['date_to'])
        if 'district' in filters:
            query = query.filter(Prediction.features['district'].astext == filters['district'])
        if 'house_type' in filters:
            query = query.filter(Prediction.features['house_type'].astext == filters['house_type'])
        if 'min_price' in filters:
            query = query.filter(Prediction.predicted_price >= filters['min_price'])
        if 'max_price' in filters:
            query = query.filter(Prediction.predicted_price <= filters['max_price'])

        return query.order_by(Prediction.created_at.desc()).all()