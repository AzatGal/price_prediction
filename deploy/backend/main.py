from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel
import yaml
import os
import sys

# Добавляем текущую директорию в путь для импортов
sys.path.append(os.path.dirname(__file__))

from database import (
    get_db, DatabaseManager, PredictionCreate, PredictionUpdate,
    PredictionResponse, User, SessionLocal
)
from model import predictor
from auth import (
    create_user, authenticate_user, create_access_token,
    get_current_active_user, ACCESS_TOKEN_EXPIRE_MINUTES
)

app = FastAPI(title="Real Estate Predictor API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Загрузка конфига
config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
config = yaml.safe_load(open(config_path, encoding='utf-8'))


# Модели Pydantic
class UserCreate(BaseModel):
    username: str
    password: str


class UserLogin(BaseModel):
    username: str
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str


class PredictionRequest(BaseModel):
    features: dict


class PredictionResult(BaseModel):
    predicted_price: float
    confidence_low: float
    confidence_high: float
    uncertainty_percent: float
    price_per_m2: float


@app.get("/")
def root():
    return {"message": "API is running", "status": "ok"}


@app.post("/auth/register", response_model=Token)
def register(user: UserCreate, db: Session = Depends(get_db)):
    if not user.username or not user.password:
        raise HTTPException(status_code=400, detail="Username and password required")

    db_user = db.query(User).filter(User.username == user.username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username already registered")

    new_user = create_user(db, user.username, user.password)
    access_token = create_access_token(
        data={"sub": new_user.username},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    return {"access_token": access_token, "token_type": "bearer"}


@app.post("/auth/login", response_model=Token)
def login(user: UserLogin, db: Session = Depends(get_db)):
    auth_user = authenticate_user(db, user.username, user.password)
    if not auth_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token(
        data={"sub": auth_user.username},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/config")
def get_config():
    return config['features']


@app.post("/predict", response_model=PredictionResult)
def predict(request: PredictionRequest, current_user: User = Depends(get_current_active_user),
            db: Session = Depends(get_db)):
    # Валидация
    for name, cfg in config['features'].items():
        if name not in request.features:
            if 'default' in cfg:
                request.features[name] = cfg['default']
            else:
                raise HTTPException(status_code=400, detail=f"Missing field: {name}")

    result = predictor.predict(request.features)

    pred_data = PredictionCreate(
        features=request.features,
        predicted_price=result['predicted_price'],
        confidence_low=result['confidence_low'],
        confidence_high=result['confidence_high']
    )
    DatabaseManager.create_prediction(db, current_user.id, pred_data)

    return result


@app.get("/predictions", response_model=List[PredictionResponse])
def get_predictions(
        skip: int = 0,
        limit: int = 100,
        current_user: User = Depends(get_current_active_user),
        db: Session = Depends(get_db)
):
    return DatabaseManager.get_predictions(db, current_user.id, skip, limit)


@app.delete("/predictions/{pred_id}")
def delete_prediction(pred_id: int, current_user: User = Depends(get_current_active_user),
                      db: Session = Depends(get_db)):
    success = DatabaseManager.delete_prediction(db, pred_id, current_user.id)
    if not success:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return {"message": "Deleted"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)