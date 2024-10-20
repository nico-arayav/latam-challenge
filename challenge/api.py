import fastapi
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, validator
from typing import List
import pandas as pd
from challenge.model import DelayModel
import joblib

app = fastapi.FastAPI()

# Initialize the model
model = DelayModel()
model._model = joblib.load('model.pkl')  # Load the model from file

VALID_AIRLINES = [
    "Grupo LATAM",
    "Sky Airline",
    "Aerolineas Argentinas",
    "Copa Air",
    "Latin American Wings",
    "Avianca",
    "JetSmart SPA",
    "Gol Trans",
    "American Airlines",
    "Air Canada",
    "Iberia",
    "Delta Air",
    "Air France",
    "Aeromexico",
    "United Airlines",
    "Oceanair Linhas Aereas",
    "Alitalia",
    "K.L.M.",
    "British Airways",
    "Qantas Airways",
    "Lacsa",
    "Austral",
    "Plus Ultra Lineas Aereas"
]

class FlightData(BaseModel):
    OPERA: str
    TIPOVUELO: str
    MES: int

    @validator('OPERA')
    def validate_opera(cls, v):
        if v not in VALID_AIRLINES:
            raise ValueError('Invalid OPERA')
        return v

    @validator('TIPOVUELO')
    def validate_tipovuelo(cls, v):
        if v not in ['N', 'I']:
            raise ValueError('Invalid TIPOVUELO')
        return v

    @validator('MES')
    def validate_mes(cls, v):
        if v < 1 or v > 12:
            raise ValueError('Invalid MES')
        return v

class PredictRequest(BaseModel):
    flights: List[FlightData]

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=400,
        content={"detail": exc.errors()},
    )

@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }

@app.post("/predict", status_code=200)
async def post_predict(request: PredictRequest) -> dict:
    try:
        # Convert request data to DataFrame
        data = pd.DataFrame([flight.dict() for flight in request.flights])
        
        # Preprocess the data
        features = model.preprocess(data)
        
        # Make predictions
        predictions = model.predict(features)
        
        return {"predict": predictions}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))