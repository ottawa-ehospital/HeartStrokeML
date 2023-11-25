from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from models import StrokePayload
import predictapi as MLApi

app = FastAPI()

origins = [
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def index():
  return {
    "name": "Heart Stroke Prediction", 
    "condition": "OK",
    "model_version": "0.1.1"
  }

@app.post("/predict")
async def predict_stroke(payload : StrokePayload):
  result = MLApi.predict_stroke(
    gender=payload.gender,
    age=payload.age,
    hypertension=payload.hypertension,
    heart_disease=payload.heart_disease,
    married=payload.ever_married,
    work_type=payload.work_type,
    Residence_type=payload.Residence_type,
    avg_glucose_level=payload.avg_glucose_level,
    bmi=payload.bmi,
    smoking_status=payload.smoking_status,
  )

  if result:
    response = {"prediction": "Positive"}
  else:
    response = {"prediction": "Negative"}
  return response
