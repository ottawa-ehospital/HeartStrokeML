from pydantic import BaseModel

class StrokePayload(BaseModel):
  gender : str
  age : int
  hypertension : int
  heart_disease : int
  ever_married : str
  work_type : str
  Residence_type : str
  avg_glucose_level : float
  bmi : float
  smoking_status : str
