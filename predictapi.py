import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

loaded_model = joblib.load('MLModels/stroke_prediction_model.pkl')

def predict_stroke(
    gender, age, hypertension, heart_disease, married, work_type,
    Residence_type, avg_glucose_level, bmi, smoking_status):

  patient_data = pd.DataFrame({
    'gender': [str(gender)], 
    'age': [int(age)], 
    'hypertension': [int(hypertension)], 
    'heart_disease': [int(heart_disease)], 
    'ever_married': [str(married)], 
    'work_type': [str(work_type)], 
    'Residence_type': [str(Residence_type)], 
    'avg_glucose_level': [float(avg_glucose_level)], 
    'bmi': [float(bmi)],
    'smoking_status': [str(smoking_status)]
  })
  
  # Load the original dataset and encode the categorical features
  original_data = pd.read_csv('MLModels/healthcare-dataset-stroke-data-ml-model.csv')
  categorical_columns = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
  label_encoders = {}
  
  for column in categorical_columns:
    label_encoder = LabelEncoder()
    label_encoder.fit(original_data[column])
    patient_data[column] = label_encoder.transform(patient_data[column])
    label_encoders[column] = label_encoder
  
  prediction = loaded_model.predict(patient_data)

  if prediction[0] == 1:
    return True
  else:
    return False    
