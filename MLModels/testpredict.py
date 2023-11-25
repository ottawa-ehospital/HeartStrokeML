import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import joblib

# Load the saved model
loaded_model = joblib.load('stroke_prediction_model.pkl')

# Prepare input data for a single prediction
# You can create a single-row DataFrame or use a dictionary
single_data = pd.DataFrame({
    'gender': ['Male'], 
    'age': [79], 
    'hypertension': [0], 
    'heart_disease': [0], 
    'ever_married': ['Yes'], 
    'work_type': ['Private'], 
    'Residence_type': ['Rural'], 
    'avg_glucose_level': [72.73], 
    'bmi': [28.4],
    'smoking_status': ['never smoked']
})

# Load the original dataset and encode the categorical features
original_data = pd.read_csv('healthcare-dataset-stroke-data-ml-model.csv')
categorical_columns = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
label_encoders = {}

for column in categorical_columns:
    label_encoder = LabelEncoder()
    label_encoder.fit(original_data[column])
    single_data[column] = label_encoder.transform(single_data[column])
    label_encoders[column] = label_encoder

# Make a single prediction
single_prediction = loaded_model.predict(single_data)

# Print the prediction
print(single_prediction[0])
if single_prediction[0] == 1:
    print("The individual is predicted to have a stroke.")
else:
    print("The individual is predicted to not have a stroke.")

from predictapi import predict_stroke
result = predict_stroke('Male', 79, 0, 0, 'Yes', 'Private', 'Rural', 72.73, 28.4, 'never smoked')
print(result)
