# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset and remove the 'id' column
data = pd.read_csv('healthcare-dataset-stroke-data-ml-model.csv')
data = data.drop(columns=['id'])

# Data preprocessing
# Encode categorical features
label_encoder = LabelEncoder()
categorical_columns = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
for column in categorical_columns:
    data[column] = label_encoder.fit_transform(data[column])

# Split the data into features (X) and target (y)
X = data.drop(columns=['stroke'])
y = data['stroke']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an XGBoost model
model = XGBClassifier()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Print the results
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", classification_rep)

# You can also save the trained model for future use
# import joblib
# joblib.dump(model, 'stroke_prediction_model.pkl')

import joblib

# Assuming you have already trained and have a model object
model = XGBClassifier()
model.fit(X_train, y_train)

# Save the model to a file
joblib.dump(model, 'stroke_prediction_model.pkl')


# predict
# Load the saved model
import joblib

# Load the saved model
loaded_model = joblib.load('stroke_prediction_model.pkl')

# Prepare input data for a single prediction
# You can create a single-row DataFrame or use a dictionary
single_data = pd.DataFrame({
    'gender': ['Male'], 
    'age': [66], 
    'hypertension': [0], 
    'heart_disease': [0], 
    'ever_married': ['Yes'], 
    'work_type': ['Private'], 
    'Residence_type': ['Rural'], 
    'avg_glucose_level': [76.0], 
    'bmi': [22.0], 
    'smoking_status': ['formerly smoked']
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
if single_prediction[0] == 1:
    print("The individual is predicted to have a stroke.")
else:
    print("The individual is predicted to not have a stroke.")
