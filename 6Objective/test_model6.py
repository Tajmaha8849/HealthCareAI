import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib

# Test data in dictionary format
test_data = [
    {
        "PatientID": 1,
        "Age": 65,
        "Gender": "M",
        "Ethnicity": "White",
        "ChronicCondition1": "Diabetes",
        "ChronicCondition2": "Hypertension",
        "BP_Systolic": 140,
        "BP_Diastolic": 90,
        "Glucose": 120,
        "BMI": 30,
        "Medication1": "Metformin",
        "Medication2": "Statins",
        "HospitalAdmissions": 2,
        "Outcome": "Survived"
    },
    {
        "PatientID": 2,
        "Age": 45,
        "Gender": "F",
        "Ethnicity": "Hispanic",
        "ChronicCondition1": "Hypertension",
        "ChronicCondition2": None,
        "BP_Systolic": None,
        "BP_Diastolic": None,
        "Glucose": 110,
        "BMI": 28,
        "Medication1": "Beta-blockers",
        "Medication2": "ACE inhibitors",
        "HospitalAdmissions": 1,
        "Outcome": "Improved"
    },
    {
        "PatientID": 3,
        "Age": 70,
        "Gender": "M",
        "Ethnicity": "Black",
        "ChronicCondition1": "Heart disease",
        "ChronicCondition2": None,
        "BP_Systolic": None,
        "BP_Diastolic": None,
        "Glucose": 130,
        "BMI": 26,
        "Medication1": "Aspirin",
        "Medication2": "Warfarin",
        "HospitalAdmissions": 3,
        "Outcome": "Died"
    },
    {
        "PatientID": 4,
        "Age": 55,
        "Gender": "F",
        "Ethnicity": "Asian",
        "ChronicCondition1": None,
        "ChronicCondition2": None,
        "BP_Systolic": 130,
        "BP_Diastolic": 85,
        "Glucose": 140,
        "BMI": 32,
        "Medication1": "Insulin",
        "Medication2": "Diuretics",
        "HospitalAdmissions": 2,
        "Outcome": "Stable"
    },
    {
        "PatientID": 5,
        "Age": 60,
        "Gender": "M",
        "Ethnicity": "White",
        "ChronicCondition1": "Diabetes",
        "ChronicCondition2": None,
        "BP_Systolic": 145,
        "BP_Diastolic": 95,
        "Glucose": 160,
        "BMI": 29,
        "Medication1": "Insulin",
        "Medication2": "Statins",
        "HospitalAdmissions": 1,
        "Outcome": "Improved"
    },
    {
        "PatientID": 6,
        "Age": 50,
        "Gender": "F",
        "Ethnicity": "Hispanic",
        "ChronicCondition1": "Heart disease",
        "ChronicCondition2": None,
        "BP_Systolic": None,
        "BP_Diastolic": None,
        "Glucose": 120,
        "BMI": 25,
        "Medication1": "Aspirin",
        "Medication2": None,
        "HospitalAdmissions": 1,
        "Outcome": "Stable"
    },
    {
        "PatientID": 7,
        "Age": 62,
        "Gender": "F",
        "Ethnicity": "White",
        "ChronicCondition1": "Diabetes",
        "ChronicCondition2": None,
        "BP_Systolic": 150,
        "BP_Diastolic": 92,
        "Glucose": 130,
        "BMI": 31,
        "Medication1": "Metformin",
        "Medication2": None,
        "HospitalAdmissions": 2,
        "Outcome": "Survived"
    },
    {
        "PatientID": 8,
        "Age": 48,
        "Gender": "M",
        "Ethnicity": "Asian",
        "ChronicCondition1": "Hypertension",
        "ChronicCondition2": None,
        "BP_Systolic": 135,
        "BP_Diastolic": 88,
        "Glucose": 115,
        "BMI": 27,
        "Medication1": "Beta-blockers",
        "Medication2": "ACE inhibitors",
        "HospitalAdmissions": 1,
        "Outcome": "Improved"
    }
]

# Convert test data to DataFrame
df_test = pd.DataFrame(test_data)

# Preprocessing for test data (same as training data)
label_encoder = LabelEncoder()
df_test['Gender'] = label_encoder.fit_transform(df_test['Gender'])
df_test['Ethnicity'] = label_encoder.fit_transform(df_test['Ethnicity'])
df_test['ChronicCondition1'] = label_encoder.fit_transform(df_test['ChronicCondition1'].astype(str))
df_test['ChronicCondition2'] = label_encoder.fit_transform(df_test['ChronicCondition2'].astype(str))
df_test['Medication1'] = label_encoder.fit_transform(df_test['Medication1'].astype(str))
df_test['Medication2'] = label_encoder.fit_transform(df_test['Medication2'].astype(str))
df_test['Outcome'] = label_encoder.fit_transform(df_test['Outcome'])

# Separate features (X_test) and target variable (y_test)
X_test = df_test.drop(columns=['PatientID', 'Outcome'])
y_test = df_test['Outcome']

# Load trained model
model = joblib.load('trained_model.pkl')

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{class_report}')
