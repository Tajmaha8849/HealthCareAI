data = [
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

import pandas as pd

# Convert data to DataFrame
df = pd.DataFrame(data)

# Save data to CSV file
df.to_csv('healthcare_data.csv', index=False)

print(f"Sample dataset successfully saved as 'healthcare_data.csv'")
