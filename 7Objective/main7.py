import pandas as pd
import numpy as np

# Generate synthetic data
num_samples = 1000

data = {
    'Age': np.random.randint(18, 80, num_samples),
    'Gender': np.random.choice(['M', 'F'], num_samples),
    'Ethnicity': np.random.choice(['Asian', 'Caucasian', 'African American', 'Hispanic'], num_samples),
    'Weight': np.random.randint(50, 100, num_samples),
    'Height': np.random.randint(150, 200, num_samples),
    'Previous Diagnoses': np.random.choice(['Hypertension', 'Diabetes', 'Arthritis', 'None'], num_samples),
    'Family History': np.random.choice(['Heart Disease', 'Cancer', 'None'], num_samples),
    'Genetic Markers': np.random.choice(['Marker1', 'Marker2', 'Marker3'], num_samples),
    'Current Diagnosis': np.random.choice(['Diabetes', 'Cancer', 'Hypertension'], num_samples),
    'Vital Signs': np.random.choice(['BP, HR', 'BP, HR, Temp'], num_samples),
    'Lab Results': np.random.choice(['Glucose', 'Cholesterol', 'CBC'], num_samples),
    'Treatment Type': np.random.choice(['Medication', 'Chemotherapy', 'Surgery'], num_samples),
    'Dosage': np.random.choice([None, '50mg', '100mg'], num_samples),
    'Frequency': np.random.choice([None, 'Daily', 'Weekly'], num_samples),
    'Duration': np.random.choice(['3 months', '6 months', '1 year'], num_samples),
    'Adherence': np.random.choice(['Yes', 'No'], num_samples),
    'Lifestyle Factors': np.random.choice(['Smoker', 'Non-smoker'], num_samples),
    'Treatment Outcome': np.random.choice(['Improved', 'No Change', 'Worsened'], num_samples),
}

df = pd.DataFrame(data)

# Save to CSV
df.to_csv('synthetic_patient_data7.csv', index=False)

print(df.head())
