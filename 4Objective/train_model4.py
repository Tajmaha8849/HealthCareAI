import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
import numpy as np

# Example synthetic dataset (replace with your actual data)
# data = {
#     'heart_rate': [70, 80, 75, 90, 95, 85, 78, 88, 92, 100,
#                    72, 82, 77, 85, 89, 91, 76, 86, 94, 102,
#                    74, 84, 79, 87, 93, 83, 80, 88, 91, 97,
#                    75, 78, 82, 85, 90, 92, 79, 84, 89, 96,
#                    73, 81, 76, 86, 92, 87, 80, 85, 90, 95],
#     'blood_pressure': [120, 130, 125, 140, 135, 128, 132, 145, 138, 150,
#                        122, 133, 127, 142, 136, 131, 129, 146, 139, 152,
#                        121, 132, 126, 141, 137, 130, 134, 148, 140, 155,
#                        124, 129, 128, 143, 138, 150, 127, 135, 142, 149,
#                        123, 134, 126, 139, 137, 132, 130, 143, 141, 147],
#     'temperature': [36.5, 37.0, 36.8, 37.2, 36.9, 37.1, 36.7, 37.3, 37.0, 37.5,
#                     36.6, 37.1, 36.9, 37.3, 37.0, 37.2, 36.8, 37.4, 37.1, 37.6,
#                     36.5, 37.0, 36.7, 37.1, 36.8, 37.2, 36.9, 37.3, 37.1, 37.4,
#                     36.6, 37.1, 36.9, 37.2, 37.0, 37.3, 36.8, 37.2, 37.0, 37.5,
#                     36.7, 37.0, 36.8, 37.1, 36.9, 37.2, 36.6, 37.3, 37.1, 37.4],
#     'at_risk': [0, 1, 0, 1, 1, 0, 0, 1, 1, 1,
#                 0, 1, 0, 1, 1, 0, 0, 1, 1, 1,
#                 0, 1, 0, 1, 1, 0, 0, 1, 1, 1,
#                 0, 1, 0, 1, 1, 0, 0, 1, 1, 1,
#                 0, 1, 0, 1, 1, 0, 0, 1, 1, 1]
# }

# Define the number of samples
num_samples = 1000

# Generate synthetic data
np.random.seed(42)  # for reproducibility

# Generate heart rate, blood pressure, and temperature data
heart_rate = np.random.normal(loc=80, scale=10, size=num_samples).astype(int)
blood_pressure = np.random.normal(loc=130, scale=15, size=num_samples).astype(int)
temperature = np.random.normal(loc=37.0, scale=0.5, size=num_samples).round(1)

# Simulate the 'at_risk' label based on heuristic thresholds
risk_factor = (heart_rate > 90) | (blood_pressure > 140) | (temperature > 37.5)
at_risk = risk_factor.astype(int)

# Create the DataFrame
data = {
    'heart_rate': heart_rate,
    'blood_pressure': blood_pressure,
    'temperature': temperature,
    'at_risk': at_risk
}
print("Length of the hear_rate",len(data['blood_pressure']))

df = pd.DataFrame(data)

# Separate features (X) and target (y)
X = df[['heart_rate', 'blood_pressure', 'temperature']]
y = df['at_risk']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a RandomForestClassifier model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate model performance (optional)
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy:.2f}")

# Save the trained model to a .pkl file
with open('remote_patient_monitoring_model4.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

print("Model saved successfully.")


