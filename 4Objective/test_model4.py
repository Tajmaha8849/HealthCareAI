import pandas as pd
import pickle

# Example test data (replace with your actual data)
test_data = {
    'heart_rate': [85, 92, 75, 100],
    'blood_pressure': [125, 140, 120, 135],
    'temperature': [37.0, 37.2, 36.8, 37.5]
}

# Create a DataFrame for the test data
test_df = pd.DataFrame(test_data)

# Load the trained model from the .pkl file
with open('remote_patient_monitoring_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Make predictions on the test data
predictions = model.predict(test_df)

# Output the predictions
print("Predictions:", predictions)

# If you want to see the probability estimates of the predictions
# probability_estimates = model.predict_proba(test_df)
# print("Probability Estimates:")
# print(probability_estimates)
