import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
import joblib
import numpy as np

# Parameters
num_rows = 1000


# Generate random data with realistic dependencies
np.random.seed(42)  # For reproducibility

# Patient count with Poisson distribution
patient_counts = np.random.poisson(lam=50, size=num_rows)  # Poisson distribution for patient counts

# Average wait time with a normal distribution and realistic bounds
average_wait_times = np.clip(np.random.normal(loc=30, scale=10, size=num_rows), 5, 60)  # Normal distribution for wait times

# Bed occupancy rate with a realistic range and some dependency on patient counts
bed_occupancy_rates = np.clip(0.5 + 0.5 * (patient_counts / patient_counts.max()), 0.5, 1.0)  # Higher patient counts may indicate higher occupancy

# Staff availability rate with a realistic range
staff_availability_rates = np.round(np.random.uniform(0.6, 1.0, num_rows), 2)  # Assuming hospitals typically have moderate to high staff availability

# Equipment utilization rate with a realistic range
equipment_utilization = np.round(np.random.uniform(0.5, 1.0, num_rows), 2)

# Generate Workflow_Status based on conditions with more granularity
workflow_status = []
for i in range(num_rows):
    if (bed_occupancy_rates[i] > 0.85 and staff_availability_rates[i] < 0.75 and average_wait_times[i] > 40 and
            patient_counts[i] > 70 and equipment_utilization[i] > 0.9):
        workflow_status.append('Poor')
    elif (bed_occupancy_rates[i] > 0.75 and average_wait_times[i] > 40 and
          patient_counts[i] > 60 and equipment_utilization[i] > 0.8):
        workflow_status.append('Average')
    elif (bed_occupancy_rates[i] <= 0.75 and staff_availability_rates[i] >= 0.85 and
          average_wait_times[i] < 30 and equipment_utilization[i] < 0.7):
        workflow_status.append('Outstanding')
    else:
        workflow_status.append('Good')

# Create DataFrame
workflow_data = {
    'Patient_Count': patient_counts,
    'Average_Wait_Time': average_wait_times,
    'Bed_Occupancy_Rate': bed_occupancy_rates,
    'Staff_Availability_Rate': staff_availability_rates,
    'Equipment_Utilization': equipment_utilization,
    'Workflow_Status': workflow_status,
}


# Sample Workflow Data
# workflow_data = {
    
#     'Patient_Count': [50, 40, 30, 45, 35, 48, 42, 32, 47, 37,
#                       45, 38, 34, 50, 32, 60, 55, 50, 53, 58,
#                       49, 51, 39, 62, 54, 65, 52, 47, 48, 44,
#                       57, 62, 55, 48, 49, 53, 38, 42, 45, 51,
#                       36, 29, 40, 44, 50, 58, 61, 47, 39, 43],
#     'Average_Wait_Time': [30, 25, 45, 35, 40, 32, 28, 48, 37, 38,
#                           28, 22, 50, 30, 42, 33, 29, 38, 41, 34,
#                           26, 27, 24, 30, 32, 35, 29, 31, 28, 30,
#                           25, 30, 29, 34, 36, 38, 32, 30, 28, 35,
#                           40, 27, 26, 35, 30, 33, 28, 29, 37, 34],
#     'Bed_Occupancy_Rate': [0.85, 0.80, 0.70, 0.88, 0.80, 0.83, 0.78, 0.68, 0.87, 0.79,
#                            0.80, 0.77, 0.65, 0.90, 0.75, 0.85, 0.88, 0.72, 0.82, 0.78,
#                            0.81, 0.76, 0.73, 0.85, 0.79, 0.90, 0.84, 0.67, 0.75, 0.80,
#                            0.82, 0.75, 0.77, 0.78, 0.80, 0.81, 0.74, 0.73, 0.68, 0.76,
#                            0.78, 0.80, 0.72, 0.79, 0.82, 0.76, 0.75, 0.78, 0.79, 0.73],
#     'Staff_Availability_Rate': [0.90, 0.85, 0.75, 0.95, 0.80, 0.92, 0.87, 0.73, 0.93, 0.79,
#                                 0.88, 0.84, 0.70, 0.97, 0.78, 0.90, 0.85, 0.80, 0.86, 0.81,
#                                 0.89, 0.84, 0.75, 0.88, 0.83, 0.76, 0.72, 0.79, 0.88, 0.82,
#                                 0.85, 0.88, 0.84, 0.80, 0.77, 0.75, 0.72, 0.69, 0.76, 0.81,
#                                 0.84, 0.86, 0.78, 0.80, 0.73, 0.82, 0.90, 0.78, 0.75, 0.88],
#     'Equipment_Utilization': [0.75, 0.80, 0.70, 0.78, 0.75, 0.77, 0.82, 0.72, 0.80, 0.76,
#                               0.78, 0.79, 0.69, 0.82, 0.74, 0.80, 0.75, 0.78, 0.81, 0.73,
#                               0.79, 0.80, 0.72, 0.74, 0.76, 0.78, 0.75, 0.77, 0.79, 0.81,
#                               0.82, 0.80, 0.77, 0.74, 0.76, 0.75, 0.72, 0.70, 0.68, 0.73,
#                               0.75, 0.76, 0.78, 0.80, 0.72, 0.74, 0.78, 0.76, 0.75, 0.80],
#     'Workflow_Status': ['Good', 'Good', 'Average', 'Outstanding', 'Average',
#                         'Good', 'Good', 'Poor', 'Outstanding', 'Average',
#                         'Good', 'Good', 'Poor', 'Outstanding', 'Poor',
#                         'Good', 'Average', 'Good', 'Outstanding', 'Average',
#                         'Good', 'Good', 'Average', 'Poor', 'Outstanding',
#                         'Good', 'Good', 'Average', 'Poor', 'Outstanding',
#                         'Average', 'Average', 'Good', 'Good', 'Poor',
#                         'Outstanding', 'Good', 'Good', 'Average', 'Poor',
#                         'Outstanding', 'Good', 'Average', 'Good', 'Good',
#                         'Outstanding', 'Average', 'Good', 'Average', 'Poor']
# }
# # Convert to DataFrame
df_workflow = pd.DataFrame(workflow_data)

# Encode the target variable
le = LabelEncoder()
df_workflow['Workflow_Status'] = le.fit_transform(df_workflow['Workflow_Status'])

# Select features and target
X = df_workflow[['Patient_Count', 'Average_Wait_Time', 'Bed_Occupancy_Rate', 'Staff_Availability_Rate', 'Equipment_Utilization']]
y = df_workflow['Workflow_Status']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2529)

# Initialize and fit the scaler
print(X_test)
scaler = StandardScaler()
X_train[['Patient_Count', 'Average_Wait_Time']] = scaler.fit_transform(X_train[['Patient_Count', 'Average_Wait_Time']])
X_test[['Patient_Count', 'Average_Wait_Time']] = scaler.transform(X_test[['Patient_Count', 'Average_Wait_Time']])

# Initialize and train the Random Forest classifier
model = RandomForestClassifier(random_state=2529)
model.fit(X_train, y_train)
print(X_test)
y_pred = model.predict(X_test)
print(y_pred)
print(y_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
# Save the model and the scaler
joblib.dump(model, 'workflow_model1.pkl')
joblib.dump(le, 'label_encoder1.pkl')
joblib.dump(scaler, 'scaler1.pkl')
