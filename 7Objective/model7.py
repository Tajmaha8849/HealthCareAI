import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
# Load data
data = pd.read_csv('synthetic_patient_data7.csv')

# Assume the last column is the target (treatment response) and the rest are features
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Handle missing values
X.fillna(method='ffill', inplace=True)

# Encode categorical variables (if any)
X = pd.get_dummies(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Define parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Initialize RandomForestClassifier
rf = RandomForestClassifier(random_state=42)

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)

# Fit the model
grid_search.fit(X_train, y_train)

# Get the best model
best_rf = grid_search.best_estimator_

# Train the best model
best_rf.fit(X_train, y_train)
# Predict on test data
y_pred = best_rf.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
# Save the trained model
with open('personalized_medicine_model7.pkl', 'wb') as f:
    pickle.dump(best_rf, f)

# Save the feature columns (for later use during prediction)
with open('feature_columns7.pkl', 'wb') as f:
    pickle.dump(X.columns.tolist(), f)
