import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression

# Load dataset (replace this with your actual data loading mechanism)
df = pd.read_csv('healthcare_data.csv')

# Preprocessing function to encode categorical variables using OneHotEncoder
def preprocess_data(df):
    # Drop columns not needed for training
    df.drop(columns=['PatientID'], inplace=True)
    
    # Fill missing numeric values with mean
    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    
    # One-hot encode categorical variables
    categorical_cols = ['Gender', 'Ethnicity', 'ChronicCondition1', 'ChronicCondition2', 'Medication1', 'Medication2']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # Encode outcome variable if needed
    
    df['Outcome'] = df['Outcome'].map({'Died': 0, 'Stable': 1, 'Improved': 2, 'Survived': 3})  # Assuming Outcome is categorical
    
    return df

# Preprocess the dataset
df = preprocess_data(df)

# Split data into features and target variable
X = df.drop(columns=['Outcome'])
y = df['Outcome']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a pipeline for preprocessing and modeling
numeric_features = X.select_dtypes(include=['number']).columns
categorical_features = X.select_dtypes(include=['object']).columns

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Append classifier to preprocessing pipeline.
# Now we have a full prediction pipeline.
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', LogisticRegression())])

# Train model using the pipeline
pipeline.fit(X_train, y_train)

# Save trained model
joblib.dump(pipeline, 'trained_model.pkl')
print("Model trained and saved successfully!")

# Make predictions on test data
y_pred = pipeline.predict(X_test)

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Optionally, you can calculate accuracy as well
accuracy = pipeline.score(X_test, y_test)
print(f"\nAccuracy: {accuracy * 100:.2f}%")
