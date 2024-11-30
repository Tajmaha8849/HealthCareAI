from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Define file paths
model_file = 'workflow_model1.pkl'
scaler_file = 'scaler1.pkl'
le_file = 'label_encoder1.pkl'

app = Flask(__name__)

# Initialize global variables
model = None
scaler = None
le = None

@app.route('/train1', methods=['POST'])
def train():
    # Check if a file was uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    # Check if the file is a CSV
    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'File must be a CSV'}), 400

    try:
        # Read the uploaded file
        df = pd.read_csv(file)

        # Check if the DataFrame contains the necessary columns
        required_columns = ['Patient_Count', 'Average_Wait_Time', 'Bed_Occupancy_Rate', 
                            'Staff_Availability_Rate', 'Equipment_Utilization', 'Workflow_Status']
        if not all(col in df.columns for col in required_columns):
            return jsonify({'error': 'CSV must contain required columns'}), 400

        # Encode the target variable
        le = LabelEncoder()
        df['Workflow_Status'] = le.fit_transform(df['Workflow_Status'])

        # Prepare features and target
        X = df[['Patient_Count', 'Average_Wait_Time', 'Bed_Occupancy_Rate', 
                'Staff_Availability_Rate', 'Equipment_Utilization']]
        y = df['Workflow_Status']

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2529)

        # Initialize and fit the scaler
        scaler = StandardScaler()
        X_train[['Patient_Count', 'Average_Wait_Time']] = scaler.fit_transform(X_train[['Patient_Count', 'Average_Wait_Time']])
        X_test[['Patient_Count', 'Average_Wait_Time']] = scaler.transform(X_test[['Patient_Count', 'Average_Wait_Time']])

        # Initialize and train the Random Forest classifier
        model = RandomForestClassifier(random_state=2529)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Save the model, label encoder, and scaler
        joblib.dump(model, model_file)
        joblib.dump(le, le_file)
        joblib.dump(scaler, scaler_file)

        return jsonify({"message": accuracy * 100})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict1', methods=['POST'])
def predict():
    try:
        if model is None or scaler is None or le is None:
            return jsonify({'error': 'Model not trained or files not found'}), 500

        # Get JSON data from the request
        data = request.json

        # Convert JSON data to DataFrame
        df = pd.DataFrame([data])

        # Preprocess data
        df[['Patient_Count', 'Average_Wait_Time']] = scaler.transform(df[['Patient_Count', 'Average_Wait_Time']])

        # Make predictions
        predictions = model.predict(df)
        predicted_status = le.inverse_transform(predictions)

        # Return JSON response
        return jsonify({'workflow_status': predicted_status[0]})
    except Exception as e:
        # Return error message
        return jsonify({'error': str(e)}), 500

@app.route('/')
def index():
    return render_template("index1.html")

if __name__ == '__main__':
    app.run(debug=True, port=4000)
