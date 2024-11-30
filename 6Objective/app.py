from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('trained_model6.pkl')

# LabelEncoder for categorical variables
label_encoders = {}

def encode_categorical(data):
    global label_encoders
    
    # Define categorical columns
    categorical_cols = ['Gender', 'Ethnicity', 'ChronicCondition1', 'ChronicCondition2', 'Medication1', 'Medication2']
    
    # Initialize or update LabelEncoders
    for col in categorical_cols:
        if col not in label_encoders:
            label_encoders[col] = LabelEncoder()
            label_encoders[col].fit(data[col].astype(str))  # Ensure consistent type
    
    # Encode categorical variables
    for col in categorical_cols:
        # Handle unseen categories by assigning a default value or NaN
        data[col] = data[col].astype(str).map(lambda s: label_encoders[col].transform([s])[0] if s in label_encoders[col].classes_ else None)
    
    return data

@app.route('/')
def index():
    # Render the HTML page
    return render_template('index6.html')

@app.route('/predict', methods=['POST'])
def predict6():
    try:
        # Get JSON data from request
        data = request.get_json()

        # Convert JSON data to DataFrame
        df = pd.DataFrame([data])

        # Encode categorical variables
        df = encode_categorical(df)

        # Ensure all columns expected by the model are present
        expected_cols = ['Age', 'Gender', 'Ethnicity', 'ChronicCondition1', 'ChronicCondition2',
                         'BP_Systolic', 'BP_Diastolic', 'Glucose', 'BMI', 'Medication1', 'Medication2',
                         'HospitalAdmissions']

        for col in expected_cols:
            if col not in df.columns:
                df[col] = None  # Assign None or NaN for missing columns

        # Reorder columns to match the order of training data
        df = df[expected_cols]

        # Make prediction
        prediction = model.predict(df)

        # Assuming 'prediction' is a numpy array or similar, convert to string or JSON format
        outcome_mapping = {
            0: 'Died',
            1: 'Stable',
            2: 'Improved',
            3: 'Survived'  # Assuming prediction[0] can go up to 3 based on your previous mapping
        }

        # Adjust the prediction_result based on the mapping
        prediction_result = {
            'prediction': outcome_mapping[prediction[0]]
        }
        # prediction_result = {
        #     'prediction': str(prediction[0])  # Adjust this based on your prediction format
        # }

        return jsonify(prediction_result)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True,port=4003)
