from flask import Flask, request, render_template
import pandas as pd
import pickle

app = Flask(__name__)

# Load the trained model
with open('remote_patient_monitoring_model4.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/')
def home():
    return render_template('index4.html')

@app.route('/predictw', methods=['POST'])
def predict():
    # Extract input data from form
    heart_rate = float(request.form['heart_rate'])
    blood_pressure = float(request.form['blood_pressure'])
    temperature = float(request.form['temperature'])
    
    # Create a DataFrame for the input data
    input_data = pd.DataFrame([[heart_rate, blood_pressure, temperature]], 
                              columns=['heart_rate', 'blood_pressure', 'temperature'])
    
    # Make predictions
    prediction = model.predict(input_data)
    probability_estimates = model.predict_proba(input_data)
    
    # Determine risk status
    risk_status = 'At Risk' if probability_estimates[0][1] > 0.5 else 'Not at Risk'
    risk_value = probability_estimates[0][1] if probability_estimates[0][1] > 0.5 else probability_estimates[0][0]
    
    # Format the result
    result = {
        'prediction': int(prediction[0]),
        'risk_status': risk_status,
        'risk_value': f"{risk_value * 100:.2f}%",
        'probability': {
            'not_at_risk': f"{probability_estimates[0][0] * 100:.2f}%",
            'at_risk': f"{probability_estimates[0][1] * 100:.2f}%"
        }
    }
    
    return render_template('predict4.html', 
                           heart_rate=heart_rate, 
                           blood_pressure=blood_pressure, 
                           temperature=temperature, 
                           result=result)

if __name__ == '__main__':
    app.run(debug=True)
