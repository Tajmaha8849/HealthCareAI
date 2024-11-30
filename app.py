from flask import Flask, request, jsonify, render_template, redirect, url_for
import pandas as pd
import joblib
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras._tf_keras.keras.models import load_model
from keras._tf_keras.keras.preprocessing import image
import numpy as np
import os
from keras._tf_keras.keras.utils import load_img

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

app = Flask(__name__)

# Global variables for 1Objective
model_1 = None
scaler_1 = None
le_1 = None

# Load models for 4Objective and 5Objective
model_4 = pickle.load(open('4Objective/remote_patient_monitoring_model4.pkl', 'rb'))
model_5 = load_model('5Objective/pneumonia_detection_model5.h5')

# Load models for 6Objective, 7Objective, and 8Objective
model_6 = joblib.load('trained_model6.pkl')
with open('personalized_medicine_model7.pkl', 'rb') as f:
    model_7 = pickle.load(f)

with open('feature_columns7.pkl', 'rb') as f:
    feature_columns_7 = pickle.load(f)

# model_7 = pickle.load(open('personalized_medicine_model7.pkl', 'rb'))
# feature_columns_7 = pickle.load(open('feature_columns7.pkl', 'rb'))
model_hp_8 = joblib.load('recommendation_model_hp8.pkl')
model_dr_8 = joblib.load('recommendation_model_dr8.pkl')
model_er_8 = joblib.load('recommendation_model_er8.pkl') 
le_8 = joblib.load('label_encoder8.pkl')

model_9 = load_model('model9.h5')
# Dictionary to map numerical labels to emotions
emotion_labels_9 = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

label_encoders_6 = {}

# Ensure the uploads directory exists for 5Objective
if not os.path.exists('5Objective/uploads'):
    os.makedirs('5Objective/uploads')

# Ensure the uploads directory exists for 9Objective
if not os.path.exists('uploads'):
    os.makedirs('uploads')

# Emotion labels for 9Objective
emotion_labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}
# model_9 = load_model('model/model9.h5')

@app.route('/')
def index():
    return render_template('index.html')

# Routes for 1Objective
@app.route('/1Objective')
def index_1Objective():
    return render_template('1Objective/index1.html')

@app.route('/train1', methods=['POST'])
def train_1Objective():
    global model_1, scaler_1, le_1

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'File must be a CSV'}), 400

    try:
        df = pd.read_csv(file)
        required_columns = ['Patient_Count', 'Average_Wait_Time', 'Bed_Occupancy_Rate', 
                            'Staff_Availability_Rate', 'Equipment_Utilization', 'Workflow_Status']
        if not all(col in df.columns for col in required_columns):
            return jsonify({'error': 'CSV must contain required columns'}), 400

        le_1 = LabelEncoder()
        df['Workflow_Status'] = le_1.fit_transform(df['Workflow_Status'])

        X = df[['Patient_Count', 'Average_Wait_Time', 'Bed_Occupancy_Rate', 
                'Staff_Availability_Rate', 'Equipment_Utilization']]
        y = df['Workflow_Status']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2529)

        scaler_1 = StandardScaler()
        X_train[['Patient_Count', 'Average_Wait_Time']] = scaler_1.fit_transform(X_train[['Patient_Count', 'Average_Wait_Time']])
        X_test[['Patient_Count', 'Average_Wait_Time']] = scaler_1.transform(X_test[['Patient_Count', 'Average_Wait_Time']])

        model_1 = RandomForestClassifier(random_state=2529)
        model_1.fit(X_train, y_train)
        y_pred = model_1.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        joblib.dump(model_1, '1Objective/workflow_model1.pkl')
        joblib.dump(le_1, '1Objective/label_encoder1.pkl')
        joblib.dump(scaler_1, '1Objective/scaler1.pkl')

        return jsonify({"message": accuracy * 100})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict1', methods=['POST'])
def predict_1Objective():
    global model_1, scaler_1, le_1

    try:
        if model_1 is None or scaler_1 is None or le_1 is None:
            model_1 = joblib.load('1Objective/workflow_model1.pkl')
            scaler_1 = joblib.load('1Objective/scaler1.pkl')
            le_1 = joblib.load('1Objective/label_encoder1.pkl')

        data = request.json
        df = pd.DataFrame([data])
        df[['Patient_Count', 'Average_Wait_Time']] = scaler_1.transform(df[['Patient_Count', 'Average_Wait_Time']])
        predictions = model_1.predict(df)
        predicted_status = le_1.inverse_transform(predictions)
        return jsonify({'workflow_status': predicted_status[0]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Routes for 4Objective
@app.route('/4Objective')
def index_4Objective():
    return render_template('4Objective/index4.html')

@app.route('/predictw', methods=['POST'])
def predict_4Objective():
    heart_rate = float(request.form['heart_rate'])
    blood_pressure = float(request.form['blood_pressure'])
    temperature = float(request.form['temperature'])

    input_data = pd.DataFrame([[heart_rate, blood_pressure, temperature]], 
                              columns=['heart_rate', 'blood_pressure', 'temperature'])
    
    prediction = model_4.predict(input_data)
    probability_estimates = model_4.predict_proba(input_data)
    
    risk_status = 'At Risk' if probability_estimates[0][1] > 0.5 else 'Not at Risk'
    risk_value = probability_estimates[0][1] if probability_estimates[0][1] > 0.5 else probability_estimates[0][0]
    
    result = {
        'prediction': int(prediction[0]),
        'risk_status': risk_status,
        'risk_value': f"{risk_value * 100:.2f}%",
        'probability': {
            'not_at_risk': f"{probability_estimates[0][0] * 100:.2f}%",
            'at_risk': f"{probability_estimates[0][1] * 100:.2f}%"
        }
    }
    
    return render_template('4Objective/predict4.html', 
                           heart_rate=heart_rate, 
                           blood_pressure=blood_pressure, 
                           temperature=temperature, 
                           result=result)

# Routes for 5Objective
@app.route('/5Objective')
def index_5Objective():
    return render_template('5Objective/index5.html')

@app.route('/predict5', methods=['POST'])
def predict_5Objective():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    img_path = f'5Objective/uploads/{file.filename}'
    file.save(img_path)

    img = preprocess_image(img_path)
    prediction = model_5.predict(img)

    result = 'Pneumonia' if prediction[0][0] > 0.5 else 'Normal'
    confidence = float(prediction[0][0])

    return jsonify({'prediction': result, 'confidence': confidence})

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.0
    return img

# Routes for 6Objective

def encode_categorical_6(df):
    global label_encoders_6
    
    # Define categorical columns
    categorical_cols = ['Gender', 'Ethnicity', 'ChronicCondition1', 'ChronicCondition2', 'Medication1', 'Medication2']
    
    # Initialize or update LabelEncoders
    for col in categorical_cols:
        if col not in label_encoders_6:
            label_encoders_6[col] = LabelEncoder()
            label_encoders_6[col].fit(df[col].astype(str))  # Ensure consistent type
    
    # Encode categorical variables
    for col in categorical_cols:
        # Handle unseen categories by assigning a default value or NaN
        df[col] = df[col].astype(str).map(lambda s: label_encoders_6[col].transform([s])[0] if s in label_encoders_6[col].classes_ else None)
    
    return df
@app.route('/6Objective')
def index_6Objective():
    return render_template('6Objective/index6.html')

@app.route('/predict6', methods=['POST'])
def predict_6Objective():
    try:
        # Get JSON data from request
        data = request.get_json()

        # Convert JSON data to DataFrame
        df = pd.DataFrame([data])

        # Encode categorical variables
        df = encode_categorical_6(df)

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
        prediction = model_6.predict(df)

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


    

# Routes for 7Objective
@app.route('/7Objective')
def index_7Objective():
    return render_template('7Objective/index7.html')

@app.route('/predict7', methods=['POST'])
def predict_7Objective():
    data = request.get_json()
    df = pd.DataFrame([data])
    df = pd.get_dummies(df)
    df = df.reindex(columns=feature_columns_7, fill_value=0)
    prediction = model_7.predict(df)
    return jsonify({'prediction': prediction[0]})
# Routes for 8Objective
nltk.download('punkt')
nltk.download('stopwords')


def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

# Load and preprocess the dataset
data = pd.read_csv('data_disease.csv')
data['processed_text'] = data['Symptoms'].apply(lambda x: preprocess_text(x))

# Train the model
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(data['processed_text'])
y = data['Disease']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)
@app.route('/11Objective')
def index_8Objective():
    return render_template('11Objective/index11.html')

@app.route('/predict_hp8', methods=['POST'])
def predict_hp_8Objective():
    symptoms = request.json['symptoms']
    symptoms = preprocess_text(symptoms)
    vectorized_input = tfidf.transform([symptoms])
    prediction = model.predict(vectorized_input)[0]
    return jsonify({'disease': prediction})


# Routes for 9Objective
# Function to preprocess the image for prediction
def preprocess_9_image(image_path):
    img = image.load_img(image_path, target_size=(48, 48), color_mode='grayscale')
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.0  # Normalize pixel values
@app.route('/9Objective')
def index_9Objective():
    return render_template('9Objective/index9.html')

@app.route('/prediction', methods=['POST'])
def predict_9Objective():
    if request.method == 'POST':
        # Check if the POST request has the file part
        if 'file' not in request.files:
            return render_template('9Objective/index9.html', prediction_result='No file part')
        
        file = request.files['file']

        # If user does not select file, browser also submit an empty part without filename
        if file.filename == '':
            return render_template('9Objective/index9.html', prediction_result='No selected file')

        if file:
            # Save the file to the uploads folder
            file_path = os.path.join('uploads', file.filename)
            file.save(file_path)

            # Preprocess the image
            processed_image = preprocess_9_image(file_path)

            # Make prediction
            prediction = model_9.predict(processed_image)
            predicted_label = np.argmax(prediction)
            predicted_emotion = emotion_labels_9[predicted_label]

            # Delete the uploaded file to save space
            os.remove(file_path)

            return render_template('9Objective/index9.html', prediction_result=predicted_emotion)
@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contactus')
def contact():
    return render_template("contactus.html")
@app.route('/disclaimer')
def discalimer():
    return render_template("disclaimer.html")

@app.route('/terms')
def terms():
    return render_template("terms.html")



#10 Objective
#mood tracker
model_10_tracker = joblib.load('model.pkl')

# Define mood labels corresponding to numerical predictions
mood_10_labels = ['happy', 'sad', 'anxious', 'angry', 'neutral']

@app.route('/predicts',methods=['POST'])
def predict():
    # Get user input from the form
    activity_level = float(request.form['activity_level'])
    sleep_quality = float(request.form['sleep_quality'])

    # Example of predicting mood based on input (replace with your actual prediction logic)
    # Assuming your model expects an array of features
    input_data = np.array([[activity_level, sleep_quality]])
    predicted_mood_index = model_10_tracker.predict(input_data)[0]  # Get the predicted mood index

    # Map the predicted index to the corresponding mood label
    predicted_mood = mood_10_labels[predicted_mood_index]

    # Render the predict.html template with prediction results
    return render_template('10Objective/result.html', 
                           activity_level=activity_level, 
                           sleep_quality=sleep_quality, 
                           predicted_mood=predicted_mood)

if __name__ == '__main__':
    app.run(debug=True,port=5000,host='0.0.0.0')
