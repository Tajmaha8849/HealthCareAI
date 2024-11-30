from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib

# Load models and label encoder
model_hp = joblib.load('recommendation_model_hp8.pkl')
model_dr = joblib.load('recommendation_model_dr8.pkl')
model_er = joblib.load('recommendation_model_er8.pkl')
le = joblib.load('label_encoder8.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index8.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    disease = data['disease']
    disease_encoded = le.transform([disease])[0]
    input_data = pd.DataFrame({'Disease': [disease_encoded]})
    precautions = model_hp.predict(input_data)[0]
    dietary_recommendations = model_dr.predict(input_data)[0]
    exercise_recommendations = model_er.predict(input_data)[0]
    return jsonify({
        'Health Precautions': precautions,
        'Dietary Recommendations': dietary_recommendations,
        'Exercise Recommendations': exercise_recommendations
    })


if __name__ == '__main__':
    app.run(debug=True,port=4005)
