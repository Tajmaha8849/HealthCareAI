from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained model and feature columns
with open('personalized_medicine_model7.pkl', 'rb') as f:
    model = pickle.load(f)

with open('feature_columns7.pkl', 'rb') as f:
    feature_columns = pickle.load(f)

@app.route('/')
def home():
    return render_template('index7.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    df = pd.DataFrame([data])
    df = pd.get_dummies(df)
    df = df.reindex(columns=feature_columns, fill_value=0)
    prediction = model.predict(df)
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True,port=4004)
