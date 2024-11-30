from flask import Flask, render_template, request
import numpy as np
from keras._tf_keras.keras.models import load_model
from keras._tf_keras.keras.preprocessing import image
import os

app = Flask(__name__)

# Load the trained model
model = load_model('model/model9.h5')

# Dictionary to map numerical labels to emotions
emotion_labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

# Function to preprocess the image for prediction
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(48, 48), color_mode='grayscale')
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array / 255.0  # Normalize pixel values

# Route to index page
@app.route('/')
def index():
    return render_template('index.html', prediction_result=None)

# Route to handle prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Check if the POST request has the file part
        if 'file' not in request.files:
            return render_template('index.html', prediction_result='No file part')
        
        file = request.files['file']

        # If user does not select file, browser also submit an empty part without filename
        if file.filename == '':
            return render_template('index.html', prediction_result='No selected file')

        if file:
            # Save the file to the uploads folder
            file_path = os.path.join('uploads', file.filename)
            file.save(file_path)

            # Preprocess the image
            processed_image = preprocess_image(file_path)

            # Make prediction
            prediction = model.predict(processed_image)
            predicted_label = np.argmax(prediction)
            predicted_emotion = emotion_labels[predicted_label]

            # Delete the uploaded file to save space
            os.remove(file_path)

            return render_template('index.html', prediction_result=predicted_emotion)

if __name__ == '__main__':
    app.run(debug=True,port=4007)
