from flask import Flask, render_template, request, redirect, url_for
from PIL import Image
import numpy as np
import tensorflow as tf
import os
import base64
import uuid

app = Flask(__name__)
model = tf.keras.models.load_model('model/shrub_model.h5')

UPLOAD_FOLDER = 'static/captured_images'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

labels = ['Hagunoy', 'Bayabas', 'Otot-otot']

def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))  # adjust based on your model input
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data_url = request.form['image']
        header, encoded = data_url.split(",", 1)
        img_bytes = base64.b64decode(encoded)

        filename = f"{uuid.uuid4().hex}.jpg"
        image_path = os.path.join(UPLOAD_FOLDER, filename)
        with open(image_path, "wb") as f:
            f.write(img_bytes)

        processed = preprocess_image(image_path)
        prediction = model.predict(processed)
        predicted_label = labels[np.argmax(prediction)]

        return render_template('result.html', prediction=predicted_label, image_path='/' + image_path)
    except Exception as e:
        return f"Prediction failed: {e}"

if __name__ == '__main__':
    app.run(debug=True)
